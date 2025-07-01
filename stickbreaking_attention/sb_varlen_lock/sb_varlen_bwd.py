import math

import torch
import triton
import triton.language as tl

from ..utils import ALLOW_TF32, inv_log2
from .sb_varlen_fwd import compute_block, load_kv

from ..utils import custom_op
from torch.library import triton_op, wrap_triton


def get_configs():
    if False:
        return [
            triton.Config(
                {},
                num_stages=s, num_warps=w, maxnreg=mnr, 
                reg_dec_producer=rdp,
                reg_inc_consumer=ric
            )
            # for mb in [32]
            # for nb in [32]
            for s in [9]
            for w in [4]
            for rdp in [2]
            for ric in [16]
            for mnr in [2048]
            # if nb <= mb and mb % nb == 0 
        ]
        # for mb in [64]
        # for nb in [32]
        # for s in [4]
        # for w in [4]]
    else:
        # num_warps: 4, num_ctas: 1, num_stages: 4, num_buffers_warp_spec: 0, num_consumer_groups: 0,
        # reg_dec_producer: 8, reg_inc_consumer: 2, maxnreg: 256; 
        return [
            triton.Config(
                {},
                num_stages=4, num_warps=4,
                maxnreg=1024, 
                reg_dec_producer=8,
                reg_inc_consumer=2
            )
        ]



@triton.autotune(
    configs=get_configs(),
    key=["token_size", "head_size"],
    reset_to_zero=["DK_ptr", "DV_ptr"]
)
@triton.jit
def _backward(
        DO_ptr, DQ_ptr, DK_ptr, DV_ptr, Q_ptr, K_ptr, V_ptr,
        head_stride: tl.constexpr, length_stride: tl.constexpr,
        DR_ptr, drh_stride: tl.constexpr, drm_stride: tl.constexpr,
        A_ptr, ah_stride: tl.constexpr, am_stride: tl.constexpr,
        # KV_Lock_ptr, KV_Count_ptr, stride_kvb: tl.constexpr, stride_kvh: tl.constexpr,
        # Att_ptr, atth_stride, attm_stride,
        CSL_ptr,
        logit_scale: tl.constexpr,
        attend_current: tl.constexpr,
        batch_size: tl.constexpr,
        token_size: tl.constexpr,
        head_size: tl.constexpr,
        num_heads: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_CSL: tl.constexpr,
        NO_M_MASK: tl.constexpr, NO_N_MASK: tl.constexpr, NO_D_MASK: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        inv_log2: tl.constexpr,
        acc_dtype: tl.constexpr = tl.float32,
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)
    if seq_id == 0:
        seq_start_offset = 0
    else:
        seq_start_offset = tl.load(CSL_ptr + seq_id - 1).to(tl.int32)
    seq_end_offset = tl.load(CSL_ptr + seq_id).to(tl.int32)
    seq_length = seq_end_offset - seq_start_offset
    M_blk_offset = BLOCK_M * block_id
    num_seq_blocks = tl.cdiv(seq_length, BLOCK_M)

    if M_blk_offset < seq_length:
        qk_scale = inv_log2 * logit_scale
        M_range = tl.arange(0, BLOCK_M).to(tl.int64)
        N_range = tl.arange(0, BLOCK_N).to(tl.int64)
        D_range = tl.arange(0, BLOCK_D).to(tl.int64)
        cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

        M_blk_idxs = M_blk_offset + M_range
        M_mask = M_blk_idxs < seq_length
        D_mask = D_range < head_size
        MD_mask = M_mask[:, None] & D_mask[None, :]
        N_blk_idxs = N_range
        ND_blk_idxs = length_stride * N_blk_idxs[:, None] + D_range[None, :]

        batch_head_offset = head_stride * head_id + length_stride * seq_start_offset
        MD_blk_idxs = length_stride * M_blk_idxs[:, None] + D_range[None, :]

        DO_blk_ptrs = DO_ptr + batch_head_offset + MD_blk_idxs
        Q_blk_ptrs = Q_ptr + batch_head_offset + MD_blk_idxs
        DQ_blk_ptrs = DQ_ptr + batch_head_offset + MD_blk_idxs
        K_blk_ptrs = K_ptr + batch_head_offset + ND_blk_idxs
        V_blk_ptrs = V_ptr + batch_head_offset + ND_blk_idxs
        DK_blk_ptrs = DK_ptr + batch_head_offset + ND_blk_idxs
        DV_blk_ptrs = DV_ptr + batch_head_offset + ND_blk_idxs

        DR_blk_ptrs = DR_ptr + drh_stride * head_id + drm_stride * (seq_start_offset + M_blk_idxs)
        A_blk_ptrs = A_ptr + ah_stride * head_id + am_stride * (seq_start_offset + M_blk_idxs)

        # KV_Lock_ptr = KV_Lock_ptr + stride_kvb * seq_id + stride_kvh * head_id
        # KV_Count_ptr = KV_Count_ptr + stride_kvb * seq_id + stride_kvh * head_id

        # Init accumulators
        q = tl.load(Q_blk_ptrs, mask=MD_mask)
        do = tl.load(DO_blk_ptrs, mask=MD_mask)
        dr = tl.load(DR_blk_ptrs, mask=M_mask)
        neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask).to(dtype=acc_dtype)
        grad_prev_acc = tl.zeros((BLOCK_M,), dtype=acc_dtype)
        dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=acc_dtype)
        iters = (M_blk_offset + BLOCK_M) // BLOCK_N

        for i in range(iters):
            N_mask = N_blk_idxs < seq_length
            ND_mask = N_mask[:, None] & D_mask[None, :]
            k, v = load_kv(
                K_blk_ptrs,
                V_blk_ptrs,
                N_mask=N_mask, NO_N_MASK=False,
                D_mask=D_mask, NO_D_MASK=False,
            )
            if attend_current:
                block_mask = M_blk_idxs[:, None] >= N_blk_idxs[None, :]
            else:
                block_mask = M_blk_idxs[:, None] > N_blk_idxs[None, :]

            p, log_om_beta, neg_log_acc = compute_block(
                q, k, qk_scale,
                neg_log_acc,
                cm,
                block_mask=block_mask,
                ALLOW_TF32=ALLOW_TF32,
                backward=True,
                is_compiling=False,
            )
            # tl.store(Att_ptr + atth_stride * head_id +  attm_stride * M_blk_idxs[:, None] + N_blk_idxs[None, :], p)

            # block_dv = tl.dot(tl.trans(p), do.to(p.dtype), allow_tf32=ALLOW_TF32)
            # tl.atomic_add(DV_blk_ptrs, block_dv, mask=ND_mask, sem='relaxed')

            # att_dA = p * (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None])
            # cumul_att_dA = tl.dot(att_dA.to(cm.dtype), tl.trans(cm), allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
            # grad_prev_acc += tl.sum(att_dA, axis=1)
            # beta = tl.exp2(log_om_beta) - 1
            # dqk = att_dA + beta * cumul_att_dA
            # dqk *= logit_scale 
            # block_dk = tl.dot(tl.trans(dqk).to(q.dtype), q, allow_tf32=ALLOW_TF32)
            # tl.atomic_add(DK_blk_ptrs, block_dk, mask=ND_mask, sem='relaxed')
            dqk = p.to(k.dtype)
            block_dv = tl.dot(tl.trans(dqk), do, allow_tf32=ALLOW_TF32)
            tl.atomic_add(DV_blk_ptrs, block_dv, mask=ND_mask, sem='relaxed')
            dqk *= (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None]).to(do.dtype)
            cumul_att_dA = tl.dot(dqk.to(cm.dtype), tl.trans(cm), allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
            grad_prev_acc += tl.sum(dqk, axis=1)
            neg_beta = tl.exp2(log_om_beta) - 1
            dqk += (neg_beta * cumul_att_dA).to(q.dtype)
            block_dk = tl.dot(tl.trans(dqk), q, allow_tf32=ALLOW_TF32) * logit_scale
            tl.atomic_add(DK_blk_ptrs, block_dk, mask=ND_mask, sem='relaxed')

            dq = tl.dot(dqk, k, acc=dq, allow_tf32=ALLOW_TF32)

            # Striding is working for k,v,dk,dv
            # TODO if block_id == num_seq_blocks - 1:
            # TODO     tl.store(DK_blk_ptrs, k, mask=ND_mask)
            # TODO     tl.store(DV_blk_ptrs, v, mask=ND_mask)


            N_blk_idxs += BLOCK_N
            K_blk_ptrs += BLOCK_N * length_stride
            V_blk_ptrs += BLOCK_N * length_stride
            DK_blk_ptrs += BLOCK_N * length_stride
            DV_blk_ptrs += BLOCK_N * length_stride

        dq *= logit_scale
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None] & D_mask[None, :])

        # Striding is working for q,dq
        # TODO tl.store(DQ_blk_ptrs, q, mask=M_mask[:, None] & D_mask[None, :])


def varlen_bwd(
    do: torch.Tensor,
    dr: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlens: int,
    neg_log_acc: torch.Tensor,
    logit_scale,
    attend_current=False,
    BLOCK_M=64,
    BLOCK_N=32,
):
    batch_size = cu_seqlens.size(0)
    num_heads, token_size, dim_size = q.size()
    if logit_scale is None:
        logit_scale = 1 / math.sqrt(dim_size)
    num_sequences = batch_size
    num_seq_blocks = triton.cdiv(max_seqlens, BLOCK_M)
    # N_count = num_seq_blocks * (BLOCK_M // BLOCK_N)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    # dkdv_lock = torch.zeros((num_sequences, num_heads, N_count), dtype=torch.int32, device=q.device)
    # dkdv_count = torch.zeros((num_sequences, num_heads, N_count), dtype=torch.int32, device=q.device)


    _compileable_backward(
        do, dr, q, k, v,
        cu_seqlens,
        neg_log_acc,
        logit_scale,
        batch_size,
        num_heads,
        token_size,
        dim_size,
        dq, dk, dv,
        # dkdv_lock,
        # dkdv_count,
        num_sequences,
        num_seq_blocks,
        attend_current=attend_current,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

    )
    # del dkdv_lock
    # del dkdv_count
    return dq, dk, dv


@custom_op("varlen_bwd", mutates_args={"dq", "dk", "dv"})
# @triton_op("sb_attn::varlen_bwd", mutates_args={"dq", "dk", "dv", "dkdv_lock", "dkdv_count"})
def _compileable_backward(
    do: torch.Tensor, dr: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens: torch.Tensor, neg_log_acc: torch.Tensor,
    logit_scale: float,
    batch_size: int, num_heads: int, token_size: int, dim_size: int,
    dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
    num_sequences: int, num_seq_blocks: int,
    attend_current: bool = False,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
) -> None:
    BLOCK_D = triton.next_power_of_2(dim_size)
    q_stride = q.stride()
    k_stride = k.stride()
    v_stride = v.stride()
    do_stride = do.stride()
    dq_stride = dq.stride()
    dk_stride = dk.stride()
    dv_stride = dv.stride()
    
    # att = torch.zeros((q.size(0), q.size(1), k.size(1)), dtype=torch.float32, device=q.device)

    assert q_stride == k_stride and k_stride == v_stride , "Forward strides need to be same"
    assert (
        v_stride == do_stride and
        dq_stride == dk_stride and
        dk_stride == dv_stride and
        dv_stride == do_stride
    ), "Backwards needs equal strides"
    assert q_stride[-1] == 1, "Final stride should be contiguous"
    # print("strides", k_stride)
    head_stride, length_stride, _ = q_stride
    _backward[num_sequences, num_heads, num_seq_blocks](
        do, dq, dk, dv, q, k, v, head_stride, length_stride,
        # DR_ptr, stride_drh, stride_drm,
        dr, dr.stride(0), dr.stride(1),
        # A_ptr, stride_ah, stride_am,
        neg_log_acc, neg_log_acc.stride(0), neg_log_acc.stride(1),
        cu_seqlens,
        logit_scale=logit_scale,
        batch_size=batch_size,
        token_size=token_size,
        head_size=dim_size,
        num_heads=num_heads,
        BLOCK_CSL=triton.next_power_of_2(batch_size),
        attend_current=attend_current,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        NO_D_MASK=BLOCK_D == dim_size,
        NO_M_MASK=False,
        NO_N_MASK=False,
        ALLOW_TF32=ALLOW_TF32,
        inv_log2=inv_log2,
    )
    # from matplotlib import pyplot as plt
    # plt.imshow(att[0].cpu().numpy(), interpolation='none', vmax=1, vmin=0)
    # plt.savefig('att.png')



