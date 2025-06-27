import math

import torch
import triton
import triton.language as tl

from ..utils import ALLOW_TF32, inv_log2
from .sb_varlen_fwd import compute_block, load_kv

from ..utils import custom_op
from torch.library import triton_op, wrap_triton

@triton.jit
def locked_add(Lock_ptr, Count_ptr, A_ptrs, a, B_ptrs, b, N_mask, NO_N_MASK, D_mask, NO_D_MASK: tl.constexpr,
               EVICTION_POLICY: tl.constexpr=tl.constexpr("")):
    lock_val = tl.atomic_cas(Lock_ptr, 0, 1)
    while lock_val == 1:
        lock_val = tl.atomic_cas(Lock_ptr, 0, 1)

    count = tl.load(Count_ptr, eviction_policy=EVICTION_POLICY)
    if NO_D_MASK and NO_N_MASK:
        if count == 0:
            tl.store(Count_ptr, 1, eviction_policy=EVICTION_POLICY)
            tl.store(A_ptrs, a, eviction_policy=EVICTION_POLICY)
            tl.store(B_ptrs, b, eviction_policy=EVICTION_POLICY)
        else:
            a += tl.load(A_ptrs, eviction_policy=EVICTION_POLICY)
            tl.store(A_ptrs, a, eviction_policy=EVICTION_POLICY)
            b += tl.load(B_ptrs, eviction_policy=EVICTION_POLICY)
            tl.store(B_ptrs, b, eviction_policy=EVICTION_POLICY)
    else:
        if N_mask is not None and D_mask is not None:
            mask = N_mask[:, None] & D_mask[None, :]
        elif D_mask is not None:
            mask = D_mask[None, :]
        else:
            mask = N_mask[:, None]

        if count == 0:
            tl.store(Count_ptr, 1, eviction_policy=EVICTION_POLICY)
            tl.store(A_ptrs, a, mask=mask, eviction_policy=EVICTION_POLICY)
            tl.store(B_ptrs, b, mask=mask, eviction_policy=EVICTION_POLICY)

        else:
            a += tl.load(A_ptrs, mask=mask, eviction_policy=EVICTION_POLICY)
            tl.store(A_ptrs, a, mask=mask, eviction_policy=EVICTION_POLICY)
            b += tl.load(B_ptrs, mask=mask, eviction_policy=EVICTION_POLICY)
            tl.store(B_ptrs, b, mask=mask, eviction_policy=EVICTION_POLICY)

    # tl.device_print("End locked add.")
    tl.atomic_xchg(Lock_ptr, 0)

@triton.jit
def _locked_add(Lock_ptr, Count_ptr, A_ptrs, a, B_ptrs, b, N_mask, NO_N_MASK, D_mask, NO_D_MASK: tl.constexpr,
                EVICTION_POLICY: tl.constexpr=""):
    # count = tl.load(Count_ptr, eviction_policy=EVICTION_POLICY)
    if NO_D_MASK:
        if NO_N_MASK:
            tl.atomic_add(A_ptrs, a)
            tl.atomic_add(B_ptrs, b)
        else:
            tl.atomic_add(A_ptrs, a, mask=N_mask[:, None])
            tl.atomic_add(B_ptrs, b, mask=N_mask[:, None])
    else:
        mask = N_mask[:, None] & D_mask[None, :]
        tl.atomic_add(A_ptrs, a, mask=mask)
        tl.atomic_add(B_ptrs, b, mask=mask)
 


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
                maxnreg=512, 
                # reg_dec_producer=1,
                # reg_inc_consumer=1
            )
        ]



@triton.autotune(
    configs=get_configs(),
    key=["token_size", "head_size"],
    reset_to_zero=["DK_ptr", "DV_ptr", "KV_Lock_ptr", "KV_Count_ptr"]
)
@triton.jit
def _backward(
    DO_ptr, stride_doh: tl.constexpr, stride_dom: tl.constexpr, stride_dod: tl.constexpr,
    DR_ptr, stride_drh: tl.constexpr, stride_drm: tl.constexpr,
    A_ptr, stride_ah: tl.constexpr, stride_am: tl.constexpr,
    Q_ptr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qd: tl.constexpr,
    K_ptr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    V_ptr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    DQ_ptr, stride_dqh: tl.constexpr, stride_dqm: tl.constexpr, stride_dqd: tl.constexpr,
    DK_ptr, stride_dkh: tl.constexpr, stride_dkn: tl.constexpr, stride_dkd: tl.constexpr,
    DV_ptr, stride_dvh: tl.constexpr, stride_dvn: tl.constexpr, stride_dvd: tl.constexpr,
    KV_Lock_ptr, KV_Count_ptr, stride_kvs: tl.constexpr, stride_kvh: tl.constexpr,
    CSL_ptr,
    logit_scale: tl.constexpr,
    batch_size: tl.constexpr,
    token_size: tl.constexpr,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CSL: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
    attend_current: tl.constexpr = False,
    shared_strides: tl.constexpr = False
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    seq_id = tl.program_id(0)
    fhead_id = tl.program_id(1)
    seq_alloc_prog_id = tl.program_id(2)
    num_seq_alloc_progs = tl.num_programs(2)

    if seq_id == 0:
        seq_start_offset = 0
    else:
        seq_start_offset = tl.load(CSL_ptr + seq_id - 1).to(tl.int32)

    seq_end_offset = tl.load(CSL_ptr + seq_id).to(tl.int32)
    seq_length = seq_end_offset - seq_start_offset
    num_seq_blocks = tl.cdiv(seq_length, BLOCK_M)

    seq_a_block_id = num_seq_blocks - seq_alloc_prog_id - 1
    seq_b_block_id = seq_alloc_prog_id - (num_seq_alloc_progs - num_seq_blocks)

    if shared_strides:
        stride_kh = stride_qh
        stride_vh = stride_qh
        stride_doh = stride_qh
        stride_dqh = stride_qh
        stride_dkh = stride_qh
        stride_dvh = stride_qh

        stride_dqm = stride_qm
        stride_kn = stride_qm
        stride_vn = stride_qm
        stride_dkn = stride_qm
        stride_dvn = stride_qm
        stride_dom = stride_qm
 
        stride_kd = None
        stride_vd = None
        stride_dqd = None
        stride_dkd = None
        stride_dvd = None
        stride_dod = None 



    if seq_a_block_id >= 0 or seq_b_block_id >= 0:
        # Universal stuff
        qk_scale = inv_log2 * logit_scale
        M_range = tl.arange(0, BLOCK_M).to(tl.int64)
        N_range = tl.arange(0, BLOCK_N).to(tl.int64)
        D_range = tl.arange(0, BLOCK_D).to(tl.int64)
        D_mask = D_range < head_size
        cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)
        

        if seq_a_block_id >= 0:
            head_id = fhead_id * 2
            DO_head_seq_ptr = DO_ptr + stride_doh * head_id + stride_dom * seq_start_offset
            DR_head_seq_ptr = DR_ptr + stride_drh * head_id + stride_drm * seq_start_offset
            A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
            Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
            K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
            V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
            DQ_head_seq_ptr = DQ_ptr + stride_dqh * head_id + stride_dqm * seq_start_offset
            DK_head_seq_ptr = DK_ptr + stride_dkh * head_id + stride_dkn * seq_start_offset
            DV_head_seq_ptr = DV_ptr + stride_dvh * head_id + stride_dvn * seq_start_offset
            KV_Lock_head_seq_ptr = KV_Lock_ptr + stride_kvs * seq_id + stride_kvh * head_id
            KV_Count_head_seq_ptr = KV_Count_ptr + stride_kvs * seq_id + stride_kvh * head_id

            _backward_one_row(
                seq_a_block_id, seq_length, qk_scale,
                M_range, N_range, D_range, D_mask,
                cm,
                DO_head_seq_ptr, stride_dom, stride_dod,
                DR_head_seq_ptr, stride_drm,
                A_head_seq_ptr, stride_am,
                Q_head_seq_ptr, stride_qm, stride_qd,
                K_head_seq_ptr, stride_kn, stride_kd,
                V_head_seq_ptr, stride_vn, stride_vd,
                DQ_head_seq_ptr, stride_dqm, stride_dqd,
                DK_head_seq_ptr, stride_dkn, stride_dkd,
                DV_head_seq_ptr, stride_dvn, stride_dvd,
                KV_Lock_head_seq_ptr,
                KV_Count_head_seq_ptr,
                logit_scale,
                BLOCK_D,
                NO_D_MASK,
                NO_M_MASK,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                acc_dtype,
                attend_current=attend_current,
                shared_strides=shared_strides
            )
        if seq_b_block_id >= 0 and fhead_id * 2 + 1 < num_heads:
            head_id = fhead_id * 2 + 1
            DO_head_seq_ptr = DO_ptr + stride_doh * head_id + stride_dom * seq_start_offset
            DR_head_seq_ptr = DR_ptr + stride_drh * head_id + stride_drm * seq_start_offset
            A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
            Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
            K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
            V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
            DQ_head_seq_ptr = DQ_ptr + stride_dqh * head_id + stride_dqm * seq_start_offset
            DK_head_seq_ptr = DK_ptr + stride_dkh * head_id + stride_dkn * seq_start_offset
            DV_head_seq_ptr = DV_ptr + stride_dvh * head_id + stride_dvn * seq_start_offset
            KV_Lock_head_seq_ptr = KV_Lock_ptr + stride_kvs * seq_id + stride_kvh * head_id
            KV_Count_head_seq_ptr = KV_Count_ptr +  stride_kvs * seq_id + stride_kvh * head_id
            _backward_one_row(
                seq_b_block_id, seq_length, qk_scale,
                M_range, N_range, D_range, D_mask,
                cm,
                DO_head_seq_ptr, stride_dom, stride_dod,
                DR_head_seq_ptr, stride_drm,
                A_head_seq_ptr, stride_am,
                Q_head_seq_ptr, stride_qm, stride_qd,
                K_head_seq_ptr, stride_kn, stride_kd,
                V_head_seq_ptr, stride_vn, stride_vd,
                DQ_head_seq_ptr, stride_dqm, stride_dqd,
                DK_head_seq_ptr, stride_dkn, stride_dkd,
                DV_head_seq_ptr, stride_dvn, stride_dvd,
                KV_Lock_head_seq_ptr,
                KV_Count_head_seq_ptr,
                logit_scale,
                BLOCK_D,
                NO_D_MASK,
                NO_M_MASK,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                acc_dtype,
                attend_current=attend_current,
                shared_strides=shared_strides
            )


@triton.jit
def _backward_one_row(
    seq_prog_id, seq_length, qk_scale,
    M_range, N_range, D_range, D_mask,
    cm,
    DO_head_seq_ptr, stride_dom: tl.constexpr, stride_dod: tl.constexpr,
    DR_head_seq_ptr, stride_drm: tl.constexpr,
    A_head_seq_ptr, stride_am: tl.constexpr,
    Q_head_seq_ptr, stride_qm: tl.constexpr, stride_qd: tl.constexpr,
    K_head_seq_ptr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    V_head_seq_ptr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    DQ_head_seq_ptr, stride_dqm: tl.constexpr, stride_dqd: tl.constexpr,
    DK_head_seq_ptr, stride_dkn: tl.constexpr, stride_dkd: tl.constexpr,
    DV_head_seq_ptr, stride_dvn: tl.constexpr, stride_dvd: tl.constexpr,
    KV_Lock_ptr, KV_Count_ptr,
    logit_scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
    is_compiling: tl.constexpr = False,
    attend_current: tl.constexpr = False,
    shared_strides: tl.constexpr = False
):
    # Loading thread information
    block_start_offset = BLOCK_M * seq_prog_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    NO_M_MASK = (block_start_offset + BLOCK_M - 1) < seq_length

    N_blk_idxs_start = 0
    N_blk_idxs = N_blk_idxs_start + N_range

    # Init pointers
    if shared_strides:
        MD_blk_idxs = stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :]
        ND_blk_idxs = stride_kn * N_blk_idxs[:, None] + stride_qd * D_range[None, :]
        # Inputs
        DO_blk_ptrs = DO_head_seq_ptr + MD_blk_idxs
        Q_blk_ptrs = Q_head_seq_ptr + MD_blk_idxs
        K_blk_ptrs = K_head_seq_ptr + ND_blk_idxs
        V_blk_ptrs = V_head_seq_ptr + ND_blk_idxs
        # Outputs
        DQ_blk_ptrs = DQ_head_seq_ptr + MD_blk_idxs
        DK_blk_ptrs = DK_head_seq_ptr + ND_blk_idxs
        DV_blk_ptrs = DV_head_seq_ptr + ND_blk_idxs
    else:
        # Inputs
        DO_blk_ptrs = DO_head_seq_ptr + (stride_dom * M_blk_idxs[:, None] + stride_dod * D_range[None, :])
        K_blk_ptrs = K_head_seq_ptr + (stride_kn * N_blk_idxs[:, None] + stride_kd * D_range[None, :])
        Q_blk_ptrs = Q_head_seq_ptr + (stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :])
        V_blk_ptrs = V_head_seq_ptr + (stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :])
        # Outputs
        DQ_blk_ptrs = DQ_head_seq_ptr + (stride_dqm * M_blk_idxs[:, None] + stride_dqd * D_range[None, :])
        DK_blk_ptrs = DK_head_seq_ptr + (stride_dkn * N_blk_idxs[:, None] + stride_dkd * D_range[None, :])
        DV_blk_ptrs = DV_head_seq_ptr + (stride_dvn * N_blk_idxs[:, None] + stride_dvd * D_range[None, :])

    A_blk_ptrs = A_head_seq_ptr + stride_am * M_blk_idxs
    DR_blk_ptrs = DR_head_seq_ptr + stride_drm * M_blk_idxs

    # --- Load band vectors ---
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
            do = tl.load(DO_blk_ptrs)
            dr = tl.load(DR_blk_ptrs)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None])
            do = tl.load(DO_blk_ptrs, mask=M_mask[:, None])
            dr = tl.load(DR_blk_ptrs, mask=M_mask)
    else:
        MD_mask = M_mask[:, None] & D_mask[None, :]
        q = tl.load(Q_blk_ptrs, mask=MD_mask)
        do = tl.load(DO_blk_ptrs, mask=MD_mask)
        dr = tl.load(DR_blk_ptrs, mask=M_mask)
    neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
    # --- End band vectors ---

    # Init accumulators
    neg_log_acc = neg_log_acc.to(dtype=acc_dtype)
    grad_prev_acc = tl.zeros((BLOCK_M,), dtype=acc_dtype)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=acc_dtype)

    # always multiple of number of blocks.
    iters = (block_start_offset + BLOCK_M) // BLOCK_N
    on_band_iters: tl.constexpr = BLOCK_M // BLOCK_N

    # Iterate only up to start of sequence
    for i in range(iters):
        NO_N_MASK = (N_blk_idxs_start + BLOCK_N - 1) < seq_length
        N_mask = N_blk_idxs < seq_length

        # --- Recompute block ---
        k, v = load_kv(
            K_blk_ptrs,
            V_blk_ptrs,
            N_mask=N_mask,
            NO_N_MASK=NO_N_MASK,
            D_mask=D_mask,
            NO_D_MASK=NO_D_MASK,
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
            is_compiling=is_compiling,
        )
        if not NO_M_MASK:
            neg_log_acc = tl.where(M_mask, neg_log_acc, 0.0)

        # --- Do gradient stuff ---

        grad_prev_acc, dq = accumulate_gradients(
            KV_Lock_ptr + i, KV_Count_ptr + i,
            DK_blk_ptrs, DV_blk_ptrs,
            log_om_beta, p, do, dr, q, k, v,
            grad_prev_acc, dq, cm,
            N_mask, NO_N_MASK,
            D_mask, NO_D_MASK,   
            logit_scale, ALLOW_TF32, 
        )
        # tl.store(GAI_head_seq_ptr + (((seq_prog_id + 1) * seq_prog_id) // 2) + i, seq_prog_id)
        # --- End gradient stuff ---

        N_blk_idxs += BLOCK_N
        N_blk_idxs_start += BLOCK_N
        if shared_strides:
            stride_size = BLOCK_N * stride_kn
            K_blk_ptrs += stride_size
            V_blk_ptrs += stride_size
            DK_blk_ptrs += stride_size
            DV_blk_ptrs += stride_size
        else:
            K_blk_ptrs += BLOCK_N * stride_kn
            V_blk_ptrs += BLOCK_N * stride_vn
            DK_blk_ptrs += BLOCK_N * stride_dkn
            DV_blk_ptrs += BLOCK_N * stride_dvn

    # dq = (logit_scale * dq).to(DQ_head_seq_ptr.type.element_ty)
    dq *= logit_scale
    if NO_D_MASK:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None])
    else:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None] & D_mask[None, :])

@triton.jit
def accumulate_gradients(
    KV_Lock_ptr, KV_Count_ptr, DK_blk_ptrs, DV_blk_ptrs,
    log_om_beta, p, do, dr, q, k, v,
    grad_prev_acc, dq, cm,
    N_mask, NO_N_MASK,
    D_mask, NO_D_MASK,
    logit_scale, ALLOW_TF32, 
):
    block_dv = tl.dot(tl.trans(p), do.to(p.dtype), allow_tf32=ALLOW_TF32)
    att_dA = p * (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None])
    cumul_att_dA =  tl.dot(att_dA.to(cm.dtype), tl.trans(cm), allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
    grad_prev_acc += tl.sum(att_dA, axis=1)
    beta = 1 - tl.exp2(log_om_beta)  # 180 -> 175
    dqk = att_dA - beta * cumul_att_dA

    block_dk = tl.dot(tl.trans(dqk).to(q.dtype), q, allow_tf32=ALLOW_TF32) * logit_scale
    dq = tl.dot(dqk.to(k.dtype), k, acc=dq, allow_tf32=ALLOW_TF32)

    # dqk = p
    # block_dv = tl.dot(tl.trans(dqk).to(do.dtype), do, allow_tf32=ALLOW_TF32)
    # dqk *= (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None]).to(do.dtype)
    # cumul_att_dA = tl.dot(dqk, tl.trans(cm), allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
    # grad_prev_acc += tl.sum(dqk, axis=1)
    # neg_beta = tl.exp2(log_om_beta) - 1
    #     # dqk = dqk + neg_beta * cumul_att_dA
    # dqk += neg_beta * cumul_att_dA
    # dqk = dqk.to(k.dtype)
    # block_dk = tl.dot(tl.trans(dqk), q, allow_tf32=ALLOW_TF32) * logit_scale
    # dq = tl.dot(dqk, k, acc=dq, allow_tf32=ALLOW_TF32)
    locked_add(
        KV_Lock_ptr, KV_Count_ptr,
        DK_blk_ptrs, block_dk,
        DV_blk_ptrs, block_dv,
        N_mask, NO_N_MASK,
        D_mask, NO_D_MASK,
    )

    return grad_prev_acc, dq


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
    num_folded_heads = triton.cdiv(num_heads, 2)
    num_seq_blocks = triton.cdiv(max_seqlens, BLOCK_M) + 1

    N_count = num_seq_blocks * (BLOCK_M // BLOCK_N)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dkdv_lock = torch.zeros((num_sequences, num_heads, N_count), dtype=torch.int32, device=q.device)
    dkdv_count = torch.zeros((num_sequences, num_heads, N_count), dtype=torch.int32, device=q.device)


    _compileable_backward(
        do, dr, q, k, v,
        cu_seqlens,
        neg_log_acc,
        dkdv_lock,
        dkdv_count,
        logit_scale,
        batch_size,
        num_heads,
        token_size,
        dim_size,
        dq, dk, dv,
        # dkdv_lock,
        # dkdv_count,
        num_sequences,
        num_folded_heads,
        num_seq_blocks,
        attend_current=attend_current,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

    )
    return dq, dk, dv


# @custom_op("varlen_bwd", mutates_args={"dq", "dk", "dv"})
@triton_op("sb_attn::varlen_bwd", mutates_args={"dq", "dk", "dv", "dkdv_lock", "dkdv_count"})
def _compileable_backward(
    do: torch.Tensor,
    dr: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    neg_log_acc: torch.Tensor,
    dkdv_lock: torch.Tensor,
    dkdv_count: torch.Tensor,
    logit_scale: float,
    batch_size: int,
    num_heads: int,
    token_size: int,
    dim_size: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    # dkdv_lock: torch.Tensor,
    # dkdv_count: torch.Tensor,
    num_sequences: int,
    num_folded_heads: int,
    num_seq_blocks: int,
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

    none_stride = (None, None, None)
    shared_strides = False and (
        (v_stride == k_stride) and 
        (q_stride == k_stride) and 
        (dk_stride == k_stride) and
        (do_stride == dk_stride) and
        (dv_stride == dq_stride)
    )
    if shared_strides:
        k_stride = none_stride
        v_stride = none_stride
        assert (dv_stride == dk_stride) and (dq_stride ==  do_stride)
        do_stride = none_stride
        dq_stride = none_stride
        dk_stride = none_stride
        dv_stride = none_stride

    wrap_triton(_backward)[num_sequences, num_folded_heads, num_seq_blocks](
        # DO_ptr, stride_doh, stride_dom, stride_dod,
        do, do_stride[0], do_stride[1], do_stride[2],
        # DR_ptr, stride_drh, stride_drm,
        dr, dr.stride(0), dr.stride(1),
        # A_ptr, stride_ah, stride_am,
        neg_log_acc, neg_log_acc.stride(0), neg_log_acc.stride(1),
        # Q_ptr, stride_qh, stride_qm, stride_qd,
        q, q_stride[0], q_stride[1], q_stride[2],
        # K_ptr, stride_kh, stride_kn, stride_kd,
        k, k_stride[0], k_stride[1], k_stride[2],
        # V_ptr, stride_vh, stride_vn, stride_vd,
        v, v_stride[0], v_stride[1], v_stride[2],
        # DQ_ptr, stride_dqh, stride_dqm, stride_dqd,
        dq, dq_stride[0], dq_stride[1], dq_stride[2],
        # DK_ptr, stride_dkh, stride_dkn, stride_dkd,
        dk, dk_stride[0], dk_stride[1], dk_stride[2],
        # DV_ptr, stride_dvh, stride_dvn, stride_dvd,
        dv, dv_stride[0], dv_stride[1], dv_stride[2],
        # KV_Lock_ptr, KV_Count_ptr, stride_kvl,
        dkdv_lock,
        dkdv_count,
        dkdv_lock.stride(0),
        dkdv_lock.stride(1),
        cu_seqlens,
        logit_scale=logit_scale,
        batch_size=batch_size,
        token_size=token_size,
        head_size=dim_size,
        num_heads=num_heads,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_CSL=triton.next_power_of_2(batch_size),
        NO_D_MASK=BLOCK_D == dim_size,
        NO_M_MASK=False,
        NO_N_MASK=False,
        ALLOW_TF32=ALLOW_TF32,
        inv_log2=inv_log2,
        attend_current=attend_current,
        shared_strides=shared_strides,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    # print("sb done.")
    del dkdv_lock
    del dkdv_count


