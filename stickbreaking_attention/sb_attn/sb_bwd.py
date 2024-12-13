import math
import torch
import triton
import triton.language as tl
from ..sb_varlen import inv_log2, ALLOW_TF32
from ..sb_varlen.sb_varlen_fwd import load_kv, compute_block
from ..sb_varlen.sb_varlen_bwd import _backward_one_row


   
def get_configs():
    return [
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [8]
        for w in [4]
    ]
@triton.autotune(configs=get_configs(), key=["token_size", "head_size"], )
                 # reset_to_zero=["DK_ptr", "DV_ptr"])
@triton.jit()
def _backward(
    DO_ptr, stride_dob, stride_doh, stride_dom: tl.constexpr, stride_dod: tl.constexpr,
    DR_ptr, stride_drb, stride_drh, stride_drm: tl.constexpr,
    A_ptr, stride_ab, stride_ah, stride_am: tl.constexpr,
    Q_ptr, stride_qb, stride_qh, stride_qm: tl.constexpr, stride_qd: tl.constexpr,
    K_ptr, stride_kb, stride_kh, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    V_ptr, stride_vb, stride_vh, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    DQ_ptr, stride_dqb, stride_dqh, stride_dqm: tl.constexpr, stride_dqd: tl.constexpr,
    DK_ptr, stride_dkb, stride_dkh, stride_dkn: tl.constexpr, stride_dkd: tl.constexpr,
    DV_ptr, stride_dvb, stride_dvh, stride_dvn: tl.constexpr, stride_dvd: tl.constexpr,
    KV_Lock_ptr, KV_Count_ptr, stride_kvb: tl.constexpr, stride_kvl: tl.constexpr,
    logit_scale,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    batch_id = tl.program_id(0)
    head_pid = tl.program_id(1)
    prog_id = tl.program_id(2)
    # Universal stuff
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    head_id = head_pid
    seq_prog_id = prog_id
    seq_length = token_size

    DO_head_seq_ptr = DO_ptr + stride_dob * batch_id + stride_doh * head_id
    DR_head_seq_ptr = DR_ptr + stride_drb * batch_id + stride_drh * head_id
    A_head_seq_ptr = A_ptr + stride_ab * batch_id + stride_ah * head_id
    Q_head_seq_ptr = Q_ptr + stride_qb * batch_id + stride_qh * head_id
    K_head_seq_ptr = K_ptr + stride_kb * batch_id + stride_kh * head_id
    V_head_seq_ptr = V_ptr + stride_vb * batch_id + stride_vh * head_id
    DQ_head_seq_ptr = DQ_ptr + stride_dqb * batch_id + stride_dqh * head_id
    DK_head_seq_ptr = DK_ptr + stride_dkb * batch_id + stride_dkh * head_id
    DV_head_seq_ptr = DV_ptr + stride_dvb * batch_id + stride_dvh * head_id
    KV_Lock_head_seq_ptr =  KV_Lock_ptr  + stride_kvb * batch_id + stride_kvl * head_id 
    KV_Count_head_seq_ptr = KV_Count_ptr + stride_kvb * batch_id + stride_kvl * head_id 
    _backward_one_row(
        seq_prog_id, seq_length,
        qk_scale,
        M_range,
        N_range,
        D_range, D_mask, cm,
        DO_head_seq_ptr, stride_dom, stride_dod,
        DR_head_seq_ptr, stride_drm,
        A_head_seq_ptr, stride_am,
        Q_head_seq_ptr, stride_qm, stride_qd,
        K_head_seq_ptr, stride_kn, stride_kd,
        V_head_seq_ptr, stride_vn, stride_vd,
        DQ_head_seq_ptr, stride_dqm, stride_dqd,
        DK_head_seq_ptr, stride_dkn, stride_dkd,
        DV_head_seq_ptr, stride_dvn, stride_dvd,
        KV_Lock_head_seq_ptr, KV_Count_head_seq_ptr,
        logit_scale,
        BLOCK_D,
        NO_D_MASK,
        NO_M_MASK,
        ALLOW_TF32,
        BLOCK_M,
        BLOCK_N,
        acc_dtype,
    )

def _bwd(do, dr, q, k, v,  neg_log_acc, logit_scale, BLOCK_M=64, BLOCK_N=32, strides=None):
    with torch.cuda.device(q.device):
        batch_size, num_heads, token_size, dim_size = q.size()
        BLOCK_D = triton.next_power_of_2(dim_size)
        M_count = triton.cdiv(token_size, BLOCK_M)
        N_count = triton.cdiv(token_size, BLOCK_N)
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        M_count = triton.cdiv(token_size, BLOCK_M)
        N_count = M_count * (BLOCK_M // BLOCK_N)
        dkdv_lock = torch.zeros((batch_size, num_heads, N_count), dtype=torch.int32, device=q.device)
        dkdv_count = torch.zeros((batch_size, num_heads, N_count), dtype=torch.bool, device=q.device)
        if strides is not None:
            q_stride, k_stride, v_stride, o_stride, rem_stride, neg_log_acc_stride = strides
        else:
            q_stride = q.stride()
            k_stride = k.stride()
            v_stride = v.stride()
            o_stride = do.stride()
            rem_stride = dr.stride()
            neg_log_acc_stride = neg_log_acc.stride()

        do = do.contiguous()
        dr = dr.contiguous()
        do_stride = (do.size(-3) * do.size(-2) * do.size(-1),
                     do.size(-2) * do.size(-1),
                     do.size(-1), 1)
        dr_stride = (dr.size(-2) * dr.size(-1), dr.size(-1), 1)

        _backward[batch_size, num_heads, M_count](
            do, do_stride[0], do_stride[1], do_stride[2], do_stride[3],
            dr, dr_stride[0], dr_stride[1], dr_stride[2],
            neg_log_acc, neg_log_acc_stride[0], neg_log_acc_stride[1], neg_log_acc_stride[2],
            q, q_stride[0], q_stride[1], q_stride[2], q_stride[3],
            k, k_stride[0], k_stride[1], k_stride[2], k_stride[3],
            v, v_stride[0], v_stride[1], v_stride[2], v_stride[3],
            dq, q_stride[0], q_stride[1], q_stride[2], q_stride[3],
            dk, k_stride[0], k_stride[1], k_stride[2], k_stride[3],
            dv, v_stride[0], v_stride[1], v_stride[2], v_stride[3],
            dkdv_lock, dkdv_count, num_heads * N_count, N_count,
            logit_scale=logit_scale,
            batch_size=batch_size,
            token_size=token_size,
            head_size=dim_size,
            num_heads=num_heads,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            NO_D_MASK=BLOCK_D == dim_size,
            NO_M_MASK=(token_size % BLOCK_M) == 0,
            NO_N_MASK=(token_size % BLOCK_N) == 0,
            ALLOW_TF32=ALLOW_TF32,
            inv_log2=inv_log2,
            acc_dtype=tl.float32
        )
        return dq, dk, dv
