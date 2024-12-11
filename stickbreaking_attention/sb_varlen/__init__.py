import math
import torch
import triton.language as tl
from torch.nn import functional as F

log2 = math.log(2)
inv_log2 = 1 / log2
ALLOW_TF32 = True

from .sb_varlen_fwd import sb_fwd
from .sb_varlen_bwd import sb_bwd


def calculate_programs_needed(cu_seqlens: torch.Tensor, BLOCK_SIZE):
    lens = cu_seqlens.clone()
    lens[1:] -= cu_seqlens[:-1]
    seq_num_programs = ((lens - 1) // BLOCK_SIZE) + 1 
    seq_program_offsets = torch.cumsum(seq_num_programs, dim=0)
    return seq_program_offsets


class StickBreakingAttention(torch.autograd.Function):

    FWD_BLOCK_M = 64
    FWD_BLOCK_N = 32
    BWD_BLOCK_M = 64
    BWD_BLOCK_N = 32
    
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, inv_temp):
        no_grad = not ctx.needs_input_grad[0]
        logit_scale = inv_temp
        BLOCK_M = StickBreakingAttention.FWD_BLOCK_M
        BLOCK_N = StickBreakingAttention.FWD_BLOCK_N
        seq_program_offsets = calculate_programs_needed(cu_seqlens, BLOCK_SIZE=BLOCK_M)
        o, rem, neg_log_acc = sb_fwd(
            q, k, v,
            cu_seqlens,
            seq_program_offsets + torch.arange(seq_program_offsets.size(0), device=q.device) + 1,
            logit_scale=inv_temp,
            no_grad=no_grad,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(q, k, v, neg_log_acc, cu_seqlens, seq_program_offsets)
        ctx.logit_scale = logit_scale
        return o, rem

    @staticmethod
    def backward(ctx, do, drem):
        logit_scale = ctx.logit_scale
        q, k, v, neg_log_acc, cu_seqlens, seq_program_offsets = ctx.saved_tensors
        BLOCK_M = StickBreakingAttention.BWD_BLOCK_M
        BLOCK_N = StickBreakingAttention.BWD_BLOCK_N

        if StickBreakingAttention.BWD_BLOCK_M != StickBreakingAttention.FWD_BLOCK_M:
            seq_program_offsets = calculate_programs_needed(cu_seqlens, BLOCK_SIZE=BLOCK_M)
        dq, dk, dv = sb_bwd(
            do, drem,
            q, k, v,
            cu_seqlens, seq_program_offsets,
            neg_log_acc, logit_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        return dq, dk, dv, None, None


def sb_attn_varlen(q, k, v, cu_seqlens, inv_temp=None, zero_start=True):
    if zero_start:
        assert cu_seqlens[0] == 0
        cu_seqlens = cu_seqlens[1:]
    if inv_temp is None:
        inv_temp = 1 / math.sqrt(q.size(-1))

    return sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens)


def sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens):
    return StickBreakingAttention.apply(q, k, v, cu_seqlens, inv_temp)