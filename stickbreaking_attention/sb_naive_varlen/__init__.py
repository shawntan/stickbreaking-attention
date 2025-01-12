from .sb_varlen_fwd import varlen_fwd
from .sb_varlen_bwd import varlen_bwd
import math

import torch
import triton.language as tl
from torch.nn import functional as F


FWD_BLOCK_M: tl.constexpr = 64
FWD_BLOCK_N: tl.constexpr = 32
BWD_BLOCK_M: tl.constexpr = 64
BWD_BLOCK_N: tl.constexpr = 32


class StickBreakingAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, log_forget, cu_seqlens, max_seqlens, inv_temp, attend_current):
        no_grad = not ctx.needs_input_grad[0]
        logit_scale = inv_temp
        o, rem, neg_log_acc = varlen_fwd(
            q,
            k,
            v,
            log_forget,
            cu_seqlens,
            max_seqlens,
            logit_scale=inv_temp,
            attend_current=attend_current,
            no_grad=no_grad,
            BLOCK_M=FWD_BLOCK_M,
            BLOCK_N=FWD_BLOCK_N,
        )
        ctx.save_for_backward(q, k, v, log_forget, neg_log_acc, cu_seqlens)
        ctx.logit_scale = logit_scale
        ctx.max_seqlens = max_seqlens
        ctx.attend_current = attend_current
        return o, rem

    @staticmethod
    def backward(ctx, do, drem):
        logit_scale = ctx.logit_scale
        max_seqlens = ctx.max_seqlens
        attend_current = ctx.attend_current
        q, k, v, log_forget, neg_log_acc, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, dlf = varlen_bwd(
            do, drem,
            q, k, v, log_forget,
            cu_seqlens,
            max_seqlens,
            neg_log_acc,
            logit_scale,
            attend_current=attend_current,
            BLOCK_M=BWD_BLOCK_M,
            BLOCK_N=BWD_BLOCK_N,
        )

        # if (
        #     torch.isinf(dq).any() or torch.isnan(dq).any() or
        #     torch.isinf(dk).any() or torch.isnan(dk).any() or
        #     torch.isinf(dv).any() or torch.isnan(dv).any() or
        #     torch.isinf(dlf).any() or torch.isnan(dlf).any()
        # ):

        #     rank = torch.distributed.get_rank()
        #     torch.save(
        #         (
        #             do, drem,
        #             q, k, v, log_forget,
        #             cu_seqlens,
        #             max_seqlens,
        #             neg_log_acc,
        #             logit_scale,
        #             attend_current,
        #             BWD_BLOCK_M,
        #             BWD_BLOCK_N,
        #         ),
        #         open(f'naninfbwd_{rank}.pkl', 'wb')
        #     )
        #     exit()

        return dq, dk, dv, dlf, None, None, None, None


def sb_attn_varlen(q, k, v, log_forget,
                   cu_seqlens, max_seqlens, inv_temp=None, zero_start=True, attend_current=False):
    if zero_start:
        assert cu_seqlens[0] == 0
        cu_seqlens = cu_seqlens[1:]
    if inv_temp is None:
        inv_temp = 1 / math.sqrt(q.size(-1))
    return sb_attn_varlen_(q, k, v, log_forget,
                           inv_temp, cu_seqlens, max_seqlens, attend_current)


def sb_attn_varlen_(q, k, v, log_forget,
                    inv_temp, cu_seqlens, max_seqlens, attend_current):
    return StickBreakingAttention.apply(q, k, v, log_forget,
                                        cu_seqlens, max_seqlens, inv_temp, attend_current)
