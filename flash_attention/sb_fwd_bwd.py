# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.profiler import cached
from tilelang.autotuner import *
import tilelang.language as T
import argparse
import math
# from stickbreaking_attention.sb_ref import stickbreaking
# for reference
def stickbreaking(q, k, v, mask, cum_weight):
    """
    Stick-breaking attention weights.
    """
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

    original_dtype = logits.dtype

    logits = logits.float()
    log_om_beta = -F.softplus(logits).masked_fill(mask, 0).to(original_dtype)
    cumsum_log_om_beta = torch.einsum("bhij,jk->bhik", log_om_beta, cum_weight) + log_om_beta
    log_att = logits + cumsum_log_om_beta
    att = torch.exp(log_att).masked_fill(mask, 0).to(original_dtype)

    return att, att @ v, 1 - att.sum(dim=-1)



def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N,
                  dtype):
    # scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    scale = 1.0 / math.sqrt(dim)
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"
    @T.prim_func
    def flash_fwd(
            Q: T.Buffer(shape, dtype),  # type: ignore
            K: T.Buffer(shape, dtype),  # type: ignore
            V: T.Buffer(shape, dtype),  # type: ignore
            A: T.Buffer([batch, heads, seq_len, seq_len]), # type: ignore
            cm_out: T.Buffer([block_N, block_N]), # type: ignore
            Output: T.Buffer(shape, dtype),  # type: ignore
            lse: T.Buffer([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):

            cm = T.alloc_shared([block_N, block_N], dtype)
            for i, j in T.Parallel(block_N, block_N):
                cm[i, j] = T.if_then_else(i < j, 0., 1.)

            cm_cast = T.alloc_shared([block_N, block_N], accum_dtype)
            T.copy(cm, cm_cast)
            T.copy(cm_cast, cm_out)
            

            # allocate shared mem
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)

            # block fragments
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            acc_log_om_beta = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(acc_log_om_beta, 0)

            qk_scale = T.alloc_fragment([block_M, block_N], accum_dtype)
            log_om_beta = T.alloc_fragment([block_M, block_N], accum_dtype)
            tile_log_om_beta_sum = T.alloc_fragment([block_M], accum_dtype)
            cum_log_om_beta = T.alloc_fragment([block_M, block_N], accum_dtype)
            log_om_beta_cast = T.alloc_fragment([block_M, block_N], dtype)
            log_p = T.alloc_fragment([block_M, block_N], accum_dtype)
            p = T.alloc_fragment([block_M, block_N], accum_dtype)
            p_cast = T.alloc_fragment([block_M, block_N], dtype)

            T.fill(cum_log_om_beta, 0.)
            loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
            for k_rev in T.Pipelined(loop_range, num_stages=1):
                k = loop_range - k_rev - 1
                # Load K and V
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)

                T.clear(qk_scale)
                
                T.gemm(Q_shared, K_shared, qk_scale, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, block_N):
                    qk_scale[i, j] = qk_scale[i, j] * scale
                    log_om_beta[i, j] = -T.if_then_else(
                        qk_scale[i, j] > 15.0,
                        qk_scale[i, j],
                        T.log(1 + T.exp(qk_scale[i, j]))
                    )

                    if is_causal:
                        log_om_beta[i, j] = T.if_then_else(
                            bx * block_M + i <= k * block_N + j,
                            0.,
                            log_om_beta[i, j],
                        )
                    
                T.fill(cum_log_om_beta, 0.)
                T.copy(log_om_beta, log_om_beta_cast)
                T.gemm(log_om_beta_cast, cm, cum_log_om_beta , policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, block_N):
                    cum_log_om_beta[i, j] = cum_log_om_beta[i, j] + acc_log_om_beta[i]

                T.reduce_sum(log_om_beta, tile_log_om_beta_sum, dim=1)
                for i in T.Parallel(block_M):
                    acc_log_om_beta[i] = acc_log_om_beta[i] + tile_log_om_beta_sum[i]

                for i, j in T.Parallel(block_M, block_N):
                    # cum_log_om_beta[i, j] += acc_log_om_beta[i]
                    # cum_log_om_beta[i, j] = cum_log_om_beta[i, j] + acc_log_om_beta[i]
                    log_p[i, j] = qk_scale[i, j] + cum_log_om_beta[i, j]
                    p[i, j] = T.exp(log_p[i, j])
                    # p[i, j] = T.exp(log_om_beta[i, j])
                    if is_causal:
                        p[i, j] = T.if_then_else(
                            bx * block_M + i <= k * block_N + j,
                            0.,
                            p[i, j],
                        )
                
                # T.copy(cum_log_om_beta[:, 0], acc_log_om_beta)
                T.copy(p, A[bz, by, bx * block_M:(bx + 1) * block_M,  k * block_N:(k + 1) * block_N])

                T.copy(p, p_cast)
                T.gemm(p_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
            T.copy(acc_log_om_beta, lse[bz, by, bx * block_M:(bx + 1) * block_M])

    return flash_fwd


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k, v, causal):
        BATCH, N_CTX, H, D_HEAD = q.shape
        block_M = 64
        block_N = 64 if D_HEAD <= 128 else 32
        dtype_str = str(q.dtype).split('.')[-1]
        mod = cached(
            flashattn_fwd, [3, 4, 5, 6],
            BATCH, H, N_CTX, D_HEAD, causal, block_M, block_N, dtype_str
        )
        A, cm, o, lse = mod(q, k, v)
        A = torch.tril(A)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return A, o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = 128
        block_N = 128 if D_HEAD <= 64 else 32
        mod_prep = cached(flashattn_bwd_preprocess, [2], BATCH, H, N_CTX, D_HEAD)
        mod_post = cached(flashattn_bwd_postprocess, [1], BATCH, H, N_CTX, D_HEAD)
        delta = mod_prep(o, do)
        mod = cached(flashattn_bwd, [6, 7, 8], BATCH, H, N_CTX, D_HEAD, ctx.causal, block_M,
                     block_N)
        dq, dk, dv = mod(q, k, v, do, lse, delta)
        dq = mod_post(dq)
        return dq, dk, dv, None


attention = _attention.apply


def ref_program(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool):
    q = Q.permute(0, 2, 1, 3)
    k = K.permute(0, 2, 1, 3)
    v = V.permute(0, 2, 1, 3)
    length = q.size(2)
    cm = torch.ones(length, length).tril(-1).to(Q)
    if is_causal:
        mask = torch.ones(length, length).triu(0).cuda().bool()
    A, o, rem = stickbreaking(q, k, v, mask, cm)
    # o = o + rem[..., None] * v
    output = o.permute(0, 2, 1, 3)
    return A, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--h', type=int, default=1, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=128, help='Context size')
    parser.add_argument('--d_head', type=int, default=64, help='Head dimension')
    parser.add_argument('--casual', type=bool, default=True, help='Casual flag')
    args = parser.parse_args()
    BATCH, H, N_CTX, D_HEAD = args.batch, args.h, args.n_ctx, args.d_head
    casual = args.casual
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 5 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    Q = torch.empty(BATCH, N_CTX, H, D_HEAD, dtype=torch.bfloat16, device="cuda").zero_().requires_grad_()
    K = torch.empty_like(Q).zero_().requires_grad_()
    V = torch.empty_like(Q).normal_().requires_grad_()
    dO = torch.randn_like(Q)
    A_ref, O_ref = ref_program(Q, K, V, casual)
    A, O = attention(Q, K, V, casual)
    def print_err(x: torch.Tensor, x_ref: torch.Tensor, rtol=1e-2, atol=1e-2):
        print((x - x_ref).abs().max())
        assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)
    print((A_ref - A)[0, 0].abs().max(dim=1))
    print((A_ref - A)[0, 0].abs().max(dim=0))
    print(A_ref[0, 0, 64-5:64+5, 64-5:64+5])
    print(A[0, 0, 64-5:64+5, 64-5:64+5])
    print_err(O, O_ref, rtol=1e-2, atol=1e-2)
    exit()
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None

    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None



    print_err(dV, dV_ref, rtol=1e-2, atol=1e-2)
    print_err(dK, dK_ref, rtol=1e-2, atol=1e-2)
    print_err(dQ, dQ_ref, rtol=1e-2, atol=1e-2)

    def run():
        O_ref.backward(dO, retain_graph=True)

    def run1():
        O.backward(dO, retain_graph=True)

    from tilelang.profiler import do_bench

    latency = do_bench(run, warmup=500)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(run1, warmup=500)
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
