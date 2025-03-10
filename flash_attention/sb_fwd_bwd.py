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
from stickbreaking_attention.sb_attn import sb_attn

import sb_mha_fwd_bshd_wgmma_pipelined

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
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"
    @T.macro
    def QK_MM(
        K: T.Buffer(shape, dtype), # type: ignore
        Q_shared: T.Buffer([block_M, dim], dtype), # type: ignore
        K_shared: T.Buffer([block_N, dim], dtype), # type: ignore
        qk_scaled: T.Buffer([block_M, block_N], accum_dtype), # type: ignore
        bx: T.int32, by: T.int32, bz: T.int32, k: T.int32,
    ):
        T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                qk_scaled[i, j] = T.if_then_else(
                    bx * block_M + i <= k * block_N + j,
                    -T.infinity(qk_scaled.dtype),
                    0
                )
        else:
            T.clear(qk_scaled)
        T.gemm(Q_shared, K_shared, qk_scaled, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)


    log2 = math.log(2)
    inv_log2 = 1 / log2
    scale = 1.0 / math.sqrt(dim)

    @T.prim_func
    def flash_fwd(
            Q: T.Buffer(shape, dtype),  # type: ignore
            K: T.Buffer(shape, dtype),  # type: ignore
            V: T.Buffer(shape, dtype),  # type: ignore
            Output: T.Buffer(shape, dtype),  # type: ignore
            lse: T.Buffer([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):


        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            cm = T.alloc_shared([block_N, block_N], dtype)
            for i, j in T.Parallel(block_N, block_N):
                cm[i, j] = T.if_then_else(i < j, 0., 1.)
            # allocate shared mem
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)

            # block fragments
            T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
           
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            acc_log_om_beta = T.alloc_fragment([block_M], accum_dtype)

            qk_scaled = T.alloc_fragment([block_M, block_N], accum_dtype)
            log_om_beta = T.alloc_fragment([block_M, block_N], accum_dtype)
            log_om_beta_cast = T.alloc_fragment([block_M, block_N], dtype)
            log_p = T.alloc_fragment([block_M, block_N], accum_dtype)
            p = T.alloc_fragment([block_M, block_N], dtype)
            tile_log_om_beta_sum = T.alloc_fragment([block_M], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(acc_log_om_beta, 0)

            loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
            for k_rev in T.Pipelined(
                loop_range, num_stages=0,
                # order=[-1, 0, 3, 1, -1, 2],
                # stage=[-1, 0, 0, 1, -1, 1],
                # group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]
            ):
                k = loop_range - k_rev - 1
                QK_MM(K, Q_shared, K_shared, qk_scaled, bx, by, bz, k)
                for i, j in T.Parallel(block_M, block_N):
                    # qk_scaled[i, j] = qk_scaled[i, j] * scale * inv_log2
                    qk_scaled[i, j] *= scale * inv_log2
                    log_om_beta[i, j] = -T.if_then_else(
                        qk_scaled[i, j] > 15.0,
                        qk_scaled[i, j],
                        T.log2(1 + T.exp2(qk_scaled[i, j]))
                    )
                    # init log_p
                    log_p[i, j] = acc_log_om_beta[i] + qk_scaled[i, j]
                T.copy(log_om_beta, log_om_beta_cast)
                T.gemm(log_om_beta_cast, cm, log_p, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(log_om_beta, tile_log_om_beta_sum, dim=1)
                for i, j in T.Parallel(block_M, block_N):
                    p[i, j] = T.exp2(log_p[i, j])
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                T.gemm(p, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i in T.Parallel(block_M):
                    acc_log_om_beta[i] = acc_log_om_beta[i] + tile_log_om_beta_sum[i]
            T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
            T.copy(acc_log_om_beta, lse[bz, by, bx * block_M:(bx + 1) * block_M])

    return flash_fwd


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k, v, causal):
        BATCH, N_CTX, H, D_HEAD = q.shape
        block_M = 64
        # block_N = 64 if D_HEAD <= 128 else 32
        block_N = 32
        dtype_str = str(q.dtype).split('.')[-1]
        mod = cached(
            flashattn_fwd, [3, 4],
            BATCH, H, N_CTX, D_HEAD, causal, block_M, block_N, dtype_str
        )
        # batch, heads, seq_len, dim, is_causal, tune=args.tune)(
        #     block_M=128, block_N=128, num_stages=2, threads=256)
        # mod = cached(
            # sb_mha_fwd_bshd_wgmma_pipelined.flashattn, [3],
            # BATCH, H, N_CTX, D_HEAD, causal, True
        # )

        o, lse = mod(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

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


def ref_program_1(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool):
    print("Running pytorch")
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
    return  output

def ref_program(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool):
    # print("Running triton")
    assert is_causal
    q = Q.permute(0, 2, 1, 3)
    k = K.permute(0, 2, 1, 3)
    v = V.permute(0, 2, 1, 3)
    length = q.size(2)
    inv_temp = math.sqrt(q.shape[-1])
    output, rem = sb_attn(q, k, v)
    output = output.permute(0, 2, 1, 3)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--h', type=int, default=32, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=4096, help='Context size')
    parser.add_argument('--d_head', type=int, default=128, help='Head dimension')
    parser.add_argument('--casual', type=bool, default=True, help='Casual flag')
 
    args = parser.parse_args()
    BATCH, H, N_CTX, D_HEAD = args.batch, args.h, args.n_ctx, args.d_head
    casual = args.casual
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 5 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    Q = torch.empty(BATCH, N_CTX, H, D_HEAD, dtype=torch.bfloat16, device="cuda").normal_().requires_grad_()
    K = torch.empty_like(Q).normal_().requires_grad_()
    V = torch.empty_like(Q).normal_().requires_grad_()
    dO = torch.randn_like(Q)
    # O_ref1 = ref_program_1(Q, K, V, casual)
    O_ref2 = ref_program(Q, K, V, casual)
    O = attention(Q, K, V, casual)
    def print_err(x: torch.Tensor, x_ref: torch.Tensor, rtol=1e-2, atol=1e-2):
        print((x - x_ref).abs().max().item())
        # assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)
    # print("Pytorch vs. Triton:")
    # print_err(O_ref1, O_ref2, rtol=1e-2, atol=1e-2)
    # print("TileLang vs. Pytorch:")
    # print_err(O, O_ref1, rtol=1e-2, atol=1e-2)
    print("TileLang vs. Triton:")
    print_err(O, O_ref2, rtol=1e-2, atol=1e-2)

    from tilelang.profiler import do_bench
    def run():
        O_ref = ref_program(Q, K, V, casual)

    def run1():
        O = attention(Q, K, V, casual)


    latency = do_bench(run, warmup=500)
    print("triton: {:.2f} ms".format(latency))
    print("triton: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(run1, warmup=500)
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))

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

if __name__ == "__main__":
    main()