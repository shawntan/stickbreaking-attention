import torch
import pytest
import math
from torch.nn import functional as F
from stickbreaking_attention.sb_varlen import sb_attn_varlen
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from transformers import set_seed


# for reference
def stickbreaking(q, k, v, mask, cum_weight):
    """
    Stick-breaking attention weights.
    """
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

    original_dtype = logits.dtype

    logits = logits.float()
    log_z = F.logsigmoid(logits).masked_fill(mask, -1e5).to(original_dtype)

    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0).to(original_dtype)

    re_cum_log_beta = torch.einsum('bhij,jk->bhik', log_beta, cum_weight.to(log_beta))
    log_att = log_z + re_cum_log_beta
    att = log_att.exp()
    return att @ v, 1 - att.sum(dim=-1)


def ref_fwd(q, k, v, lengths):
    splits = list(lengths.cpu().numpy())
    max_len = max(splits)
    cm = torch.ones(max_len, max_len).tril(-1).to(q)
    mask = torch.ones(max_len, max_len).triu(0).cuda().bool()
    outputs = []
    for q_chunk, k_chunk, v_chunk in zip(q.split(splits, 1), k.split(splits, 1), v.split(splits, 1)):
        len = q_chunk.size(1)
        o, rem = stickbreaking(
            q_chunk[None, :],
            k_chunk[None, :],
            v_chunk[None, :],
            mask[:len, :len], cm[:len, :len]
        )

        o = o + rem[..., None] * v_chunk[None]
        outputs.append(o[0])
    return torch.cat(outputs, 1)

def ref_fwdbwd(do, q, k, v, lengths):
    o = ref_fwd(q, k, v, lengths)
    dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
    return o, dq, dk, dv

def tri_fwdbwd(do, q, k, v, lengths):
    cu_seqlens = torch.cumsum(lengths, dim=-1)
    o, rem = sb_attn_varlen(q, k, v, cu_seqlens,
                            inv_temp=1 / math.sqrt(q.size(-1)),
                            zero_start=False)
    o = o + rem[..., None] * v
    dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
    return o, dq, dk, dv

def flash_fwdbwd(rope, position_ids, do, q, k, v, lengths):
    cos, sin = rope(v, position_ids)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    lengths = lengths.to(torch.int32)
    cu_seqlens = torch.cumsum(lengths, dim=-1)
    cu_seqlens = F.pad(cu_seqlens, (1, 0)).to(torch.int32)
    max_len = torch.max(lengths)

    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
        causal=True
    )
    o = o.permute(1, 0, 2)
    dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
    return o, dq, dk, dv


providers = [
    ("reference", "Stickbreaking (ref.)", ("red", "-")),
    ("triton", "Stickbreaking", ("blue", "-")),
    ("flash", "Flash Attention", ("green", "-")),
]
@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["length"],
        x_vals=[4096, 8192, 8192 * 2],
        line_arg="provider",
        line_vals=[x[0] for x in providers],
        line_names=[x[1] for x in providers],
        styles=[x[2] for x in providers],
        ylabel="ms",
        plot_name=f"triton v torch",
        args={"batch_size": 2, "num_heads": 8, "head_dim": 64, "dtype": torch.bfloat16}
    )
])
def benchmark_varlen(batch_size, num_heads, head_dim, length, dtype, provider):
    device = torch.device('cuda:0')
    set_seed(1337)
    lengths = torch.randint(length // 2, length, (batch_size,)).to(device=device, dtype=torch.int32)
    print(lengths)
    total_length = lengths.sum()
    warmup = 100
    rep = 1000

    q = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    k = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    v = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    do = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(q.size(1), device=device, dtype=torch.int32)[None, :]

    if provider== "reference":
        fun = lambda: ref_fwdbwd(do, q, k, v, lengths)
    elif provider == "triton":
        fun = lambda: tri_fwdbwd(do, q, k, v, lengths)
    elif provider == "flash":
        rope = LlamaRotaryEmbedding(dim=head_dim).to(device)
        fun = lambda: flash_fwdbwd(rope, position_ids, do, q, k, v, lengths)

    return triton.testing.do_bench(fun, warmup=warmup, rep=rep)



if __name__ == "__main__":
    benchmark_varlen.run(save_path=None, print_data=True)
