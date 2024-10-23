import torch
import pytest
import math
from torch.nn import functional as F
from stickbreaking_attention.sb_varlen import sb_attn_varlen

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

def ref_bwd(do, q, k, v, lengths):
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    output = ref_fwd(q, k, v, lengths)
    output.backward(do)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    q.grad = None
    k.grad = None
    v.grad = None
    return output, dq, dk, dv

class TestClass:

    @pytest.mark.parametrize('batch_size', [2, 4])
    @pytest.mark.parametrize('num_heads', [1, 2, 4, 8, 7])
    @pytest.mark.parametrize('head_dim', [50, 16, 32, 64])
    @pytest.mark.parametrize('length', [512, 1024, 2048, 4096])
    @pytest.mark.parametrize('dtype', [torch.float32])
    def test_varlen(self, batch_size, num_heads, head_dim, length, dtype):
        torch.set_printoptions(linewidth=1024, edgeitems=500)
        device = torch.device('cuda:0')
        lengths = torch.randint(length // 2, length, (batch_size,)).to(device=device, dtype=torch.int32)
        total_length = lengths.sum()
        cu_seqlens = torch.cumsum(lengths, dim=-1)

        q = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        k = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        v = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        do = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        o, rem = sb_attn_varlen(q, k, v, cu_seqlens,
                                inv_temp=1 / math.sqrt(q.size(-1)),
                                zero_start=False)
        o = o + rem[..., None] * v
        dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
        torch.cuda.synchronize()

        ref_out, ref_dq, ref_dk, ref_dv = ref_bwd(do, q, k, v, lengths)
        print("o", (ref_out - o).abs().max())
        print("dq", (ref_dq - dq).abs().max())
        print("dk", (ref_dk - dk).abs().max())
        print("dv", (ref_dv - dv).abs().max())
        assert (ref_out - o).abs().max() < 1e-5,  (ref_out - o).abs().max()
        assert (ref_dq - dq).abs().max() < 1e-5, ((ref_dq - dq).abs().max())
        assert (ref_dk - dk).abs().max() < 1e-5, (ref_dk - dk).abs().max()
        assert (ref_dv - dv).abs().max() < 1e-5, (ref_dv - dv).abs().max()
