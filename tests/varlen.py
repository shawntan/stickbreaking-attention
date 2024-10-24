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

def assert_close(varname, a, b, eps):
    if torch.isnan(a).any():
        print("Reference is nan")
        return 
    diff = (a - b).abs().max()
    print(varname, diff.item())
    assert diff < eps, diff



class TestClass:

    @pytest.mark.parametrize('batch_size', [2, 4])
    @pytest.mark.parametrize('num_heads', [8, 4, 2, 1, 7])
    @pytest.mark.parametrize('head_dim', [64, 32, 16, 50])
    @pytest.mark.parametrize('length', [4096, 2048, 1024, 512, 500])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_varlen(self, batch_size, num_heads, head_dim, length, dtype):
        torch.set_printoptions(linewidth=1024, edgeitems=500)
        device = torch.device('cuda:0')
        lengths = torch.randint(length // 2, length, (batch_size,)).to(device=device, dtype=torch.int32)
        total_length = lengths.sum()
        cu_seqlens = torch.cumsum(lengths, dim=-1)

        q = 0.5 * torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        k = 0.5 * torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        v = 0.5 * torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        do = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        o, rem = sb_attn_varlen(q, k, v, cu_seqlens,
                                inv_temp=1 / math.sqrt(q.size(-1)),
                                zero_start=False)
        o = o + rem[..., None] * v
        ref_out, ref_dq, ref_dk, ref_dv = ref_bwd(do, q, k, v, lengths)
        eps = 0.05
        assert_close("o", ref_out, o, eps)
        dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
        assert_close("dq", ref_dq, dq, eps)
        assert_close("dk", ref_dk, dk, eps)
        assert_close("dv", ref_dv, dv, eps)
