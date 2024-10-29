import torch
import pytest
import math
from torch.nn import functional as F
from stickbreaking_attention.sb_varlen import sb_attn_varlen
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
    diff = (a - b).abs()
    from matplotlib import pyplot as plt
    plt.imshow(diff.float().detach().cpu().numpy()[0], interpolation='none')
    plt.savefig('diff.png')

    max_diff= diff.max()
    if max_diff < eps:
        print(varname, max_diff.item())
    else:
        print(varname, max_diff.item(), diff.median().item())
        print((diff.sum(0).median(dim=0)[0] > eps).int())
        err_locs = (diff.sum(0).median(dim=1)[0] > eps).int()
        print(err_locs, err_locs.sum())
        assert max_diff < eps, max_diff



class TestClass:

    @pytest.mark.parametrize('batch_size', [4, 2, 1])
    @pytest.mark.parametrize('num_heads', [8, 4, 2, 1, 7])
    @pytest.mark.parametrize('head_dim', [64, 32, 16, 50])
    @pytest.mark.parametrize('length', [4096, 2048, 1024, 512, 256, 500])
    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize('forward_only', [False, True])
    def test_varlen(self, batch_size, num_heads, head_dim, length, dtype, forward_only):
        set_seed(1337)
        torch.set_printoptions(linewidth=110, edgeitems=30)
        device = torch.device('cuda:0')
        lengths = torch.randint(length // 2, length + 1, (batch_size,)).to(device=device, dtype=torch.int32)
        print(lengths)
        total_length = lengths.sum()
        cu_seqlens = torch.cumsum(lengths, dim=-1)
        q = 0.5 * torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype) - 0.5
        k = 0.5 * torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype) + 0.5
        v = 0.5 * torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        do = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
        o, rem = sb_attn_varlen(q, k, v, cu_seqlens,
                                inv_temp=1 / math.sqrt(q.size(-1)),
                                zero_start=False)
        o = o + rem[..., None] * v
        torch.cuda.synchronize()
        ref_out, ref_dq, ref_dk, ref_dv = ref_bwd(do, q, k, v, lengths)
        eps = 0.05
        assert_close("o", ref_out, o, eps)
        if not forward_only:
            dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
            torch.cuda.synchronize()
            assert_close("dq", ref_dq, dq, eps)
            assert_close("dk", ref_dk, dk, eps)
            assert_close("dv", ref_dv, dv, eps)
        torch.cuda.empty_cache()
