import torch
import pytest
import math
from torch.nn import functional as F
from stickbreaking_attention import sb_attn_varlen
from transformers import set_seed
from stickbreaking_attention.sb_ref import stickbreaking


def ref_fwd(q, k, v, lengths, attend_current=False):
    splits = list(lengths.cpu().numpy())
    max_len = max(splits)
    cm = torch.ones(max_len, max_len).tril(-1).to(q)
    mask = torch.ones(max_len, max_len).triu(0 if not attend_current else 1).cuda().bool()
    outputs = []
    for q_chunk, k_chunk, v_chunk in zip(q.split(splits, 1), k.split(splits, 1), v.split(splits, 1)):
        len = q_chunk.size(1)
        o, rem = stickbreaking(
            q_chunk[None, :],
            k_chunk[None, :],
            v_chunk[None, :],
            mask[:len, :len], cm[:len, :len]
        )

        # o = o + rem[..., None] * v_chunk[None]
        outputs.append(o[0])
    return torch.cat(outputs, 1)

def ref_bwd(do, q, k, v, lengths, attend_current=False):
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    q_ = q.permute(1, 0, 2)
    k_ = k.permute(1, 0, 2)
    v_ = v.permute(1, 0, 2)
    output = ref_fwd(q_, k_, v_, lengths, attend_current=attend_current)
    do_ = do.permute(1, 0, 2)
    output.backward(do_)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    q.grad = None
    k.grad = None
    v.grad = None
    output = output.permute(1, 0, 2)
    return output, dq, dk, dv

def assert_close(varname, a, b, eps):
    if torch.isnan(a).any():
        print("Reference is nan")
        return
    assert not torch.isnan(b).any()
    diff = (a - b).abs()

    max_diff = diff.max().item()
    median_diff = diff.median().item()
    if max_diff < eps:
        print(varname, max_diff)
    else:
        print(varname, max_diff, median_diff)
        # print((diff.sum(0).median(dim=0)[0] > eps).int())
        err_locs = (diff > eps).max(dim=-1)[0].int()
        print(err_locs)
        print(torch.sum(1 - err_locs))
        # print(diff)
        assert max_diff < eps, max_diff



class TestClass:

    @pytest.mark.parametrize('batch_size', [4, 2, 1])
    @pytest.mark.parametrize('num_heads', [12, 24, 8, 4, 2, 1, 7])
    @pytest.mark.parametrize('head_dim', [128, 64, 32, 16, 50])
    @pytest.mark.parametrize('length', [4096, 2048, 1024, 512, 256, 128, 64, 500])
    # @pytest.mark.parametrize('batch_size', [1])
    # @pytest.mark.parametrize('num_heads', [12, 3])
    # @pytest.mark.parametrize('head_dim', [128])
    # @pytest.mark.parametrize('length', [4096, 8192, 8192 * 2])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    @pytest.mark.parametrize('forward_only', [False])
    @pytest.mark.parametrize('attend_current', [False, True])
    def test_varlen(self, batch_size, num_heads, head_dim, length, attend_current, dtype, forward_only):
        set_seed(1337)
        torch.set_printoptions(threshold=2 * 8192, linewidth=110, edgeitems=30)
        device = torch.device('cuda:0')
        lengths = torch.randint(length, length + 1, (batch_size,)).to(device=device, dtype=torch.int32)
        total_length = lengths.sum()
        cu_seqlens = torch.cumsum(lengths, dim=-1)
        q = 0.25 * (torch.randn((total_length, num_heads, head_dim), device=device, dtype=torch.float32) + 1)
        k = 0.25 * (torch.randn((total_length, num_heads, head_dim), device=device, dtype=torch.float32) - 1) 
        v = 0.25 * torch.randn((total_length, num_heads,  head_dim), device=device, dtype=torch.float32) 
        print(q.max(), k.max(), v.max())

        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        do = torch.randn((total_length, num_heads, head_dim), device=device, dtype=dtype) # * 0. \
        #     + 0.001 * torch.arange(total_length, device=device, dtype=torch.float32)[None, :, None]
        with torch.cuda.device(device):
            q_ = q.permute(1, 0, 2)
            k_ = k.permute(1, 0, 2)
            v_ = v.permute(1, 0, 2)
            o_, rem = sb_attn_varlen(
                q_,
                k_,
                v_,
                cu_seqlens=cu_seqlens,
                max_seqlens=torch.max(lengths).item(),
                inv_temp=1 / math.sqrt(q.size(-1)),
                zero_start=False,
                attend_current=attend_current
            )
            o = o_.permute(1, 0, 2)
            # o = o + rem[..., None] * v
            ref_out, ref_dq, ref_dk, ref_dv = ref_bwd(do, q, k, v, lengths, attend_current=attend_current)
        eps = 0.05
        assert_close("o", ref_out, o, eps)
        if not forward_only:
            dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
            assert_close("dq", ref_dq, dq, eps)
            assert_close("dk", ref_dk, dk, eps)
            assert_close("dv", ref_dv, dv, eps)
