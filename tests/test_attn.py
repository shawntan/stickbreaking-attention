import torch
import pytest
import math
from stickbreaking_attention.sb_attn import sb_attn
from transformers import set_seed
from .test_varlen import assert_close

def ref_fwd(q, k, v, length, attend_current=False):
    cm = torch.ones(length, length).tril(-1).to(q)
    if attend_current:
        mask = torch.ones(length, length).triu(1).cuda().bool()
    else:
        mask = torch.ones(length, length).triu(0).cuda().bool()
    o, rem = stickbreaking(q, k, v, mask, cm)
    o = o + rem[..., None] * v
    return o

def ref_fwdbwd(do, q, k, v, length, attend_current=False):
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    output = ref_fwd(q, k, v, length, attend_current)
    output.backward(do)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    q.grad = None
    k.grad = None
    v.grad = None
    return output, dq, dk, dv


class TestClass:

    @pytest.mark.parametrize('batch_size', [4, 2, 1])
    @pytest.mark.parametrize('num_heads', [24, 8, 4, 2, 1, 7])
    @pytest.mark.parametrize('head_dim', [64, 32, 16, 50])
    @pytest.mark.parametrize('length', [4096, 2048, 1024, 512, 256, 500])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    @pytest.mark.parametrize('forward_only', [False])
    @pytest.mark.parametrize('attend_current', [False, True])
    def test_varlen(self, batch_size, num_heads, head_dim, attend_current, length, dtype, forward_only):
        set_seed(1337)
        torch.set_printoptions(linewidth=110, edgeitems=30)
        device = torch.device('cuda:0')
        input_dims = (batch_size, num_heads, length, head_dim)
        v = 0.25 * torch.randn(input_dims, device=device, dtype=torch.float32)
        q = 0.25 * (torch.randn(input_dims, device=device, dtype=torch.float32) + 1) 
        k = 0.25 * (torch.randn(input_dims, device=device, dtype=torch.float32) - 1)
        print(q.max(), k.max(), v.max())
        q = q.to(dtype).requires_grad_()
        k = k.to(dtype).requires_grad_()
        v = v.to(dtype).requires_grad_()
        do = torch.randn(input_dims, device=device, dtype=dtype)

        with torch.cuda.device(device):
            o, rem= sb_attn(
                q, k, v,
                inv_temp=1 / math.sqrt(q.size(-1)),
                attend_current=attend_current
            )
            o = o + rem[..., None] * v
            ref_out, ref_dq, ref_dk, ref_dv = ref_fwdbwd(do, q, k, v, length,
                                                         attend_current=attend_current)
        eps = 0.05
        torch.cuda.synchronize()
        assert_close("o", ref_out, o, eps)
        if not forward_only:
            dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
            assert_close("dq", ref_dq, dq, eps)
            assert_close("dk", ref_dk, dk, eps)
            assert_close("dv", ref_dv, dv, eps)
