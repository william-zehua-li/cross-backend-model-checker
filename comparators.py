import torch
import numpy as np

def tensors_allclose(t1, t2, atol=1e-5, rtol=1e-5):
    if t1.shape != t2.shape:
        return False, {"reason": "shape_mismatch", "shape1": t1.shape, "shape2": t2.shape}
    if not torch.isfinite(t1).all() or not torch.isfinite(t2).all():
        return False, {"reason": "non_finite_values"}
    diff = torch.abs(t1 - t2)
    max_err = diff.max().item()
    passed = torch.allclose(t1, t2, atol=atol, rtol=rtol)
    return passed, {"max_err": max_err}

def compare_outputs(ref_out, tgt_out, atol, rtol):
    # 支持 tuple/list 结构
    if isinstance(ref_out, (tuple, list)):
        if len(ref_out) != len(tgt_out):
            return False, {"reason": "length_mismatch"}
        for r, t in zip(ref_out, tgt_out):
            passed, info = compare_outputs(r, t, atol, rtol)
            if not passed:
                return False, info
        return True, {}
    elif isinstance(ref_out, torch.Tensor):
        return tensors_allclose(ref_out, tgt_out, atol, rtol)
    else:
        return True, {}  # 忽略非 tensor
