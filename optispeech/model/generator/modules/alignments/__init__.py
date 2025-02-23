import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit

# Reexport
try:
    print("Using Triton implementation of monotonic align")
    from .S_monotonic_align_Triton import maximum_path
except ImportError:
    from .S_monotonic_align import maximum_path2 as maximum_path


@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    xs_avg = np.zeros_like(ds)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """Average frame-level features into token-level according to durations

    Args:
        ds (Tensor): Batched token duration (B, T_text).
        xs (Tensor): Batched feature sequences to be averaged (B, T_feats).
        text_lengths (Tensor): Text length tensor (B,).
        feats_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched feature averaged according to the token duration (B, T_text).

    """
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().float().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg).to(device)
    return xs_avg


def expand_by_duration(x, durations):
    dtype = x.dtype
    lengths = durations.sum(dim=1)
    max_len = lengths.max()
    dur_cumsum = torch.cumsum(F.pad(durations, (1, 0, 0, 0), value=0.0), dim=1)
    dur_cumsum = dur_cumsum[:, None, :]
    dur_cumsum = dur_cumsum.to(dtype)
    range_ = torch.arange(max_len, device=x.device)[None, :, None]
    mult = (
        (dur_cumsum[:, :, :-1] <= range_)
        &(dur_cumsum[:, :, 1:] > range_)
    )
    mult = mult.to(dtype)
    expanded = torch.matmul(mult, x)
    return expanded, lengths

