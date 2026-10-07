"""Logit-histogram sufficient statistics for the exact Bernoulli subgroup gain.

For a subgroup S, gain(S,d) = sum_{i in S} [ y_i d - softplus(z_i+d) + softplus(z_i) ].
Group rows by quantised base logit z (B bins). Per bin keep n_b, e_b (events), and
zbar_b (mean logit). Then gain ~= sum_b [ e_b d - n_b (softplus(zbar_b+d) - softplus(zbar_b)) ].
The error is second order in the bin width (we use the bin MEAN, not centre).
The exact gain is then a 1-D concave Newton problem over B numbers, not n_S rows,
and it is batchable over many candidate subgroups at once.
"""
import numpy as np

def softplus(x): return np.logaddexp(0.0, x)
def sig(x): return 0.5*(1+np.tanh(0.5*x))

class LogitBins:
    def __init__(self, z, B=64):
        qs = np.quantile(z, np.linspace(0,1,B+1)[1:-1])
        self.edges = np.unique(qs)
        self.bin = np.searchsorted(self.edges, z)          # row -> bin
        self.B = len(self.edges)+1
    def stats(self, y, z, mask):
        b = self.bin[mask]
        n = np.bincount(b, minlength=self.B).astype(float)
        e = np.bincount(b, weights=y[mask], minlength=self.B)
        sz = np.bincount(b, weights=z[mask], minlength=self.B)
        return n, e, sz

def batched_gain(n, e, sz, iters=30, bound=12.0):
    """n,e,sz: arrays (C,B). Returns gain (C,), delta (C,). Vectorised Newton."""
    n = np.atleast_2d(n); e = np.atleast_2d(e); sz = np.atleast_2d(sz)
    zbar = np.where(n>0, sz/np.maximum(n,1), 0.0)
    d = np.zeros(n.shape[0])
    E = e.sum(1)
    for _ in range(iters):
        p = sig(zbar + d[:,None])
        g = E - (n*p).sum(1)
        h = (n*p*(1-p)).sum(1)
        step = np.where(h>0, g/np.maximum(h,1e-300), 0.0)
        d = np.clip(d+step, -bound, bound)
        if np.max(np.abs(step)) < 1e-10: break
    gain = (E*d) - (n*(softplus(zbar+d[:,None]) - softplus(zbar))).sum(1)
    return np.maximum(gain,0.0), d

def all_threshold_gains(xcol, y, z, binidx, B, parent_idx, min_size):
    """Exact-binned gain for EVERY feasible '<= t' and '> t' split of one feature
    inside a parent subgroup. Returns (positions, thresholds, gain_le, gain_gt, d_le, d_gt)."""
    idx = parent_idx[np.isfinite(xcol[parent_idx])]
    o = np.argsort(xcol[idx], kind="mergesort"); idx = idx[o]; x = xcol[idx]; m = len(idx)
    ch = np.flatnonzero(x[:-1] < x[1:]) + 1
    ch = ch[(ch >= min_size) & (m - ch >= min_size)]
    if len(ch) == 0: return None
    b = binidx[idx]
    # cumulative per-bin statistics along sorted x  (m x B, built with one scatter)
    def cum(w):
        A = np.zeros((m, B)); A[np.arange(m), b] = w; return np.cumsum(A, axis=0)
    Cn, Ce, Cz = cum(np.ones(m)), cum(y[idx]), cum(z[idx])
    lo = (Cn[ch-1], Ce[ch-1], Cz[ch-1])
    hi = tuple(C[-1] - L for C, L in zip((Cn, Ce, Cz), lo))
    gl, dl = batched_gain(*lo); gh, dh = batched_gain(*hi)
    thr = 0.5*(x[ch-1] + x[ch])
    return ch, thr, gl, gh, dl, dh, idx
