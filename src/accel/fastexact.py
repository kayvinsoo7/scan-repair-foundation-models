"""Drop-in, numerically-equivalent replacement for exact_subgroup_logit_correction.
Same Newton iteration, same stopping rule, same clipping; fused into one compiled loop
per iteration (no temporaries, no repeated sigmoid arrays)."""
import numpy as np, numba as nb, math
import mdss_treescan_stage1 as M

@nb.njit(cache=True, fastmath=False)
def _core(y, z, idx, l2, bound, tol, max_iter):
    n = idx.size
    delta = 0.0
    for _ in range(max_iter):
        g = 0.0; h = 0.0
        for k in range(n):
            i = idx[k]; x = z[i] + delta
            # same branch-stable sigmoid as M.sigmoid
            if x >= 0: p = 1.0/(1.0+math.exp(-x))
            else:
                e = math.exp(x); p = e/(1.0+e)
            g += y[i] - p; h += p*(1.0-p)
        g -= l2*delta; h += l2
        if h <= 0: break
        nd = min(max(delta + g/h, -bound), bound)
        if abs(nd-delta) < tol:
            delta = nd; break
        delta = nd
    gain = 0.0; ev = 0.0; pb = 0.0; pa = 0.0
    for k in range(n):
        i = idx[k]; zi = z[i]; yi = y[i]; za = zi + delta
        sb = max(zi,0.0) + math.log1p(math.exp(-abs(zi)))
        sa = max(za,0.0) + math.log1p(math.exp(-abs(za)))
        gain += (sb - yi*zi) - (sa - yi*za)
        ev += yi
        pb += 1.0/(1.0+math.exp(-zi)) if zi>=0 else math.exp(zi)/(1.0+math.exp(zi))
        pa += 1.0/(1.0+math.exp(-za)) if za>=0 else math.exp(za)/(1.0+math.exp(za))
    gain -= 0.5*l2*delta*delta
    return gain, delta, ev, pb, pa

_orig = M.exact_subgroup_logit_correction
def fast_exact(y, logits, mask, *, l2=0.0, delta_bound=12.0, min_size=25, min_events=2,
               min_nonevents=2, tol=1e-10, max_iter=100):
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask); n = idx.size
    if n < min_size: return None
    y = np.asarray(y, dtype=float); z = np.asarray(logits, dtype=float)
    gain, delta, ev, pb, pa = _core(y, z, idx, float(l2), float(delta_bound), float(tol), int(max_iter))
    events = int(round(ev)); non = n - events
    if events < min_events or non < min_nonevents: return None
    return M.SubgroupScore(gain=max(0.0, gain), delta=float(delta), size=int(n), events=events,
        nonevents=non, mean_observed=ev/n, mean_predicted_before=pb/n, mean_predicted_after=pa/n)

def install():
    """Patch every module that imported the scorer by name."""
    import sys
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "exact_subgroup_logit_correction", None) is _orig:
            mod.exact_subgroup_logit_correction = fast_exact
