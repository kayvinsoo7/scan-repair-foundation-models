"""Batched, output-identical acceleration of the two hot search routines.

1. AdaptiveStructuredScannerV2._numeric_candidates: the <=6 shortlisted thresholds of one
   (feature, parent rule) pair are scored in ONE compiled call instead of six Python calls,
   each of which built a full-length mask.
2. StructuredBeamMDSS.scan (fixed five-bin guard candidate): all tail literals of one
   (feature, parent rule) pair are scored in ONE compiled call, and masks come from a
   lookup table instead of np.isin.

Exactness: every candidate is scored with the same Newton iteration, clipping, stopping rule
and row summation order (ascending row index) as fastexact.fast_exact, so scores are bit-identical
to the compiled single-candidate scorer; control flow (support checks, ranking, ties, dedup)
is copied line for line from the original methods. install() patches the classes in place."""
import math, numpy as np, numba as nb
import mdss_treescan_stage1 as M
from mdss_treescan_stage1 import SubgroupScore, sigmoid

@nb.njit(cache=True)
def _sig(x):
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x); return e / (1.0 + e)

@nb.njit(cache=True)
def _score_one(y, z, rows, include, l2, bound, tol, max_iter, out, c):
    """Exact Newton score over rows[k] with include[k] True; rows ascending. Writes out[c, :]."""
    n = 0; ev = 0.0
    for k in range(rows.size):
        if include[k]:
            n += 1; ev += y[rows[k]]
    out[c, 0] = n; out[c, 1] = ev
    if n == 0:
        return
    delta = 0.0
    for _ in range(max_iter):
        g = 0.0; h = 0.0
        for k in range(rows.size):
            if include[k]:
                i = rows[k]; p = _sig(z[i] + delta)
                g += y[i] - p; h += p * (1.0 - p)
        g -= l2 * delta; h += l2
        if h <= 0: break
        nd = min(max(delta + g / h, -bound), bound)
        if abs(nd - delta) < tol:
            delta = nd; break
        delta = nd
    gain = 0.0; pb = 0.0; pa = 0.0
    for k in range(rows.size):
        if include[k]:
            i = rows[k]; zi = z[i]; yi = y[i]; za = zi + delta
            sb = max(zi, 0.0) + math.log1p(math.exp(-abs(zi)))
            sa = max(za, 0.0) + math.log1p(math.exp(-abs(za)))
            gain += (sb - yi * zi) - (sa - yi * za)
            pb += _sig(zi); pa += _sig(za)
    gain -= 0.5 * l2 * delta * delta
    out[c, 2] = gain; out[c, 3] = delta; out[c, 4] = pb; out[c, 5] = pa

@nb.njit(cache=True, parallel=True)
def score_thresholds(y, z, rows, rank, cuts, upper, l2, bound, tol, max_iter):
    """rows: parent rows ascending; rank[k]: position of rows[k] in x-sorted order.
    Candidate c keeps rank < cuts[c] (upper[c]==0, '<=') or rank >= cuts[c] ('>').
    Candidates are scored in parallel; each one's summation order is unchanged."""
    C = cuts.size; out = np.zeros((C, 6))
    for c in nb.prange(C):
        inc = np.empty(rows.size, np.bool_)
        for k in range(rows.size):
            inc[k] = (rank[k] >= cuts[c]) if upper[c] else (rank[k] < cuts[c])
        _score_one(y, z, rows, inc, l2, bound, tol, max_iter, out, c)
    return out

@nb.njit(cache=True)
def score_bin_literals(y, z, rows, bins, table, l2, bound, tol, max_iter, min_size):
    """rows ascending; bins[k] = bin code of rows[k]; table[c, b] True if literal c allows bin b.
    Candidates with support < min_size are only counted (gain left at 0)."""
    C = table.shape[0]; out = np.zeros((C, 6)); inc = np.empty(rows.size, np.bool_)
    for c in range(C):
        n = 0
        for k in range(rows.size):
            inc[k] = table[c, bins[k]]
            if inc[k]: n += 1
        if n < min_size:
            out[c, 0] = n; continue
        _score_one(y, z, rows, inc, l2, bound, tol, max_iter, out, c)
    return out

@nb.njit(cache=True, parallel=True)
def score_bin_literals_multi(y, z, rows, Xb_rows, cand_j, cand_c, tables, l2, bound, tol, max_iter, min_size):
    """All literals of all features for one parent rule. Xb_rows[k, j] = bin code of rows[k];
    candidate c tests feature cand_j[c] with literal row cand_c[c] of tables[cand_j[c]]."""
    C = cand_j.size; out = np.zeros((C, 6))
    for c in nb.prange(C):
        j = cand_j[c]; lc = cand_c[c]; inc = np.empty(rows.size, np.bool_); n = 0
        for k in range(rows.size):
            inc[k] = tables[j, lc, Xb_rows[k, j]]
            if inc[k]: n += 1
        if n < min_size:
            out[c, 0] = n
        else:
            _score_one(y, z, rows, inc, l2, bound, tol, max_iter, out, c)
    return out

def _to_score(row, min_size, min_events, min_nonevents):
    n = int(row[0])
    if n < min_size: return None
    events = int(round(row[1])); non = n - events
    if events < min_events or non < min_nonevents: return None
    return SubgroupScore(gain=max(0.0, float(row[2])), delta=float(row[3]), size=n, events=events, nonevents=non,
                         mean_observed=float(row[1]) / n, mean_predicted_before=float(row[4]) / n, mean_predicted_after=float(row[5]) / n)

# ---------------------------------------------------------------- numeric candidates (adaptive scan)
def _numeric_candidates_batched(self, X, y, logits, parent_mask, j):
    from adaptive_structured_scanner_v2_1 import NumericLiteral
    # parent-level quantities are identical for every feature j: compute once per parent mask object
    cache = getattr(self, "_fs_parent_cache", None)
    if cache is None or cache[0] is not parent_mask or cache[1] is not logits or cache[2] is not y:
        idx_all = np.flatnonzero(parent_mask)
        cache = (parent_mask, logits, y, idx_all, np.asarray(y[idx_all], dtype=float), sigmoid(np.asarray(logits[idx_all], dtype=float)),
                 np.asarray(y, float), np.asarray(logits, float))
        self._fs_parent_cache = cache
    idx = cache[3]
    if len(idx) < 2 * self.min_size: return []
    x = np.asarray(X[idx, j], dtype=float); yy = cache[4]; pp = cache[5]
    finite = np.isfinite(x)
    if finite.sum() < 2 * self.min_size: return []
    idx = idx[finite]; x = x[finite]; yy = yy[finite]; pp = pp[finite]
    order = np.argsort(x, kind="mergesort")
    x = x[order]; yy = yy[order]; pp = pp[order]; idx = idx[order]
    m = len(x)
    change = np.flatnonzero(x[:-1] < x[1:]) + 1
    if len(change) == 0: return []
    change = change[(change >= self.min_size) & ((m - change) >= self.min_size)]
    if len(change) == 0: return []
    g = pp - yy; h = pp * (1.0 - pp)
    G = np.cumsum(g); H = np.cumsum(h)
    total_G = G[-1]; total_H = H[-1]; t = change
    lower_G = G[t - 1]; lower_H = H[t - 1]; upper_G = total_G - lower_G; upper_H = total_H - lower_H
    lower_proxy = 0.5 * lower_G**2 / np.maximum(lower_H + max(self.l2, 1e-12), 1e-12)
    upper_proxy = 0.5 * upper_G**2 / np.maximum(upper_H + max(self.l2, 1e-12), 1e-12)
    self.screened_numeric_candidates_ += int(2 * len(change))
    shortlist = set()
    k = min(self.numeric_shortlist_per_direction, len(change))
    if k > 0:
        low_top = np.argpartition(-lower_proxy, k - 1)[:k]; up_top = np.argpartition(-upper_proxy, k - 1)[:k]
        for pos in low_top: shortlist.add(("le", int(pos)))
        for pos in up_top: shortlist.add(("gt", int(pos)))
    for q in self.numeric_anchor_quantiles:
        target = int(round(q * m)); nearest = int(np.argmin(np.abs(change - target)))
        shortlist.add(("le", nearest)); shortlist.add(("gt", nearest))
    ranked = []
    for direction, pos in shortlist:
        proxy = float(lower_proxy[pos]) if direction == "le" else float(upper_proxy[pos])
        ranked.append((proxy, direction, int(pos)))
    ranked.sort(key=lambda z_: z_[0], reverse=True)
    ranked = ranked[: self.numeric_exact_pool]
    if not ranked: return []
    # ---- batched exact scoring (rows in ascending index order, as the mask-based scorer uses)
    asc = np.argsort(idx, kind="mergesort"); rows = idx[asc].astype(np.int64)
    rank = asc.astype(np.int64)   # rows[k] = idx[asc[k]], so its position in x-sorted order is asc[k]
    cuts = np.array([int(change[pos]) for _, _, pos in ranked], np.int64)
    upper = np.array([d_ == "gt" for _, d_, _ in ranked], np.bool_)
    res = score_thresholds(cache[6], cache[7], rows, rank, cuts, upper, float(self.l2), 12.0, 1e-10, 100)
    exact_candidates = []
    for r_i, (_, direction, pos) in enumerate(ranked):
        split_t = int(change[pos]); threshold = float(0.5 * (x[split_t - 1] + x[split_t]))
        self.evaluated_exact_candidates_ += 1
        sc = _to_score(res[r_i], self.min_size, self.min_events, self.min_nonevents)
        if sc is None: continue
        exact_candidates.append((NumericLiteral(feature=j, direction=direction, threshold=threshold), (direction, split_t), sc))
    exact_candidates.sort(key=lambda z_: z_[2].gain, reverse=True)
    out = []
    for lit, (direction, split_t), sc in exact_candidates[: self.per_feature_exact_candidates]:
        mask = np.zeros(len(X), dtype=bool); mask[idx[:split_t] if direction == "le" else idx[split_t:]] = True
        out.append((lit, mask, sc))
    return out

# ---------------------------------------------------------------- fixed five-bin guard scan (legacy)
def _legacy_scan_batched(self, X, y, logits):
    from structured_mdss import candidate_bin_subsets, BeamState, BeamMDSSResult
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int); logits = np.asarray(logits, dtype=float)
    if self.binner_ is None: self.fit_binner(X)
    Xb = self.binner_.transform(X); n, p = Xb.shape
    levels = [tuple(int(v) for v in np.unique(Xb[:, j])) for j in range(p)]
    literals = [candidate_bin_subsets(levels[j], mode=self.literal_mode) for j in range(p)]
    Xb_i = np.asarray(Xb, dtype=np.int64); yf = y.astype(float)
    # per-feature lookup tables: table[c, code] = code allowed by literal c
    Kmax = int(Xb_i.max()) + 1 if n else 1; Cmax = max([len(l) for l in literals] + [1])
    tables = np.zeros((p, Cmax, Kmax), np.bool_)
    for j in range(p):
        for c, allowed in enumerate(literals[j]):
            for v in allowed:
                if 0 <= v < Kmax: tables[j, c, v] = True
    best_global = None; best_depth = 0; evaluated = 0
    beam_rules = [({}, np.ones(n, dtype=bool))]
    for depth in range(1, self.max_rule_features + 1):
        child_states = {}
        for restrictions, parent_mask in beam_rules:
            used = set(restrictions.keys())
            rows = np.flatnonzero(parent_mask).astype(np.int64)
            cj, cc = [], []
            for j in range(p):
                if j in used: continue
                for c in range(len(literals[j])): cj.append(j); cc.append(c)
            if not cj: continue
            res = score_bin_literals_multi(yf, logits, rows, np.ascontiguousarray(Xb_i[rows]), np.array(cj, np.int64), np.array(cc, np.int64),
                                           tables, float(self.l2), 12.0, 1e-10, 100, int(self.min_size))
            for r_i, (j, c) in enumerate(zip(cj, cc)):      # same (feature, literal) order as the original loops
                allowed = literals[j][c]
                if int(res[r_i, 0]) < self.min_size: continue
                evaluated += 1
                sc = _to_score(res[r_i], self.min_size, self.min_events, self.min_nonevents)
                if sc is None: continue
                new_restrictions = dict(restrictions); new_restrictions[j] = tuple(allowed)
                key = self._rule_key(new_restrictions)
                old = child_states.get(key)
                if old is None or sc.gain > old.score.gain:
                    mask = parent_mask & tables[j, c][Xb_i[:, j]]
                    child_states[key] = BeamState(restrictions=new_restrictions, mask=mask, score=sc)
        if not child_states: break
        ranked = sorted(child_states.values(), key=lambda st: st.score.gain, reverse=True)
        beam = ranked[: self.beam_width]
        if best_global is None or beam[0].score.gain > best_global.score.gain:
            best_global = beam[0]; best_depth = depth
        beam_rules = [(state.restrictions, state.mask) for state in beam]
    if best_global is None: raise RuntimeError("No valid structured subgroup found.")
    return BeamMDSSResult(gain=float(best_global.score.gain), delta=float(best_global.score.delta),
                          restrictions=dict(best_global.restrictions), mask=best_global.mask, score=best_global.score,
                          depth=best_depth, evaluated_candidates=evaluated)

_ORIG = {}
def install():
    import adaptive_structured_scanner_v2_1 as A, structured_mdss as S
    _ORIG.setdefault("num", A.AdaptiveStructuredScannerV2._numeric_candidates)
    _ORIG.setdefault("leg", S.StructuredBeamMDSS.scan)
    A.AdaptiveStructuredScannerV2._numeric_candidates = _numeric_candidates_batched
    S.StructuredBeamMDSS.scan = _legacy_scan_batched
def uninstall():
    import adaptive_structured_scanner_v2_1 as A, structured_mdss as S
    if _ORIG: A.AdaptiveStructuredScannerV2._numeric_candidates = _ORIG["num"]; S.StructuredBeamMDSS.scan = _ORIG["leg"]
