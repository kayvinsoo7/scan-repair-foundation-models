"""Tuned tree base models (XGBoost, CatBoost) on exactly the ladder's splits, saved in the ladder's .npz format so
corrections.py applies every correction to them unchanged.

Protocol ("each model at its best"):
  * the model sees only the TRAINING split (the same rows TabPFN/TabICL get as their full context);
  * hyperparameters: 5-fold cross-validated random search (N_CONFIGS configurations, log-loss, early stopping)
    on the first seed's training split of each dataset, then fixed for all seeds;
  * number of trees: the cross-validated best iteration (mean over folds) x 1.1, refit on the whole training split.
Usage: python base_trees.py xgboost|catboost [out_root]      (default out_root: <artistry>/logits)
Discovery-size arm: DATASETS="synth8x4d4 synth1x4d4" python base_trees.py xgboost <artistry>/logits/discovery
  (reuses the hyperparameters tuned for synth8/synth1, read from TUNED_DIR, default <artistry>/logits/<model>)."""
import os, sys, time, json, numpy as np, pandas as pd
from common import *
from ladder_data import load_dataset, DATASET_SEEDS

MODEL = sys.argv[1]; OUT_ROOT = sys.argv[2] if len(sys.argv) > 2 else str(ARTISTRY / "logits")
OUT = os.path.join(OUT_ROOT, MODEL); os.makedirs(OUT, exist_ok=True)
N_CONFIGS = int(os.environ.get("N_CONFIGS", "25")); THREADS = int(os.environ.get("THREADS", "2"))
DATASETS = os.environ.get("DATASETS", "synth8 synth1 credit").split()

def xgb_space(rng):
    return dict(max_depth=int(rng.integers(2, 9)), learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                min_child_weight=float(np.exp(rng.uniform(0, np.log(30)))), subsample=float(rng.uniform(0.6, 1.0)),
                colsample_bytree=float(rng.uniform(0.5, 1.0)), reg_lambda=float(np.exp(rng.uniform(np.log(0.1), np.log(20)))),
                reg_alpha=float(np.exp(rng.uniform(np.log(1e-3), np.log(5)))))
def cat_space(rng):
    return dict(depth=int(rng.integers(3, 9)), learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                l2_leaf_reg=float(np.exp(rng.uniform(0, np.log(30)))), random_strength=float(rng.uniform(0, 2)),
                bagging_temperature=float(rng.uniform(0, 1)))

def frame(D, idx):
    df = pd.DataFrame(D.X[idx], columns=D.feature_names)
    for c in D.categorical_names: df[c] = df[c].astype(int)
    return df

def tune(D):
    rng = np.random.default_rng(12345); X, y = D.X[D.train], D.y[D.train]; best = None
    for i in range(N_CONFIGS):
        if MODEL == "xgboost":
            import xgboost as xgb
            p = xgb_space(rng); params = dict(p, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                              nthread=THREADS, seed=0, verbosity=0)
            cv = xgb.cv(params, xgb.DMatrix(X, label=y), num_boost_round=3000, nfold=5, stratified=True,
                        early_stopping_rounds=50, seed=0, verbose_eval=False)
            score, n = float(cv["test-logloss-mean"].iloc[-1]), len(cv)
        else:
            from catboost import Pool, cv as ccv
            p = cat_space(rng)
            params = dict(p, loss_function="Logloss", iterations=3000, thread_count=THREADS, random_seed=0, logging_level="Silent",
                          od_type="Iter", od_wait=50)
            cvr = ccv(Pool(frame(D, D.train), y, cat_features=D.categorical_names or None), params, fold_count=5,
                      stratified=True, seed=0, early_stopping_rounds=50, logging_level="Silent")
            score = float(cvr["test-Logloss-mean"].min()); n = int(cvr["test-Logloss-mean"].idxmin()) + 1
        print(f"    config {i+1:2d}/{N_CONFIGS}: cv log-loss {score:.5f}, trees {n}", flush=True)
        if best is None or score < best[0]: best = (score, p, n)
    return dict(cv_logloss=best[0], params=best[1], n_trees=int(np.ceil(best[2] * 1.1)))

def fit_predict(D, cfg, seed):
    t0 = time.perf_counter()
    if MODEL == "xgboost":
        import xgboost as xgb
        params = dict(cfg["params"], objective="binary:logistic", tree_method="hist", nthread=THREADS, seed=seed, verbosity=0)
        m = xgb.train(params, xgb.DMatrix(D.X[D.train], label=D.y[D.train]), num_boost_round=cfg["n_trees"])
        t_fit = time.perf_counter() - t0; t0 = time.perf_counter()
        z = {p: m.predict(xgb.DMatrix(D.X[getattr(D, p)]), output_margin=True) for p in ("discovery", "selection", "test")}
    else:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(**cfg["params"], iterations=cfg["n_trees"], loss_function="Logloss", thread_count=THREADS,
                               random_seed=seed, verbose=False, cat_features=D.categorical_names or None)
        m.fit(frame(D, D.train), D.y[D.train]); t_fit = time.perf_counter() - t0; t0 = time.perf_counter()
        z = {p: m.predict(frame(D, getattr(D, p)), prediction_type="RawFormulaVal") for p in ("discovery", "selection", "test")}
    return z, t_fit, time.perf_counter() - t0

info = dict(backend=MODEL, version=MODEL, device=f"cpu ({THREADS} threads)")
for ds in DATASETS:
    seeds = DATASET_SEEDS[ds]
    # discovery-size arm: same generator and training size as the main synthetic data -> reuse its tuned settings
    cfg_f = os.path.join(os.environ.get("TUNED_DIR", str(ARTISTRY / "logits" / MODEL)), f"tuned_{ds[:ds.index('x4')] if 'x4' in ds else ds}.json")
    if os.path.exists(cfg_f): cfg = json.load(open(cfg_f))
    else:
        print(f"[{MODEL}] tuning on {ds} (seed {seeds[0]} training split)", flush=True)
        t0 = time.perf_counter(); cfg = tune(load_dataset(ds, seeds[0])); cfg["tuning_seconds"] = time.perf_counter() - t0
        json.dump(cfg, open(cfg_f, "w"), indent=1); print(f"  best: {cfg}", flush=True)
    for seed in seeds:
        f = os.path.join(OUT, f"{ds}_seed{seed}_ctxfull.npz")
        if os.path.exists(f): continue
        D = load_dataset(ds, seed); z, t_fit, t_pred = fit_predict(D, cfg, seed)
        np.savez_compressed(f, zd=z["discovery"], zs=z["selection"], zt=z["test"], ctx_idx=D.train,
                            yd=D.y[D.discovery], ys=D.y[D.selection], yt=D.y[D.test], fit_seconds=t_fit, predict_seconds=t_pred,
                            rows_predicted=len(D.discovery) + len(D.selection) + len(D.test), predict_seconds_test=0.0,
                            context_rows=len(D.train), info=json.dumps(info))
        print(f"  {MODEL} {ds} seed {seed}: fit {t_fit:.1f}s", flush=True)
print("done", MODEL)
