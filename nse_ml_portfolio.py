#!/usr/bin/env python3
"""
Reference implementation of the seven-layer walk-forward framework in
"Machine Learning for Portfolio Selection and Risk Management on the NSE".

  L1 data  ->  L2 features  ->  L3 six learners + rank-average ensemble
  ->  L4 top-K pre-selection + Grinold alphas  ->  L5 fuzzy c-means + real-coded GA
  ->  L6 EWMA volatility-targeting overlay  ->  L7 metrics, bootstrap, PSR/DSR

Usage
-----
  python nse_ml_portfolio.py --synthetic --models ridge,svr,rf,lgbm,mlp   # smoke test
  python nse_ml_portfolio.py --data-dir ./eod2_data/daily --universe symbols.txt \
         --index "NIFTY 50" --gold GOLDBEES

Data layout: one CSV per symbol in --data-dir with columns Date,Open,High,Low,Close,Volume.
Notes: written from the paper's specification, not the authors' original code; the
benchmarks min-variance, max-Sharpe and HRP are not included (1/N, NIFTY 50 and top-K EW are).
"""
import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

FEATURES = ["mom1", "mom3", "mom6", "mom12_1", "vol1m", "vol3m", "ivol", "beta",
            "maxret", "skew3m", "rsi14", "macd_h", "dist52w", "vlm_ratio"]


@dataclass
class Config:
    first_oos_year: int = 2015
    seq_len: int = 6
    val_months: int = 12
    K: int = 16
    w_max: float = 0.15
    n_clusters: int = 5
    fuzzifier: float = 2.0
    lam_c: float = 0.25
    lam_t: float = 0.10
    ga_pop: int = 80
    ga_gens: int = 60
    cost: float = 0.0015          # 15 bp one-way
    rf: float = 0.06
    ewma_lambda: float = 0.94
    exp_floor: float = 0.25
    exp_band: float = 0.10
    seed: int = 0


# --------------------------------------------------------------------------- L1: data
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_data(data_dir, universe_file, index_name, gold_name, demergers=None):
    symbols = [s.strip() for s in open(universe_file) if s.strip()]
    closes, vols = {}, {}
    for s in symbols:
        d = load_csv(os.path.join(data_dir, f"{s}.csv"))
        closes[s], vols[s] = d["close"], d["volume"]
    close, vol = pd.DataFrame(closes), pd.DataFrame(vols)
    nifty = load_csv(os.path.join(data_dir, f"{index_name}.csv"))["close"]
    gold = load_csv(os.path.join(data_dir, f"{gold_name}.csv"))["close"]
    if demergers:                                   # {symbol: [ex-dates]}
        close = neutralise_demergers(close, demergers)
    return close, vol, nifty, gold


def neutralise_demergers(close, fixes):
    """Replace the ex-date return by the equal-weighted cross-sectional return of that day."""
    r = close.pct_change()
    for sym, dates in fixes.items():
        for d in dates:
            d = pd.Timestamp(d)
            if d in r.index:
                r.loc[d, sym] = r.loc[d].drop(sym).mean()
    base = close.bfill().iloc[0]
    out = base * (1 + r.fillna(0)).cumprod()
    return out.where(close.notna())


def synthetic_data(n=48, seed=0):
    """Toy universe with a weak persistent-drift signal so the pipeline can be smoke-tested."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-06-01", "2026-08-31")
    T = len(dates)
    mkt = rng.standard_t(5, T) * 0.008 + 0.0004
    beta = rng.uniform(0.6, 1.3, n)
    months = dates.to_period("M")
    codes = pd.factorize(months)[0]
    a = np.zeros((codes.max() + 1, n))
    for m in range(1, len(a)):
        a[m] = 0.9 * a[m - 1] + rng.normal(0, 0.0004, n)
    drift = a[codes]
    r = mkt[:, None] * beta + drift + rng.normal(0, 0.012, (T, n))
    syms = [f"S{i:02d}" for i in range(n)]
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=syms)
    vol = pd.DataFrame(rng.lognormal(12, 0.4, (T, n)), index=dates, columns=syms)
    nifty = pd.Series(1000 * np.cumprod(1 + r.mean(1) * 0.9), index=dates)
    gold = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, T)), index=dates)
    return close, vol, nifty, gold


# --------------------------------------------------------------------------- L2: features
def daily_features(close, vol):
    r = close.pct_change()
    mkt = r.mean(axis=1)
    F = {}
    F["mom1"], F["mom3"], F["mom6"] = (close.pct_change(k) for k in (21, 63, 126))
    F["mom12_1"] = close.shift(21) / close.shift(252) - 1
    F["vol1m"] = r.rolling(21).std() * np.sqrt(252)
    F["vol3m"] = r.rolling(63).std() * np.sqrt(252)
    beta = r.rolling(252).cov(mkt).div(mkt.rolling(252).var(), axis=0)
    F["beta"] = beta
    F["ivol"] = (r - beta.mul(mkt, axis=0)).rolling(63).std() * np.sqrt(252)
    F["maxret"] = r.rolling(21).max()
    F["skew3m"] = r.rolling(63).skew()
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    F["rsi14"] = 100 - 100 / (1 + up / dn)
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    macd = ema(close, 12) - ema(close, 26)
    F["macd_h"] = (macd - ema(macd, 9)) / close
    F["dist52w"] = close / close.rolling(252).max() - 1
    F["vlm_ratio"] = vol.rolling(21).mean() / vol.rolling(126).mean() - 1
    return F


def rank01(df):
    """Eq. (1): cross-sectional percentile rank, min-max scaled to [0, 1]."""
    rk = df.rank(axis=1)
    mn, mx = rk.min(axis=1), rk.max(axis=1)
    return rk.sub(mn, axis=0).div((mx - mn).replace(0, np.nan), axis=0)


def build_panel(close, vol):
    F = daily_features(close, vol)
    me = close.index.to_series().groupby(close.index.to_period("M")).last().values
    me = pd.DatetimeIndex(me)
    X = np.stack([rank01(F[k].loc[me]).values for k in FEATURES], axis=-1)     # [T, N, F]
    avail = ~np.isnan(X).any(axis=-1) & close.loc[me].notna().values
    X = np.nan_to_num(X, nan=0.5)
    px = close.loc[me]
    nxt = px.shift(-1) / px - 1
    y = (nxt.rank(axis=1) / nxt.notna().sum(axis=1).values[:, None] - 0.5).values  # centred rank
    return me, X, y, avail


# --------------------------------------------------------------------------- L3: learners
def fit_ridge(p, X, y, Xv, yv, n):
    m = Ridge(alpha=p["alpha"]).fit(X, y)
    return m.predict, None


def fit_svr(p, X, y, Xv, yv, n):
    m = SVR(kernel="rbf", gamma="scale", epsilon=0.05, C=p["C"]).fit(X, y)
    return m.predict, None


def fit_rf(p, X, y, Xv, yv, n):
    m = RandomForestRegressor(300, max_depth=p["depth"], min_samples_leaf=20,
                              max_features=0.5, n_jobs=-1, random_state=0).fit(X, y)
    return m.predict, None


def fit_lgbm(p, X, y, Xv, yv, n):
    import lightgbm as lgb
    kw = dict(learning_rate=0.02, num_leaves=p["leaves"], min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1)
    if n is None:
        m = lgb.LGBMRegressor(n_estimators=2000, **kw)
        m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return m.predict, max(int(m.best_iteration_ or 1), 1)
    return lgb.LGBMRegressor(n_estimators=n, **kw).fit(X, y).predict, n


def _keras(kind, p, X, y, Xv, yv, n):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    tf.keras.utils.set_random_seed(0)
    l2 = regularizers.l2(p["l2"])
    if kind == "mlp":
        net = [layers.Input((X.shape[1],)), layers.Dense(32, "relu", kernel_regularizer=l2),
               layers.Dropout(0.2), layers.Dense(16, "relu", kernel_regularizer=l2),
               layers.Dropout(0.2), layers.Dense(1)]
    else:
        net = [layers.Input(X.shape[1:]), layers.LSTM(16, dropout=0.2, kernel_regularizer=l2),
               layers.Dense(1)]
    m = tf.keras.Sequential(net)
    m.compile(tf.keras.optimizers.Adam(1e-3), "mse")
    if n is None:
        es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        h = m.fit(X, y, validation_data=(Xv, yv), epochs=300, batch_size=128,
                  callbacks=[es], verbose=0)
        n = int(np.argmin(h.history["val_loss"])) + 1
    else:
        m.fit(X, y, epochs=n, batch_size=128, verbose=0)
    return (lambda Z: m.predict(Z, verbose=0).ravel()), n


def fit_mlp(p, X, y, Xv, yv, n):
    try:
        return _keras("mlp", p, X, y, Xv, yv, n)
    except ImportError:                                  # fallback if TensorFlow is missing
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=p["l2"], learning_rate_init=1e-3, batch_size=128,
                         early_stopping=True, n_iter_no_change=10, max_iter=300,
                         random_state=0).fit(X, y)
        return m.predict, None


def fit_lstm(p, X, y, Xv, yv, n):
    return _keras("lstm", p, X, y, Xv, yv, n)


SPECS = {
    "ridge": dict(grid=[{"alpha": a} for a in (1, 10, 100, 1000)], fit=fit_ridge, seq=False),
    "svr":   dict(grid=[{"C": c} for c in (0.1, 1)], fit=fit_svr, seq=False),
    "rf":    dict(grid=[{"depth": d} for d in (3, 5, 8)], fit=fit_rf, seq=False),
    "lgbm":  dict(grid=[{"leaves": k} for k in (7, 15)], fit=fit_lgbm, seq=False),
    "mlp":   dict(grid=[{"l2": a} for a in (1e-4, 1e-3)], fit=fit_mlp, seq=False),
    "lstm":  dict(grid=[{"l2": 1e-3}], fit=fit_lstm, seq=True),
}


def make_samples(X, y, avail, ts, seq_len=None, need_y=True):
    xs, ys, tix, six = [], [], [], []
    for t in ts:
        if seq_len and t < seq_len - 1:
            continue
        ok = avail[t] & (~np.isnan(y[t]) if need_y else True)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            continue
        xs.append(X[t - seq_len + 1:t + 1][:, idx, :].transpose(1, 0, 2) if seq_len else X[t, idx])
        ys.append(y[t, idx])
        tix.append(np.full(len(idx), t))
        six.append(idx)
    if not xs:
        return None, None, None, None
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(tix), np.concatenate(six)


def mean_ic(pred, yv, tix):
    ics = [spearmanr(pred[tix == t], yv[tix == t])[0] for t in np.unique(tix) if (tix == t).sum() > 5]
    v = np.nanmean(ics) if len(ics) else np.nan
    return -1.0 if np.isnan(v) else v


def walk_forward(X, y, avail, me, model_names, cfg):
    """Annual (December) re-training on an expanding window; last 12 months = validation."""
    T, N, _ = X.shape
    preds = {m: np.full((T, N), np.nan) for m in model_names}
    train_ic = {m: [] for m in model_names}
    years = sorted(set(me.year))
    for Y in range(cfg.first_oos_year, years[-1] + 1):
        d0 = np.where((me.year == Y - 1) & (me.month == 12))[0]
        if len(d0) == 0:
            continue
        d0 = d0[0]
        fdates = list(range(d0, min(d0 + 12, T - 1)))             # Dec(Y-1) .. Nov(Y)
        ts = [t for t in range(d0) if avail[t].any()]             # labels observable at d0
        ts_val, ts_fit = ts[-cfg.val_months:], ts[:-cfg.val_months]
        print(f"  model {Y}: train {len(ts_fit)} + val {len(ts_val)} months, "
              f"{len(fdates)} forecasts", flush=True)
        for name in model_names:
            spec = SPECS[name]
            sl = cfg.seq_len if spec["seq"] else None
            Xf, yf, _, _ = make_samples(X, y, avail, ts_fit, sl)
            Xv, yv, tv, _ = make_samples(X, y, avail, ts_val, sl)
            best = None
            for p in spec["grid"]:
                pr, n = spec["fit"](p, Xf, yf, Xv, yv, None)
                ic = mean_ic(pr(Xv), yv, tv)
                if best is None or ic > best[0]:
                    best = (ic, p, n)
            Xa, ya, ta, _ = make_samples(X, y, avail, ts, sl)
            pr, _ = spec["fit"](best[1], Xa, ya, None, None, best[2])
            train_ic[name].append(mean_ic(pr(Xa), ya, ta))
            for t in fdates:
                Xt, _, _, st = make_samples(X, y, avail, [t], sl, need_y=False)
                if Xt is not None:
                    preds[name][t, st] = pr(Xt)
    return preds, train_ic


def rank_average(preds, names):
    """Eq. (2): equal-weighted average of cross-sectional ranks of the forecasts."""
    out = 0
    for m in names:
        p = pd.DataFrame(preds[m])
        out = out + p.rank(axis=1).div(p.notna().sum(axis=1), axis=0).values
    return out / len(names)


def ic_series(score, y):
    ic = np.full(len(score), np.nan)
    for t in range(len(score)):
        ok = ~np.isnan(score[t]) & ~np.isnan(y[t])
        if ok.sum() > 5:
            ic[t] = spearmanr(score[t, ok], y[t, ok])[0]
    return ic


# --------------------------------------------------------------------------- L5: FCM + GA
def fcm_embedding(R):
    C = np.corrcoef(np.nan_to_num(R).T)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    return V[:, 1:6] * np.sqrt(np.maximum(w[1:6], 0))          # drop market mode, keep next 5


def fuzzy_cmeans(E, c=5, m=2.0, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.dirichlet(np.ones(c), size=len(E))
    pw = 2.0 / (m - 1.0)
    for _ in range(iters):
        Um = U ** m
        V = (Um.T @ E) / Um.sum(0)[:, None]
        D = np.linalg.norm(E[:, None, :] - V[None], axis=2) + 1e-12
        inv = D ** (-pw)
        Un = inv / inv.sum(1, keepdims=True)
        done = np.abs(Un - U).max() < 1e-5
        U = Un
        if done:
            break
    return U


def project_capped_simplex(V, cap):
    """Row-wise projection onto {0 <= w <= cap, sum w = 1} by bisection on the shift tau."""
    V = np.atleast_2d(V)
    lo, hi = V.min(1) - cap, V.max(1)
    for _ in range(60):
        tau = (lo + hi) / 2
        s = np.clip(V - tau[:, None], 0, cap).sum(1)
        lo, hi = np.where(s > 1, tau, lo), np.where(s > 1, hi, tau)
    return np.clip(V - ((lo + hi) / 2)[:, None], 0, cap)


def ga_optimise(alpha, Sigma, U, wprev, outside_prev, cfg, rng):
    """Real-coded GA maximising Eq. (5); 1/N is seeded into the initial population."""
    K, C = len(alpha), U.shape[1]
    cap = cfg.w_max

    def fitness(W):
        vol = np.sqrt(np.einsum("pi,ij,pj->p", W, Sigma, W)) + 1e-12
        conc = ((W @ U) ** 2).sum(1) - 1.0 / C
        turn = np.abs(W - wprev).sum(1) + outside_prev
        return W @ alpha / vol - cfg.lam_c * conc - cfg.lam_t * turn

    pop = np.vstack([np.full((1, K), 1.0 / K),
                     project_capped_simplex(rng.dirichlet(np.ones(K), cfg.ga_pop - 1), cap)])
    for _ in range(cfg.ga_gens):
        f = fitness(pop)
        elite = pop[np.argsort(f)[::-1][:2]]
        n_ch = cfg.ga_pop - 2
        pick = lambda: pop[(lambda i: i[np.arange(n_ch), np.argmax(f[i], 1)])(
            rng.integers(0, cfg.ga_pop, (n_ch, 3)))]
        p1, p2 = pick(), pick()
        lo, d = np.minimum(p1, p2), np.abs(p1 - p2)
        blx = lo - 0.3 * d + rng.random((n_ch, K)) * d * (1 + 2 * 0.3)       # BLX-alpha, 0.3
        child = np.where(rng.random((n_ch, 1)) < 0.9, blx, p1)
        child = child + (rng.random((n_ch, K)) < 0.1) * rng.normal(0, 0.05, (n_ch, K))
        pop = np.vstack([elite, project_capped_simplex(np.clip(child, 0, None), cap)])
    return pop[np.argmax(fitness(pop))]


# --------------------------------------------------------------------------- L6: overlay
def vol_target(r_eq, r_def, sigma_star, cfg, seed_var):
    """Eqs. (6)-(8): EWMA volatility targeting with floor, band and costs."""
    var, e_prev = seed_var, 1.0
    out, exp = np.empty(len(r_eq)), np.empty(len(r_eq))
    for k in range(len(r_eq)):
        e_raw = min(1.0, max(cfg.exp_floor, sigma_star / (np.sqrt(252 * var) + 1e-12)))
        at_bound = e_raw in (1.0, cfg.exp_floor)
        e = e_raw if (abs(e_raw - e_prev) >= cfg.exp_band or at_bound) else e_prev
        out[k] = e * r_eq[k] + (1 - e) * r_def[k] - cfg.cost * abs(e - e_prev)
        exp[k] = e
        e_prev = e
        var = cfg.ewma_lambda * var + (1 - cfg.ewma_lambda) * r_eq[k] ** 2
    return out, exp


# --------------------------------------------------------------------------- L7: evaluation
def metrics(r, rf=0.06, turnover=np.nan):
    r = pd.Series(r).dropna()
    yrs = len(r) / 252
    nav = (1 + r).cumprod()
    ex = r - rf / 252
    dd = nav / nav.cummax() - 1
    down = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    return dict(CAGR=100 * cagr, Vol=100 * r.std() * np.sqrt(252),
                Sharpe=ex.mean() / r.std() * np.sqrt(252),
                Sortino=ex.mean() / down * np.sqrt(252), MDD=100 * dd.min(),
                Calmar=cagr / abs(dd.min()), CVaR95=100 * r[r <= r.quantile(0.05)].mean(),
                Turnover=turnover)


def stationary_bootstrap_idx(n, mean_block, rng):
    p = 1.0 / mean_block
    new = rng.random(n) < p
    new[0] = True
    pos = np.arange(n)
    start = rng.integers(0, n, n)
    last = np.maximum.accumulate(np.where(new, pos, 0))
    return (start[last] + pos - last) % n


def sharpe_diff_test(r1, r2, rf=0.06, B=2000, block=21, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(r1) - rf / 252, np.asarray(r2) - rf / 252
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)
    d0 = sr(a) - sr(b)
    ds = np.empty(B)
    for i in range(B):
        ix = stationary_bootstrap_idx(len(a), block, rng)
        ds[i] = sr(a[ix]) - sr(b[ix])
    p = np.mean(np.abs(ds - d0) >= abs(d0))
    return d0, np.percentile(ds, [2.5, 97.5]), p


def psr(r, sr0=0.0, rf=0.06):
    """Eq. (9) on daily excess returns (raw kurtosis)."""
    x = np.asarray(r) - rf / 252
    s = x.mean() / x.std()
    g3 = pd.Series(x).skew()
    g4 = pd.Series(x).kurt() + 3
    return norm.cdf((s - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * s + (g4 - 1) / 4 * s ** 2))


def dsr(r, trial_daily_sharpes, rf=0.06):
    """Eq. (10): deflate by the expected max Sharpe of N zero-skill trials."""
    N, g = len(trial_daily_sharpes), 0.5772156649
    sr0 = np.sqrt(np.var(trial_daily_sharpes, ddof=1)) * (
        (1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
    return psr(r, sr0, rf)


# --------------------------------------------------------------------------- orchestration
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir")
    ap.add_argument("--universe")
    ap.add_argument("--index", default="NIFTY 50")
    ap.add_argument("--gold", default="GOLDBEES")
    ap.add_argument("--models", default="ridge,svr,rf,lgbm,mlp,lstm")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = Config()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    names = [m for m in a.models.split(",") if m]

    if a.synthetic:
        close, vol, nifty, gold = synthetic_data()
    else:
        demergers = {"GRASIM": ["2017-07-19"], "RELIANCE": ["2023-07-20"], "ITC": ["2025-01-06"]}
        listed = open(a.universe).read().split()
        demergers = {k: v for k, v in demergers.items() if k in listed}
        close, vol, nifty, gold = load_data(a.data_dir, a.universe, a.index, a.gold, demergers)
    close, vol = close.sort_index(), vol.reindex(close.index)
    R = close.pct_change().fillna(0.0).values
    mkt = close.pct_change().mean(axis=1)
    nifty_r = nifty.pct_change().reindex(close.index).fillna(0.0)
    gold_r = gold.pct_change().reindex(close.index).fillna(0.0).values
    N = close.shape[1]

    print("Building features ...")
    me, X, y, avail = build_panel(close, vol)
    print("Walk-forward training ...")
    preds, train_ic = walk_forward(X, y, avail, me, names, cfg)
    ens = rank_average(preds, names)

    # ---- L3 diagnostics: Table 6
    rows = {}
    for m in names + ["ensemble"]:
        ic = ic_series(ens if m == "ensemble" else preds[m], y)
        v = ic[~np.isnan(ic)]
        rows[m] = dict(meanIC=v.mean(), ICIR_ann=v.mean() / v.std() * np.sqrt(12),
                       t=v.mean() / v.std() * np.sqrt(len(v)), hit=(v > 0).mean(),
                       train_IC=np.mean(train_ic[m]) if m in train_ic else np.nan)
    table6 = pd.DataFrame(rows).T
    print("\nOut-of-sample forecast quality (Table 6)\n", table6.round(3))
    table6.to_csv(f"{a.out}/table6_forecast_quality.csv")
    ic_ens = ic_series(ens, y)

    # ---- L4/L5: portfolio construction at each month-end
    pos = close.index.get_indexer(me)
    oos = [t for t in range(len(me) - 1) if not np.isnan(ens[t]).all()]

    def ic_prior(t):
        past = ic_ens[:t][~np.isnan(ic_ens[:t])]            # only fully observed months
        return 0.03 if len(past) < 12 else max(past.mean(), 0.01)

    def top_k(t, scores):
        s = np.where(np.isnan(scores), -np.inf, scores)
        return np.argsort(-s)[:cfg.K]

    def w_equal(t, wprev):
        w = np.zeros(N); ok = avail[t]; w[ok] = 1 / ok.sum(); return w

    def make_topk_ew(scores):
        def f(t, wprev):
            w = np.zeros(N); w[top_k(t, scores[t])] = 1 / cfg.K; return w
        return f

    def w_ga(t, wprev, use_fcm=True, use_tc=True):
        p = pos[t]
        win = R[p - 251:p + 1]
        cand = top_k(t, ens[t])
        s = ens[t]
        z = (s - np.nanmean(s)) / np.nanstd(s)
        sig = win.std(0) * np.sqrt(252)
        alpha = ic_prior(t) * sig[cand] * z[cand]
        Sigma = LedoitWolf().fit(win[:, cand]).covariance_ * 252
        U = fuzzy_cmeans(fcm_embedding(win), cfg.n_clusters, cfg.fuzzifier,
                         seed=cfg.seed)[cand] if use_fcm else np.ones((len(cand), 1))
        cfg_local = Config(**cfg.__dict__)
        if not use_fcm:
            cfg_local.lam_c = 0.0
        if not use_tc:
            cfg_local.lam_t = 0.0
        outside = wprev.sum() - wprev[cand].sum()
        wc = ga_optimise(alpha, Sigma, U, wprev[cand], outside, cfg_local, rng)
        w = np.zeros(N); w[cand] = wc
        return w

    def run(construct):
        """Daily simulation with drifting weights and proportional costs."""
        w = np.zeros(N); rets, turns = [], []
        for t in oos:
            target = construct(t, w)
            turn = np.abs(target - w).sum(); turns.append(turn)
            w = target.copy()
            for k, d in enumerate(range(pos[t] + 1, pos[t + 1] + 1)):
                rd = R[d]
                rets.append(w @ rd - (cfg.cost * turn if k == 0 else 0.0))
                w = w * (1 + rd); w = w / w.sum()
        idx = close.index[pos[oos[0]] + 1: pos[oos[-1] + 1] + 1]
        yrs = len(idx) / 252
        return pd.Series(rets, index=idx), sum(turns[1:]) / yrs

    strategies = {}
    print("\nSimulating strategies ...")
    strategies["1/N"] = run(w_equal)
    for m in names:
        strategies[f"{m} top-{cfg.K} EW"] = run(make_topk_ew(
            np.where(np.isnan(preds[m]), np.nan, pd.DataFrame(preds[m]).rank(axis=1).values)))
    strategies["Ensemble top-K EW"] = run(make_topk_ew(ens))
    strategies["Ensemble GA+FCM+TC"] = run(w_ga)

    # ---- L6: overlay
    base = strategies["Ensemble GA+FCM+TC"][0]
    sigma_star = nifty_r.loc["2012-03":"2014-12"].std() * np.sqrt(252)
    seed_var = mkt.loc[:base.index[0]].tail(63).var()
    g = pd.Series(gold_r, index=close.index).reindex(base.index).values
    rf_d = np.full(len(base), cfg.rf / 252)
    for lab, rdef in (("cash", rf_d), ("gold", g)):
        r_o, exp = vol_target(base.values, rdef, sigma_star, cfg, seed_var)
        strategies[f"Proposed (overlay -> {lab})"] = (pd.Series(r_o, index=base.index),
                                                       strategies["Ensemble GA+FCM+TC"][1])
        if lab == "cash":
            print(f"Average exposure {exp.mean():.2f}; fully invested {np.mean(exp == 1):.1%}; "
                  f"at floor {np.mean(exp == cfg.exp_floor):.1%}")
    ew = strategies["1/N"][0]
    r_o, _ = vol_target(ew.values, rf_d, sigma_star, cfg, seed_var)
    strategies["1/N + overlay -> cash"] = (pd.Series(r_o, index=ew.index), strategies["1/N"][1])
    strategies["NIFTY 50"] = (nifty_r.reindex(base.index), np.nan)

    # ---- L7: reporting
    tab = pd.DataFrame({k: metrics(v[0], cfg.rf, v[1]) for k, v in strategies.items()}).T
    print("\nNet performance, 15 bp one-way (Table 7)\n", tab.round(2))
    tab.to_csv(f"{a.out}/table7_performance.csv")

    ml_keys = [k for k in strategies if k not in ("1/N", "NIFTY 50", "1/N + overlay -> cash")]
    trial_sr = np.array([(strategies[k][0] - cfg.rf / 252).mean() / strategies[k][0].std()
                         for k in ml_keys])
    inf = []
    for k, (r, _) in strategies.items():
        d, ci, p = sharpe_diff_test(r.values, ew.values) if k != "1/N" else (0, (0, 0), np.nan)
        inf.append(dict(strategy=k, dSharpe_vs_1N=d, CI_lo=ci[0], CI_hi=ci[1], p=p,
                        PSR0=psr(r.values), DSR=dsr(r.values, trial_sr)))
    inf = pd.DataFrame(inf).set_index("strategy")
    print("\nInference vs 1/N (Table 10)\n", inf.round(3))
    inf.to_csv(f"{a.out}/table10_inference.csv")
    pd.DataFrame({k: v[0] for k, v in strategies.items()}).to_csv(f"{a.out}/daily_returns.csv")


if __name__ == "__main__":
    main()
