from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2, jarque_bera, kurtosis, norm, skew

warnings.filterwarnings("ignore")

# KONFIGURASI

START_DATE = "2012-10-01"
END_DATE   = "2024-04-29"

# Stock split MAPI — rasio 1:10 pada 2, 4, 14 Mei 2018
MAPI_SPLIT_DATES = ["2018-05-02", "2018-05-04", "2018-05-14"]

TRADING_DAYS = 252
ALPHA = 0.05            # Tingkat signifikansi untuk VaR/CVaR
BACKTEST_WINDOW = 252   # Rolling window untuk VaR backtest

# Bobot Stability Score (konsisten dengan paper)
WEIGHTS = {
    "Volatility_Pct": 0.20,
    "VaR_Pct":        0.10,
    "CVaR_Pct":       0.30,
    "DD_Pct":         0.15,
    "MDD_Pct":        0.25,
}

SECTOR_MAP = {
    "BBCA": "Financials", "BBNI": "Financials", "BBRI": "Financials", "BMRI": "Financials",
    "ADRO": "Energy", "AKRA": "Energy", "ITMG": "Energy", "MEDC": "Energy",
    "PGAS": "Energy", "PTBA": "Energy",
    "ANTM": "Basic Materials", "BRPT": "Basic Materials", "INCO": "Basic Materials",
    "INKP": "Basic Materials", "INTP": "Basic Materials", "SMGR": "Basic Materials",
    "AMRT": "Consumer Non-Cyclicals", "CPIN": "Consumer Non-Cyclicals",
    "GGRM": "Consumer Non-Cyclicals", "ICBP": "Consumer Non-Cyclicals",
    "INDF": "Consumer Non-Cyclicals", "UNVR": "Consumer Non-Cyclicals",
    "ACES": "Consumer Cyclicals", "MAPI": "Consumer Cyclicals",
    "ASII": "Industrials", "UNTR": "Industrials",
    "KLBF": "Healthcare",
    "TLKM": "Infrastructures", "EXCL": "Infrastructures",
}


# UTILITIES

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(iterable, desc: str = ""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable


# STEP 1: LOAD & CLEAN

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    log(f"Memuat {csv_path}...")
    df = pd.read_csv(csv_path, sep=";")
    log(f"  baris awal: {len(df):,}")

    # --- Parse & normalisasi ---
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df["Stock_Name"] = df["Stock_Name"].str.strip()

    # --- Hapus inkonsistensi OHLC ---
    inconsistent = (
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"]) |
        (df["Low"]  > df["Open"]) |
        (df["Low"]  > df["Close"]) |
        (df["High"] < df["Low"])
    )
    n_bad = int(inconsistent.sum())
    df = df[~inconsistent].copy()
    log(f"  dihapus {n_bad} baris inkonsisten, tersisa {len(df):,}")

    # --- Filter periode ---
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].copy()
    log(f"  setelah filter periode [{START_DATE} — {END_DATE}]: {len(df):,} baris")

    # --- Fix MAPI stock split (1:10, Mei 2018) ---
    mapi_mask = (df["Stock_Name"] == "MAPI") & (
        df["Date"].dt.strftime("%Y-%m-%d").isin(MAPI_SPLIT_DATES)
    )
    n_mapi = int(mapi_mask.sum())
    if n_mapi > 0:
        df.loc[mapi_mask, ["Open", "High", "Low", "Close"]] *= 10
        log(f"  fix MAPI stock split: {n_mapi} baris dikalikan 10")

    # --- Tambah sektor ---
    df["Sector"] = df["Stock_Name"].map(SECTOR_MAP)

    # --- Sort final ---
    df = df.sort_values(["Stock_Name", "Date"]).reset_index(drop=True)

    return df


# STEP 2: FEATURE ENGINEERING

def add_returns_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    log("Menghitung return & rolling volatility...")

    g = df.groupby("Stock_Name")["Close"]
    df["Return"] = g.pct_change()
    df["Log_Return"] = g.transform(lambda x: np.log(x / x.shift(1)))

    lr = df.groupby("Stock_Name")["Log_Return"]
    for window in (5, 20, 60):
        df[f"Volatility_{window}d"] = lr.transform(
            lambda x: x.rolling(window=window, min_periods=window).std()
        )

    return df


# STEP 3: JARQUE-BERA

def compute_jarque_bera(df: pd.DataFrame) -> pd.DataFrame:
    log("Menghitung Jarque-Bera per saham...")
    rows = []
    for stock, grp in df.groupby("Stock_Name"):
        data = grp["Log_Return"].dropna()
        jb_stat, p_value = jarque_bera(data)
        rows.append({
            "Stock_Name": stock,
            "JB_Stat": jb_stat,
            "p_value": p_value,
            "Skewness": float(data.skew()),
            "Excess_Kurtosis": float(data.kurtosis()),
            "Normal_Distribution": "Yes" if p_value > 0.05 else "No",
        })
    return pd.DataFrame(rows).sort_values("JB_Stat", ascending=False).reset_index(drop=True)


# STEP 4: GARCH VOLATILITY SERIES

def compute_garch_vol_series(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit GARCH(1,1) per saham (mean='Constant', dist='normal'), simpan
    conditional volatility time series + mean annualized per saham.

    Return:
        - vol_df: Date, Stock_Name, garch_vol, garch_vol_annual
        - mean_annual_map: {stock: mean annualized vol}  (untuk metrik akhir)
    """
    log("Fitting GARCH(1,1) per saham (conditional vol)...")

    all_records = []
    mean_annual = {}

    for stock in progress(df["Stock_Name"].unique(), desc="GARCH vol"):
        grp = df[df["Stock_Name"] == stock].sort_values("Date").reset_index(drop=True)
        x = grp["Log_Return"].dropna()
        if len(x) < 50:
            continue

        try:
            model = arch_model(
                x * 100,  # arch lebih stabil di skala persen
                vol="Garch", p=1, q=1, mean="Constant", dist="normal",
            )
            res = model.fit(disp="off")
            cond_vol = res.conditional_volatility.values / 100.0
            cond_vol_ann = cond_vol * np.sqrt(TRADING_DAYS)

            # Align ke tanggal non-NaN
            valid_idx = x.index
            dates_valid = grp.loc[valid_idx, "Date"].values

            for d, v, va in zip(dates_valid, cond_vol, cond_vol_ann):
                all_records.append({
                    "Date": d,
                    "Stock_Name": stock,
                    "garch_vol": v,
                    "garch_vol_annual": va,
                })

            mean_annual[stock] = float(np.mean(cond_vol_ann))
        except Exception as e:
            log(f"  ! GARCH gagal untuk {stock}: {e}")

    vol_df = pd.DataFrame(all_records)
    log(f"  selesai: {len(mean_annual)} saham, {len(vol_df):,} baris time series")
    return vol_df, mean_annual


# STEP 5: VaR POINT ESTIMATES (untuk ranking akhir)

def _garch_var_last(series: pd.Series, alpha: float = ALPHA) -> float:
    x = series.dropna()
    if len(x) < 50:
        return np.nan
    try:
        model = arch_model(x * 100, vol="Garch", p=1, q=1,
                          mean="Constant", dist="normal")
        res = model.fit(disp="off")
        mu = res.params.get("mu", 0.0)
        sigma_t = res.conditional_volatility.iloc[-1]
        z = norm.ppf(alpha)
        return (mu + z * sigma_t) / 100.0
    except Exception:
        return np.nan


def _historical_var(series: pd.Series, alpha: float = ALPHA) -> float:
    x = series.dropna()
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, alpha))


def _cornish_fisher_var(series: pd.Series, alpha: float = ALPHA) -> float:
    x = series.dropna()
    if len(x) < 20:
        return np.nan
    mu = x.mean()
    sigma = x.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    S = skew(x, bias=False)
    K = kurtosis(x, fisher=True, bias=False)
    z = norm.ppf(alpha)
    z_cf = (
        z
        + (1/6)  * (z**2 - 1)      * S
        + (1/24) * (z**3 - 3*z)    * K
        - (1/36) * (2*z**3 - 5*z)  * (S**2)
    )
    return mu + z_cf * sigma


def compute_var_point_estimates(df: pd.DataFrame) -> pd.DataFrame:
    log("Menghitung VaR point estimates (historical, CF, GARCH)...")
    rows = []
    for stock, grp in progress(df.groupby("Stock_Name"), desc="VaR point"):
        x = grp["Log_Return"]
        hv = _historical_var(x, ALPHA)
        cf = _cornish_fisher_var(x, ALPHA)
        gv = _garch_var_last(x, ALPHA)
        rows.append({
            "Stock_Name": stock,
            "Risk_HistVaR":  abs(hv) if pd.notna(hv) else np.nan,
            "Risk_CFVaR":    abs(cf) if pd.notna(cf) else np.nan,
            "Risk_GARCHVaR": abs(gv) if pd.notna(gv) else np.nan,
        })
    return pd.DataFrame(rows)


# STEP 6: CVaR HISTORIS

def compute_cvar(df: pd.DataFrame) -> pd.DataFrame:
    log("Menghitung CVaR historis...")
    rows = []
    for stock, grp in df.groupby("Stock_Name"):
        x = grp["Log_Return"].dropna()
        if len(x) == 0:
            rows.append({"Stock_Name": stock, "Risk_CVaR": np.nan})
            continue
        var_h = np.quantile(x, ALPHA)
        tail = x[x <= var_h]
        cvar = tail.mean() if len(tail) > 0 else var_h
        rows.append({"Stock_Name": stock, "Risk_CVaR": abs(cvar)})
    return pd.DataFrame(rows)


# STEP 7: DOWNSIDE DEVIATION

def compute_downside_deviation(df: pd.DataFrame, tau: float = 0.0) -> pd.DataFrame:
    log("Menghitung Downside Deviation (τ = 0)...")
    rows = []
    for stock, grp in df.groupby("Stock_Name"):
        r = grp["Log_Return"].dropna().values
        T = len(r)
        if T == 0:
            continue
        semi_var = np.sum(np.minimum(r - tau, 0) ** 2) / T
        dd_daily = np.sqrt(semi_var)
        dd_ann   = dd_daily * np.sqrt(TRADING_DAYS)
        neg_freq = np.sum(r < tau) / T
        neg_returns = r[r < tau]
        mean_neg = float(np.mean(neg_returns)) if len(neg_returns) else 0.0

        rows.append({
            "Stock_Name":      stock,
            "SemiVariance":    semi_var,
            "DD_daily":        dd_daily,
            "DD_ann":          dd_ann,
            "Neg_freq":        neg_freq,
            "Mean_neg_return": mean_neg,
        })
    return pd.DataFrame(rows)


# STEP 8: DRAWDOWN

def _drawdown_series_for(log_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cum = np.cumsum(log_returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = np.exp(cum - running_max) - 1
    return cum, running_max, drawdown


def _mdd_and_duration(
    dates: pd.Series, drawdown: np.ndarray,
    cum: np.ndarray, running_max: np.ndarray,
) -> dict:
    mdd_idx = int(np.argmin(drawdown))
    mdd = float(drawdown[mdd_idx])
    trough_date = dates.iloc[mdd_idx]

    peak_val = running_max[mdd_idx]
    peak_cands = np.where(np.abs(cum[: mdd_idx + 1] - peak_val) < 1e-10)[0]
    peak_idx = int(peak_cands[-1]) if len(peak_cands) > 0 else 0
    peak_date = dates.iloc[peak_idx]

    post = cum[mdd_idx:]
    rec_cands = np.where(post >= peak_val - 1e-10)[0]
    if len(rec_cands) > 1:
        recovery_idx = mdd_idx + int(rec_cands[1])
        recovery_date = dates.iloc[recovery_idx]
        recovered = True
        duration = recovery_idx - peak_idx
    else:
        recovery_date = dates.iloc[-1]
        recovered = False
        duration = len(dates) - peak_idx

    return {
        "MDD":           mdd,
        "MDD_Pct":       abs(mdd) * 100,
        "Peak_date":     peak_date,
        "Trough_date":   trough_date,
        "Recovery_date": recovery_date,
        "Duration_days": int(duration),
        "Recovered":     bool(recovered),
    }


def compute_drawdowns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Menghitung drawdown series dan MDD...")
    summary = []
    series_records = []

    for stock, grp in df.groupby("Stock_Name"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        r = grp["Log_Return"].dropna().values
        dates = grp.loc[grp["Log_Return"].notna(), "Date"].reset_index(drop=True)
        if len(r) < 2:
            continue
        sector = grp["Sector"].iloc[0]

        cum, running_max, dd_series = _drawdown_series_for(r)
        info = _mdd_and_duration(dates, dd_series, cum, running_max)

        summary.append({
            "Stock_Name": stock, "Sector": sector,
            **info,
        })
        for d, v in zip(dates.values, dd_series):
            series_records.append({
                "Date": d, "Stock_Name": stock, "Drawdown": float(v),
            })

    dd_summary = pd.DataFrame(summary).sort_values("MDD_Pct").reset_index(drop=True)
    dd_series_df = pd.DataFrame(series_records)
    return dd_summary, dd_series_df


# STEP 9: SORTINO RATIO

def compute_sortino(df: pd.DataFrame, risk_free: float = 0.0) -> pd.DataFrame:
    log("Menghitung Sortino ratio...")
    rows = []
    for stock, grp in df.groupby("Stock_Name"):
        r = grp["Log_Return"].dropna().values
        T = len(r)
        if T == 0:
            continue
        sector = grp["Sector"].iloc[0]
        return_ann = np.mean(r) * TRADING_DAYS
        semi_var = np.sum(np.minimum(r, 0) ** 2) / T
        dd_ann = np.sqrt(semi_var) * np.sqrt(TRADING_DAYS)
        sortino = (return_ann - risk_free) / dd_ann if dd_ann > 0 else np.nan
        rows.append({
            "Stock_Name": stock, "Sector": sector,
            "Return_ann": return_ann, "DD_ann": dd_ann, "Sortino": sortino,
        })

    out = pd.DataFrame(rows).sort_values("Sortino", ascending=False).reset_index(drop=True)
    out["Sortino_rank"] = np.arange(1, len(out) + 1)
    return out


# STEP 10: VaR BACKTEST (OPSIONAL, LAMBAT)

def _kupiec_pof(breaches: np.ndarray, alpha: float) -> dict:
    T = len(breaches)
    x = int(breaches.sum())
    if T == 0 or x == 0 or x == T:
        return {"LR_pof": np.nan, "p_value_pof": np.nan,
                "n_obs": T, "n_fail": x,
                "fail_rate": x / T if T else np.nan}
    p_hat = x / T
    logL_null = (T - x) * np.log(1 - alpha) + x * np.log(alpha)
    logL_alt  = (T - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    LR = -2 * (logL_null - logL_alt)
    return {
        "LR_pof": LR, "p_value_pof": 1 - chi2.cdf(LR, df=1),
        "n_obs": T, "n_fail": x, "fail_rate": p_hat,
    }


def _christoffersen_ind(breaches: np.ndarray) -> dict:
    if len(breaches) < 2:
        return {"LR_ind": np.nan, "p_value_ind": np.nan}
    prev, curr = breaches[:-1], breaches[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    denom0, denom1 = n00 + n01, n10 + n11
    denom = denom0 + denom1
    if denom == 0:
        return {"LR_ind": np.nan, "p_value_ind": np.nan}

    pi01 = n01 / denom0 if denom0 > 0 else 0
    pi11 = n11 / denom1 if denom1 > 0 else 0
    pi   = (n01 + n11) / denom

    def safe(c, p):
        if c == 0: return 0.0
        if p <= 0 or p >= 1: return np.nan
        return c * np.log(p)

    ll_ind    = safe(n00 + n10, 1 - pi) + safe(n01 + n11, pi)
    ll_markov = safe(n00, 1 - pi01) + safe(n01, pi01) + safe(n10, 1 - pi11) + safe(n11, pi11)

    if np.isnan(ll_ind) or np.isnan(ll_markov):
        return {"LR_ind": np.nan, "p_value_ind": np.nan}
    LR = -2 * (ll_ind - ll_markov)
    return {"LR_ind": LR, "p_value_ind": 1 - chi2.cdf(LR, df=1)}


def _quantile_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    e = y - q
    return float(np.mean(np.where(e < 0, (alpha - 1) * e, alpha * e)))


def run_var_backtest(df: pd.DataFrame, window: int = BACKTEST_WINDOW) -> pd.DataFrame:
    """
    Rolling 1-day-ahead VaR backtest per saham × 3 metode.
    LAMBAT: ~1-2 jam untuk 29 saham. Hanya aggregate summary yang disimpan.
    """
    log(f"VaR backtest (rolling window={window}) — ini memakan waktu...")
    all_eval = []

    for stock in progress(df["Stock_Name"].unique(), desc="Backtest"):
        grp = df[df["Stock_Name"] == stock].sort_values("Date").reset_index(drop=True)
        r = grp["Log_Return"].dropna().values
        if len(r) <= window + 10:
            continue

        for method in ("historical", "cornish_fisher", "garch"):
            vars_, actuals = [], []
            for i in range(window, len(r)):
                train = pd.Series(r[i - window:i])
                actual = r[i]
                if method == "historical":
                    v = _historical_var(train, ALPHA)
                elif method == "cornish_fisher":
                    v = _cornish_fisher_var(train, ALPHA)
                else:
                    v = _garch_var_last(train, ALPHA)
                if pd.notna(v):
                    vars_.append(abs(v))
                    actuals.append(actual)

            vars_ = np.array(vars_)
            actuals = np.array(actuals)
            threshold = -vars_
            breaches = (actuals < threshold).astype(int)

            pof = _kupiec_pof(breaches, ALPHA)
            ind = _christoffersen_ind(breaches)
            qloss = _quantile_loss(actuals, threshold, ALPHA)

            lr_cc = (
                pof["LR_pof"] + ind["LR_ind"]
                if pd.notna(pof["LR_pof"]) and pd.notna(ind["LR_ind"])
                else np.nan
            )
            p_cc = 1 - chi2.cdf(lr_cc, df=2) if pd.notna(lr_cc) else np.nan

            all_eval.append({
                "Stock_Name": stock, "Method": method,
                **pof, **ind, "LR_cc": lr_cc, "p_value_cc": p_cc,
                "Quantile_Loss": qloss,
            })

    eval_df = pd.DataFrame(all_eval)

    # Aggregate summary (mirror paper Gambar 8)
    summary = (
        eval_df.groupby("Method", as_index=False)
        .agg(
            n_obs=("n_obs", "sum"),
            n_fail=("n_fail", "sum"),
            p_value_pof_uc=("p_value_pof", "mean"),
            p_value_ind=("p_value_ind", "mean"),
            p_value_cc=("p_value_cc", "mean"),
            Quantile_Loss=("Quantile_Loss", "mean"),
        )
    )
    summary["fail_rate"] = summary["n_fail"] / summary["n_obs"]
    summary["expected_fail_rate"] = ALPHA
    summary = summary[["Method", "n_obs", "n_fail", "fail_rate", "expected_fail_rate",
                       "p_value_pof_uc", "p_value_ind", "p_value_cc", "Quantile_Loss"]]
    return summary


# STEP 11: FINAL METRICS TABLE + STABILITY SCORE

def build_final_metrics(
    vol_mean_annual: dict,
    var_point: pd.DataFrame,
    cvar_df: pd.DataFrame,
    dd_df: pd.DataFrame,
    mdd_df: pd.DataFrame,
) -> pd.DataFrame:
    log("Menyusun tabel metrik final + Stability Score...")

    # Mulai dengan volatilitas GARCH (mean annualized), dalam %
    out = pd.DataFrame([
        {"Stock_Name": s, "Volatility_Pct": v * 100}
        for s, v in vol_mean_annual.items()
    ])

    # Merge VaR (GARCH) — sudah absolut, konversi ke %
    var_g = var_point[["Stock_Name", "Risk_GARCHVaR"]].copy()
    var_g["VaR_Pct"] = var_g["Risk_GARCHVaR"] * 100
    out = out.merge(var_g[["Stock_Name", "VaR_Pct"]], on="Stock_Name", how="left")

    # CVaR
    cv = cvar_df.copy()
    cv["CVaR_Pct"] = cv["Risk_CVaR"] * 100
    out = out.merge(cv[["Stock_Name", "CVaR_Pct"]], on="Stock_Name", how="left")

    # Downside Deviation
    ddx = dd_df.copy()
    ddx["DD_Pct"] = ddx["DD_ann"] * 100
    out = out.merge(ddx[["Stock_Name", "DD_Pct"]], on="Stock_Name", how="left")

    # MDD
    mx = mdd_df[["Stock_Name", "MDD_Pct"]].copy()
    out = out.merge(mx, on="Stock_Name", how="left")

    # Sektor
    out["Sector"] = out["Stock_Name"].map(SECTOR_MAP)

    # --- Rank normalization (pct rank) per metrik, ascending ---
    # Semua metrik: nilai lebih kecil = lebih stabil, sehingga rank ascending cocok.
    metric_cols = list(WEIGHTS.keys())
    ranks = out[metric_cols].rank(pct=True)
    ranks.columns = [f"{c}_norm" for c in metric_cols]

    # Stability Score = weighted sum
    stability = sum(ranks[f"{c}_norm"] * w for c, w in WEIGHTS.items())
    out["Stability_Score"] = stability.values

    # Gabungkan kolom norm untuk transparansi (dipakai di halaman sensitivitas)
    out = pd.concat([out, ranks], axis=1)

    out = out.sort_values("Stability_Score").reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)

    # Urutkan kolom rapih
    ordered = (
        ["Rank", "Stock_Name", "Sector"]
        + metric_cols
        + [f"{c}_norm" for c in metric_cols]
        + ["Stability_Score"]
    )
    out = out[ordered]
    return out


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline dashboard risiko saham IDX")
    parser.add_argument("--input", required=True, type=Path,
                       help="Path ke CSV mentah (separator ';')")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                       help="Direktori output parquet (default: ./data)")
    parser.add_argument("--skip-backtest", action="store_true",
                       help="Skip VaR rolling backtest (menghemat ~1-2 jam)")
    args = parser.parse_args()

    if not args.input.exists():
        log(f"ERROR: file input tidak ditemukan: {args.input}")
        sys.exit(1)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir.resolve()}")

    t0 = time.time()

    # === 1. Load & clean ===
    df = load_and_clean(args.input)

    # === 2. Feature engineering ===
    df = add_returns_and_volatility(df)

    # Simpan prices parquet
    log("Menyimpan prices.parquet...")
    df.to_parquet(out_dir / "prices.parquet", index=False)

    # === 3. Jarque-Bera ===
    jb = compute_jarque_bera(df)
    jb.to_parquet(out_dir / "jarque_bera.parquet", index=False)

    # === 4. GARCH vol series ===
    vol_series, vol_mean = compute_garch_vol_series(df)
    vol_series.to_parquet(out_dir / "garch_vol_series.parquet", index=False)

    # === 5. VaR point estimates ===
    var_point = compute_var_point_estimates(df)

    # === 6. CVaR ===
    cvar_df = compute_cvar(df)

    # === 7. Downside Deviation ===
    dd_df = compute_downside_deviation(df, tau=0.0)

    # === 8. Drawdown ===
    mdd_df, dd_series_df = compute_drawdowns(df)
    mdd_df.to_parquet(out_dir / "drawdowns.parquet", index=False)
    dd_series_df.to_parquet(out_dir / "drawdown_series.parquet", index=False)

    # === 9. Sortino ===
    sortino_df = compute_sortino(df, risk_free=0.0)
    sortino_df.to_parquet(out_dir / "sortino.parquet", index=False)

    # === 10. Final metrics + Stability Score ===
    metrics = build_final_metrics(vol_mean, var_point, cvar_df, dd_df, mdd_df)
    metrics.to_parquet(out_dir / "risk_metrics.parquet", index=False)

    # === 11. VaR backtest (opsional) ===
    if args.skip_backtest:
        log("✓ Backtest di-SKIP (--skip-backtest)")
    else:
        backtest = run_var_backtest(df)
        backtest.to_parquet(out_dir / "var_backtest_summary.parquet", index=False)

    elapsed = time.time() - t0
    log(f"✓ SELESAI dalam {elapsed/60:.1f} menit")

    # === Preview hasil akhir ===
    print("\n" + "=" * 60)
    print("PREVIEW RISK_METRICS (10 teratas)")
    print("=" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 140):
        cols_show = ["Rank", "Stock_Name", "Sector",
                     "Volatility_Pct", "VaR_Pct", "CVaR_Pct",
                     "DD_Pct", "MDD_Pct", "Stability_Score"]
        print(metrics[cols_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()