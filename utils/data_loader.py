"""
Data loader untuk dashboard analisis risiko saham IDX.

Modul ini membaca hasil preprocessing (parquet files) dan menyediakan
fungsi cached untuk setiap halaman dashboard. Semua fungsi decorator
@st.cache_data agar hanya di-load sekali per session.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

# KONFIGURASI PATH & KONSTANTA

# PLACEHOLDER
DATA_DIR = Path(__file__).parent.parent / "data"

# Palet warna per sektor
SECTOR_COLORS = {
    "Financials":             "#2563EB",
    "Healthcare":              "#16A34A",
    "Consumer Non-Cyclicals":  "#D97706",
    "Industrials":             "#7C3AED",
    "Infrastructures":         "#0891B2",
    "Consumer Cyclicals":      "#DB2777",
    "Basic Materials":         "#B45309",
    "Energy":                  "#DC2626",
}

# Mapping sektor
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

# Nama perusahaan penuh
STOCK_FULL_NAMES = {
    "BBCA": "Bank Central Asia",
    "BBNI": "Bank Negara Indonesia",
    "BBRI": "Bank Rakyat Indonesia",
    "BMRI": "Bank Mandiri",
    "ADRO": "Adaro Energy",
    "AKRA": "AKR Corporindo",
    "ITMG": "Indo Tambangraya Megah",
    "MEDC": "Medco Energi Internasional",
    "PGAS": "Perusahaan Gas Negara",
    "PTBA": "Bukit Asam",
    "ANTM": "Aneka Tambang",
    "BRPT": "Barito Pacific",
    "INCO": "Vale Indonesia",
    "INKP": "Indah Kiat Pulp & Paper",
    "INTP": "Indocement Tunggal Prakarsa",
    "SMGR": "Semen Indonesia",
    "AMRT": "Sumber Alfaria Trijaya",
    "CPIN": "Charoen Pokphand Indonesia",
    "GGRM": "Gudang Garam",
    "ICBP": "Indofood CBP Sukses Makmur",
    "INDF": "Indofood Sukses Makmur",
    "UNVR": "Unilever Indonesia",
    "ACES": "Ace Hardware Indonesia",
    "MAPI": "Mitra Adiperkasa",
    "ASII": "Astra International",
    "UNTR": "United Tractors",
    "KLBF": "Kalbe Farma",
    "TLKM": "Telkom Indonesia",
    "EXCL": "XL Axiata",
}

# Bobot default untuk Stability Score
DEFAULT_WEIGHTS = {
    "Volatility_Pct": 0.20,
    "VaR_Pct":        0.10,
    "CVaR_Pct":       0.30,
    "DD_Pct":         0.15,
    "MDD_Pct":        0.25,
}

# Periode COVID untuk shading di chart
COVID_START = pd.Timestamp("2020-02-01")
COVID_END   = pd.Timestamp("2021-05-31")


# LOADERS

def _check_file(filename: str) -> Path:
    """Pastikan file ada; raise dengan pesan user-friendly jika tidak."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            f"Jalankan preprocess.py terlebih dahulu untuk menghasilkan data."
        )
    return path


@st.cache_data(show_spinner="Memuat data harga...")
def load_prices() -> pd.DataFrame:
    """Time series OHLCV + return + rolling volatility untuk 29 saham."""
    path = _check_file("prices.parquet")
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Memuat metrik risiko...")
def load_risk_metrics() -> pd.DataFrame:
    """Tabel final 29 baris dengan 5 metrik + Stability Score + Rank."""
    path = _check_file("risk_metrics.parquet")
    df = pd.read_parquet(path)
    # Pastikan terurut berdasarkan rank
    df = df.sort_values("Rank").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Memuat data drawdown...")
def load_drawdowns() -> pd.DataFrame:
    """Ringkasan MDD per saham (peak, trough, recovery, duration)."""
    path = _check_file("drawdowns.parquet")
    df = pd.read_parquet(path)
    for col in ["Peak_date", "Trough_date", "Recovery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data
def load_drawdown_series() -> pd.DataFrame:
    """Time series drawdown untuk underwater plot."""
    path = _check_file("drawdown_series.parquet")
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_garch_vol_series() -> pd.DataFrame:
    """Time series volatilitas GARCH (conditional)."""
    path = _check_file("garch_vol_series.parquet")
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_sortino() -> pd.DataFrame:
    """Sortino ratio + return tahunan per saham."""
    path = _check_file("sortino.parquet")
    return pd.read_parquet(path)


@st.cache_data
def load_jarque_bera() -> pd.DataFrame:
    """Hasil uji Jarque-Bera per saham."""
    path = _check_file("jarque_bera.parquet")
    return pd.read_parquet(path)


# HELPERS

def get_stocks_by_sector() -> dict:
    """Return dict {sector: [list of stocks]} untuk grouped dropdown."""
    result = {}
    for stock, sector in SECTOR_MAP.items():
        result.setdefault(sector, []).append(stock)
    for sector in result:
        result[sector] = sorted(result[sector])
    return dict(sorted(result.items()))


def get_stock_label(stock: str) -> str:
    """Format '<KODE> — <Nama Perusahaan>' untuk display."""
    full = STOCK_FULL_NAMES.get(stock, stock)
    return f"{stock} — {full}"


def check_data_available() -> tuple[bool, list[str]]:
    """
    Periksa apakah semua file parquet tersedia.
    Return (all_ok, list_of_missing_files).
    """
    required = [
        "prices.parquet", "risk_metrics.parquet", "drawdowns.parquet",
        "drawdown_series.parquet", "garch_vol_series.parquet",
        "sortino.parquet", "jarque_bera.parquet",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    return (len(missing) == 0, missing)

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color (e.g. '#2563EB') to rgba string for Plotly.

    Plotly's color properties don't accept 8-digit hex (#RRGGBBAA);
    use this helper whenever you need transparency.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"