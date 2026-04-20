"""
Profil Saham, deep dive per saham.
- Apakah saham ini berisiko atau stabil?
- Apa kekuatan dan kelemahan risikonya?
- Bagaimana perilakunya sepanjang periode?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from utils.data_loader import (
    COVID_END, COVID_START, SECTOR_COLORS, STOCK_FULL_NAMES,
    check_data_available, load_drawdown_series, load_drawdowns,
    load_garch_vol_series, load_jarque_bera, load_prices,
    load_risk_metrics, load_sortino, hex_to_rgba
)

# CONFIG & DATA CHECK

st.set_page_config(page_title="Profil Saham", page_icon="📊", layout="wide")

data_ok, missing = check_data_available()
if not data_ok:
    st.error("⚠️ Data belum tersedia. Jalankan preprocess.py terlebih dahulu.")
    st.stop()

# Metric metadata — dikelola di satu tempat
METRIC_COLS = ["Volatility_Pct", "VaR_Pct", "CVaR_Pct", "DD_Pct", "MDD_Pct"]
METRIC_SHORT = {
    "Volatility_Pct": "Volatilitas",
    "VaR_Pct":        "VaR",
    "CVaR_Pct":       "CVaR",
    "DD_Pct":         "Downside Deviation",
    "MDD_Pct":        "Max Drawdown",
}
METRIC_FULL = {
    "Volatility_Pct": "Volatilitas GARCH (tahunan)",
    "VaR_Pct":        "Value at Risk (GARCH-based, 5%)",
    "CVaR_Pct":       "Conditional Value at Risk (5%)",
    "DD_Pct":         "Downside Deviation (tahunan)",
    "MDD_Pct":        "Maximum Drawdown",
}
METRIC_TOOLTIP = {
    "Volatility_Pct": "Seberapa besar harga bergerak setiap hari (diestimasi GARCH, dianualisasi).",
    "VaR_Pct":        "Batas kerugian pada 5% hari terburuk (model GARCH).",
    "CVaR_Pct":       "Rata-rata kerugian justru pada 5% hari terburuk — ukuran risiko ekor.",
    "DD_Pct":         "Rata-rata kerugian di seluruh hari negatif (τ = 0), dianualisasi.",
    "MDD_Pct":        "Kejatuhan kumulatif terparah dari harga puncak historis.",
}

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# LOAD DATA

risk = load_risk_metrics().rename(columns={"Stability_Score": "Risk_Score"})
prices_all = load_prices()
drawdowns = load_drawdowns()
dd_series_all = load_drawdown_series()
vol_series_all = load_garch_vol_series()
sortino_all = load_sortino()
jb_all = load_jarque_bera()

N_STOCKS = len(risk)
ALL_STOCKS = sorted(risk["Stock_Name"].tolist())

# HELPERS

def badge(text: str, color: str = "#666", bg: str | None = None,
          bold: bool = True) -> str:
    if bg is None:
        # 15% opacity of color
        bg = color + "22"
    weight = "600" if bold else "400"
    return (
        f'<span style="background:{bg}; color:{color}; '
        f'padding:4px 12px; border-radius:14px; font-size:0.85em; '
        f'font-weight:{weight}; white-space:nowrap;">{text}</span>'
    )


def rank_category(rank: int, total: int = N_STOCKS) -> tuple[str, str, str]:
    if rank <= 3:
        return "🟢", "Paling Stabil", "#059669"
    elif rank <= 10:
        return "🟢", "Stabil", "#10B981"
    elif rank < total - 10:
        return "🟡", "Menengah", "#D97706"
    elif rank < total - 3:
        return "🟠", "Cenderung Berisiko", "#EA580C"
    else:
        return "🔴", "Paling Berisiko", "#DC2626"


def metric_tone(pct_rank: float) -> tuple[str, str]:
    if pct_rank <= 0.33:
        return "#10B981", "Rendah"
    elif pct_rank <= 0.67:
        return "#F59E0B", "Sedang"
    else:
        return "#EF4444", "Tinggi"


def make_verdict(rank: int, score: float, total: int = N_STOCKS) -> str:
    if rank <= 3:
        return (
            f"Termasuk **3 saham paling stabil** selama periode pengamatan. "
            f"Profil risikonya rendah di hampir semua dimensi."
        )
    elif rank <= 10:
        return (
            f"**Peringkat #{rank}** — lebih stabil dari {total - rank} saham lainnya. "
            f"Termasuk saham dengan profil risiko yang terkelola baik."
        )
    elif rank >= total - 2:
        return (
            f"Termasuk **3 saham paling berisiko** selama periode pengamatan. "
            f"Profil risikonya tinggi di mayoritas dimensi."
        )
    elif rank >= total - 9:
        return (
            f"**Peringkat #{rank}** — lebih berisiko dari {rank - 1} saham lainnya. "
            f"Perlu perhatian khusus pada salah satu atau lebih dimensi risiko."
        )
    else:
        return (
            f"**Peringkat #{rank}** — posisi menengah di antara {total} saham. "
            f"Profil risikonya campuran; detail per dimensi ada di bawah."
        )


def sortino_interpretation(sortino: float) -> str:
    if pd.isna(sortino):
        return "Tidak dapat dihitung."
    if sortino >= 1.0:
        return (
            "Return tahunannya melebihi downside deviation — "
            "**return sangat baik dibandingkan risiko yang ditanggung**."
        )
    elif sortino >= 0.5:
        return (
            "Return tahunan signifikan relatif terhadap risiko downside-nya."
        )
    elif sortino >= 0:
        return (
            "Return tahunan positif, tetapi **tidak sebanding** dengan "
            "risiko downside yang ditanggung."
        )
    else:
        return (
            "Return tahunan **negatif** sepanjang periode. Investor "
            "saham ini rata-rata mengalami kerugian."
        )


def format_rupiah(x: float) -> str:
    if pd.isna(x):
        return "-"
    if x >= 1_000_000:
        return f"Rp {x/1_000_000:.2f}jt"
    elif x >= 1_000:
        return f"Rp {x/1_000:,.0f}rb".replace(",", ".")
    return f"Rp {x:,.0f}".replace(",", ".")


# PAGE HEADER & SELECTOR

st.title("Profil Saham")
st.caption("Analisis detail per saham — posisi risiko, distribusi return, drawdown, dan volatilitas dinamis.")

# Query param support untuk link cross-page
qp_stock = st.query_params.get("stock", "BBCA")
if qp_stock not in ALL_STOCKS:
    qp_stock = "BBCA"

selected = st.selectbox(
    "Pilih saham untuk dianalisis",
    options=ALL_STOCKS,
    index=ALL_STOCKS.index(qp_stock),
    format_func=lambda s: f"{s} — {STOCK_FULL_NAMES.get(s, s)}",
    key="stock_selector",
)

# Sync ke query param agar URL bisa di-share
if st.query_params.get("stock") != selected:
    st.query_params["stock"] = selected

# Ekstrak data saham ini
row = risk[risk["Stock_Name"] == selected].iloc[0]
sector = row["Sector"]
stock_rank = int(row["Rank"])
stock_score = float(row["Risk_Score"])
sector_color = SECTOR_COLORS.get(sector, "#666")

stock_prices = prices_all[prices_all["Stock_Name"] == selected] \
    .sort_values("Date").reset_index(drop=True)
stock_dd = drawdowns[drawdowns["Stock_Name"] == selected].iloc[0]
stock_dd_series = dd_series_all[dd_series_all["Stock_Name"] == selected] \
    .sort_values("Date").reset_index(drop=True)
stock_vol_ts = vol_series_all[vol_series_all["Stock_Name"] == selected] \
    .sort_values("Date").reset_index(drop=True)
stock_jb = jb_all[jb_all["Stock_Name"] == selected].iloc[0]
stock_sortino = sortino_all[sortino_all["Stock_Name"] == selected].iloc[0]

# HERO SECTION

st.divider()

cat_emoji, cat_label, cat_color = rank_category(stock_rank)
verdict = make_verdict(stock_rank, stock_score)

col_hero_l, col_hero_r = st.columns([3, 2])

with col_hero_l:
    full_name = STOCK_FULL_NAMES.get(selected, selected)
    st.markdown(f"## {selected} — {full_name}")

    st.markdown(
        badge(sector, sector_color) + " &nbsp; "
        + badge(f"Peringkat #{stock_rank} dari {N_STOCKS}", "#475569") + " &nbsp; "
        + badge(f"{cat_emoji} {cat_label}", cat_color) + " &nbsp; "
        + badge(f"Risk Score {stock_score:.4f}", "#475569"),
        unsafe_allow_html=True,
    )
    st.markdown("")  # spacer
    st.markdown(verdict)

with col_hero_r:
    # Key stats: harga terakhir, total return, Sortino
    last_price = stock_prices["Close"].iloc[-1]
    first_price = stock_prices["Close"].iloc[0]
    total_return = (last_price / first_price - 1) * 100
    last_date = stock_prices["Date"].iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Harga Terakhir",
        format_rupiah(last_price),
        help=f"Per {last_date.strftime('%d %b %Y')}",
    )
    c2.metric(
        "Total Return",
        f"{total_return:+.1f}%",
        help="Perubahan harga sejak awal periode (Okt 2012)",
    )
    c3.metric(
        "Sortino Ratio",
        f"{stock_sortino['Sortino']:.2f}",
        help=f"Peringkat Sortino #{int(stock_sortino['Sortino_rank'])} dari {N_STOCKS}",
    )

st.divider()

# METRIC CARDS

st.markdown("### Profil Risiko dalam 5 Dimensi")
st.caption(
    "Setiap kartu menunjukkan nilai metrik dan **posisinya relatif** "
    "terhadap 29 saham lain. Nilai rendah = lebih stabil."
)

metric_cols = st.columns(5)

for col, mcol in zip(metric_cols, METRIC_COLS):
    value = float(row[mcol])
    pct_rank = float(row[f"{mcol}_norm"])
    # Rank ascending: nilai kecil = rank kecil
    metric_rank = int(risk[mcol].rank(method="min").loc[risk["Stock_Name"] == selected].iloc[0])
    tone_color, tone_label = metric_tone(pct_rank)

    # Persentase "lebih rendah dari X% saham"
    pct_lower_than = (1 - pct_rank) * 100

    short = METRIC_SHORT[mcol]
    full = METRIC_FULL[mcol]
    tooltip = METRIC_TOOLTIP[mcol]

    with col:
        st.markdown(
            f"""
            <div style="
                padding: 16px;
                border: 1px solid rgba(128,128,128,0.2);
                border-left: 4px solid {tone_color};
                border-radius: 8px;
                background: rgba(128,128,128,0.03);
                height: 100%;
            " title="{tooltip}">
                <div style="color: #888; font-size: 0.8em; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.5px;
                            margin-bottom: 4px;">
                    {short}
                </div>
                <div style="font-size: 1.6em; font-weight: 700; margin-bottom: 2px;">
                    {value:.2f}%
                </div>
                <div style="color: #888; font-size: 0.78em; margin-bottom: 10px;">
                    {full}
                </div>
                <div style="background: #e5e7eb; height: 6px; border-radius: 3px;
                            position: relative; margin-bottom: 8px;">
                    <div style="background: {tone_color}; height: 100%; border-radius: 3px;
                                width: {pct_rank*100}%;"></div>
                </div>
                <div style="font-size: 0.78em; line-height: 1.4;">
                    <span style="color: {tone_color}; font-weight: 600;">
                        #{metric_rank} dari {N_STOCKS}
                    </span>
                    <br>
                    <span style="color: #666;">
                        Lebih rendah dari {pct_lower_than:.0f}% saham lain
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

# RADAR + RISK-RETURN CONTEXT

col_radar, col_rr = st.columns([3, 2])

with col_radar:
    st.markdown("#### Profil Relatif vs Rata-rata Sektor")
    st.caption(
        "Sumbu = persentil rank di antara 29 saham (0 = paling stabil, "
        "1 = paling berisiko). Profil yang mengumpul ke pusat = lebih stabil."
    )

    # Ambil norm values
    this_values = [float(row[f"{c}_norm"]) for c in METRIC_COLS]
    axis_labels = [METRIC_SHORT[c] for c in METRIC_COLS]

    # Sector average norms (exclude saham ini sendiri untuk baseline?)
    sector_stocks = risk[risk["Sector"] == sector]
    sector_size = len(sector_stocks)
    show_sector_trace = sector_size > 1
    if show_sector_trace:
        sector_values = [
            float(sector_stocks[f"{c}_norm"].mean()) for c in METRIC_COLS
        ]

    # Close the polygon by repeating first value
    def closed(arr):
        return arr + [arr[0]]

    theta_closed = closed(axis_labels)

    fig_radar = go.Figure()

    # IDX median reference (always 0.5-ish since it's percentile)
    fig_radar.add_trace(go.Scatterpolar(
        r=closed([0.5] * 5),
        theta=theta_closed,
        line=dict(color="#cbd5e1", dash="dot", width=1.5),
        name="Median IDX",
        hoverinfo="skip",
    ))

    if show_sector_trace:
        fig_radar.add_trace(go.Scatterpolar(
            r=closed(sector_values),
            theta=theta_closed,
            line=dict(color=sector_color, dash="dash", width=2),
            name=f"Rata-rata sektor {sector}",
            fill="toself",
            fillcolor = hex_to_rgba(sector_color, 0.08),
            hovertemplate="<b>%{theta}</b><br>Rata-rata sektor: %{r:.2f}<extra></extra>",
        ))

    fig_radar.add_trace(go.Scatterpolar(
        r=closed(this_values),
        theta=theta_closed,
        line=dict(color=sector_color, width=3),
        marker=dict(size=8, color=sector_color),
        name=selected,
        fill="toself",
        fillcolor=hex_to_rgba(sector_color, 0.25),
        hovertemplate="<b>%{theta}</b><br>" + selected + ": %{r:.2f}<extra></extra>",
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=["0", "", "0.5", "", "1"],
                tickfont=dict(size=9, color="#888"),
                gridcolor="rgba(128,128,128,0.2)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#374151"),
                gridcolor="rgba(128,128,128,0.2)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(size=10)),
        height=400,
        margin=dict(l=40, r=40, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    if not show_sector_trace:
        st.caption(f"ℹ️ Sektor **{sector}** hanya memiliki 1 saham ({selected}), jadi tidak ada rata-rata sektor untuk dibandingkan.")

with col_rr:
    st.markdown("#### Risk vs Return")
    st.caption("Apakah return-nya sebanding dengan risiko downside?")

    return_ann = float(stock_sortino["Return_ann"]) * 100
    dd_ann = float(stock_sortino["DD_ann"]) * 100
    sortino_val = float(stock_sortino["Sortino"])
    sortino_rank = int(stock_sortino["Sortino_rank"])

    # Return color
    ret_color = "#10B981" if return_ann > 0 else "#EF4444"

    st.markdown(
        f"""
        <div style="padding: 14px; border: 1px solid rgba(128,128,128,0.2);
                    border-radius: 8px; background: rgba(128,128,128,0.03);
                    margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;
                        margin-bottom: 8px;">
                <span style="color: #666;">Return tahunan</span>
                <span style="color: {ret_color}; font-weight: 600;">
                    {return_ann:+.2f}%
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;
                        margin-bottom: 8px;">
                <span style="color: #666;">Downside Deviation tahunan</span>
                <span style="font-weight: 600;">{dd_ann:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;
                        padding-top: 8px; border-top: 1px solid rgba(128,128,128,0.2);">
                <span style="color: #666;">Sortino Ratio</span>
                <span style="font-weight: 700; font-size: 1.1em;">
                    {sortino_val:.3f}
                </span>
            </div>
            <div style="color: #888; font-size: 0.8em; margin-top: 4px; text-align: right;">
                Peringkat #{sortino_rank} dari {N_STOCKS}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**Interpretasi:** {sortino_interpretation(sortino_val)}")

st.divider()

# TABS: HARGA | RETURN | DRAWDOWN | VOLATILITAS

tab_price, tab_return, tab_dd, tab_vol = st.tabs([
    "📈 Harga & Volume",
    "📊 Return & Distribusi",
    "📉 Drawdown",
    "🔥 Volatilitas Dinamis",
])

# TAB 1: HARGA & VOLUME
with tab_price:
    fig_candle = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("Harga (OHLC)", "Volume"),
    )

    fig_candle.add_trace(
        go.Candlestick(
            x=stock_prices["Date"],
            open=stock_prices["Open"],
            high=stock_prices["High"],
            low=stock_prices["Low"],
            close=stock_prices["Close"],
            name=selected,
            increasing=dict(line=dict(color="#10B981"), fillcolor="#10B981"),
            decreasing=dict(line=dict(color="#EF4444"), fillcolor="#EF4444"),
            showlegend=False,
        ),
        row=1, col=1,
    )

    up_down = np.where(
        stock_prices["Close"] >= stock_prices["Open"],
        hex_to_rgba("#10B981", 0.33),
        hex_to_rgba("#EF4444", 0.33),
    )
    
    fig_candle.add_trace(
        go.Bar(
            x=stock_prices["Date"],
            y=stock_prices["Volume"],
            marker=dict(color=up_down),
            name="Volume",
            showlegend=False,
            hovertemplate="%{x|%d %b %Y}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # COVID shading
    fig_candle.add_vrect(
        x0=COVID_START, x1=COVID_END,
        fillcolor="#F59E0B", opacity=0.08, line_width=0,
        annotation_text="COVID-19", annotation_position="top left",
        annotation=dict(font=dict(size=10, color="#D97706")),
        row=1, col=1,
    )

    fig_candle.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis2=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]),
                bgcolor="rgba(128,128,128,0.1)",
                activecolor="#3B82F6",
            ),
            type="date",
        ),
        xaxis=dict(rangeslider=dict(visible=False)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    # Default view: last 3 years
    three_yr_ago = stock_prices["Date"].iloc[-1] - pd.DateOffset(years=3)
    fig_candle.update_xaxes(range=[three_yr_ago, stock_prices["Date"].iloc[-1]])

    st.plotly_chart(fig_candle, use_container_width=True)
    st.caption("""
    🟢 Hijau: hari naik (Close ≥ Open)\n
    🔴 Merah: hari turun (Close < Open)\n
    🟠 Oranye: periode COVID-19\n
    Gunakan tombol **1Y / 3Y / 5Y / All** untuk mengubah rentang waktu
    """)

# TAB 2: RETURN & DISTRIBUSI
with tab_return:
    returns = stock_prices["Log_Return"].dropna()
    mu = returns.mean()
    sigma = returns.std()
    sk = float(stock_jb["Skewness"])
    kurt = float(stock_jb["Excess_Kurtosis"])
    jb_pval = float(stock_jb["p_value"])

    col_hist, col_stat = st.columns([2, 1])

    with col_hist:
        fig_hist = go.Figure()

        fig_hist.add_trace(go.Histogram(
            x=returns,
            nbinsx=80,
            marker=dict(color=sector_color, line=dict(color="white", width=0.3)),
            opacity=0.75,
            name="Log Return",
            hovertemplate="Return: %{x:.3f}<br>Frekuensi: %{y}<extra></extra>",
        ))

        # Scale normal PDF to match histogram counts
        x_range = np.linspace(returns.min(), returns.max(), 300)
        pdf_normal = stats.norm.pdf(x_range, loc=mu, scale=sigma)
        bin_width = (returns.max() - returns.min()) / 80
        pdf_scaled = pdf_normal * bin_width * len(returns)

        fig_hist.add_trace(go.Scatter(
            x=x_range, y=pdf_scaled,
            mode="lines",
            line=dict(color="#111827", width=2, dash="dash"),
            name="Normal (teoretis)",
            hovertemplate="Return: %{x:.3f}<br>Prediksi normal: %{y:.1f}<extra></extra>",
        ))

        fig_hist.update_layout(
            title="Distribusi Log-Return Harian",
            xaxis_title="Log Return",
            yaxis_title="Frekuensi",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
            bargap=0.02,
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig_hist.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stat:
        st.markdown("##### Statistik Distribusi")

        c1, c2 = st.columns(2)
        c1.metric("Mean Harian", f"{mu*100:.3f}%")
        c2.metric("Std Harian", f"{sigma*100:.2f}%")

        c3, c4 = st.columns(2)
        c3.metric("Skewness", f"{sk:+.3f}",
                  help="0 = simetris. Positif = ekor kanan lebih panjang.")
        c4.metric("Excess Kurtosis", f"{kurt:+.2f}",
                  help="0 = seperti normal. Tinggi = ekor lebih tebal (fat tails).")

        if jb_pval < 0.05:
            st.error(
                f"**Distribusi TIDAK normal** (Jarque-Bera p = {jb_pval:.1e}). "
                f"Ekor tebal berarti kejadian ekstrem (kerugian/keuntungan besar) "
                f"**lebih sering** terjadi dibandingkan perkiraan distribusi normal."
            )
        else:
            st.success(
                f"**Distribusi konsisten dengan normal** (Jarque-Bera p = {jb_pval:.3f}). "
                f"Ini langka untuk data saham."
            )

    st.caption(
        "Garis hitam putus-putus = distribusi normal teoretis dengan mean & std yang sama. "
        "Perhatikan puncak di tengah lebih tinggi dan ekor lebih 'gemuk' dari garis — "
        "karakteristik khas return saham (*fat tails*)."
    )

# TAB 3: DRAWDOWN
with tab_dd:
    # Key stats di atas
    mdd_pct = float(stock_dd["MDD_Pct"])
    duration = int(stock_dd["Duration_days"])
    peak_date = pd.to_datetime(stock_dd["Peak_date"])
    trough_date = pd.to_datetime(stock_dd["Trough_date"])
    recovered = bool(stock_dd["Recovered"])
    recovery_date = pd.to_datetime(stock_dd["Recovery_date"])

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Max Drawdown", f"-{mdd_pct:.1f}%",
                  help="Penurunan kumulatif terparah dari puncak ke titik terendah.")
    col_s2.metric("Tanggal Puncak", peak_date.strftime("%b %Y"))
    col_s3.metric("Tanggal Lembah", trough_date.strftime("%b %Y"))
    if recovered:
        col_s4.metric("Durasi Pemulihan",
                      f"{duration} hari",
                      help=f"Pulih pada {recovery_date.strftime('%b %Y')}.")
    else:
        col_s4.metric("Status", "⏳ Belum Pulih",
                      delta=f"{duration} hari sejak puncak",
                      delta_color="off")

    # Narrative
    if recovered:
        st.info(
            f"📉 Puncak historis **{selected}** terjadi di {peak_date.strftime('%B %Y')}. "
            f"Setelah itu, harga jatuh hingga **-{mdd_pct:.1f}%** dari puncaknya "
            f"pada {trough_date.strftime('%B %Y')}, dan butuh **{duration} hari perdagangan** "
            f"(≈ {duration/252:.1f} tahun) untuk kembali ke level puncak di {recovery_date.strftime('%B %Y')}."
        )
    else:
        years = duration / 252
        st.warning(
            f"⚠️ **{selected} belum pulih** ke puncak historisnya di {peak_date.strftime('%B %Y')}. "
            f"Harga pernah jatuh **-{mdd_pct:.1f}%** ke titik terendah pada "
            f"{trough_date.strftime('%B %Y')}, dan sudah **{duration} hari perdagangan** "
            f"(≈ {years:.1f} tahun) underwater hingga akhir periode."
        )

    # Underwater plot: harga terindeks + drawdown
    price_indexed = stock_prices["Close"] / stock_prices["Close"].iloc[0] * 100

    fig_dd = make_subplots(specs=[[{"secondary_y": True}]])

    fig_dd.add_trace(
        go.Scatter(
            x=stock_prices["Date"], y=price_indexed,
            name="Harga (indeks = 100)",
            line=dict(color=sector_color, width=2),
            hovertemplate="%{x|%d %b %Y}<br>Harga terindeks: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig_dd.add_trace(
        go.Scatter(
            x=stock_dd_series["Date"], y=stock_dd_series["Drawdown"] * 100,
            name="Drawdown (%)",
            line=dict(color="#DC2626", width=0.8),
            fill="tozeroy", fillcolor = hex_to_rgba("#DC2626", 0.12),
            hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # Mark MDD point
    fig_dd.add_trace(
        go.Scatter(
            x=[trough_date], y=[-mdd_pct],
            mode="markers+text",
            marker=dict(color="#DC2626", size=12, symbol="x", line=dict(width=2, color="white")),
            text=[f"MDD -{mdd_pct:.1f}%"],
            textposition="bottom center",
            textfont=dict(size=10, color="#DC2626"),
            name="Titik MDD",
            hoverinfo="skip",
            showlegend=False,
        ),
        secondary_y=True,
    )

    # COVID shade
    fig_dd.add_vrect(
        x0=COVID_START, x1=COVID_END,
        fillcolor="#F59E0B", opacity=0.08, line_width=0,
    )

    fig_dd.update_xaxes(title_text="")
    fig_dd.update_yaxes(title_text="Harga Terindeks", secondary_y=False,
                        showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig_dd.update_yaxes(title_text="Drawdown (%)", secondary_y=True,
                        showgrid=False, range=[-100, 5])

    fig_dd.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    st.caption(
        "Garis berwarna = harga terindeks 100 di awal periode. "
        "Area merah = drawdown dari puncak berjalan (sumbu kanan, negatif). "
        "Area terisi = periode 'underwater'."
    )

# TAB 4: VOLATILITAS DINAMIS

with tab_vol:
    if len(stock_vol_ts) == 0:
        st.warning("Data volatilitas GARCH tidak tersedia untuk saham ini.")
    else:
        vol_ann = stock_vol_ts["garch_vol_annual"] * 100
        mean_vol = vol_ann.mean()
        peak_vol = vol_ann.max()
        peak_date_vol = stock_vol_ts.loc[vol_ann.idxmax(), "Date"]
        recent_vol = vol_ann.iloc[-1]

        # IDX-wide average for reference
        idx_mean_vol = vol_series_all["garch_vol_annual"].mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rata-rata Volatilitas", f"{mean_vol:.1f}%",
                  help="Rata-rata volatilitas tahunan sepanjang periode.")
        c2.metric("Puncak Volatilitas", f"{peak_vol:.1f}%",
                  help=f"Tercapai pada {pd.to_datetime(peak_date_vol).strftime('%b %Y')}.")
        c3.metric("Volatilitas Terkini", f"{recent_vol:.1f}%",
                  delta=f"{recent_vol - mean_vol:+.1f}% vs rata-rata",
                  delta_color="inverse")
        c4.metric("Rata-rata IDX", f"{idx_mean_vol:.1f}%",
                  help="Rata-rata 29 saham sepanjang periode.")

        fig_vol = go.Figure()

        fig_vol.add_trace(go.Scatter(
            x=stock_vol_ts["Date"], y=vol_ann,
            mode="lines",
            line=dict(color=sector_color, width=1.5),
            fill="tozeroy",
            fillcolor=hex_to_rgba(sector_color, 0.12),
            name=f"Volatilitas {selected}",
            hovertemplate="%{x|%d %b %Y}<br>Vol tahunan: %{y:.1f}%<extra></extra>",
        ))

        # Reference lines
        fig_vol.add_hline(
            y=idx_mean_vol, line_dash="dash", line_color="#888",
            annotation_text=f"Rata-rata IDX ({idx_mean_vol:.1f}%)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#888"),
        )
        fig_vol.add_hline(
            y=mean_vol, line_dash="dot", line_color=sector_color,
            annotation_text=f"Rata-rata {selected} ({mean_vol:.1f}%)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color=sector_color),
        )

        # COVID shade
        fig_vol.add_vrect(
            x0=COVID_START, x1=COVID_END,
            fillcolor="#F59E0B", opacity=0.10, line_width=0,
            annotation_text="COVID-19", annotation_position="top left",
            annotation_font=dict(size=10, color="#D97706"),
        )

        fig_vol.update_layout(
            title="Volatilitas Kondisional GARCH(1,1) — Tahunan",
            xaxis_title="",
            yaxis_title="Volatilitas Tahunan (%)",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hovermode="x unified",
        )
        fig_vol.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        fig_vol.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

        st.plotly_chart(fig_vol, use_container_width=True)

        # Interpretasi
        if recent_vol < mean_vol * 0.85:
            st.info(
                f"📉 Volatilitas terkini **{recent_vol:.1f}%** "
                f"lebih rendah dari rata-rata historisnya sendiri ({mean_vol:.1f}%), "
                f"mengindikasikan kondisi relatif tenang."
            )
        elif recent_vol > mean_vol * 1.15:
            st.warning(
                f"📈 Volatilitas terkini **{recent_vol:.1f}%** "
                f"lebih tinggi dari rata-rata historisnya ({mean_vol:.1f}%). "
                f"Fluktuasi harga sedang meningkat."
            )

        st.caption(
            "Volatilitas kondisional dari model GARCH(1,1) menangkap *volatility clustering* — "
            "periode fluktuasi besar cenderung diikuti periode fluktuasi besar lainnya."
        )

# FOOTER

st.divider()

# with st.expander("📚 Cara membaca halaman ini"):
#     st.markdown(
#         """
#         **Posisi rank** di setiap metrik menunjukkan dari mana saham ini dibandingkan
#         28 saham lainnya. Rank 1 = paling kecil (paling stabil untuk metrik itu), rank 29 = paling besar.

#         **Radar chart** menggunakan *persentil rank* (0–1) di tiap metrik, bukan nilai aktual.
#         Ini memungkinkan 5 metrik dengan satuan berbeda ditampilkan di satu chart.
#         Profil yang mengumpul ke pusat = stabil; profil yang mekar keluar = berisiko.

#         **Sortino Ratio** hanya menjadi konteks (bukan komponen Risk Score). Sortino yang tinggi
#         berarti return-nya tinggi relatif terhadap *downside deviation*.

#         **Semua metrik bersifat historis** dan menggunakan data Oktober 2012 – April 2024.
#         Tidak ada jaminan kinerja yang sama akan berlanjut di masa depan.
#         """
#     )