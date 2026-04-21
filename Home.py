import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import jarque_bera as jb

from utils.data_loader import (
    load_risk_metrics, load_sortino, load_drawdowns,
    SECTOR_COLORS, STOCK_FULL_NAMES, check_data_available, DATA_DIR,
)

# KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Analisis Risiko Saham IDX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# DATA AVAILABILITY CHECK

data_ok, missing = check_data_available()
if not data_ok:
    st.error("⚠️ Data belum tersedia")
    st.markdown(
        f"""
        File tidak ditemukan di `{DATA_DIR}`:

        {chr(10).join(f"- `{f}`" for f in missing)}
        """
    )
    st.stop()

# LOAD DATA

risk = load_risk_metrics().rename(columns={"Stability_Score": "Risk_Score"})
sortino = load_sortino()
drawdowns = load_drawdowns()

# HEADER

st.title("Analisis Risiko Saham Bursa Efek Indonesia")
st.markdown(
    """
    Dashboard analisis stabilitas **29 saham IDX** periode **Oktober 2012 – April 2024**,
    berbasis kerangka pengukuran risiko multidimensi.   
    """
)
st.caption(
    "Disclaimer: Dashboard ini merupakan analisis risiko historis semata dan bukan merupakan saran investasi."
)

st.divider()


st.subheader("Ringkasan Dataset")

# Baris 1 — deskriptor data mentah
row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.metric(
        label="Jumlah Saham",
        value=f"{len(risk)}",
        help="Saham IDX yang dianalisis, lintas 8 sektor.",
    )

with row1_col2:
    st.metric(
        label="Jumlah Sektor",
        value=f"{risk['Sector'].nunique()}",
        help="Sektor industri yang tercakup, sesuai klasifikasi BEI.",
    )

with row1_col3:
    st.metric(
        label="Periode Data",
        value="Okt 2012 – Apr 2024",
        help="Periode pengamatan setelah filter konsistensi OHLC.",
    )

# Baris 2 — insight statistik
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.metric(
        label="Jumlah Observasi",
        value="80.786",
        help="Total baris data (saham × tanggal perdagangan) setelah pembersihan.",
    )

with row2_col2:
    avg_vol = risk["Volatility_Pct"].mean()
    st.metric(
        label="Rata-rata Volatilitas Tahunan",
        value=f"{avg_vol:.1f}%",
        help="Rata-rata volatilitas GARCH tahunan dari 29 saham.",
    )

with row2_col3:
    vol_min = risk["Volatility_Pct"].min()
    vol_max = risk["Volatility_Pct"].max()
    st.metric(
        label="Rentang Volatilitas",
        value=f"{vol_min:.0f}% – {vol_max:.0f}%",
        help=(
            "Volatilitas tahunan terendah hingga tertinggi di antara 29 saham. "
            "Menunjukkan betapa beragamnya profil risiko di IDX."
        ),
    )

st.divider()

# LEADERBOARD: TOP 5 vs BOTTOM 5

st.subheader("Peringkat Teratas dan Terbawah")

col_top, col_bot = st.columns(2)

def render_leaderboard_card(df_slice: pd.DataFrame, title: str, icon: str):
    st.markdown(f"#### {icon} {title}")
    for _, row in df_slice.iterrows():
        stock = row["Stock_Name"]
        full = STOCK_FULL_NAMES.get(stock, "")
        score = row["Risk_Score"]
        sector = row["Sector"]
        color = SECTOR_COLORS.get(sector, "#888")

        st.markdown(
            f"""
            <div style="
                padding: 10px 14px; margin-bottom: 6px;
                border-left: 4px solid {color};
                background: rgba(128,128,128,0.05);
                border-radius: 4px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>#{row['Rank']} {stock}</strong>
                        <span style="color: #888; font-size: 0.85em;"> — {full}</span>
                    </div>
                    <div style="font-family: monospace; font-size: 0.9em;">
                        Risk Score: <strong>{score:.4f}</strong>
                    </div>
                </div>
                <div style="font-size: 0.8em; color: #888; margin-top: 2px;">
                    {sector}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_top:
    render_leaderboard_card(
        risk.head(5),
        "5 Saham Paling Stabil",
        "🟢",
    )

with col_bot:
    render_leaderboard_card(
        risk.tail(5).iloc[::-1],
        "5 Saham Paling Berisiko",
        "🔴",
    )

with st.expander("❓ Bagaimana peringkat ini dihitung?", expanded=False):
    st.markdown(
        """
        Peringkat di atas dihitung dari **Risk Score** — kombinasi berbobot dari
        5 metrik risiko yang masing-masing menangkap dimensi berbeda:
        """
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(
            """
            **Volatilitas GARCH**
            <span style="color:#888; font-size:0.85em;">(bobot 20%)</span>

            <span style="font-size:0.9em;">Seberapa besar harga bergerak naik-turun setiap hari.</span>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            **Value at Risk**
            <span style="color:#888; font-size:0.85em;">(bobot 10%)</span>

            <span style="font-size:0.9em;">Batas kerugian pada 5% hari terburuk.</span>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            """
            **Conditional Value at Risk**
            <span style="color:#888; font-size:0.85em;">(bobot 30%)</span>

            <span style="font-size:0.9em;">Rata-rata kerugian pada 5% hari terburuk.</span>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            """
            **Downside Deviation**
            <span style="color:#888; font-size:0.85em;">(bobot 15%)</span>

            <span style="font-size:0.9em;">Rata-rata kerugian di semua hari merugi.</span>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            """
            **Maximum Drawdown**
            <span style="color:#888; font-size:0.85em;">(bobot 25%)</span>

            <span style="font-size:0.9em;">Kejatuhan terparah dari harga puncaknya.</span>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Skor akhir dihitung dengan *rank normalization* tiap metrik lalu dijumlahkan "
        "menurut bobot di atas. Skor rendah = saham lebih stabil. Detail metodologi "
        "di halaman **Peringkat** dan **Sensitivitas Bobot**."
    )

st.info(
    "💡 Buka halaman **Profil Saham** (sidebar) untuk analisis detail per saham, "
    "atau **Peringkat** untuk tabel lengkap 29 saham."
)

st.divider()

# VISUALISASI RATA-RATA RISK SCORE PER SEKTOR

st.subheader("Profil Risiko per Sektor")

sector_avg = (
    risk.groupby("Sector", as_index=False)
    .agg(
        Avg_Score=("Risk_Score", "mean"),
        N_Stocks=("Stock_Name", "count"),
    )
    .sort_values("Avg_Score")
)

fig_sector = go.Figure()
fig_sector.add_trace(
    go.Bar(
        y=sector_avg["Sector"],
        x=sector_avg["Avg_Score"],
        orientation="h",
        marker=dict(
            color=[SECTOR_COLORS.get(s, "#888") for s in sector_avg["Sector"]],
            line=dict(color="white", width=1),
        ),
        text=[f"  {v:.3f}  (n={n})" for v, n in zip(sector_avg["Avg_Score"], sector_avg["N_Stocks"])],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Rata-rata Risk Score: %{x:.4f}<br>"
            "Jumlah saham: %{customdata}"
            "<extra></extra>"
        ),
        customdata=sector_avg["N_Stocks"],
    )
)
fig_sector.update_layout(
    title=dict(
        text="Rata-rata Risk Score per Sektor <span style='font-size:12px;color:#888'>(rendah = lebih stabil)</span>",
        font=dict(size=16),
    ),
    xaxis=dict(title="Rata-rata Risk Score", range=[0, sector_avg["Avg_Score"].max() * 1.25]),
    yaxis=dict(title=""),
    height=420,
    margin=dict(l=10, r=10, t=60, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
fig_sector.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)
fig_sector.update_yaxes(showgrid=False)

st.plotly_chart(fig_sector, use_container_width=True)

st.caption(
    "Sektor **Financials** konsisten tercatat sebagai sektor paling stabil, "
    "sedangkan **Energy** dan **Basic Materials** memiliki profil risiko tertinggi."
)

st.divider()

# TAHUKAH ANDA?

# Generate insights dari data
def generate_insights() -> list[dict]:
    """Generate insight cards dari data yang sudah dimuat."""
    insights = []

    # Insight 1: BBCA sebagai satu-satunya Sortino > 1
    best_sortino = sortino.sort_values("Sortino", ascending=False).iloc[0]
    n_sortino_gt_1 = (sortino["Sortino"] > 1).sum()
    if n_sortino_gt_1 <= 1:
        insights.append({
            "icon": "🏆",
            "title": "Hanya satu saham dengan Sortino Ratio di atas 1",
            "body": (
                f"{best_sortino['Stock_Name']} ({STOCK_FULL_NAMES.get(best_sortino['Stock_Name'], '')}) "
                f"adalah satu-satunya saham dengan Sortino Ratio > 1 (nilai: {best_sortino['Sortino']:.3f}), "
                f"artinya return tahunannya melebihi downside deviation-nya."
            ),
        })

    # Insight 2: Jumlah saham belum pulih
    not_recovered = drawdowns[~drawdowns["Recovered"]]
    if len(not_recovered) > 0:
        longest = not_recovered.sort_values("Peak_date").iloc[0]
        insights.append({
            "icon": "⏳",
            "title": f"{len(not_recovered)} dari 29 saham belum pulih ke puncaknya",
            "body": (
                f"Per April 2024, {len(not_recovered)} saham masih berada di bawah harga tertinggi historisnya. "
                f"Yang paling lama, {longest['Stock_Name']}, sudah underwater sejak "
                f"{longest['Peak_date'].strftime('%B %Y')}."
            ),
        })

    # Insight 3: Sektor paling stabil vs paling berisiko
    sec_sorted = sector_avg.copy()
    best_sec = sec_sorted.iloc[0]
    worst_sec = sec_sorted.iloc[-1]
    insights.append({
        "icon": "📈",
        "title": "Jurang risiko antar sektor cukup besar",
        "body": (
            f"Rata-rata Risk Score sektor {best_sec['Sector']} ({best_sec['Avg_Score']:.3f}) "
            f"hanya sekitar {(best_sec['Avg_Score'] / worst_sec['Avg_Score'] * 100):.0f}% "
            f"dari sektor {worst_sec['Sector']} ({worst_sec['Avg_Score']:.3f})."
        ),
    })

    # Insight 4: MDD range
    avg_mdd = risk["MDD_Pct"].mean()
    insights.append({
        "icon": "📉",
        "title": f"Rata-rata MDD 29 saham: {avg_mdd:.1f}%",
        "body": (
            f"Artinya secara rata-rata setiap saham pernah turun {avg_mdd:.0f}% dari puncaknya selama "
            f"periode pengamatan."
        ),
    })

    # Insight 5: BBCA vs MEDC (spread terbesar)
    most_stable = risk.iloc[0]
    least_stable = risk.iloc[-1]
    insights.append({
        "icon": "🎯",
        "title": "Rentang volatilitas antar saham sangat lebar",
        "body": (
            f"{least_stable['Stock_Name']} memiliki volatilitas tahunan {least_stable['Volatility_Pct']:.1f}%, "
            f"lebih dari dua kali lipat {most_stable['Stock_Name']} sebesar {most_stable['Volatility_Pct']:.1f}%."
        ),
    })

    # Insight 6: Durasi pemulihan rata-rata (shocking number)
    recovery_days = drawdowns[drawdowns["Recovered"]]["Duration_days"].mean()
    if pd.notna(recovery_days):
        years = recovery_days / 252
        insights.append({
            "icon": "⏱️",
            "title": f"Butuh rata-rata {years:.1f} tahun untuk pulih dari kejatuhan terburuk",
            "body": (
                f"Saham yang berhasil pulih ke puncaknya memerlukan rata-rata "
                f"{recovery_days:.0f} hari perdagangan (sekitar {years:.1f} tahun). "
                f"Ini gambaran tentang horizon investasi yang realistis ketika pasar sedang tertekan."
            ),
        })

    # Insight 8: BBCA sebagai pemimpin absolut
    top_stock = risk.iloc[0]
    second_stock = risk.iloc[1]
    score_gap = (second_stock["Risk_Score"] - top_stock["Risk_Score"]) / top_stock["Risk_Score"] * 100
    insights.append({
        "icon": "👑",
        "title": f"{top_stock['Stock_Name']} unggul jauh dari pesaingnya",
        "body": (
            f"Risk Score {top_stock['Stock_Name']} ({top_stock['Risk_Score']:.4f}) adalah "
            f"{score_gap:.0f}% lebih rendah dari saham #2 ({second_stock['Stock_Name']}, "
            f"{second_stock['Risk_Score']:.4f}). Bukan hanya paling stabil, tapi stabil "
            f"dengan selisih yang substansial."
        ),
    })

    # Insight 10: MDD terburuk konkret
    worst_mdd = drawdowns.sort_values("MDD_Pct", ascending=False).iloc[0]
    insights.append({
        "icon": "💔",
        "title": f"Kejatuhan terparah: {worst_mdd['Stock_Name']} turun {worst_mdd['MDD_Pct']:.0f}%",
        "body": (
            f"{worst_mdd['Stock_Name']} pernah turun {worst_mdd['MDD_Pct']:.1f}% dari "
            f"puncaknya di {pd.to_datetime(worst_mdd['Peak_date']).strftime('%B %Y')}. "
            f"Pemegang saham yang beli di puncak sampai sekarang "
            f"{'sudah ' if worst_mdd['Recovered'] else 'belum '}"
            f"{'pulih' if worst_mdd['Recovered'] else 'kembali ke modal awal'}."
        ),
    })

    # Insight 11: Sektor winner konkret
    # Cari saham paling stabil di tiap sektor dominan
    top_per_sector = risk.groupby("Sector").apply(
        lambda g: g.sort_values("Risk_Score").iloc[0]
    ).reset_index(drop=True)
    financial_winner = top_per_sector[top_per_sector["Sector"] == "Financials"].iloc[0] \
        if "Financials" in top_per_sector["Sector"].values else None
    if financial_winner is not None:
        insights.append({
            "icon": "🏦",
            "title": "Sektor perbankan mendominasi papan atas",
            "body": (
                f"Dari 10 saham paling stabil, {(risk.head(10)['Sector'] == 'Financials').sum()} "
                f"berasal dari sektor Financials. {financial_winner['Stock_Name']} memimpin "
                f"dengan rank #{int(financial_winner['Rank'])}, dan sektor ini secara konsisten "
                f"menunjukkan profil risiko rendah."
            ),
        })

    return insights

insights = generate_insights()



st.subheader("💡 Tahukah Anda?")

# Convert markdown **bold** ke <strong> karena tidak render di dalam HTML
import re
def md_bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

# Render 1 container dengan horizontal scroll
cards_html = ""
for ins in insights:
    cards_html += f"""
    <div style="
        flex: 0 0 320px;
        padding: 16px;
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        background: rgba(128,128,128,0.03);
        scroll-snap-align: start;
    ">
        <div style="font-size: 1.5em; margin-bottom: 8px;">{ins['icon']}</div>
        <div style="font-weight: 600; margin-bottom: 8px; line-height: 1.3;">
            {md_bold_to_html(ins['title'])}
        </div>
        <div style="font-size: 0.9em; color: rgba(128,128,128,0.9); line-height: 1.5;">
            {md_bold_to_html(ins['body'])}
        </div>
    </div>
    """

st.markdown(
    f"""
    <div style="
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding-bottom: 12px;
        scroll-snap-type: x mandatory;
        -webkit-overflow-scrolling: touch;
    ">
        {cards_html}
    </div>
    <style>
        /* Scrollbar styling */
        div[data-testid="stMarkdownContainer"] > div::-webkit-scrollbar {{
            height: 8px;
        }}
        div[data-testid="stMarkdownContainer"] > div::-webkit-scrollbar-thumb {{
            background: rgba(128,128,128,0.3);
            border-radius: 4px;
        }}
        div[data-testid="stMarkdownContainer"] > div::-webkit-scrollbar-thumb:hover {{
            background: rgba(128,128,128,0.5);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("💫 Geser ke kanan untuk melihat insight lainnya.")


# FOOTER

st.divider()

# with st.expander("ℹ️ Tentang dashboard ini"):
#     st.markdown(
#         """
#         Dashboard ini merupakan implementasi interaktif dari analisis yang diajukan
#         dalam lomba **Data Analysis Competition Matematika Fair 2026** (Universitas
#         Negeri Medan), dengan judul *Pendekatan Multidimensi dalam Analisis Risiko
#         Saham: Integrasi Volatilitas, VaR, CVaR, Downside Deviation, dan Drawdown
#         untuk Pemeringkatan Saham Bursa Efek Indonesia*.

#         **Disclaimer.** Dashboard ini merupakan analisis risiko historis semata dan
#         **tidak merupakan saran investasi**. Kinerja masa lalu tidak menjamin kinerja
#         masa depan.
#         """
#     )