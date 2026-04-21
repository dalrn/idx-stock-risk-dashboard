"""
Sensitivitas Bobot — bagaimana ranking berubah jika bobot metrik diubah?
- Apakah peringkat akan berubah kalau saya lebih peduli pada dimensi risiko tertentu?
- Seberapa robust hasil peringkat di paper terhadap pilihan bobot?
- Saham mana yang posisinya paling sensitif terhadap bobot?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    SECTOR_COLORS, STOCK_FULL_NAMES,
    check_data_available, hex_to_rgba,
    load_risk_metrics,
)

# ============================================================
# CONFIG & DATA
# ============================================================

st.set_page_config(page_title="Sensitivitas Bobot",
                   page_icon="🎛️", layout="wide")

data_ok, _ = check_data_available()
if not data_ok:
    st.error("⚠️ Data belum tersedia. Jalankan preprocess.py terlebih dahulu.")
    st.stop()

risk = load_risk_metrics().rename(columns={"Stability_Score": "Risk_Score"})
N_STOCKS = len(risk)

# Metric metadata
METRIC_COLS = ["Volatility_Pct", "VaR_Pct", "CVaR_Pct", "DD_Pct", "MDD_Pct"]
METRIC_SHORT = {
    "Volatility_Pct": "Volatilitas",
    "VaR_Pct":        "VaR",
    "CVaR_Pct":       "CVaR",
    "DD_Pct":         "Downside Dev.",
    "MDD_Pct":        "Max Drawdown",
}
METRIC_DESC = {
    "Volatility_Pct": "Fluktuasi harian harga",
    "VaR_Pct":        "Batas kerugian 5% hari terburuk",
    "CVaR_Pct":       "Rata-rata kerugian 5% hari terburuk",
    "DD_Pct":         "Rata-rata kerugian hari negatif",
    "MDD_Pct":        "Kejatuhan kumulatif terparah",
}

# Bobot penelitian (baseline)
PAPER_WEIGHTS = {
    "Volatility_Pct": 20,
    "VaR_Pct":        10,
    "CVaR_Pct":       30,
    "DD_Pct":         15,
    "MDD_Pct":        25,
}

# Presets
PRESETS = {
    "📄 Penelitian (20/10/30/15/25)": {
        "Volatility_Pct": 20, "VaR_Pct": 10, "CVaR_Pct": 30,
        "DD_Pct": 15, "MDD_Pct": 25,
    },
    "⚖️ Setara (20/20/20/20/20)": {
        "Volatility_Pct": 20, "VaR_Pct": 20, "CVaR_Pct": 20,
        "DD_Pct": 20, "MDD_Pct": 20,
    },
    "💥 Fokus Risiko Ekor (0/20/50/10/20)": {
        "Volatility_Pct": 0, "VaR_Pct": 20, "CVaR_Pct": 50,
        "DD_Pct": 10, "MDD_Pct": 20,
    },
    "📊 Fokus Fluktuasi Harian (40/10/10/30/10)": {
        "Volatility_Pct": 40, "VaR_Pct": 10, "CVaR_Pct": 10,
        "DD_Pct": 30, "MDD_Pct": 10,
    },
    "⏳ Fokus Jangka Panjang (15/5/15/15/50)": {
        "Volatility_Pct": 15, "VaR_Pct": 5, "CVaR_Pct": 15,
        "DD_Pct": 15, "MDD_Pct": 50,
    },
}

# ============================================================
# STATE INIT
# ============================================================

# Inisialisasi slider values kalau belum ada
for col in METRIC_COLS:
    key = f"weight_{col}"
    if key not in st.session_state:
        st.session_state[key] = PAPER_WEIGHTS[col]

# ============================================================
# HEADER
# ============================================================

st.title("🎛️ Sensitivitas Bobot")
st.caption(
    "Bobot metrik risiko di paper dipilih berdasarkan pertimbangan tertentu "
    "(20% Volatilitas, 10% VaR, 30% CVaR, 15% Downside Dev., 25% MDD). "
    "Di sini kamu bisa memodifikasi bobot dan melihat bagaimana ranking berubah."
)

# ============================================================
# PRESET BUTTONS
# ============================================================

st.markdown("**Mulai dari preset:**")

preset_cols = st.columns(len(PRESETS))
for col, (preset_name, preset_weights) in zip(preset_cols, PRESETS.items()):
    with col:
        if st.button(preset_name, use_container_width=True, key=f"preset_{preset_name}"):
            for mcol, w in preset_weights.items():
                st.session_state[f"weight_{mcol}"] = w
            st.rerun()

st.divider()

# ============================================================
# SLIDERS
# ============================================================

st.markdown("### Atur Bobot Metrik")
st.caption(
    "Geser slider untuk mengubah bobot tiap metrik. Nilai ditampilkan dalam persen. "
    "Bobot yang tidak total 100% akan **dinormalisasi otomatis** — yang penting "
    "adalah proporsi relatif antar metrik."
)

slider_cols = st.columns(5)
raw_weights = {}

for col, mcol in zip(slider_cols, METRIC_COLS):
    with col:
        short = METRIC_SHORT[mcol]
        desc = METRIC_DESC[mcol]

        st.markdown(
            f"<div style='font-size: 0.95em; font-weight: 600;'>{short}</div>"
            f"<div style='font-size: 0.75em; color: #888; margin-bottom: 4px;'>{desc}</div>",
            unsafe_allow_html=True,
        )
        raw_weights[mcol] = st.slider(
            f"Bobot {short}",
            min_value=0, max_value=100,
            step=5,
            key=f"weight_{mcol}",
            label_visibility="collapsed",
        )

# --- Normalize ---
total_raw = sum(raw_weights.values())
if total_raw == 0:
    st.error("⚠️ Semua bobot tidak boleh nol. Setidaknya satu metrik harus punya bobot > 0.")
    st.stop()

norm_weights = {k: v / total_raw for k, v in raw_weights.items()}

# --- Indicator bar: visualisasi bobot ternormalisasi ---
st.markdown(
    f"<div style='font-size: 0.85em; color: #888; margin-top: 8px;'>"
    f"Total mentah: {total_raw}%. Ditampilkan setelah normalisasi ke 100%:"
    f"</div>",
    unsafe_allow_html=True,
)

# Stacked bar horizontal sebagai indikator
fig_bar = go.Figure()
# Plotly-friendly color per metric
METRIC_BAR_COLORS = {
    "Volatility_Pct": "#3B82F6",
    "VaR_Pct":        "#8B5CF6",
    "CVaR_Pct":       "#EC4899",
    "DD_Pct":         "#F59E0B",
    "MDD_Pct":        "#EF4444",
}
for mcol in METRIC_COLS:
    w = norm_weights[mcol] * 100
    if w < 0.01:
        continue
    fig_bar.add_trace(go.Bar(
        y=["Bobot"],
        x=[w],
        name=f"{METRIC_SHORT[mcol]} ({w:.0f}%)",
        orientation="h",
        marker=dict(color=METRIC_BAR_COLORS[mcol]),
        text=f"{METRIC_SHORT[mcol]}<br>{w:.0f}%",
        textposition="inside",
        textfont=dict(size=10, color="white"),
        hovertemplate=f"<b>{METRIC_SHORT[mcol]}</b><br>Bobot: {w:.1f}%<extra></extra>",
    ))
fig_bar.update_layout(
    barmode="stack",
    height=70,
    margin=dict(l=10, r=10, t=0, b=0),
    xaxis=dict(visible=False, range=[0, 100]),
    yaxis=dict(visible=False),
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ============================================================
# KOMPUTASI RANKING BARU
# ============================================================

def compute_ranking(weights: dict[str, float], df: pd.DataFrame) -> pd.DataFrame:
    """Compute Risk Score dan rank baru berdasarkan bobot yang diberikan.

    weights: normalized weights (sum = 1) untuk setiap metric
    df: risk DataFrame dengan kolom _norm yang sudah ada dari preprocess
    """
    result = df[["Stock_Name", "Sector"] + METRIC_COLS].copy()

    # Gunakan kolom _norm yang sudah dihitung di preprocess (percentile rank)
    score = sum(df[f"{c}_norm"] * w for c, w in weights.items())
    result["Risk_Score"] = score.values

    result = result.sort_values("Risk_Score").reset_index(drop=True)
    result["Rank"] = np.arange(1, len(result) + 1)
    return result

# Ranking dengan bobot saat ini (dari slider)
new_ranking = compute_ranking(norm_weights, risk)

# Ranking dengan bobot penelitian (baseline)
paper_weights_norm = {k: v / sum(PAPER_WEIGHTS.values()) for k, v in PAPER_WEIGHTS.items()}
paper_ranking = compute_ranking(paper_weights_norm, risk)

# Merge untuk delta comparison
merged = new_ranking[["Stock_Name", "Sector", "Rank", "Risk_Score"]].merge(
    paper_ranking[["Stock_Name", "Rank", "Risk_Score"]].rename(
        columns={"Rank": "Rank_Paper", "Risk_Score": "Score_Paper"}
    ),
    on="Stock_Name", how="left",
)
merged["Rank_Delta"] = merged["Rank_Paper"] - merged["Rank"]  # + = naik, - = turun

# Cek apakah bobot saat ini identik dengan penelitian
is_at_baseline = all(
    abs(norm_weights[c] - paper_weights_norm[c]) < 0.001 for c in METRIC_COLS
)

# ============================================================
# RANKING TABLE + DELTA VIEWER
# ============================================================

col_rank, col_delta = st.columns([3, 2])

# ------------------- RANKING TABLE -------------------
with col_rank:
    st.markdown("### 🏆 Ranking Baru")

    if is_at_baseline:
        st.info("Saat ini menggunakan **bobot penelitian**. Geser slider atau pilih preset lain untuk melihat perubahan.")

    # Tampilkan top 15, lalu expander untuk sisanya
    top_n = 15
    show_df = merged.head(top_n).copy()

    # Tambahkan kolom nama perusahaan
    show_df.insert(2, "Nama",
                   show_df["Stock_Name"].map(STOCK_FULL_NAMES))

    # Render sebagai dataframe dengan column config
    # Delta column sebagai text dengan arrow
    def delta_display(d):
        if pd.isna(d) or d == 0:
            return "—"
        elif d > 0:
            return f"▲ +{int(d)}"
        else:
            return f"▼ {int(d)}"

    show_df["Perubahan"] = show_df["Rank_Delta"].apply(delta_display)

    st.dataframe(
        show_df[["Rank", "Stock_Name", "Nama", "Sector",
                 "Risk_Score", "Rank_Paper", "Perubahan"]],
        column_config={
            "Rank": st.column_config.NumberColumn("Rank Baru", format="#%d", width="small"),
            "Stock_Name": st.column_config.TextColumn("Kode", width="small"),
            "Nama": st.column_config.TextColumn("Nama Perusahaan", width="medium"),
            "Sector": st.column_config.TextColumn("Sektor", width="medium"),
            "Risk_Score": st.column_config.ProgressColumn(
                "Risk Score", format="%.4f",
                min_value=0.0, max_value=1.0,
            ),
            "Rank_Paper": st.column_config.NumberColumn("Rank Paper", format="#%d", width="small"),
            "Perubahan": st.column_config.TextColumn("Δ vs Paper", width="small"),
        },
        hide_index=True,
        use_container_width=True,
        height=min(580, (len(show_df) + 1) * 35 + 3),
    )

    with st.expander(f"Lihat rank #{top_n+1} – #{N_STOCKS}"):
        rest_df = merged.iloc[top_n:].copy()
        rest_df.insert(2, "Nama", rest_df["Stock_Name"].map(STOCK_FULL_NAMES))
        rest_df["Perubahan"] = rest_df["Rank_Delta"].apply(delta_display)
        st.dataframe(
            rest_df[["Rank", "Stock_Name", "Nama", "Sector",
                     "Risk_Score", "Rank_Paper", "Perubahan"]],
            column_config={
                "Rank": st.column_config.NumberColumn("Rank Baru", format="#%d"),
                "Risk_Score": st.column_config.ProgressColumn(
                    "Risk Score", format="%.4f", min_value=0.0, max_value=1.0,
                ),
                "Rank_Paper": st.column_config.NumberColumn("Rank Paper", format="#%d"),
                "Perubahan": st.column_config.TextColumn("Δ vs Paper"),
            },
            hide_index=True, use_container_width=True,
        )

# ------------------- DELTA VIEWER -------------------
with col_delta:
    st.markdown("### 🔀 Perubahan Terbesar")
    st.caption("Saham yang rank-nya paling berubah dibanding bobot penelitian.")

    if is_at_baseline:
        st.markdown(
            """
            <div style="padding: 20px; text-align: center; color: #888;
                        border: 1px dashed rgba(128,128,128,0.3); border-radius: 8px;">
                <div style="font-size: 2em; margin-bottom: 8px;">⚖️</div>
                <div>Ranking identik dengan bobot penelitian.<br>
                Geser slider untuk melihat perubahan.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Naik peringkat (Rank_Delta > 0, rank kecil = peringkat lebih baik)
        movers_up = merged.sort_values("Rank_Delta", ascending=False).head(5)
        movers_down = merged.sort_values("Rank_Delta", ascending=True).head(5)

        # --- NAIK ---
        st.markdown("#### 🟢 Naik Peringkat")
        for _, r in movers_up.iterrows():
            if r["Rank_Delta"] <= 0:
                continue
            sector_color = SECTOR_COLORS.get(r["Sector"], "#666")
            st.markdown(
                f"""
                <div style="padding: 8px 12px; margin-bottom: 6px;
                            border-left: 3px solid {sector_color};
                            background: rgba(16,185,129,0.06);
                            border-radius: 4px;
                            display: flex; justify-content: space-between;
                            align-items: center;">
                    <div>
                        <strong>{r['Stock_Name']}</strong>
                        <span style="color: #888; font-size: 0.85em;">
                            &nbsp;#{int(r['Rank_Paper'])} → #{int(r['Rank'])}
                        </span>
                    </div>
                    <span style="color: #10B981; font-weight: 600;">
                        ▲ +{int(r['Rank_Delta'])}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- TURUN ---
        st.markdown("#### 🔴 Turun Peringkat")
        for _, r in movers_down.iterrows():
            if r["Rank_Delta"] >= 0:
                continue
            sector_color = SECTOR_COLORS.get(r["Sector"], "#666")
            st.markdown(
                f"""
                <div style="padding: 8px 12px; margin-bottom: 6px;
                            border-left: 3px solid {sector_color};
                            background: rgba(239,68,68,0.06);
                            border-radius: 4px;
                            display: flex; justify-content: space-between;
                            align-items: center;">
                    <div>
                        <strong>{r['Stock_Name']}</strong>
                        <span style="color: #888; font-size: 0.85em;">
                            &nbsp;#{int(r['Rank_Paper'])} → #{int(r['Rank'])}
                        </span>
                    </div>
                    <span style="color: #EF4444; font-weight: 600;">
                        ▼ {int(r['Rank_Delta'])}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Ringkasan perubahan top-3 & bottom-3
        n_changed_top3 = (merged.head(3)["Stock_Name"].tolist() !=
                          paper_ranking.head(3)["Stock_Name"].tolist())
        n_changed_bot3 = (merged.tail(3)["Stock_Name"].tolist() !=
                          paper_ranking.tail(3)["Stock_Name"].tolist())

        summary_lines = []
        if not n_changed_top3:
            summary_lines.append("✓ Top 3 **tetap sama**")
        else:
            summary_lines.append("⚠ Top 3 **berubah**")

        if not n_changed_bot3:
            summary_lines.append("✓ Bottom 3 **tetap sama**")
        else:
            summary_lines.append("⚠ Bottom 3 **berubah**")

        st.info(" · ".join(summary_lines))

st.divider()

# ============================================================
# DUMBBELL CHART: RANK NEW VS PAPER
# ============================================================

if not is_at_baseline:
    st.markdown("### 📊 Visualisasi Perubahan Rank")
    st.caption(
        "Setiap baris adalah satu saham. Titik abu-abu = rank di bobot penelitian. "
        "Titik berwarna = rank dengan bobot kamu saat ini. Garis = arah perubahan."
    )

    # Hanya tampilkan saham yang rank-nya berubah
    changed = merged[merged["Rank_Delta"] != 0].copy()

    if len(changed) == 0:
        st.info("Tidak ada saham yang berpindah rank dengan konfigurasi ini.")
    else:
        # Sort by rank_paper supaya urutan y-axis rapi dari stabil ke berisiko
        changed = changed.sort_values("Rank_Paper").reset_index(drop=True)

        fig_dumb = go.Figure()

        for _, r in changed.iterrows():
            sector_color = SECTOR_COLORS.get(r["Sector"], "#666")
            direction_color = "#10B981" if r["Rank_Delta"] > 0 else "#EF4444"

            # Garis konektor
            fig_dumb.add_trace(go.Scatter(
                x=[r["Rank_Paper"], r["Rank"]],
                y=[r["Stock_Name"], r["Stock_Name"]],
                mode="lines",
                line=dict(color=hex_to_rgba(direction_color, 0.4), width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

            # Titik paper (abu-abu)
            fig_dumb.add_trace(go.Scatter(
                x=[r["Rank_Paper"]], y=[r["Stock_Name"]],
                mode="markers",
                marker=dict(color="#94A3B8", size=10,
                           line=dict(color="white", width=1)),
                hovertemplate=(
                    f"<b>{r['Stock_Name']}</b><br>"
                    f"Rank penelitian: #{int(r['Rank_Paper'])}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

            # Titik baru (warna sektor)
            fig_dumb.add_trace(go.Scatter(
                x=[r["Rank"]], y=[r["Stock_Name"]],
                mode="markers",
                marker=dict(color=sector_color, size=12,
                           line=dict(color="white", width=1.5)),
                hovertemplate=(
                    f"<b>{r['Stock_Name']}</b><br>"
                    f"Rank baru: #{int(r['Rank'])}<br>"
                    f"Perubahan: {'+' if r['Rank_Delta'] > 0 else ''}{int(r['Rank_Delta'])}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        # Legend dummy
        fig_dumb.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="#94A3B8", size=10),
            name="Rank Penelitian",
        ))
        fig_dumb.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="#3B82F6", size=12),
            name="Rank Baru (warna = sektor)",
        ))

        fig_dumb.update_layout(
            height=max(400, 25 * len(changed) + 100),
            margin=dict(l=10, r=10, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Rank", range=[0, N_STOCKS + 1],
                showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                dtick=5,
            ),
            yaxis=dict(
                title="", autorange="reversed",  # stabil di atas
                showgrid=False,
            ),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        )

        st.plotly_chart(fig_dumb, use_container_width=True)

    st.divider()

# ============================================================
# ROBUSTNESS CHECK (MONTE CARLO)
# ============================================================

st.markdown("### 🎲 Analisis Ketahanan (Robustness)")
st.caption(
    "Seberapa sering setiap saham menempati rank tertentu jika bobot diambil "
    "secara acak dari ribuan konfigurasi? Ini menjawab: *apakah hasil ranking "
    "di paper bergantung pada bobot spesifik yang dipilih?*"
)

@st.cache_data(show_spinner="Menjalankan simulasi ribuan konfigurasi bobot...")
def monte_carlo_robustness(n_simulations: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Jalankan n_simulations dengan bobot acak (Dirichlet).
    Return DataFrame dengan proporsi saham menduduki rank tertentu.
    """
    rng = np.random.default_rng(seed)
    # Dirichlet dengan alpha = 1 (uniform atas simplex)
    random_weights = rng.dirichlet(alpha=np.ones(len(METRIC_COLS)), size=n_simulations)

    # Hitung rank untuk tiap simulasi
    # Vectorized: norm_matrix (29 × 5) @ weights.T (5 × n_sim) = scores (29 × n_sim)
    norm_matrix = risk[[f"{c}_norm" for c in METRIC_COLS]].values  # (29, 5)
    scores = norm_matrix @ random_weights.T  # (29, n_sim)

    # Untuk setiap simulasi, cari rank tiap saham
    # argsort along axis=0 → indices sorted; argsort lagi → rank
    ranks = scores.argsort(axis=0).argsort(axis=0) + 1  # (29, n_sim)

    stock_names = risk["Stock_Name"].values

    # Statistik per saham
    records = []
    for i, stock in enumerate(stock_names):
        stock_ranks = ranks[i, :]  # shape (n_sim,)
        records.append({
            "Stock_Name": stock,
            "Mean_Rank": float(np.mean(stock_ranks)),
            "Median_Rank": float(np.median(stock_ranks)),
            "Std_Rank": float(np.std(stock_ranks)),
            "P25_Rank": float(np.percentile(stock_ranks, 25)),
            "P75_Rank": float(np.percentile(stock_ranks, 75)),
            "P5_Rank": float(np.percentile(stock_ranks, 5)),
            "P95_Rank": float(np.percentile(stock_ranks, 95)),
            "Pct_Top1": float(np.mean(stock_ranks == 1) * 100),
            "Pct_Top3": float(np.mean(stock_ranks <= 3) * 100),
            "Pct_Top5": float(np.mean(stock_ranks <= 5) * 100),
            "Pct_Top10": float(np.mean(stock_ranks <= 10) * 100),
            "Pct_Bottom3": float(np.mean(stock_ranks >= N_STOCKS - 2) * 100),
            "Pct_Bottom10": float(np.mean(stock_ranks >= N_STOCKS - 9) * 100),  
        })

    return pd.DataFrame(records)

robust_df = monte_carlo_robustness(n_simulations=1000)
robust_df = robust_df.sort_values("Mean_Rank").reset_index(drop=True)

# --- Key robustness numbers ---
top1_stock = robust_df.sort_values("Pct_Top1", ascending=False).iloc[0]
most_volatile_rank = robust_df.sort_values("Std_Rank", ascending=False).iloc[0]

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric(
        label=f"{top1_stock['Stock_Name']} menempati rank #1",
        value=f"{top1_stock['Pct_Top1']:.0f}% dari simulasi",
        help="Persentase konfigurasi bobot acak di mana saham ini menjadi yang paling stabil.",
    )

with kpi_cols[1]:
    pct_top10_stable = (robust_df.head(10)["Pct_Top10"] > 70).sum()
    st.metric(
        label="Saham top-10 yang 'kokoh'",
        value=f"{pct_top10_stable} dari 10",
        help="Saham yang >95% simulasi tetap menempati rank 1-10, terlepas dari bobot apa pun.",
    )

with kpi_cols[2]:
    pct_bot10_stable = (robust_df.tail(10)["Pct_Bottom10"] > 70).sum()
    st.metric(
        label="Saham bottom-10 yang 'kokoh'",
        value=f"{pct_bot10_stable} dari 10",
        help="Saham yang >95% simulasi tetap menempati rank 20-29, terlepas dari bobot apa pun.",
    )

with kpi_cols[3]:
    st.metric(
        label="Rank paling labil",
        value=f"{most_volatile_rank['Stock_Name']}",
        delta=f"± {most_volatile_rank['Std_Rank']:.1f} rank",
        delta_color="off",
        help="Saham dengan deviasi rank terbesar — posisinya sangat tergantung pada bobot.",
    )

# --- Range chart: distribusi rank per saham ---
st.markdown("#### Rentang Rank Tiap Saham")
st.caption(
    "Box = rentang rank dari 25%–75% simulasi. Whisker = 5%–95%. "
    "Titik = median. Saham dengan rentang pendek = rank stabil lintas konfigurasi bobot."
)

# Sort by median rank untuk urutan yang logis
robust_sorted = robust_df.sort_values("Median_Rank").reset_index(drop=True)

fig_range = go.Figure()

for _, r in robust_sorted.iterrows():
    stock = r["Stock_Name"]
    sector = risk[risk["Stock_Name"] == stock].iloc[0]["Sector"]
    sector_color = SECTOR_COLORS.get(sector, "#666")

    # Whisker (5-95 percentile)
    fig_range.add_trace(go.Scatter(
        x=[r["P5_Rank"], r["P95_Rank"]], y=[stock, stock],
        mode="lines",
        line=dict(color=hex_to_rgba(sector_color, 0.3), width=1.5),
        showlegend=False, hoverinfo="skip",
    ))

    # Box (25-75 percentile)
    fig_range.add_trace(go.Scatter(
        x=[r["P25_Rank"], r["P75_Rank"]], y=[stock, stock],
        mode="lines",
        line=dict(color=sector_color, width=6),
        showlegend=False,
        hovertemplate=(
            f"<b>{stock}</b><br>"
            f"Median rank: {r['Median_Rank']:.1f}<br>"
            f"Rentang 25%-75%: {r['P25_Rank']:.0f} – {r['P75_Rank']:.0f}<br>"
            f"Rentang 5%-95%: {r['P5_Rank']:.0f} – {r['P95_Rank']:.0f}<br>"
            f"<i>{sector}</i>"
            "<extra></extra>"
        ),
    ))

    # Median marker
    fig_range.add_trace(go.Scatter(
        x=[r["Median_Rank"]], y=[stock],
        mode="markers",
        marker=dict(color="white", size=8,
                   line=dict(color=sector_color, width=2)),
        showlegend=False, hoverinfo="skip",
    ))

fig_range.update_layout(
    height=max(600, 22 * N_STOCKS + 100),
    margin=dict(l=10, r=10, t=10, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title="Rank (1 = paling stabil, 29 = paling berisiko)",
        range=[0, N_STOCKS + 1],
        showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        dtick=5,
    ),
    yaxis=dict(
        title="", autorange="reversed",
        showgrid=False, tickfont=dict(size=10),
    ),
)

st.plotly_chart(fig_range, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.divider()

with st.expander("📚 Cara membaca halaman ini"):
    st.markdown(
        """
        **Apa itu "bobot"?** Risk Score adalah *weighted sum* dari 5 metrik risiko
        yang sudah dinormalisasi ke rank persentil. Bobot menentukan seberapa berpengaruh
        tiap metrik terhadap skor akhir.

        **Mengapa penelitian pakai 20/10/30/15/25?**
        - **CVaR (30%)** tertinggi karena menangkap risiko ekor (standar regulasi perbankan FRTB)
        - **MDD (25%)** tinggi karena satu-satunya metrik yang memperhitungkan urutan waktu
        - **Volatilitas (20%)** sebagai ukuran umum fluktuasi
        - **Downside Deviation (15%)** melengkapi volatilitas dengan fokus kerugian
        - **VaR (10%)** rendah karena banyak informasinya sudah tercakup CVaR

        **Analisis Ketahanan** menjalankan 1.000 simulasi dengan bobot acak dari
        distribusi Dirichlet (uniform atas simplex 5-dimensi). Box plot menunjukkan
        rentang rank tiap saham di seluruh simulasi. Saham dengan box pendek =
        rank konsisten lintas segala konfigurasi bobot = robust finding.

        **Implikasi metodologis.** Kalau rank suatu saham sangat sensitif terhadap bobot,
        itu tanda bahwa *penilaian risiko saham itu dimensi-dependent* — tidak bisa
        dirangkum ke satu skor tunggal tanpa menimbulkan bias.
        """
    )