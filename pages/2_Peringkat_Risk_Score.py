"""
Peringkat & Perbandingan — tabel lengkap 29 saham + mode perbandingan.
- Saham mana yang paling stabil / berisiko secara keseluruhan?
- Bagaimana saham X vs Y (vs Z) berbeda profilnya?
- Dari 29 saham, pola mana yang menonjol secara visual?
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    SECTOR_COLORS, STOCK_FULL_NAMES,
    check_data_available, load_prices, load_risk_metrics, hex_to_rgba
)

# CONFIG & DATA CHECK

st.set_page_config(page_title="Peringkat & Perbandingan",
                   page_icon="🏆", layout="wide")

data_ok, missing = check_data_available()
if not data_ok:
    st.error("⚠️ Data belum tersedia. Jalankan preprocess.py terlebih dahulu.")
    st.stop()

# Metric metadata — shared dengan halaman profil
METRIC_COLS = ["Volatility_Pct", "VaR_Pct", "CVaR_Pct", "DD_Pct", "MDD_Pct"]
METRIC_SHORT = {
    "Volatility_Pct": "Volatilitas",
    "VaR_Pct":        "VaR",
    "CVaR_Pct":       "CVaR",
    "DD_Pct":         "Downside Dev.",
    "MDD_Pct":        "Max Drawdown",
}
METRIC_FULL = {
    "Volatility_Pct": "Volatilitas GARCH (tahunan)",
    "VaR_Pct":        "Value at Risk (GARCH, 5%)",
    "CVaR_Pct":       "Conditional Value at Risk (5%)",
    "DD_Pct":         "Downside Deviation (tahunan)",
    "MDD_Pct":        "Maximum Drawdown",
}
METRIC_HELP = {
    "Volatility_Pct": "Seberapa besar harga bergerak setiap hari (tahunan).",
    "VaR_Pct":        "Batas kerugian pada 5% hari terburuk.",
    "CVaR_Pct":       "Rata-rata kerugian pada 5% hari terburuk.",
    "DD_Pct":         "Rata-rata kerugian di hari-hari negatif (tahunan).",
    "MDD_Pct":        "Kejatuhan kumulatif terparah dari puncak.",
}

# LOAD DATA

risk = load_risk_metrics().rename(columns={"Stability_Score": "Risk_Score"})
prices_all = load_prices()
N_STOCKS = len(risk)

# HEADER & MODE SELECTOR

st.title("🏆 Peringkat & Perbandingan")
st.caption(
    "Telusuri 29 saham berdasarkan Risk Score, atau bandingkan 2–5 saham "
    "secara berdampingan untuk melihat perbedaan profil risiko."
)

mode = st.radio(
    "Mode",
    options=["🔎 Jelajahi Semua", "⚖️ Bandingkan Saham"],
    horizontal=True,
    label_visibility="collapsed",
    key="rank_mode",
)

st.divider()

# MODE 1: JELAJAHI SEMUA

if mode == "🔎 Jelajahi Semua":

    # --- Filter bar ---
    fc1, fc2, fc3 = st.columns([2, 2, 1])

    with fc1:
        all_sectors = sorted(risk["Sector"].unique())
        selected_sectors = st.multiselect(
            "Filter Sektor",
            options=all_sectors,
            default=all_sectors,
            help="Pilih satu atau lebih sektor. Kosongkan = tidak ada.",
        )

    with fc2:
        score_min, score_max = 0.04, 0.96
        score_range = st.slider(
            "Rentang Risk Score",
            min_value=round(score_min, 2),
            max_value=round(score_max, 2),
            value=(round(score_min, 2), round(score_max, 2)),
            step=0.05,
            help="0 = paling stabil, 1 = paling berisiko.",
        )

    with fc3:
        st.write("")  # spacer
        st.write("")  # spacer
        if st.button("🔄 Reset Filter", use_container_width=True):
            st.rerun()

    # Apply filters
    filtered = risk[
        (risk["Sector"].isin(selected_sectors)) &
        (risk["Risk_Score"].between(*score_range))
    ].copy().reset_index(drop=True)

    if len(filtered) == 0:
        st.warning("Tidak ada saham yang memenuhi filter. Coba perluas kriteria.")
        st.stop()

    st.caption(f"Menampilkan **{len(filtered)} dari {N_STOCKS}** saham sesuai filter.")

    # ========================================================
    # TABEL INTERAKTIF
    # ========================================================

    st.markdown("### Tabel Peringkat")

    display_df = filtered[
        ["Rank", "Stock_Name", "Sector"] + METRIC_COLS + ["Risk_Score"]
    ].copy()

    # Tambah kolom nama perusahaan
    display_df.insert(2, "Nama Perusahaan",
                     display_df["Stock_Name"].map(STOCK_FULL_NAMES))

    # Column config: progress bars untuk metrik, bar untuk score
    column_config = {
        "Rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
        "Stock_Name": st.column_config.TextColumn("Kode", width="small"),
        "Nama Perusahaan": st.column_config.TextColumn("Nama Perusahaan", width="medium"),
        "Sector": st.column_config.TextColumn("Sektor", width="medium"),
        "Volatility_Pct": st.column_config.ProgressColumn(
            "Volatilitas",
            help=METRIC_HELP["Volatility_Pct"],
            format="%.1f%%",
            min_value=0.0,
            max_value=float(risk["Volatility_Pct"].max() * 1.05),
        ),
        "VaR_Pct": st.column_config.ProgressColumn(
            "VaR",
            help=METRIC_HELP["VaR_Pct"],
            format="%.2f%%",
            min_value=0.0,
            max_value=float(risk["VaR_Pct"].max() * 1.05),
        ),
        "CVaR_Pct": st.column_config.ProgressColumn(
            "CVaR",
            help=METRIC_HELP["CVaR_Pct"],
            format="%.2f%%",
            min_value=0.0,
            max_value=float(risk["CVaR_Pct"].max() * 1.05),
        ),
        "DD_Pct": st.column_config.ProgressColumn(
            "Downside Dev.",
            help=METRIC_HELP["DD_Pct"],
            format="%.2f%%",
            min_value=0.0,
            max_value=float(risk["DD_Pct"].max() * 1.05),
        ),
        "MDD_Pct": st.column_config.ProgressColumn(
            "Max Drawdown",
            help=METRIC_HELP["MDD_Pct"],
            format="%.1f%%",
            min_value=0.0,
            max_value=float(risk["MDD_Pct"].max() * 1.05),
        ),
        "Risk_Score": st.column_config.ProgressColumn(
            "Risk Score",
            help="Skor komposit dari 5 metrik. Rendah = stabil.",
            format="%.4f",
            min_value=0.0,
            max_value=1.0,
        ),
    }

    st.dataframe(
        display_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=min(600, (len(display_df) + 1) * 35 + 3),
    )

    st.caption(
        "💡 Klik header kolom untuk mengurutkan berdasarkan kolom itu. "
        "Bar menunjukkan posisi relatif terhadap nilai maksimum di 29 saham."
    )

    # Download button
    csv_bytes = filtered[["Rank", "Stock_Name", "Sector"] + METRIC_COLS + ["Risk_Score"]]\
        .to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh CSV",
        data=csv_bytes,
        file_name=f"peringkat_saham_idx_{len(filtered)}_saham.csv",
        mime="text/csv",
    )

    st.divider()

    # ========================================================
    # HEATMAP
    # ========================================================

    st.markdown("### Heatmap Profil Risiko")
    st.caption(
        "Setiap sel menunjukkan persentil rank saham pada metrik itu "
        "(0 = paling stabil, 1 = paling berisiko). "
        "Baris diurutkan berdasarkan Risk Score (paling stabil di atas)."
    )

    # Ambil kolom _norm (sudah dihitung di preprocess)
    heatmap_df = filtered[["Stock_Name", "Rank"] +
                          [f"{c}_norm" for c in METRIC_COLS]].copy()
    heatmap_df = heatmap_df.sort_values("Rank").reset_index(drop=True)

    # Matrix untuk plotly
    z = heatmap_df[[f"{c}_norm" for c in METRIC_COLS]].values
    y_labels = [f"#{int(r['Rank'])} {r['Stock_Name']}"
                for _, r in heatmap_df.iterrows()]
    x_labels = [METRIC_SHORT[c] for c in METRIC_COLS]

    # Customdata: actual values untuk hover
    actual_values = filtered.sort_values("Rank")[METRIC_COLS].values

    hover_text = [
        [
            f"<b>{heatmap_df.iloc[i]['Stock_Name']}</b><br>"
            f"{METRIC_FULL[METRIC_COLS[j]]}<br>"
            f"Nilai: <b>{actual_values[i, j]:.2f}%</b><br>"
            f"Persentil: <b>{z[i, j]:.2f}</b>"
            for j in range(len(METRIC_COLS))
        ]
        for i in range(len(heatmap_df))
    ]

    fig_hm = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "#10B981"],    # green - low risk
            [0.33, "#86EFAC"],
            [0.5, "#FEF3C7"],    # cream - mid
            [0.67, "#FCA5A5"],
            [1.0, "#DC2626"],    # red - high risk
        ],
        zmin=0, zmax=1,
        colorbar=dict(
            title=dict(text="Persentil<br>Rank", font=dict(size=11)),
            thickness=15,
            tickvals=[0, 0.5, 1],
            ticktext=["Stabil", "Menengah", "Berisiko"],
        ),
        hoverinfo="text",
        text=hover_text,
        xgap=2, ygap=2,
    ))

    fig_hm.update_layout(
        height=max(450, 22 * len(heatmap_df) + 100),
        margin=dict(l=10, r=10, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )

    st.plotly_chart(fig_hm, use_container_width=True)

# MODE 2: BANDINGKAN SAHAM

else:  # Bandingkan Saham
    all_stocks = sorted(risk["Stock_Name"].tolist())

    # Default: BBCA (paling stabil) + MEDC (paling berisiko) untuk kontras
    default_stocks = ["BBCA", "MEDC"]

    selected_stocks = st.multiselect(
        "Pilih 2–5 saham untuk dibandingkan",
        options=all_stocks,
        default=default_stocks,
        max_selections=5,
        format_func=lambda s: f"{s} — {STOCK_FULL_NAMES.get(s, s)}",
        help="Pilih minimal 2, maksimal 5 saham. Default: saham paling stabil vs paling berisiko.",
    )

    if len(selected_stocks) < 2:
        st.info("ℹ️ Pilih minimal 2 saham untuk memulai perbandingan.")
        st.stop()

    # Ambil data saham terpilih
    cmp = risk[risk["Stock_Name"].isin(selected_stocks)].copy()
    # Sortir sesuai urutan user pilih
    cmp = cmp.set_index("Stock_Name").loc[selected_stocks].reset_index()

    # Warna per saham — pakai palette yang distinctive, bukan sector colors
    # (karena kalau 2 saham sektor sama, warnanya jadi sama)
    COMPARE_PALETTE = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED"]
    stock_colors = {s: COMPARE_PALETTE[i]
                    for i, s in enumerate(selected_stocks)}

    st.divider()

    # ========================================================
    # RINGKASAN CARDS (1 card per saham)
    # ========================================================

    st.markdown("### Ringkasan")

    cols = st.columns(len(selected_stocks))

    for col, (_, r) in zip(cols, cmp.iterrows()):
        stock = r["Stock_Name"]
        color = stock_colors[stock]
        rank = int(r["Rank"])
        sector = r["Sector"]
        score = float(r["Risk_Score"])

        # Verdict label sederhana
        if rank <= 10:
            label, label_color = "🟢 Stabil", "#10B981"
        elif rank <= 20:
            label, label_color = "🟡 Menengah", "#F59E0B"
        else:
            label, label_color = "🔴 Berisiko", "#EF4444"

        with col:
            st.markdown(
                f"""
                <div style="
                    padding: 14px;
                    border-top: 4px solid {color};
                    border-radius: 6px;
                    background: rgba(128,128,128,0.04);
                    height: 100%;
                ">
                    <div style="font-size: 1.2em; font-weight: 700;">{stock}</div>
                    <div style="font-size: 0.8em; color: #888; margin-bottom: 8px;">
                        {STOCK_FULL_NAMES.get(stock, "")}
                    </div>
                    <div style="font-size: 0.78em; color: #666; margin-bottom: 6px;">
                        {sector}
                    </div>
                    <div style="font-size: 0.85em; color: {label_color};
                                font-weight: 600; margin-bottom: 6px;">
                        {label} · Rank #{rank}
                    </div>
                    <div style="font-size: 0.78em; color: #666;">
                        Risk Score: <strong>{score:.4f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacer

    # ========================================================
    # RADAR CHART OVERLAY
    # ========================================================

    col_radar, col_table = st.columns([3, 2])

    with col_radar:
        st.markdown("#### Profil Risiko Relatif")
        st.caption(
            "Sumbu = persentil rank (0 = paling stabil, 1 = paling berisiko). "
            "Profil yang lebih kecil / mengumpul ke pusat = lebih stabil."
        )

        def closed(arr):
            return list(arr) + [arr[0]]

        axis_labels = [METRIC_SHORT[c] for c in METRIC_COLS]
        theta_closed = closed(axis_labels)

        fig_radar = go.Figure()

        # Median reference
        fig_radar.add_trace(go.Scatterpolar(
            r=closed([0.5] * 5),
            theta=theta_closed,
            line=dict(color="#cbd5e1", dash="dot", width=1.2),
            name="Median IDX",
            hoverinfo="skip",
        ))

        for _, r in cmp.iterrows():
            stock = r["Stock_Name"]
            color = stock_colors[stock]
            values = [float(r[f"{c}_norm"]) for c in METRIC_COLS]

            fig_radar.add_trace(go.Scatterpolar(
                r=closed(values),
                theta=theta_closed,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color),
                name=stock,
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.12),
                hovertemplate="<b>%{theta}</b><br>"
                              + stock + ": %{r:.2f}<extra></extra>",
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
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                        font=dict(size=10)),
            height=420,
            margin=dict(l=40, r=40, t=20, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_table:
        st.markdown("#### Perbandingan Metrik")
        st.caption("Nilai aktual tiap metrik (dalam %) — lebih rendah = lebih stabil.")

        # Transpose: metrik sebagai baris, saham sebagai kolom
        tbl = cmp.set_index("Stock_Name")[METRIC_COLS + ["Risk_Score"]].T
        tbl.index = [METRIC_SHORT.get(c, "Risk Score")
                     for c in tbl.index]
        tbl.index.name = "Metrik"

        # Highlight nilai terbaik (terkecil) per baris
        def highlight_min(row):
            min_val = row.min()
            return [
                "background-color: rgba(16, 185, 129, 0.15); font-weight: 600;"
                if v == min_val else ""
                for v in row
            ]

        styled = (
            tbl.style
            .apply(highlight_min, axis=1)
            .format("{:.4f}", subset=pd.IndexSlice["Risk Score", :])
            .format("{:.2f}%", subset=pd.IndexSlice[tbl.index[:-1], :])
        )

        st.dataframe(styled, use_container_width=True, height=min(280, 48 * len(tbl) + 40))

        st.caption(
            "🟢 Sel hijau = nilai terbaik (terendah) di baris itu."
        )

    st.divider()

    # ========================================================
    # HARGA TERINDEKS OVERLAY
    # ========================================================

    st.markdown("#### Pergerakan Harga (Terindeks)")
    st.caption(
        "Semua saham diset ke nilai 100 di awal periode, agar pergerakan relatif "
        "dapat dibandingkan terlepas dari perbedaan harga nominal."
    )

    fig_price = go.Figure()

    for stock in selected_stocks:
        sp = prices_all[prices_all["Stock_Name"] == stock]\
            .sort_values("Date")
        if len(sp) == 0:
            continue
        indexed = sp["Close"] / sp["Close"].iloc[0] * 100

        fig_price.add_trace(go.Scatter(
            x=sp["Date"], y=indexed,
            mode="lines",
            line=dict(color=stock_colors[stock], width=1.8),
            name=stock,
            hovertemplate="<b>" + stock + "</b><br>"
                          "%{x|%d %b %Y}<br>"
                          "Indeks: %{y:.1f}<extra></extra>",
        ))

    # Baseline 100
    fig_price.add_hline(
        y=100, line_dash="dot", line_color="#888",
        annotation_text="Baseline (100)",
        annotation_position="right",
        annotation_font=dict(size=10, color="#888"),
    )

    fig_price.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        yaxis=dict(title="Harga Terindeks",
                   showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        xaxis=dict(
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
            rangeslider=dict(visible=False),
            type="date",
            showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        ),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Cumulative return summary
    st.markdown("##### Total Return Sepanjang Periode")

    return_cols = st.columns(len(selected_stocks))
    for col, stock in zip(return_cols, selected_stocks):
        sp = prices_all[prices_all["Stock_Name"] == stock].sort_values("Date")
        if len(sp) == 0:
            continue
        total_return = (sp["Close"].iloc[-1] / sp["Close"].iloc[0] - 1) * 100
        arrow = "▲" if total_return >= 0 else "▼"
        ret_color = "#10B981" if total_return >= 0 else "#EF4444"

        with col:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 10px;
                            background: rgba(128,128,128,0.03);
                            border-left: 3px solid {stock_colors[stock]};
                            border-radius: 4px;">
                    <div style="font-size: 0.85em; color: #888;">{stock}</div>
                    <div style="font-size: 1.4em; font-weight: 700; color: {ret_color};">
                        {arrow} {total_return:+.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# FOOTER

st.divider()

# with st.expander("📚 Cara membaca halaman ini"):
#     st.markdown(
#         """
#         **Mode "Jelajahi Semua"** menampilkan seluruh 29 saham dalam bentuk tabel
#         dan heatmap. Filter sektor dan rentang Risk Score dapat menyempitkan pandangan.

#         **Mode "Bandingkan Saham"** cocok untuk menelaah 2–5 saham yang sedang kamu
#         pertimbangkan. Radar chart menunjukkan profil risiko lintas-dimensi, tabel
#         metrik menyoroti nilai terbaik (hijau), dan grafik harga terindeks menunjukkan
#         pergerakan relatif sejak awal periode.

#         **Kolom bar (Tabel)** — panjang bar menunjukkan posisi relatif nilai
#         terhadap maksimum di 29 saham. Bar panjang di metrik risiko = nilai tinggi = berisiko.

#         **Heatmap (Jelajahi Semua)** menggunakan *persentil rank*, bukan nilai aktual.
#         Jadi warna merah paling pekat berarti "paling berisiko di metrik itu di antara
#         29 saham" — bukan "lebih dari 90%".

#         **Semua analisis bersifat historis** (Okt 2012 – Apr 2024) dan tidak merupakan
#         saran investasi.
#         """
#     )