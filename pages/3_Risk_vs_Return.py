"""
Risk-Return Trade-off — apakah return sebanding dengan risiko?
- Saham mana yang paling menarik (rendah risiko, return baik)?
- Saham mana yang harus dihindari (tinggi risiko, return rendah)?
- Bagaimana posisi saham favorit saya dibanding yang lain?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import re

from utils.data_loader import (
    SECTOR_COLORS, STOCK_FULL_NAMES,
    check_data_available, hex_to_rgba,
    load_risk_metrics, load_sortino,
)

# CONFIG & DATA

st.set_page_config(page_title="Risk-Return Trade-off",
                   page_icon="⚖️", layout="wide")

data_ok, _ = check_data_available()
if not data_ok:
    st.error("⚠️ Data belum tersedia. Jalankan preprocess.py terlebih dahulu.")
    st.stop()

risk = load_risk_metrics().rename(columns={"Stability_Score": "Risk_Score"})
sortino = load_sortino()

# Merge kedua sumber
data = risk[["Stock_Name", "Sector", "Rank", "Risk_Score"]].merge(
    sortino[["Stock_Name", "Return_ann", "DD_ann", "Sortino", "Sortino_rank"]],
    on="Stock_Name",
    how="left",
)

N_STOCKS = len(data)

# Kuadran colors (semantic)
QUAD_COLORS = {
    "ideal":     "#10B981", # Low Risk, High Return
    "accept":    "#F59E0B", # High Risk, High Return (compensated)
    "defensive": "#3B82F6", # Low Risk, Low Return
    "avoid":     "#EF4444", # High Risk, Low Return
}

# HEADER

st.title("⚖️ Risk-Return Trade-off")
st.caption(
    "Apakah saham berisiko tinggi memberikan return yang sepadan? "
    "Halaman ini mengevaluasi hubungan antara risiko historis dan "
    "kompensasi return yang diterima investor."
)

# CHART TYPE TOGGLE

chart_type = st.radio(
    "Perspektif Analisis",
    options=[
        "Risk Score × Sortino Ratio (multidimensi)",
        "Downside Deviation × Return Tahunan (nilai absolut)",
    ],
    horizontal=True,
    help=(
        "Perspektif 1: posisi saham berdasarkan skor risiko komposit vs "
        "kompensasi return-per-risiko. "
        "Perspektif 2: angka mentah — berapa return tahunannya, berapa risk downside-nya."
    ),
)

is_multidim = chart_type.startswith("Risk Score")

# HIGHLIGHT SELECTOR

all_stocks = sorted(data["Stock_Name"].tolist())

col_hl1, col_hl2 = st.columns([3, 1])
with col_hl1:
    highlighted = st.multiselect(
        "🎯 Tandai saham tertentu (opsional)",
        options=all_stocks,
        default=[],
        format_func=lambda s: f"{s} — {STOCK_FULL_NAMES.get(s, s)}",
        help="Saham yang dipilih akan diberi lingkaran penekanan di chart.",
    )

with col_hl2:
    color_by = st.selectbox(
        "Warnai berdasarkan",
        options=["Sektor", "Kuadran"],
        help="Warna sektor = eksposur industri. Warna kuadran = klasifikasi risk-return.",
    )

st.divider()

# KOMPUTASI MEDIAN UNTUK KUADRAN

if is_multidim:
    x_col, y_col = "Risk_Score", "Sortino"
    x_label = "Risk Score (rendah = stabil)"
    y_label = "Sortino Ratio (tinggi = return baik per unit risiko)"
    x_fmt = ".4f"
    y_fmt = ".3f"
else:
    x_col, y_col = "DD_ann", "Return_ann"
    x_label = "Downside Deviation Tahunan"
    y_label = "Return Tahunan"
    x_fmt = ".2%"
    y_fmt = ".2%"

med_x = data[x_col].median()
med_y = data[y_col].median()

# Tentukan kuadran tiap saham
def assign_quadrant(x, y):
    """Kuadran berdasarkan median.
    Perhatikan: sumbu X adalah 'risiko' → kiri = low risk.
                Sumbu Y adalah 'return' → atas = high return.
    """
    low_risk = x <= med_x
    high_return = y >= med_y
    if low_risk and high_return:
        return "ideal"    # kiri-atas
    elif not low_risk and high_return:
        return "accept"   # kanan-atas
    elif low_risk and not high_return:
        return "defensive"  # kiri-bawah
    else:
        return "avoid"    # kanan-bawah

data["Quadrant"] = data.apply(
    lambda r: assign_quadrant(r[x_col], r[y_col]), axis=1
)

QUAD_LABELS = {
    "ideal":     "Low Risk, High Return",
    "accept":    "High Risk, High Return",
    "defensive": "Low Risk, Low Return",
    "avoid":     "High Risk, Low Return",
}

# SCATTER PLOT UTAMA

# Padding untuk axes
x_min_plot = data[x_col].min()
x_max_plot = data[x_col].max()
y_min_plot = data[y_col].min()
y_max_plot = data[y_col].max()
x_pad = (x_max_plot - x_min_plot) * 0.08
y_pad = (y_max_plot - y_min_plot) * 0.08

xmin, xmax = x_min_plot - x_pad, x_max_plot + x_pad
ymin, ymax = y_min_plot - y_pad, y_max_plot + y_pad

fig = go.Figure()

# --- Quadrant background fills ---
quadrant_shapes = [
    # ideal (kiri-atas): hijau
    dict(x0=xmin, x1=med_x, y0=med_y, y1=ymax, color=QUAD_COLORS["ideal"]),
    # accept (kanan-atas): orange
    dict(x0=med_x, x1=xmax, y0=med_y, y1=ymax, color=QUAD_COLORS["accept"]),
    # defensive (kiri-bawah): biru
    dict(x0=xmin, x1=med_x, y0=ymin, y1=med_y, color=QUAD_COLORS["defensive"]),
    # avoid (kanan-bawah): merah
    dict(x0=med_x, x1=xmax, y0=ymin, y1=med_y, color=QUAD_COLORS["avoid"]),
]

for q in quadrant_shapes:
    fig.add_shape(
        type="rect",
        x0=q["x0"], x1=q["x1"], y0=q["y0"], y1=q["y1"],
        fillcolor=hex_to_rgba(q["color"], 0.06),
        line=dict(width=0),
        layer="below",
    )

# Median lines
fig.add_shape(
    type="line", x0=med_x, x1=med_x, y0=ymin, y1=ymax,
    line=dict(color="#94A3B8", width=1, dash="dash"),
    layer="below",
)
fig.add_shape(
    type="line", x0=xmin, x1=xmax, y0=med_y, y1=med_y,
    line=dict(color="#94A3B8", width=1, dash="dash"),
    layer="below",
)

# Quadrant labels (corners)
label_positions = [
    dict(x=xmin + (med_x - xmin) / 2, y=ymax - y_pad * 0.4,
         text="🟢 <b>Low Risk<br>High Return</b>",
         color=QUAD_COLORS["ideal"]),
    dict(x=med_x + (xmax - med_x) / 2, y=ymax - y_pad * 0.4,
         text="🟠 <b>High Risk<br>High Return</b>",
         color=QUAD_COLORS["accept"]),
    dict(x=xmin + (med_x - xmin) / 2, y=ymin + y_pad * 0.4,
         text="🔵 <b>Low Risk<br>Low Return</b>",
         color=QUAD_COLORS["defensive"]),
    dict(x=med_x + (xmax - med_x) / 2, y=ymin + y_pad * 0.4,
         text="🔴 <b>High Risk<br>Low Return</b>",
         color=QUAD_COLORS["avoid"]),
]
for lp in label_positions:
    fig.add_annotation(
        x=lp["x"], y=lp["y"], text=lp["text"],
        showarrow=False,
        font=dict(size=10, color=lp["color"]),
        opacity=0.75,
        align="center",
    )

# --- Scatter points ---
# Warna per point tergantung mode
if color_by == "Sektor":
    point_colors = [SECTOR_COLORS.get(s, "#666") for s in data["Sector"]]
    color_legend_map = SECTOR_COLORS
else:
    point_colors = [QUAD_COLORS[q] for q in data["Quadrant"]]
    color_legend_map = {QUAD_LABELS[k]: v for k, v in QUAD_COLORS.items()}

# Hover text
if is_multidim:
    hover_text = [
        f"<b>{r['Stock_Name']}</b> — {STOCK_FULL_NAMES.get(r['Stock_Name'], '')}<br>"
        f"Sektor: {r['Sector']}<br>"
        f"Risk Score: {r['Risk_Score']:.4f} (rank #{int(r['Rank'])})<br>"
        f"Sortino: {r['Sortino']:.3f} (rank #{int(r['Sortino_rank'])})<br>"
        f"Return tahunan: {r['Return_ann']*100:+.2f}%"
        for _, r in data.iterrows()
    ]
else:
    hover_text = [
        f"<b>{r['Stock_Name']}</b> — {STOCK_FULL_NAMES.get(r['Stock_Name'], '')}<br>"
        f"Sektor: {r['Sector']}<br>"
        f"Return tahunan: {r['Return_ann']*100:+.2f}%<br>"
        f"Downside deviation: {r['DD_ann']*100:.2f}%<br>"
        f"Sortino: {r['Sortino']:.3f}"
        for _, r in data.iterrows()
    ]

# Determine which points get labels: highlighted + extremes
labels_to_show = set(highlighted)

# Auto-label extremes (paling ekstrem di tiap arah)
if not highlighted:  # hanya kalau user tidak sudah pilih
    labels_to_show.add(data.loc[data[x_col].idxmin(), "Stock_Name"])  # paling stabil
    labels_to_show.add(data.loc[data[x_col].idxmax(), "Stock_Name"])  # paling berisiko
    labels_to_show.add(data.loc[data[y_col].idxmax(), "Stock_Name"])  # return tertinggi
    labels_to_show.add(data.loc[data[y_col].idxmin(), "Stock_Name"])  # return terendah

text_labels = [
    r["Stock_Name"] if r["Stock_Name"] in labels_to_show else ""
    for _, r in data.iterrows()
]

# Marker sizes: highlighted = besar, others = normal
marker_sizes = [
    16 if s in highlighted else 10
    for s in data["Stock_Name"]
]
marker_borders = [
    3 if s in highlighted else 1
    for s in data["Stock_Name"]
]

fig.add_trace(go.Scatter(
    x=data[x_col], y=data[y_col],
    mode="markers+text",
    marker=dict(
        color=point_colors,
        size=marker_sizes,
        line=dict(color="white", width=marker_borders),
    ),
    text=text_labels,
    textposition="top center",
    textfont=dict(size=9),
    hovertext=hover_text,
    hoverinfo="text",
    showlegend=False,
))

# --- Axis styling ---
if is_multidim:
    x_tickformat = ".2f"
    y_tickformat = ".2f"
else:
    x_tickformat = ".1%"
    y_tickformat = ".1%"

fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=40, b=60),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title=dict(text=x_label, font=dict(size=12)),
        range=[xmin, xmax],
        tickformat=x_tickformat,
        showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        zeroline=False,
    ),
    yaxis=dict(
        title=dict(text=y_label, font=dict(size=12)),
        range=[ymin, ymax],
        tickformat=y_tickformat,
        showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        zeroline=(not is_multidim),  # untuk DD-Return, zero return adalah referensi penting
        zerolinecolor="rgba(128,128,128,0.3)",
    ),
    hovermode="closest",
)

if not is_multidim:
    dd_range = np.linspace(max(xmin, 0.001), xmax, 50)
    for iso_val, dash_style in [(1.0, "dot"), (0.5, "dot")]:
        fig.add_trace(go.Scatter(
            x=dd_range, y=dd_range * iso_val,
            mode="lines",
            line=dict(color="rgba(148,163,184,0.5)",
                      width=1, dash=dash_style),
            name=f"Sortino = {iso_val}",
            showlegend=True,
            hoverinfo="skip",
        ))

    fig.update_layout(
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1,
            font=dict(size=10),
        )
    )

st.plotly_chart(fig, use_container_width=True)

# LEGEND WARNA

st.markdown("**Keterangan warna:**")
legend_cols = st.columns(len(color_legend_map))
for col, (label, color) in zip(legend_cols, color_legend_map.items()):
    with col:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px;
                        font-size: 0.85em;">
                <span style="display: inline-block; width: 14px; height: 14px;
                             background: {color}; border-radius: 3px;"></span>
                <span>{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.divider()

# RANKING TABLE PER KUADRAN

st.markdown("### Saham Menonjol per Kuadran")

quad_cols = st.columns(4)

quad_display_order = ["ideal", "accept", "defensive", "avoid"]
quad_icons = {
    "ideal": "🟢", "accept": "🟠",
    "defensive": "🔵", "avoid": "🔴",
}
quad_subtitles = {
    "ideal":     "Stabil & kompensasi return baik",
    "accept":    "Berisiko, tapi return sepadan",
    "defensive": "Stabil, tapi return terbatas",
    "avoid":     "Berisiko tanpa return memadai",
}

for col, quad in zip(quad_cols, quad_display_order):
    quad_data = data[data["Quadrant"] == quad].copy()

    # Sort: ideal & accept → return tertinggi dulu; defensive & avoid → risiko terendah dulu
    if quad in ("ideal", "accept"):
        quad_data = quad_data.sort_values(y_col, ascending=False)
    else:
        quad_data = quad_data.sort_values(x_col, ascending=True)

    with col:
        st.markdown(
            f"""
            <div style="padding: 10px; border-top: 3px solid {QUAD_COLORS[quad]};
                        margin-bottom: 8px;">
                <div style="font-weight: 600;">
                    {quad_icons[quad]} {QUAD_LABELS[quad]}
                </div>
                <div style="font-size: 0.78em; color: #888;">
                    {quad_subtitles[quad]} · {len(quad_data)} saham
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(quad_data) == 0:
            st.caption("_Tidak ada saham di kuadran ini._")
            continue

        # Tampilkan top 5 per kuadran
        for _, r in quad_data.head(5).iterrows():
            stock = r["Stock_Name"]
            sector = r["Sector"]
            sector_color = SECTOR_COLORS.get(sector, "#666")

            if is_multidim:
                primary = f"Sortino {r['Sortino']:.2f}"
                secondary = f"Rank #{int(r['Rank'])}"
            else:
                primary = f"Return {r['Return_ann']*100:+.1f}%"
                secondary = f"DD {r['DD_ann']*100:.1f}%"

            st.markdown(
                f"""
                <div style="padding: 6px 10px; margin-bottom: 4px;
                            border-left: 3px solid {sector_color};
                            background: rgba(128,128,128,0.04);
                            font-size: 0.85em;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>{stock}</strong>
                        <span style="color: #666; font-size: 0.9em;">{primary}</span>
                    </div>
                    <div style="color: #888; font-size: 0.78em;">
                        {sector} · {secondary}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if len(quad_data) > 5:
            st.caption(f"_+{len(quad_data) - 5} saham lainnya_")

st.divider()

# INSIGHTS OTOMATIS

st.markdown("### 💡 Observasi dari Data")

insights = []

# --- Insight 1: Apakah ada saham "sempurna"? ---
ideal_stocks = data[data["Quadrant"] == "ideal"]
if len(ideal_stocks) > 0:
    top_ideal = ideal_stocks.sort_values(y_col, ascending=False).iloc[0]
    insights.append({
        "icon": "🎯",
        "title": f"{len(ideal_stocks)} saham menempati kuadran ideal",
        "body": (
            f"Saham dengan risiko di bawah median dan return di atas median: "
            f"{', '.join(ideal_stocks['Stock_Name'].tolist())}**. "
            f"Yang paling menonjol: {top_ideal['Stock_Name']} "
            f"({STOCK_FULL_NAMES.get(top_ideal['Stock_Name'], '')}) "
            f", Sortino {top_ideal['Sortino']:.2f}, "
            f"return tahunan {top_ideal['Return_ann']*100:+.1f}%."
        ),
    })

# --- Insight 2: Saham yang "salah harga" — risiko tinggi tanpa return ---
avoid_stocks = data[data["Quadrant"] == "avoid"]
if len(avoid_stocks) > 0:
    worst_avoid = avoid_stocks.sort_values("Sortino", ascending=True).iloc[0]
    n_negative = (avoid_stocks["Return_ann"] < 0).sum()
    insights.append({
        "icon": "⚠️",
        "title": f"{len(avoid_stocks)} saham: risiko tinggi tanpa kompensasi return",
        "body": (
            f"Saham di kuadran High Risk–Low Return, investor menanggung risiko "
            f"tanpa kompensasi yang memadai. "
            f"**{n_negative}** dari grup ini bahkan memiliki return tahunan negatif. "
            f"Yang paling bermasalah: **{worst_avoid['Stock_Name']}** "
            f"(Sortino {worst_avoid['Sortino']:.2f}, "
            f"return {worst_avoid['Return_ann']*100:+.1f}%)."
        ),
    })

# --- Insight 3: Linearitas risk-return ---
# Korelasi Pearson antara Risk_Score dan Sortino / Return_ann
if is_multidim:
    corr = data[["Risk_Score", "Sortino"]].corr().iloc[0, 1]
    corr_label = "Risk Score dan Sortino Ratio"
else:
    corr = data[["DD_ann", "Return_ann"]].corr().iloc[0, 1]
    corr_label = "Downside Deviation dan Return Tahunan"

if abs(corr) < 0.3:
    linearity_msg = (
        f"Korelasi antara {corr_label} hanya **{corr:+.2f}**, "
        f"menunjukkan bahwa **asumsi 'high risk, high return' tidak konsisten** "
        f"di observasi saham ini. Investor tidak bisa mengandalkan heuristik itu "
        f"untuk memilih saham."
    )
    corr_icon = "📊"
elif corr > 0.3:
    linearity_msg = (
        f"Korelasi antara {corr_label} adalah **{corr:+.2f}** — "
        f"risiko lebih tinggi memang cenderung diiringi return lebih tinggi, "
        f"tapi dispersinya besar (banyak pengecualian)."
    )
    corr_icon = "📈"
else:
    linearity_msg = (
        f"Korelasi negatif **{corr:.2f}** — risiko lebih tinggi justru cenderung "
        f"diiringi return lebih rendah. Ini kontras dengan intuisi pasar yang umum."
    )
    corr_icon = "⚡"

insights.append({
    "icon": corr_icon,
    "title": "Tidak ada hubungan linear yang kuat",
    "body": linearity_msg,
})

# --- Insight 4: Saham dengan Sortino > 1 ---
high_sortino = data[data["Sortino"] > 1.0]
if len(high_sortino) > 0:
    stocks_list = ", ".join(f"**{s}**" for s in high_sortino["Stock_Name"])
    insights.append({
        "icon": "🏆",
        "title": f"{len(high_sortino)} saham dengan Sortino > 1",
        "body": (
            f"Saham yang return tahunannya melebihi downside deviation-nya: "
            f"{stocks_list}. Ini ambang yang langka, sehingga "
            f"dianggap profil return-per-risiko yang sangat baik."
        ),
    })
else:
    insights.append({
        "icon": "📉",
        "title": "Tidak ada saham dengan Sortino > 1",
        "body": (
            f"Tidak ada saham dalam observasi ini yang return tahunannya melebihi "
            f"downside deviation-nya — ambang yang sering dianggap sebagai profil "
            f"risk-return yang 'sangat baik'. Ini refleksi kondisi pasar IDX selama "
            f"periode pengamatan."
        ),
    })

# --- Insight 5: Saham return negatif ---
neg_return = data[data["Return_ann"] < 0]
if len(neg_return) > 0:
    stocks_list = ", ".join(f"**{s}**" for s in neg_return["Stock_Name"])
    insights.append({
        "icon": "🔻",
        "title": f"{len(neg_return)} saham dengan return tahunan negatif",
        "body": (
            f"Investor yang hold saham-saham berikut sepanjang periode "
            f"(Okt 2012 – Apr 2024) rata-rata mengalami kerugian: {stocks_list}."
        ),
    })

# Render insights as cards
def md_bold_to_html(text: str) -> str:
    """Convert **bold** markdown syntax to <strong> HTML tag."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

n = len(insights)
ins_cols = st.columns(n)
for col, ins in zip(ins_cols, insights):
    with col:
        st.markdown(
            f"""
            <div style="padding: 16px;
                        border: 1px solid rgba(128,128,128,0.2);
                        border-radius: 8px;
                        background: rgba(128,128,128,0.03);
                        height: 100%;">
                <div style="font-size: 1.4em; margin-bottom: 6px;">{ins['icon']}</div>
                <div style="font-weight: 600; margin-bottom: 8px;
                            font-size: 0.95em; line-height: 1.3;">
                    {md_bold_to_html(ins['title'])}
                </div>
                <div style="font-size: 0.85em; color: rgba(128,128,128,0.95);
                            line-height: 1.5;">
                    {md_bold_to_html(ins['body'])}
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
#         **Dua perspektif chart:**

#         - **Risk Score × Sortino Ratio** — posisi multidimensi:
#           Risk Score adalah komposit 5 metrik risiko (hasil Halaman Peringkat).
#           Sortino Ratio mengukur return per unit *downside risk*.
#           Posisi ideal: kiri-atas (low risk, high Sortino).

#         - **Downside Deviation × Return** — nilai absolut:
#           Sumbu X = risk downside tahunan. Sumbu Y = return tahunan.
#           Garis diagonal = iso-Sortino (semua titik di atas Sortino=1 berada di atas
#           garis y = x, misalnya).

#         **Median lines** membagi chart menjadi 4 kuadran:

#         - 🟢 **Low Risk, High Return** — profil ideal
#         - 🟠 **High Risk, High Return** — risiko tinggi, tapi return sepadan
#         - 🔵 **Low Risk, Low Return** — defensif, kurang agresif
#         - 🔴 **High Risk, Low Return** — **hindari** kecuali ada alasan spesifik

#         **Sortino Ratio** adalah varian dari Sharpe Ratio yang hanya memperhitungkan
#         volatilitas sisi bawah (kerugian), bukan semua volatilitas. Formulanya:
#         `(Return_tahunan − Risk_free) / Downside_Deviation_tahunan`.
#         Penelitian ini menggunakan `Risk_free = 0` agar fokus pada perbandingan relatif.

#         **Asumsi "high risk, high return"** adalah heuristik populer, tapi tidak selalu
#         valid secara empiris. Chart ini membantu memvisualisasikan bagaimana heuristik
#         itu berlaku (atau tidak) di universe saham IDX yang dianalisis.
#         """
#     )