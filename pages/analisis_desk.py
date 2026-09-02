import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy import stats
from statsmodels.tsa.stattools import acf

from utils import (
    clean_commodity_series,
    format_id,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_dataset,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Analisis Deskriptif",
    page_icon="📊",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / analisis deskriptif",
    title="Analisis Deskriptif",
    caption=(
        "Analisis karakteristik data harga, pola waktu, "
        "dependensi lag, nonlinearitas, dan kesiapan fitur "
        "sebelum pemodelan SVR."
    ),
)


# ============================================================
# CONSTANT
# ============================================================

RANDOM_STATE = 42
MAX_LAG = 30
ROLLING_WINDOWS = [7, 14, 30]


# ============================================================
# VALIDASI & PEMBERSIHAN DATA
# ============================================================

df, date_column, commodity_column = require_dataset()

working_df = df[[date_column, commodity_column]].copy()

working_df[date_column] = pd.to_datetime(
    working_df[date_column],
    dayfirst=True,
    errors="coerce",
)

working_df = (
    working_df
    .dropna(subset=[date_column])
    .sort_values(date_column)
)

st.session_state.df = working_df

df[commodity_column] = clean_commodity_series(
    df,
    commodity_column,
)

df = df.dropna(
    subset=[commodity_column]
).copy()

df = df.sort_values(
    date_column
).reset_index(drop=True)

harga = df[commodity_column].astype(float)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_rupiah(value):
    """Format angka menjadi Rupiah."""
    return f"Rp {format_id(value, 0)}"


def make_lag_features(
    series: pd.Series,
    max_lag: int = 30,
):
    """
    Membuat fitur lag untuk kebutuhan analisis SVR.
    """
    features = pd.DataFrame(index=series.index)

    for lag in range(1, max_lag + 1):
        features[f"lag_{lag}"] = series.shift(lag)

    return features


def make_svr_features(
    series: pd.Series,
    dates: pd.Series,
    max_lag: int = 30,
):
    """
    Membuat representasi fitur yang relevan dengan
    model SVR time series.
    """

    features = pd.DataFrame(index=series.index)

    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    for lag in range(1, max_lag + 1):
        features[f"lag_{lag}"] = series.shift(lag)

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    for window in ROLLING_WINDOWS:

        features[f"rolling_mean_{window}"] = (
            series
            .shift(1)
            .rolling(window)
            .mean()
        )

        features[f"rolling_std_{window}"] = (
            series
            .shift(1)
            .rolling(window)
            .std()
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for span in ROLLING_WINDOWS:

        features[f"ema_{span}"] = (
            series
            .shift(1)
            .ewm(
                span=span,
                adjust=False,
            )
            .mean()
        )

    # --------------------------------------------------------
    # DIFFERENCE
    # --------------------------------------------------------

    features["diff_1"] = (
        series
        .shift(1)
        .diff(1)
    )

    features["diff_7"] = (
        series
        .shift(1)
        .diff(7)
    )

    # --------------------------------------------------------
    # CALENDAR FEATURES
    # --------------------------------------------------------

    dates = pd.to_datetime(dates)

    features["day_of_week"] = (
        dates.dt.dayofweek
    )

    features["day_of_month"] = (
        dates.dt.day
    )

    features["month"] = (
        dates.dt.month
    )

    features["quarter"] = (
        dates.dt.quarter
    )

    features["is_weekend"] = (
        dates.dt.dayofweek >= 5
    ).astype(int)

    return features


def detect_outliers_iqr(series):
    """
    Deteksi outlier menggunakan metode IQR.
    """

    x = series.dropna()

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (
        (x < lower)
        | (x > upper)
    )

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "count": int(mask.sum()),
        "percentage": (
            mask.mean() * 100
        ),
    }


def correlation_table(
    series: pd.Series,
    max_lag: int = 30,
):
    """
    Menghitung korelasi harga saat ini
    dengan lag sebelumnya.
    """

    rows = []

    for lag in range(1, max_lag + 1):

        lagged = series.shift(lag)

        valid = pd.concat(
            [series, lagged],
            axis=1,
        ).dropna()

        if len(valid) < 3:
            continue

        pearson = (
            valid.iloc[:, 0]
            .corr(
                valid.iloc[:, 1],
                method="pearson",
            )
        )

        spearman = (
            valid.iloc[:, 0]
            .corr(
                valid.iloc[:, 1],
                method="spearman",
            )
        )

        rows.append(
            {
                "Lag": lag,
                "Pearson": pearson,
                "Spearman": spearman,
                "Abs Pearson": abs(pearson),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BASIC STATISTICS
# ============================================================

mean_val = harga.mean()
median_val = harga.median()
std_val = harga.std()
min_val = harga.min()
max_val = harga.max()

skew_val = stats.skew(harga)
kurt_val = stats.kurtosis(
    harga,
    fisher=False,
)

cv = (
    std_val / mean_val
    if mean_val != 0
    else np.nan
)


# ============================================================
# MONTHLY CHANGE
# ============================================================

plot_df = (
    df
    .sort_values(date_column)
    .copy()
)

monthly = (
    plot_df
    .set_index(date_column)[commodity_column]
    .resample("MS")
    .mean()
)

monthly_change = (
    monthly
    .pct_change()
    .mean()
    * 100
)

if pd.isna(monthly_change):

    mean_delta = "-"

elif monthly_change > 0:

    mean_delta = (
        f"▲ {monthly_change:.2f}% / bulan"
    )

elif monthly_change < 0:

    mean_delta = (
        f"▼ {abs(monthly_change):.2f}% / bulan"
    )

else:

    mean_delta = "Tidak berubah"


# ============================================================
# VOLATILITY INTERPRETATION
# ============================================================

if cv < 0.10:

    volatility_delta = "▼ Rendah"
    volatility_color = "inverse"

elif cv < 0.20:

    volatility_delta = "■ Sedang"
    volatility_color = "off"

else:

    volatility_delta = "▲ Tinggi"
    volatility_color = "normal"


# ============================================================
# SKEWNESS INTERPRETATION
# ============================================================

if abs(skew_val) < 0.50:

    skew_delta = "● Simetris"

elif skew_val > 0:

    skew_delta = "▶ Miring ke kanan"

else:

    skew_delta = "◀ Miring ke kiri"


# ============================================================
# KURTOSIS INTERPRETATION
# ============================================================

if kurt_val < 3:

    kurt_delta = "▼ Platykurtic"

elif kurt_val <= 3.5:

    kurt_delta = "● Mesokurtic"

else:

    kurt_delta = "▲ Leptokurtic"


# ============================================================
# SECTION 1
# RINGKASAN DATA
# ============================================================

st.markdown("### Ringkasan Data")

st.caption(
    "Karakteristik dasar harga yang digunakan "
    "untuk memahami skala, volatilitas, dan "
    "distribusi data sebelum pemodelan SVR."
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Rata-rata Harga",
        format_rupiah(mean_val),
        mean_delta,
        delta_color="off",
    )

with m2:

    st.metric(
        "Volatilitas",
        format_id(std_val, 1),
        volatility_delta,
        delta_color=volatility_color,
    )

with m3:

    st.metric(
        "Skewness",
        f"{skew_val:.2f}",
        skew_delta,
    )

with m4:

    st.metric(
        "Kurtosis",
        f"{kurt_val:.2f}",
        kurt_delta,
    )


# ============================================================
# SECTION 2
# TREND HARGA
# ============================================================

st.markdown("### 📈 Tren Harga Historis")

st.caption(
    "Pergerakan harga aktual dan moving average "
    "untuk membantu mengidentifikasi pola tren "
    "yang dapat dipelajari oleh SVR."
)

with st.container(border=True):

    trend_df = plot_df.copy()

    trend_df["MA_7"] = (
        trend_df[commodity_column]
        .rolling(7)
        .mean()
    )

    trend_df["MA_30"] = (
        trend_df[commodity_column]
        .rolling(30)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_df[date_column],
            y=trend_df[commodity_column],
            mode="lines",
            name="Harga Aktual",
            line=dict(
                color="#FF4B4B",
                width=1.5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trend_df[date_column],
            y=trend_df["MA_7"],
            mode="lines",
            name="Moving Average 7",
            line=dict(
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trend_df[date_column],
            y=trend_df["MA_30"],
            mode="lines",
            name="Moving Average 30",
            line=dict(
                width=2,
            ),
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        yaxis_title="Rp/kg",
        xaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        showgrid=True,
        zeroline=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# SECTION 3
# DISTRIBUSI HARGA
# ============================================================

st.markdown("### 📊 Distribusi Harga")

st.caption(
    "Distribusi harga digunakan untuk melihat "
    "rentang data, penyebaran, dan indikasi "
    "observasi ekstrem."
)

col1, col2 = st.columns(
    2,
    gap="medium",
)


# ------------------------------------------------------------
# STATISTICS TABLE
# ------------------------------------------------------------

with col1:

    with st.container(border=True):

        st.markdown(
            "#### Statistik Deskriptif"
        )

        stat_table = pd.DataFrame(
            {
                "Statistik": [
                    "Mean",
                    "Median",
                    "Std. Deviasi",
                    "Minimum",
                    "Maksimum",
                    "Skewness",
                    "Kurtosis",
                    "Coefficient of Variation",
                ],
                "Nilai": [
                    f"{mean_val:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{median_val:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{std_val:,.1f}".replace(
                        ",",
                        ".",
                    ),
                    f"{min_val:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{max_val:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{skew_val:.2f}",
                    f"{kurt_val:.2f}",
                    f"{cv:.2%}",
                ],
            }
        )

        st.dataframe(
            stat_table,
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------
# HISTOGRAM
# ------------------------------------------------------------

with col2:

    with st.container(border=True):

        st.markdown(
            "#### Histogram Harga"
        )

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=df[commodity_column],
                nbinsx=12,
            )
        )

        fig_hist.update_layout(
            height=300,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            xaxis_title="Rp/kg",
            yaxis_title="Frekuensi",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )

        fig_hist.update_xaxes(
            showgrid=False,
        )

        fig_hist.update_yaxes(
            showgrid=True,
            zeroline=False,
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )


# ============================================================
# SECTION 4
# OUTLIER ANALYSIS
# ============================================================

st.markdown("### ⚠️ Analisis Outlier")

st.caption(
    "Identifikasi observasi ekstrem menggunakan "
    "metode Interquartile Range (IQR)."
)

outlier = detect_outliers_iqr(harga)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Q1",
        format_rupiah(outlier["q1"]),
    )

with c2:

    st.metric(
        "Q3",
        format_rupiah(outlier["q3"]),
    )

with c3:

    st.metric(
        "Jumlah Outlier",
        f"{outlier['count']:,}".replace(
            ",",
            ".",
        ),
    )

with c4:

    st.metric(
        "Persentase Outlier",
        f"{outlier['percentage']:.2f}%",
    )

with st.container(border=True):

    st.caption(
        "Outlier tidak otomatis dihapus karena "
        "perubahan harga ekstrem dapat merepresentasikan "
        "kondisi pasar yang sebenarnya. Penanganannya "
        "perlu dipertimbangkan pada tahap preprocessing."
    )


# ============================================================
# SECTION 5
# ANALISIS LAG & AUTOKORELASI
# ============================================================

st.markdown("### 🔗 Analisis Lag & Autokorelasi")

st.caption(
    "Analisis hubungan harga saat ini dengan "
    "harga pada periode sebelumnya untuk menentukan "
    "informasi historis yang potensial digunakan "
    "sebagai fitur SVR."
)

lag_corr = correlation_table(
    harga,
    max_lag=MAX_LAG,
)

if not lag_corr.empty:

    best_lag_row = (
        lag_corr
        .sort_values(
            "Abs Pearson",
            ascending=False,
        )
        .iloc[0]
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Lag Terkuat",
            f"Lag-{int(best_lag_row['Lag'])}",
        )

    with c2:

        st.metric(
            "Pearson",
            f"{best_lag_row['Pearson']:.3f}",
        )

    with c3:

        st.metric(
            "Spearman",
            f"{best_lag_row['Spearman']:.3f}",
        )


# ------------------------------------------------------------
# ACF
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):

        st.markdown(
            "#### Autocorrelation Function"
        )

        acf_values = acf(
            harga.dropna(),
            nlags=MAX_LAG,
            fft=True,
        )

        fig_acf = go.Figure()

        fig_acf.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name="ACF",
            )
        )

        fig_acf.add_hline(
            y=0,
            line_width=1,
        )

        fig_acf.update_layout(
            height=300,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            xaxis_title="Lag",
            yaxis_title="Autocorrelation",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig_acf.update_yaxes(
            range=[-1, 1],
        )

        st.plotly_chart(
            fig_acf,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )


# ------------------------------------------------------------
# LAG CORRELATION
# ------------------------------------------------------------

with col2:

    with st.container(border=True):

        st.markdown(
            "#### Korelasi Harga dengan Lag"
        )

        fig_corr = go.Figure()

        fig_corr.add_trace(
            go.Bar(
                x=lag_corr["Lag"],
                y=lag_corr["Pearson"],
                name="Pearson",
            )
        )

        fig_corr.update_layout(
            height=300,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            xaxis_title="Lag",
            yaxis_title="Korelasi",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig_corr.update_yaxes(
            range=[-1, 1],
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True,
        )


# ============================================================
# LAG TABLE
# ============================================================

with st.expander(
    "📋 Detail Korelasi Lag",
    expanded=False,
):

    display_lag = lag_corr.copy()

    display_lag["Pearson"] = (
        display_lag["Pearson"]
        .map(lambda x: f"{x:.4f}")
    )

    display_lag["Spearman"] = (
        display_lag["Spearman"]
        .map(lambda x: f"{x:.4f}")
    )

    display_lag["Abs Pearson"] = (
        display_lag["Abs Pearson"]
        .map(lambda x: f"{x:.4f}")
    )

    st.dataframe(
        display_lag,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SECTION 6
# NONLINEARITY
# ============================================================

st.markdown("### 📐 Analisis Hubungan Nonlinear")

st.caption(
    "SVR dengan kernel non-linear dapat mempelajari "
    "hubungan kompleks antara nilai historis dan "
    "harga target. Analisis ini digunakan untuk "
    "memahami pola hubungan tersebut."
)

available_lags = [
    lag
    for lag in [1, 7, 14, 30]
    if lag < len(harga)
]

selected_lag = st.selectbox(
    "Pilih Lag",
    available_lags,
    index=0,
)

nonlinear_df = pd.DataFrame(
    {
        "lag": harga.shift(
            selected_lag
        ),
        "target": harga,
    }
).dropna()

pearson_nl = nonlinear_df[
    "lag"
].corr(
    nonlinear_df["target"],
    method="pearson",
)

spearman_nl = nonlinear_df[
    "lag"
].corr(
    nonlinear_df["target"],
    method="spearman",
)

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Pearson Correlation",
        f"{pearson_nl:.3f}",
    )

with c2:

    st.metric(
        "Spearman Correlation",
        f"{spearman_nl:.3f}",
    )

with st.container(border=True):

    st.markdown(
        f"#### Hubungan Harga t-{selected_lag} dengan Harga t"
    )

    fig_scatter = go.Figure()

    fig_scatter.add_trace(
        go.Scatter(
            x=nonlinear_df["lag"],
            y=nonlinear_df["target"],
            mode="markers",
            opacity=0.45,
            showlegend=False,
        )
    )

    fig_scatter.update_layout(
        height=300,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        xaxis_title=f"Harga t-{selected_lag}",
        yaxis_title="Harga t",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig_scatter.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )

    fig_scatter.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


# ============================================================
# SECTION 7
# KESIAPAN FITUR SVR
# ============================================================

st.markdown("### 🧩 Kesiapan Fitur SVR")

st.caption(
    "Karakteristik fitur yang berpotensi digunakan "
    "sebagai input model SVR untuk memprediksi harga."
)

feature_df = make_svr_features(
    harga,
    df[date_column],
    max_lag=MAX_LAG,
)

target_df = pd.DataFrame(
    {
        "target": harga,
    },
    index=harga.index,
)

feature_analysis = pd.concat(
    [
        target_df,
        feature_df,
    ],
    axis=1,
)

feature_corr = (
    feature_analysis
    .corr()["target"]
    .drop("target")
    .dropna()
    .reset_index()
)

feature_corr.columns = [
    "Feature",
    "Correlation",
]

feature_corr["Abs Correlation"] = (
    feature_corr["Correlation"]
    .abs()
)

feature_corr = (
    feature_corr
    .sort_values(
        "Abs Correlation",
        ascending=False,
    )
)

top_features = feature_corr.head(10)


# ------------------------------------------------------------
# TOP FEATURES
# ------------------------------------------------------------

with st.container(border=True):

    st.markdown(
        "#### 10 Fitur dengan Korelasi Terkuat"
    )

    fig_features = go.Figure()

    fig_features.add_trace(
        go.Bar(
            x=top_features["Correlation"],
            y=top_features["Feature"],
            orientation="h",
        )
    )

    fig_features.update_layout(
        height=420,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        xaxis_title="Correlation",
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig_features.update_xaxes(
        range=[-1, 1],
    )

    st.plotly_chart(
        fig_features,
        use_container_width=True,
    )


# ============================================================
# FEATURE GROUP SUMMARY
# ============================================================

feature_groups = {
    "Lag": [
        c
        for c in feature_df.columns
        if c.startswith("lag_")
    ],
    "Rolling": [
        c
        for c in feature_df.columns
        if c.startswith("rolling_")
    ],
    "EMA": [
        c
        for c in feature_df.columns
        if c.startswith("ema_")
    ],
    "Difference": [
        c
        for c in feature_df.columns
        if c.startswith("diff_")
    ],
    "Calendar": [
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "is_weekend",
    ],
}

feature_summary = pd.DataFrame(
    [
        {
            "Kelompok Fitur": group,
            "Jumlah Fitur": len(features),
            "Contoh Fitur": ", ".join(
                features[:5]
            ),
        }
        for group, features
        in feature_groups.items()
    ]
)

with st.container(border=True):

    st.markdown(
        "#### Kelompok Fitur"
    )

    st.dataframe(
        feature_summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FEATURE PREVIEW
# ============================================================

with st.expander(
    "📋 Preview Fitur SVR",
    expanded=False,
):

    st.dataframe(
        feature_df.tail(10),
        use_container_width=True,
    )
