import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import loguniform

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import (
    clean_commodity_series,
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
    page_title="Input Parameter Model - KomoditasAI",
    page_icon="⚙️",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / input parameter",
    title=" Input Parameter Model",
    caption=(
        "Konfigurasi pembagian data, feature engineering, "
        "dan tuning Support Vector Regression (SVR)."
    ),
)


# ============================================================
# GLOBAL CONFIG
# ============================================================

RANDOM_STATE = 42

MIN_LAG = 1
MAX_LAG = 30

DEFAULT_TRAIN_END_DATE = pd.Timestamp(
    "2025-12-31"
)

SVR_ITERATIONS_BY_PROFILE = {
    "Fast": 15,
    "Balanced": 28,
    "Thorough": 50,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_time_series_features(
    series: pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    """
    Membuat feature time series untuk SVR.

    Feature terdiri dari:
    - Lag
    - Rolling statistics
    - Exponential moving average
    - Difference
    - Calendar features
    """

    s = series.astype(float).copy()

    frame = pd.DataFrame(index=s.index)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    frame["target"] = s

    # --------------------------------------------------------
    # Lag Features
    # --------------------------------------------------------

    for lag in range(1, max_lag + 1):
        frame[f"lag_{lag}"] = s.shift(lag)

    # --------------------------------------------------------
    # Rolling Features
    # --------------------------------------------------------

    shifted = s.shift(1)

    for window in [3, 5, 7, 10, 14, 21, 30]:

        frame[f"roll_mean_{window}"] = (
            shifted
            .rolling(window)
            .mean()
        )

        frame[f"roll_std_{window}"] = (
            shifted
            .rolling(window)
            .std()
        )

        frame[f"roll_min_{window}"] = (
            shifted
            .rolling(window)
            .min()
        )

        frame[f"roll_max_{window}"] = (
            shifted
            .rolling(window)
            .max()
        )

    # --------------------------------------------------------
    # Exponential Moving Average
    # --------------------------------------------------------

    frame["ewm_mean_5"] = (
        shifted
        .ewm(
            span=5,
            adjust=False,
        )
        .mean()
    )

    frame["ewm_mean_14"] = (
        shifted
        .ewm(
            span=14,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Difference Features
    # --------------------------------------------------------

    frame["diff_1"] = (
        s.shift(1)
        - s.shift(2)
    )

    frame["diff_5"] = (
        s.shift(1)
        - s.shift(6)
    )

    # --------------------------------------------------------
    # Calendar Features
    # --------------------------------------------------------

    idx = pd.DatetimeIndex(frame.index)

    frame["day_of_week"] = idx.dayofweek
    frame["day_of_month"] = idx.day
    frame["month"] = idx.month
    frame["quarter"] = idx.quarter

    frame["day_of_year_sin"] = np.sin(
        2 * np.pi * idx.dayofyear / 365.25
    )

    frame["day_of_year_cos"] = np.cos(
        2 * np.pi * idx.dayofyear / 365.25
    )

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------

    return (
        frame
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


def make_svr_estimator(
    C=100.0,
    gamma="scale",
    epsilon=0.05,
):
    """
    Membuat estimator SVR.

    Pipeline:
    StandardScaler -> SVR RBF

    Target juga ditransformasi menggunakan
    StandardScaler melalui TransformedTargetRegressor.
    """

    x_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    epsilon=epsilon,
                    cache_size=1000,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=x_pipeline,
        transformer=StandardScaler(),
    )


def valid_tscv(
    n_samples,
    requested_splits,
):
    """
    Menyesuaikan jumlah fold TimeSeriesSplit
    berdasarkan jumlah observasi.
    """

    n_splits = min(
        requested_splits,
        max(
            2,
            n_samples // 60,
        ),
    )

    n_splits = min(
        n_splits,
        n_samples - 1,
    )

    return TimeSeriesSplit(
        n_splits=n_splits
    )


# ============================================================
# DATASET
# ============================================================

df, date_column, commodity_column = (
    require_dataset()
)

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce",
    dayfirst=True,
)

df[commodity_column] = (
    clean_commodity_series(
        df,
        commodity_column,
    )
)

df = (
    df
    .dropna(
        subset=[
            date_column,
            commodity_column,
        ]
    )
    .sort_values(date_column)
)

harga = (
    df
    .set_index(date_column)[commodity_column]
    .astype(float)
    .sort_index()
)

harga = harga[
    ~harga.index.duplicated(
        keep="last"
    )
]


# ============================================================
# SECTION 01
# DATASET AKTIF
# ============================================================

st.markdown(
    "### 01 · Dataset Aktif"
)

st.caption(
    "Ringkasan dataset yang digunakan "
    "sebagai dasar pemodelan SVR."
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Komoditas",
        commodity_column,
    )

with m2:
    st.metric(
        "Observasi",
        f"{len(harga):,}".replace(
            ",",
            ".",
        ),
    )

with m3:
    st.metric(
        "Tanggal Mulai",
        harga.index.min().strftime(
            "%d %b %Y"
        ),
    )

with m4:
    st.metric(
        "Tanggal Akhir",
        harga.index.max().strftime(
            "%d %b %Y"
        ),
    )


# ============================================================
# SECTION 02
# TRAIN TEST SPLIT
# ============================================================

st.markdown(
    "### 02 · Pembagian Data Train & Test"
)

with st.container(border=True):

    split_col, summary_col = st.columns(
        [1.25, 1],
        gap="large",
    )

    # --------------------------------------------------------
    # SPLIT METHOD
    # --------------------------------------------------------

    with split_col:

        split_method = st.radio(
            "Metode Pembagian Data",
            [
                "Persentase",
                "Tanggal (Advanced)",
            ],
            horizontal=True,
        )

        # ----------------------------------------------------
        # PERCENTAGE SPLIT
        # ----------------------------------------------------

        if split_method == "Persentase":

            train_ratio = st.slider(
                "Proporsi Data Train (%)",
                min_value=50,
                max_value=95,
                value=80,
                step=5,
            )

            test_ratio = (
                100 - train_ratio
            )

            split_index = int(
                len(harga)
                * train_ratio
                / 100
            )

            train_series = (
                harga
                .iloc[:split_index]
                .copy()
            )

            test_series = (
                harga
                .iloc[split_index:]
                .copy()
            )

            split_note = (
                f"Data dibagi secara kronologis "
                f"{train_ratio}% train dan "
                f"{test_ratio}% test."
            )

        # ----------------------------------------------------
        # DATE SPLIT
        # ----------------------------------------------------

        else:

            train_end_date = st.date_input(
                "Tanggal Akhir Data Train",
                value=min(
                    DEFAULT_TRAIN_END_DATE.date(),
                    harga.index.max().date(),
                ),
                min_value=harga.index.min().date(),
                max_value=harga.index.max().date(),
            )

            train_series = (
                harga.loc[
                    harga.index
                    <= pd.Timestamp(
                        train_end_date
                    )
                ]
                .copy()
            )

            test_series = (
                harga.loc[
                    harga.index
                    > pd.Timestamp(
                        train_end_date
                    )
                ]
                .copy()
            )

            total = len(harga)

            train_ratio = round(
                len(train_series)
                / total
                * 100,
                1,
            )

            test_ratio = round(
                len(test_series)
                / total
                * 100,
                1,
            )

            split_note = (
                "Data dibagi berdasarkan "
                "tanggal yang dipilih."
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        len(train_series) == 0
        or len(test_series) == 0
    ):

        st.error(
            "Pembagian data menghasilkan "
            "train atau test kosong. "
            "Silakan ubah rasio atau tanggal."
        )

        st.stop()

    effective_train_end = (
        train_series.index.max()
    )

    effective_test_start = (
        test_series.index.min()
    )

    # --------------------------------------------------------
    # SPLIT SUMMARY
    # --------------------------------------------------------

    with summary_col:

        st.caption(
            "RINGKASAN PEMBAGIAN"
        )

        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                "Train",
                f"{len(train_series):,}".replace(
                    ",",
                    ".",
                ),
            )

        with m2:
            st.metric(
                "Test",
                f"{len(test_series):,}".replace(
                    ",",
                    ".",
                ),
            )

        st.caption(
            "Train sampai: "
            f"{effective_train_end.strftime('%d %b %Y')}"
        )

        st.caption(
            "Test mulai: "
            f"{effective_test_start.strftime('%d %b %Y')}"
        )


st.info(split_note)


# ============================================================
# SPLIT VISUALIZATION
# ============================================================

with st.expander(
    "📈 Lihat Visualisasi Pembagian Data",
    expanded=False,
):

    fig_split, ax_split = plt.subplots(
        figsize=(13, 4)
    )

    ax_split.plot(
        train_series.index,
        train_series,
        label="Train",
        linewidth=1.3,
    )

    ax_split.plot(
        test_series.index,
        test_series,
        label="Test",
        linewidth=1.3,
    )

    ax_split.axvline(
        effective_test_start,
        color="red",
        linestyle="--",
        linewidth=1.3,
    )

    ax_split.set_title(
        "Pembagian Data Train dan Test"
    )

    ax_split.set_xlabel(
        "Tanggal"
    )

    ax_split.set_ylabel(
        f"Harga {commodity_column}"
    )

    ax_split.legend()
    ax_split.grid(
        alpha=0.25
    )

    fig_split.tight_layout()

    st.pyplot(
        fig_split,
        use_container_width=True,
    )

    plt.close(fig_split)


st.write("")


# ============================================================
# SECTION 03
# FEATURE ENGINEERING
# ============================================================

st.markdown(
    "### 03 · Feature Engineering"
)

with st.container(border=True):

    feature_col, cv_col = st.columns(
        2,
        gap="large",
    )

    # --------------------------------------------------------
    # AUTOMATIC LAG
    # --------------------------------------------------------

    with feature_col:

        st.markdown(
            "#### Automatic Lag Selection"
        )

        st.markdown(
            """
            Lag optimum akan dipilih otomatis
            menggunakan **SVR Baseline** dengan
            evaluasi **TimeSeriesSplit Cross Validation**.
            """
        )

        st.info(
            "Metode: SVR Baseline + CV RMSE"
        )

        st.caption(
            f"Rentang kandidat lag: "
            f"{MIN_LAG} sampai {MAX_LAG}"
        )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    with cv_col:

        st.markdown(
            "#### Cross Validation"
        )

        cv_splits = st.slider(
            "TimeSeriesSplit (Fold)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key="svr_cv_split",
        )

        effective_cv = min(
            cv_splits,
            max(
                2,
                len(train_series) // 60,
            ),
        )

        effective_cv = min(
            effective_cv,
            len(train_series) - 1,
        )

        st.metric(
            "Effective Fold",
            effective_cv,
        )

        st.caption(
            "Jumlah fold akan disesuaikan "
            "apabila jumlah data tidak mencukupi."
        )


st.write("")


# ============================================================
# SECTION 04
# SVR CONFIGURATION
# ============================================================

st.markdown(
    "### 04 · Konfigurasi SVR"
)

with st.container(border=True):

    st.markdown(
        "#### Support Vector Regression"
    )

    svr_profile = st.radio(
        "Search Profile",
        [
            "Fast",
            "Balanced",
            "Thorough",
        ],
        index=1,
        horizontal=True,
        key="svr_profile",
    )

    svr_iterations = (
        SVR_ITERATIONS_BY_PROFILE[
            svr_profile
        ]
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Search Iteration",
            svr_iterations,
        )

    with c2:

        st.metric(
            "CV Fold",
            effective_cv,
        )

    st.caption(
        "RandomizedSearchCV digunakan untuk "
        "mencari kombinasi hyperparameter SVR terbaik."
    )


# ============================================================
# RUN BUTTON
# ============================================================

st.write("")
st.write("")

st.markdown(
    """
    <div style="
        text-align:center;
        margin:10px 0 14px 0;
    ">
        <div style="
            font-size:13px;
            color:#747784;
            margin-bottom:8px;
        ">
            Semua konfigurasi SVR sudah siap
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

jalankan = st.button(
    "▶  Jalankan SVR",
    type="primary",
    use_container_width=True,
)


# ============================================================
# SVR EXECUTION
# ============================================================

if jalankan:

    # ========================================================
    # STEP 1
    # AUTOMATIC LAG OPTIMIZATION
    # ========================================================

    st.markdown(
        "### 🔎 Optimasi Lag SVR"
    )

    with st.spinner(
        "Mencari lag optimum menggunakan SVR..."
    ):

        lag_results = []

        max_candidate = min(
            MAX_LAG,
            max(
                MIN_LAG,
                len(train_series) // 10,
            ),
        )

        for lag in range(
            MIN_LAG,
            max_candidate + 1,
        ):

            feature_data = (
                make_time_series_features(
                    train_series,
                    max_lag=lag,
                )
            )

            if len(feature_data) < 80:
                continue

            X_lag = (
                feature_data
                .drop(columns="target")
            )

            y_lag = (
                feature_data["target"]
            )

            cv = valid_tscv(
                len(X_lag),
                effective_cv,
            )

            scores = cross_val_score(
                make_svr_estimator(),
                X_lag,
                y_lag,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )

            lag_results.append(
                {
                    "Lag": lag,
                    "CV RMSE": -scores.mean(),
                    "CV RMSE Std": scores.std(),
                    "Jumlah Fitur": X_lag.shape[1],
                    "Jumlah Observasi": len(X_lag),
                }
            )

        if not lag_results:

            st.error(
                "Penentuan lag gagal karena "
                "data terlalu sedikit."
            )

            st.stop()

        lag_table = (
            pd.DataFrame(
                lag_results
            )
            .sort_values(
                "CV RMSE",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

        optimal_lag = int(
            lag_table.iloc[0]["Lag"]
        )

        best_lag_cv_rmse = float(
            lag_table.iloc[0]["CV RMSE"]
        )

    st.success(
        f"Lag optimum yang dipilih: "
        f"{optimal_lag}"
    )


    # ========================================================
    # STEP 2
    # CREATE SUPERVISED DATASET
    # ========================================================

    st.markdown(
        "### 📊 Menyiapkan Data Supervised"
    )

    supervised = (
        make_time_series_features(
            harga,
            max_lag=optimal_lag,
        )
    )

    X_all = (
        supervised
        .drop(columns="target")
    )

    y_all = (
        supervised["target"]
    )

    # --------------------------------------------------------
    # TRAIN TEST MASK
    # --------------------------------------------------------

    train_mask = (
        X_all.index
        <= effective_train_end
    )

    test_mask = (
        X_all.index
        >= effective_test_start
    )

    X_train = (
        X_all.loc[train_mask]
        .copy()
    )

    y_train = (
        y_all.loc[train_mask]
        .copy()
    )

    X_test = (
        X_all.loc[test_mask]
        .copy()
    )

    y_test = (
        y_all.loc[test_mask]
        .copy()
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        len(X_train) < 100
        or len(X_test) < 10
    ):

        st.error(
            f"Data supervised tidak cukup. "
            f"Train={len(X_train)}, "
            f"Test={len(X_test)}"
        )

        st.stop()

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Optimal Lag",
            optimal_lag,
        )

    with c2:
        st.metric(
            "Train Observasi",
            f"{len(X_train):,}".replace(
                ",",
                ".",
            ),
        )

    with c3:
        st.metric(
            "Test Observasi",
            f"{len(X_test):,}".replace(
                ",",
                ".",
            ),
        )


    # ========================================================
    # STEP 3
    # SVR RANDOM SEARCH
    # ========================================================

    st.markdown(
        "### 🔎 Tuning Hyperparameter SVR"
    )

    svr_search_space = {

        "regressor__svr__C": loguniform(
            1e-1,
            2e3,
        ),

        "regressor__svr__gamma": loguniform(
            1e-5,
            1.0,
        ),

        "regressor__svr__epsilon": loguniform(
            1e-3,
            0.5,
        ),
    }

    svr_search = RandomizedSearchCV(
        estimator=make_svr_estimator(),

        param_distributions=svr_search_space,

        n_iter=svr_iterations,

        scoring="neg_root_mean_squared_error",

        cv=valid_tscv(
            len(X_train),
            effective_cv,
        ),

        random_state=RANDOM_STATE,

        n_jobs=-1,

        refit=True,

        verbose=0,
    )

    with st.spinner(
        "Melakukan tuning hyperparameter SVR..."
    ):

        start = time.time()

        svr_search.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.time() - start
        ) / 60

    best_svr = (
        svr_search.best_estimator_
    )

    best_svr_params = (
        svr_search.best_params_
    )

    best_svr_cv_rmse = (
        -svr_search.best_score_
    )

    st.success(
        f"Tuning SVR selesai "
        f"({elapsed:.2f} menit)"
    )


    # ========================================================
    # STEP 4
    # BEST PARAMETER SUMMARY
    # ========================================================

    st.markdown(
        "### 🎯 Parameter SVR Terbaik"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "C",
            f"{best_svr_params['regressor__svr__C']:.4f}",
        )

    with c2:

        gamma_value = (
            best_svr_params[
                "regressor__svr__gamma"
            ]
        )

        if isinstance(
            gamma_value,
            str,
        ):
            gamma_display = gamma_value

        else:
            gamma_display = (
                f"{gamma_value:.6f}"
            )

        st.metric(
            "Gamma",
            gamma_display,
        )

    with c3:

        st.metric(
            "Epsilon",
            f"{best_svr_params['regressor__svr__epsilon']:.4f}",
        )


    # ========================================================
    # STEP 5
    # MODEL SUMMARY
    # ========================================================

    st.markdown(
        "### 📋 Ringkasan Konfigurasi SVR"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Optimal Lag",
            optimal_lag,
        )

    with c2:

        st.metric(
            "Best CV RMSE",
            f"{best_svr_cv_rmse:.3f}",
        )

    with c3:

        st.metric(
            "Train",
            f"{len(X_train):,}".replace(
                ",",
                ".",
            ),
        )

    with c4:

        st.metric(
            "Test",
            f"{len(X_test):,}".replace(
                ",",
                ".",
            ),
        )


    # ========================================================
    # STEP 6
    # LAG OPTIMIZATION RESULT
    # ========================================================

    st.markdown(
        "### 📈 Hasil Optimasi Lag"
    )

    chart_col1, chart_col2, chart_col3 = (
        st.columns(
            [1, 2.5, 1]
        )
    )

    with chart_col2:

        fig_lag, ax_lag = plt.subplots(
            figsize=(8, 4)
        )

        ax_lag.plot(
            lag_table["Lag"],
            lag_table["CV RMSE"],
            marker="o",
            linewidth=1.5,
        )

        ax_lag.scatter(
            optimal_lag,
            best_lag_cv_rmse,
            s=70,
            zorder=3,
        )

        ax_lag.set_xlabel(
            "Lag"
        )

        ax_lag.set_ylabel(
            "CV RMSE"
        )

        ax_lag.set_title(
            "Optimasi Lag SVR"
        )

        ax_lag.grid(
            alpha=0.25
        )

        fig_lag.tight_layout()

        st.pyplot(
            fig_lag,
            use_container_width=True,
        )

        plt.close(fig_lag)


    # ========================================================
    # STEP 7
    # BEST PARAMETER DETAIL
    # ========================================================

    with st.expander(
        "Lihat Detail Parameter SVR",
        expanded=False,
    ):

        parameter_table = pd.DataFrame(
            {
                "Parameter": [
                    "Kernel",
                    "C",
                    "Gamma",
                    "Epsilon",
                    "Optimal Lag",
                    "Search Profile",
                    "Search Iteration",
                    "CV Fold",
                ],
                "Value": [
                    "RBF",
                    f"{best_svr_params['regressor__svr__C']:.6f}",
                    gamma_display,
                    f"{best_svr_params['regressor__svr__epsilon']:.6f}",
                    optimal_lag,
                    svr_profile,
                    svr_iterations,
                    effective_cv,
                ],
            }
        )

        st.dataframe(
            parameter_table,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # STEP 8
    # LAG CANDIDATES
    # ========================================================

    with st.expander(
        "Lihat Kandidat Lag SVR",
        expanded=False,
    ):

        st.dataframe(
            lag_table,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # SAVE MODEL PARAMETERS
    # ========================================================

    st.session_state.model_params = {

        "commodity_column":
            commodity_column,

        "date_column":
            date_column,

        "train_end":
            effective_train_end,

        "test_start":
            effective_test_start,

        "train_size":
            len(X_train),

        "test_size":
            len(X_test),

        "train_ratio":
            train_ratio,

        "test_ratio":
            test_ratio,

        "max_lag":
            MAX_LAG,

        "optimal_lag":
            optimal_lag,

        "svr_profile":
            svr_profile,

        "svr_iterations":
            svr_iterations,

        "cv_splits":
            effective_cv,

        "random_state":
            RANDOM_STATE,
    }


    # ========================================================
    # SAVE MODEL RESULT
    # ========================================================

    st.session_state.model_result = {

        "best_svr":
            best_svr,

        "svr_best_params":
            best_svr_params,

        "svr_best_score":
            best_svr_cv_rmse,
    }


    # ========================================================
    # SAVE MODEL DATA
    # ========================================================

    st.session_state.model_data = {

        "harga":
            harga,

        "train_series":
            train_series,

        "test_series":
            test_series,

        "X_train":
            X_train,

        "y_train":
            y_train,

        "X_test":
            X_test,

        "y_test":
            y_test,

        "optimal_lag":
            optimal_lag,

        "best_cv_rmse":
            best_lag_cv_rmse,

        "best_svr":
            best_svr,

        "best_svr_params":
            best_svr_params,

        "svr_best_score":
            best_svr_cv_rmse,

        "lag_table":
            lag_table,

        "effective_train_end":
            effective_train_end,

        "effective_test_start":
            effective_test_start,
    }

    st.success(
        "Konfigurasi dan hasil tuning SVR "
        "berhasil disimpan."
    )