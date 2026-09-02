import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    evaluate_prediction,
    format_rupiah,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_trained_model,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Output Model - KomoditasAI",
    page_icon="📈",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / output",
    title="Hasil Forecasting",
    caption=(
        "Evaluasi performa model SVR, hasil prediksi harga, "
        "dan analisis risiko komoditas."
    ),
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

trained, data, params = require_trained_model()


# ============================================================
# MODEL DATA
# ============================================================

best_svr = trained["best_svr"]

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

train_series = data["train_series"]
test_series = data["test_series"]

harga = data["harga"]

commodity_name = params.get(
    "commodity_column",
    "Komoditas",
)

optimal_lag = params.get(
    "optimal_lag",
    data.get("optimal_lag"),
)

svr_profile = params.get(
    "svr_profile",
    "Balanced",
)

svr_iterations = params.get(
    "svr_iterations",
    None,
)

cv_splits = params.get(
    "cv_splits",
    None,
)


# ============================================================
# SECTION 01
# DATASET & MODEL OVERVIEW
# ============================================================

st.markdown(
    "### 01 · Dataset & Model Overview"
)

with st.container(border=True):

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Komoditas",
            commodity_name,
            "Dataset aktif",
        )

    with c2:

        st.metric(
            "Data Train",
            f"{len(train_series):,}".replace(
                ",",
                ".",
            ),
            "Observasi",
        )

    with c3:

        st.metric(
            "Data Test",
            f"{len(test_series):,}".replace(
                ",",
                ".",
            ),
            "Observasi",
        )

    with c4:

        st.metric(
            "Model",
            "SVR",
            "Support Vector Regression",
        )


st.write("")


# ============================================================
# SECTION 02
# MODEL CONFIGURATION
# ============================================================

st.markdown(
    "### 02 · Konfigurasi Model"
)

with st.container(border=True):

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Kernel",
            "RBF",
        )

    with c2:

        st.metric(
            "Optimal Lag",
            optimal_lag,
        )

    with c3:

        st.metric(
            "Search Profile",
            svr_profile,
        )

    with c4:

        st.metric(
            "CV Fold",
            cv_splits if cv_splits else "-",
        )


# ============================================================
# SECTION 03
# FIT SVR
# ============================================================

st.markdown(
    "### 03 · Membangun Model SVR"
)

with st.spinner(
    "Membangun model SVR..."
):

    best_svr.fit(
        X_train,
        y_train,
    )

st.success(
    "✓ Model SVR berhasil dibangun."
)


# ============================================================
# SECTION 04
# PREDICTION
# ============================================================

st.markdown(
    "### 04 · Hasil Prediksi SVR"
)

with st.spinner(
    "Menghasilkan prediksi SVR..."
):

    # --------------------------------------------------------
    # TRAIN PREDICTION
    # --------------------------------------------------------

    svr_train_pred = pd.Series(
        best_svr.predict(
            X_train
        ),
        index=X_train.index,
        name="SVR",
    )

    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------

    svr_test_pred = pd.Series(
        best_svr.predict(
            X_test
        ),
        index=X_test.index,
        name="SVR",
    )

st.success(
    "✓ Prediksi SVR selesai."
)


# ============================================================
# SECTION 05
# PREDICTION CHART
# ============================================================

st.markdown(
    "### 05 · Aktual vs Prediksi"
)

with st.container(border=True):

    col1, col2, col3 = st.columns(
        [1, 8, 1]
    )

    with col2:

        fig, ax = plt.subplots(
            figsize=(10, 5),
            dpi=120,
        )

        ax.plot(
            test_series.index,
            test_series,
            linewidth=2.8,
            label="Actual",
        )

        ax.plot(
            svr_test_pred.index,
            svr_test_pred,
            linewidth=2.0,
            label="SVR",
        )

        ax.set_xlabel(
            "Tanggal"
        )

        ax.set_ylabel(
            f"Harga {commodity_name}"
        )

        ax.set_title(
            "Aktual vs Prediksi SVR"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


# ============================================================
# PREDICTION TABLE
# ============================================================

with st.expander(
    "📋 Lihat Detail Prediksi SVR",
    expanded=False,
):

    prediction_df = pd.DataFrame(
        {
            "Actual": y_test,
            "SVR": svr_test_pred,
        }
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SECTION 06
# MODEL EVALUATION
# ============================================================

st.markdown(
    "### 06 · Evaluasi Performa SVR"
)

svr_metric = evaluate_prediction(
    y_test,
    svr_test_pred,
    y_train,
    "SVR",
)

METRIC_FORMAT = {
    "RMSE": "{:.2f}",
    "MAE": "{:.2f}",
    "MAPE (%)": "{:.2f}",
    "sMAPE (%)": "{:.2f}",
    "MASE": "{:.3f}",
    "R2": "{:.4f}",
    "Bias": "{:.2f}",
}


with st.container(border=True):

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "RMSE",
            f"{svr_metric['RMSE']:.2f}",
        )

    with c2:

        st.metric(
            "MAE",
            f"{svr_metric['MAE']:.2f}",
        )

    with c3:

        st.metric(
            "MAPE",
            f"{svr_metric['MAPE (%)']:.2f}%",
        )

    with c4:

        st.metric(
            "R²",
            f"{svr_metric['R2']:.4f}",
        )


with st.expander(
    "📋 Lihat Seluruh Metrik",
    expanded=False,
):

    metric_table = pd.DataFrame(
        [svr_metric]
    )

    st.dataframe(
        metric_table.style.format(
            METRIC_FORMAT
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SECTION 07
# BEST SVR PARAMETERS
# ============================================================

st.markdown(
    "### 07 · Parameter SVR Terbaik"
)

best_svr_params = trained.get(
    "svr_best_params",
    {},
)

if not best_svr_params:

    best_svr_params = (
        data.get(
            "best_svr_params",
            {},
        )
    )


with st.container(border=True):

    if best_svr_params:

        c1, c2, c3 = st.columns(3)

        # ----------------------------------------------------
        # C
        # ----------------------------------------------------

        with c1:

            c_value = best_svr_params.get(
                "regressor__svr__C"
            )

            st.metric(
                "C",
                f"{c_value:.4f}"
                if c_value is not None
                else "-",
            )

        # ----------------------------------------------------
        # GAMMA
        # ----------------------------------------------------

        with c2:

            gamma_value = best_svr_params.get(
                "regressor__svr__gamma"
            )

            if isinstance(
                gamma_value,
                str,
            ):

                gamma_display = (
                    gamma_value
                )

            elif gamma_value is not None:

                gamma_display = (
                    f"{gamma_value:.6f}"
                )

            else:

                gamma_display = "-"

            st.metric(
                "Gamma",
                gamma_display,
            )

        # ----------------------------------------------------
        # EPSILON
        # ----------------------------------------------------

        with c3:

            epsilon_value = (
                best_svr_params.get(
                    "regressor__svr__epsilon"
                )
            )

            st.metric(
                "Epsilon",
                f"{epsilon_value:.4f}"
                if epsilon_value is not None
                else "-",
            )

    else:

        st.info(
            "Parameter SVR terbaik tidak tersedia "
            "pada session state."
        )


# ============================================================
# SECTION 08
# VALUE AT RISK
# ============================================================

st.markdown(
    "### 08 · Analisis Risiko"
)

st.caption(
    "Analisis VaR menggunakan distribusi absolute error "
    "dari prediksi SVR pada data test."
)


# ============================================================
# SVR FORECAST ERROR
# ============================================================

svr_error = (
    y_test
    - svr_test_pred
)

absolute_error = (
    svr_error
    .abs()
)


# ============================================================
# VAR CALCULATION
# ============================================================

var90 = absolute_error.quantile(
    0.90
)

var95 = absolute_error.quantile(
    0.95
)

var99 = absolute_error.quantile(
    0.99
)


# ============================================================
# VAR HORIZON
# ============================================================

var_table = pd.DataFrame(
    {
        "Hari": range(
            1,
            6,
        )
    }
)

var_table["VaR 90%"] = (
    var90
    * np.sqrt(
        var_table["Hari"]
    )
)

var_table["VaR 95%"] = (
    var95
    * np.sqrt(
        var_table["Hari"]
    )
)

var_table["VaR 99%"] = (
    var99
    * np.sqrt(
        var_table["Hari"]
    )
)


# ============================================================
# VAR SUMMARY
# ============================================================

with st.container(border=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "VaR 90% · 1 Hari",
            format_rupiah(
                var90
            ),
        )

    with c2:

        st.metric(
            "VaR 95% · 1 Hari",
            format_rupiah(
                var95
            ),
        )

    with c3:

        st.metric(
            "VaR 99% · 1 Hari",
            format_rupiah(
                var99
            ),
        )


st.write("")


# ============================================================
# VAR HORIZON TABLE
# ============================================================

with st.container(border=True):

    st.markdown(
        "#### VaR Berdasarkan Horizon"
    )

    st.dataframe(
        var_table.style.format(
            {
                "VaR 90%": "Rp{:,.2f}",
                "VaR 95%": "Rp{:,.2f}",
                "VaR 99%": "Rp{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# VAR CHART
# ============================================================

st.markdown(
    "#### 📈 Perkembangan Risiko 1-5 Hari"
)

with st.container(border=True):

    col1, col2, col3 = st.columns(
        [1, 8, 1]
    )

    with col2:

        fig_var, ax_var = plt.subplots(
            figsize=(9, 5),
            dpi=120,
        )

        for column in [
            "VaR 90%",
            "VaR 95%",
            "VaR 99%",
        ]:

            ax_var.plot(
                var_table["Hari"],
                var_table[column],
                marker="o",
                linewidth=2.5,
                label=column,
            )

        ax_var.set_xlabel(
            "Horizon Risiko (Hari)"
        )

        ax_var.set_ylabel(
            "Nilai VaR (Rp)"
        )

        ax_var.set_xticks(
            var_table["Hari"]
        )

        ax_var.set_title(
            "Estimasi VaR Berdasarkan Horizon"
        )

        ax_var.grid(
            alpha=0.25
        )

        ax_var.legend(
            frameon=False
        )

        fig_var.tight_layout()

        st.pyplot(
            fig_var,
            use_container_width=True,
        )

        plt.close(fig_var)


# ============================================================
# RISK INTERPRETATION
# ============================================================

st.markdown(
    "### 09 · Interpretasi Risiko"
)

with st.container(border=True):

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Estimasi VaR 95% · 1 Hari",
            format_rupiah(
                var95
            ),
        )

    with c2:

        var95_5day = (
            var_table.iloc[-1][
                "VaR 95%"
            ]
        )

        st.metric(
            "Estimasi VaR 95% · 5 Hari",
            format_rupiah(
                var95_5day
            ),
        )

    st.write("")

    st.write(
        f"Pada tingkat kepercayaan 95%, "
        f"estimasi deviasi prediksi SVR dalam "
        f"1 hari mencapai sekitar "
        f"**{format_rupiah(var95)}**."
    )

    st.write(
        f"Dengan pendekatan square-root-of-time, "
        f"estimasi risiko meningkat menjadi sekitar "
        f"**{format_rupiah(var95_5day)}** "
        f"untuk horizon 5 hari."
    )

    st.caption(
        "VaR dihitung berdasarkan distribusi absolute error "
        "prediksi SVR dan diperluas ke beberapa horizon "
        "menggunakan pendekatan square-root-of-time."
    )


# ============================================================
# SAVE OUTPUT RESULT
# ============================================================

st.session_state.output_result = {

    "svr_train_pred":
        svr_train_pred,

    "svr_test_pred":
        svr_test_pred,

    "svr_metric":
        svr_metric,

    "prediction_result":
        prediction_df,

    "svr_error":
        svr_error,

    "absolute_error":
        absolute_error,

    "var90":
        var90,

    "var95":
        var95,

    "var99":
        var99,

    "var_table":
        var_table,

    "best_svr_params":
        best_svr_params,
}