import pandas as pd
import streamlit as st

from utils import (
    init_session_state,
    inject_custom_css,
    render_sidebar,
    page_header,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Input Data",
    page_icon="📥",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / input data",
    title="Input Data",
    caption=(
        "Unggah data historis harga dan tentukan "
        "periode data yang akan digunakan untuk analisis."
    ),
)


# ============================================================
# INITIAL SESSION STATE
# ============================================================

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "df" not in st.session_state:
    st.session_state.df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "date_column" not in st.session_state:
    st.session_state.date_column = None

if "commodity_column" not in st.session_state:
    st.session_state.commodity_column = None

if "analysis_range" not in st.session_state:
    st.session_state.analysis_range = None

if "analysis_start_date" not in st.session_state:
    st.session_state.analysis_start_date = None

if "analysis_end_date" not in st.session_state:
    st.session_state.analysis_end_date = None


# ============================================================
# UPLOAD DATASET
# ============================================================

st.markdown("### 📂 Dataset")

st.markdown(
    """
    <div style="
        color:#747784;
        font-size:13px;
        margin-top:-8px;
        margin-bottom:16px;
    ">
        Masukkan dataset historis dalam format CSV atau Excel.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.container(border=True):

    uploaded_file = st.file_uploader(
        "Unggah dataset harga",
        type=["csv", "xlsx"],
        help="Format yang didukung: CSV dan Excel.",
    )


    # ========================================================
    # LOAD DATASET
    # ========================================================

    if uploaded_file is not None:

        try:

            # ------------------------------------------------
            # LOAD FILE
            # ------------------------------------------------

            if uploaded_file.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)

            # ------------------------------------------------
            # RESET SESSION JIKA DATASET BERUBAH
            # ------------------------------------------------

            dataset_changed = (
                st.session_state.dataset_name
                != uploaded_file.name
            )

            if dataset_changed:

                st.session_state.dataset_name = (
                    uploaded_file.name
                )

                st.session_state.original_df = (
                    df_raw.copy()
                )

                st.session_state.df = (
                    df_raw.copy()
                )

                # Reset mapping
                st.session_state.date_column = None
                st.session_state.commodity_column = None

                # Reset range
                st.session_state.analysis_range = None
                st.session_state.analysis_start_date = None
                st.session_state.analysis_end_date = None

            else:

                # Gunakan dataset yang sudah tersimpan
                df_raw = st.session_state.original_df.copy()


            st.success(
                f"Dataset **{uploaded_file.name}** berhasil dimuat."
            )


            # ==================================================
            # DATASET SUMMARY
            # ==================================================

            st.markdown("#### 📊 Ringkasan Dataset")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Jumlah Baris",
                    f"{len(df_raw):,}".replace(",", "."),
                )

            with c2:
                st.metric(
                    "Jumlah Kolom",
                    f"{len(df_raw.columns):,}".replace(",", "."),
                )

            with c3:
                st.metric(
                    "Missing Value",
                    f"{int(df_raw.isna().sum().sum()):,}"
                    .replace(",", "."),
                )


            # ==================================================
            # COLUMN MAPPING
            # ==================================================

            st.markdown("#### 🗂️ Pemetaan Kolom")

            st.caption(
                "Tentukan kolom yang digunakan sebagai tanggal "
                "dan harga."
            )

            col1, col2 = st.columns(2)

            columns = list(df_raw.columns)

            # ------------------------------------------------
            # DEFAULT DATE COLUMN
            # ------------------------------------------------

            if (
                st.session_state.date_column in columns
            ):
                date_default_index = columns.index(
                    st.session_state.date_column
                )
            else:
                date_default_index = 0


            # ------------------------------------------------
            # DEFAULT COMMODITY COLUMN
            # ------------------------------------------------

            if (
                st.session_state.commodity_column in columns
            ):
                commodity_default_index = columns.index(
                    st.session_state.commodity_column
                )
            else:

                commodity_default_index = (
                    1
                    if len(columns) > 1
                    else 0
                )


            with col1:

                date_column = st.selectbox(
                    "Kolom Tanggal",
                    columns,
                    index=date_default_index,
                    key="input_date_column",
                )


            with col2:

                commodity_column = st.selectbox(
                    "Kolom Harga",
                    columns,
                    index=commodity_default_index,
                    key="input_commodity_column",
                )


            # Simpan mapping
            st.session_state.date_column = date_column
            st.session_state.commodity_column = (
                commodity_column
            )


            # ==================================================
            # DATE PROCESSING
            # ==================================================

            df = df_raw.copy()

            df[date_column] = pd.to_datetime(
                df[date_column],
                dayfirst=True,
                errors="coerce",
            )

            # Hapus tanggal invalid
            df = df.dropna(
                subset=[date_column]
            )

            # Urutkan berdasarkan tanggal
            df = df.sort_values(
                date_column
            ).reset_index(drop=True)


            if df.empty:

                st.error(
                    "Tidak terdapat tanggal valid pada "
                    "kolom yang dipilih."
                )

                st.stop()


            # Simpan dataset hasil preprocessing tanggal
            st.session_state.original_df = df.copy()


            # ==================================================
            # AVAILABLE DATE RANGE
            # ==================================================

            min_date = (
                df[date_column]
                .min()
                .date()
            )

            max_date = (
                df[date_column]
                .max()
                .date()
            )


            # ==================================================
            # ANALYSIS RANGE
            # ==================================================

            st.markdown("#### 📅 Rentang Analisis")

            st.caption(
                "Pilih periode data yang akan digunakan "
                "untuk seluruh proses analisis dan pemodelan."
            )


            # ------------------------------------------------
            # DEFAULT DATE
            # ------------------------------------------------

            saved_start = (
                st.session_state.analysis_start_date
            )

            saved_end = (
                st.session_state.analysis_end_date
            )


            # Pastikan tanggal tersimpan masih berada
            # dalam range dataset terbaru.

            if (
                saved_start is None
                or saved_start < min_date
                or saved_start > max_date
            ):
                default_start = min_date
            else:
                default_start = saved_start


            if (
                saved_end is None
                or saved_end < min_date
                or saved_end > max_date
            ):
                default_end = max_date
            else:
                default_end = saved_end


            col1, col2 = st.columns(2)


            with col1:

                start_date = st.date_input(
                    "Tanggal Awal",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="analysis_start_input",
                )


            with col2:

                end_date = st.date_input(
                    "Tanggal Akhir",
                    value=default_end,
                    min_value=min_date,
                    max_value=max_date,
                    key="analysis_end_input",
                )


            # ==================================================
            # VALIDATE RANGE
            # ==================================================

            if start_date > end_date:

                st.error(
                    "Tanggal awal tidak boleh melebihi "
                    "tanggal akhir."
                )

                st.session_state.analysis_range = None
                st.session_state.analysis_start_date = None
                st.session_state.analysis_end_date = None

            else:

                # ------------------------------------------------
                # SIMPAN RANGE UTAMA
                # ------------------------------------------------

                st.session_state.analysis_range = (
                    start_date,
                    end_date,
                )

                st.session_state.analysis_start_date = (
                    start_date
                )

                st.session_state.analysis_end_date = (
                    end_date
                )


                # ==================================================
                # FILTER DATA AKTIF
                # ==================================================

                filtered_df = df[
                    (
                        df[date_column].dt.date
                        >= start_date
                    )
                    &
                    (
                        df[date_column].dt.date
                        <= end_date
                    )
                ].copy()


                # ------------------------------------------------
                # RESET INDEX
                # ------------------------------------------------

                filtered_df = (
                    filtered_df
                    .sort_values(date_column)
                    .reset_index(drop=True)
                )


                # ------------------------------------------------
                # SIMPAN DATA AKTIF
                # ------------------------------------------------

                st.session_state.df = (
                    filtered_df
                )


                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                st.success(
                    f"Rentang analisis berhasil ditetapkan: "
                    f"**{start_date.strftime('%d %b %Y')}** "
                    f"sampai **{end_date.strftime('%d %b %Y')}**."
                )


                st.info(
                    f"Data aktif untuk proses berikutnya: "
                    f"**{len(filtered_df):,} observasi**."
                )


            # ==================================================
            # AVAILABLE DATA INFORMATION
            # ==================================================

            st.caption(
                f"Data tersedia dari "
                f"**{min_date.strftime('%d %b %Y')}** "
                f"sampai "
                f"**{max_date.strftime('%d %b %Y')}**."
            )


        except Exception as e:

            st.error(
                f"Dataset tidak dapat diproses: {e}"
            )


# ============================================================
# ACTIVE ANALYSIS RANGE
# ============================================================

if (
    st.session_state.analysis_range
    is not None
):

    start_date, end_date = (
        st.session_state.analysis_range
    )

    st.markdown("### 🎯 Periode Aktif")

    with st.container(border=True):

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Tanggal Awal",
                start_date.strftime(
                    "%d %b %Y"
                ),
            )

        with c2:

            st.metric(
                "Tanggal Akhir",
                end_date.strftime(
                    "%d %b %Y"
                ),
            )

        with c3:

            active_df = st.session_state.df

            st.metric(
                "Observasi Aktif",
                f"{len(active_df):,}"
                .replace(",", "."),
            )


# ============================================================
# PREVIEW DATA
# ============================================================

st.markdown("### Pratinjau Data")

df = st.session_state.df


if (
    df is not None
    and len(df)
):

    preview = df.copy()


    if (
        st.session_state.date_column
        in preview.columns
        and
        st.session_state.commodity_column
        in preview.columns
    ):

        preview = preview[
            [
                st.session_state.date_column,
                st.session_state.commodity_column,
            ]
        ]


    st.markdown(
        f"""
        <div style="
            color:#747784;
            font-size:13px;
            margin-top:-8px;
            margin-bottom:12px;
        ">
            Menampilkan data yang akan digunakan
            dalam proses analisis.
            Total <b>{len(preview):,}</b> baris.
        </div>
        """,
        unsafe_allow_html=True,
    )


    with st.container(border=True):

        st.dataframe(
            preview,
            height=500,
            use_container_width=True,
            hide_index=True,
        )


else:

    with st.container(border=True):

        st.html(
            """
            <div style="
                text-align:center;
                padding:45px 20px;
            ">

                <div style="
                    font-size:42px;
                    margin-bottom:12px;
                ">
                    📊
                </div>

                <div style="
                    font-size:17px;
                    font-weight:700;
                    color:#31333F;
                    margin-bottom:6px;
                ">
                    Belum Ada Dataset
                </div>

                <div style="
                    font-size:13px;
                    color:#747784;
                ">
                    Unggah dataset di bagian atas
                    untuk mulai melakukan analisis.
                </div>

            </div>
            """
        )