import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64
from pathlib import Path

# ======================================================================
# SESSION STATE
# ======================================================================

# Nilai default seluruh session_state yang dipakai lintas halaman.
SESSION_DEFAULTS = {
    "df": None,
    "original_df": None,
    "dataset_name": None,
    "date_column": None,
    "commodity_column": None,
    "analysis_range": None,
    "model_result": None,
    "model_data": None,
    "model_params": None,
}


def init_session_state():
    """Pastikan seluruh key session_state sudah terdaftar sebelum dipakai halaman mana pun."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ======================================================================
# STYLING (CSS GLOBAL)
# ======================================================================

_CUSTOM_CSS = """
<style>

/* ==========================================================
   KOMODITASAI COLOR PALETTE
========================================================== */

:root{

    /* PRIMARY */
    --olive:#566834;
    --olive-dark:#46562A;
    --olive-light:#718A42;

    /* GREEN */
    --forest:#006738;
    --forest-dark:#004D2A;
    --leaf:#87BD43;
    --leaf-light:#B7D97A;

    /* GOLD */
    --gold:#E5B043;
    --gold-dark:#C99528;
    --gold-light:#F4D98B;

    /* BACKGROUND */
    --bg:#FFFFFF;
    --bg-soft:#F7F9F3;
    --bg-green:#F1F5E9;

    /* BORDER */
    --border:#DDE4D3;
    --border-dark:#C9D3BC;

    /* TEXT */
    --text:#30372B;
    --text-secondary:#66705D;
    --text-muted:#8A9282;
}


/* ==========================================================
   GLOBAL
========================================================== */

html{
    font-size:14px;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"]{

    background:#FFFFFF;

    color:var(--text);

    font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


/* ==========================================================
   STREAMLIT HEADER
========================================================== */

header[data-testid="stHeader"]{

    background:#FFFFFF !important;

    border-bottom:1px solid var(--border);
}

[data-testid="stToolbar"]{

    background:#FFFFFF !important;
}


/* ==========================================================
   MAIN CONTAINER
========================================================== */

.block-container{

    max-width:1400px;

    padding:1rem 2rem 1.5rem;
}


/* ==========================================================
   TYPOGRAPHY
========================================================== */

h1{

    font-size:42px !important;

    font-weight:800 !important;

    color:var(--text);

    margin-bottom:.5rem;
}

h2{

    font-size:32px !important;

    font-weight:700 !important;

    color:var(--text);

    margin-top:1.4rem;

    margin-bottom:.6rem;
}

h3{

    font-size:24px !important;

    font-weight:700 !important;

    color:var(--text);
}

h4{

    font-size:18px !important;

    font-weight:600 !important;

    color:var(--text);
}


/* ==========================================================
   HOMEPAGE HERO
========================================================== */

.homepage-hero{

    position:relative;

    overflow:hidden;

    min-height:690px;

    background:

        radial-gradient(
            circle at 78% 22%,
            rgba(135,189,67,.15),
            transparent 25%
        ),

        radial-gradient(
            circle at 12% 88%,
            rgba(0,103,56,.07),
            transparent 27%
        ),

        radial-gradient(
            circle at 92% 85%,
            rgba(229,176,67,.08),
            transparent 22%
        ),

        linear-gradient(
            135deg,
            #FFFFFF 0%,
            #FBFCF8 52%,
            #F3F7EC 100%
        );

    border:1px solid var(--border);

    border-radius:28px;

    padding:65px 70px 45px;

    box-shadow:
        0 8px 30px rgba(70,86,42,.07);
}


/* ==========================================================
   DECORATIVE GRID
========================================================== */

.homepage-hero::before{

    content:"";

    position:absolute;

    top:45px;
    left:45px;

    width:150px;
    height:120px;

    opacity:.55;

    background-image:

        radial-gradient(
            var(--leaf) 1.6px,
            transparent 1.6px
        );

    background-size:24px 24px;

    mask-image:

        linear-gradient(
            135deg,
            black 0%,
            transparent 90%
        );

    -webkit-mask-image:

        linear-gradient(
            135deg,
            black 0%,
            transparent 90%
        );
}


/* ==========================================================
   RIGHT GREEN GLOW
========================================================== */

.homepage-hero::after{

    content:"";

    position:absolute;

    width:430px;
    height:430px;

    right:-170px;
    top:110px;

    border-radius:50%;

    background:

        radial-gradient(
            circle,
            rgba(135,189,67,.15) 0%,
            rgba(135,189,67,.06) 38%,
            transparent 72%
        );

    pointer-events:none;
}


/* ==========================================================
   HERO CONTENT
========================================================== */

.hero-content{

    position:relative;

    z-index:5;

    text-align:center;

    max-width:1100px;

    margin:0 auto;
}


/* ==========================================================
   HERO LOGO
========================================================== */

.hero-logo{

    position:relative;

    width:155px;
    height:155px;

    margin:0 auto 22px;

    display:flex;

    align-items:center;

    justify-content:center;

    background:transparent;

    border:none;

    border-radius:0;

    overflow:visible;

    z-index:5;
}


/* ==========================================================
   LOGO GLOW
========================================================== */

.hero-logo::before{

    content:"";

    position:absolute;

    width:130px;
    height:130px;

    border-radius:50%;

    background:

        radial-gradient(
            circle,
            rgba(135,189,67,.18),
            rgba(135,189,67,.06) 45%,
            transparent 72%
        );

    filter:blur(10px);

    z-index:-1;
}


/* ==========================================================
   HERO LOGO IMAGE
========================================================== */

.hero-logo-image{

    width:155px;
    height:155px;

    object-fit:contain;

    display:block;

    background:transparent;

    filter:

        drop-shadow(
            0 10px 18px
            rgba(86,104,52,.15)
        );
}


/* ==========================================================
   KICKER
========================================================== */

.hero-kicker{

    position:relative;

    display:inline-flex;

    align-items:center;

    justify-content:center;

    padding:8px 17px;

    margin-bottom:22px;

    border-radius:30px;

    background:

        linear-gradient(
            135deg,
            #F1F5E9,
            #E8EFDB
        );

    border:1px solid #D3DEC2;

    color:var(--forest);

    font-size:12px;

    font-weight:750;

    letter-spacing:.8px;

    text-transform:uppercase;

    box-shadow:

        0 4px 12px
        rgba(86,104,52,.06);
}


/* ==========================================================
   HERO TITLE
========================================================== */

.hero-title{

    font-size:43px;

    font-weight:800;

    line-height:1.18;

    letter-spacing:-1.4px;

    color:var(--olive-dark);

    margin:0 auto;

    max-width:1000px;
}


/* ==========================================================
   HERO ACCENT
========================================================== */

.hero-accent{

    width:105px;

    height:5px;

    border-radius:10px;

    background:

        linear-gradient(
            90deg,
            var(--forest) 0%,
            var(--olive) 50%,
            var(--gold) 100%
        );

    margin:28px auto 26px;

    box-shadow:

        0 4px 10px
        rgba(86,104,52,.15);
}


/* ==========================================================
   HERO AUTHORS
========================================================== */

.hero-authors{

    display:flex;

    align-items:center;

    justify-content:center;

    gap:12px;

    font-size:18px;

    font-weight:600;

    color:#596351;
}


/* ==========================================================
   AUTHOR ICON
========================================================== */

.hero-author-icon{

    width:40px;
    height:40px;

    border-radius:50%;

    display:flex;

    align-items:center;

    justify-content:center;

    background:

        linear-gradient(
            135deg,
            #F0F5E8,
            #E5EDD8
        );

    border:1px solid #CFDABE;

    color:var(--forest);

    font-size:21px;

    box-shadow:

        0 5px 14px
        rgba(86,104,52,.10);
}


/* ==========================================================
   EXTRA DECORATIVE ELEMENT
========================================================== */

.homepage-hero .hero-content::after{

    content:"";

    position:absolute;

    width:8px;
    height:8px;

    right:-130px;
    top:120px;

    border-radius:50%;

    background:var(--gold);

    box-shadow:

        26px 45px 0 var(--leaf),
        -35px 80px 0 var(--forest),
        55px 105px 0 var(--olive-light);

    opacity:.60;
}


/* ==========================================================
   BOTTOM DECORATIVE LINE
========================================================== */

.homepage-hero{

    isolation:isolate;
}

.homepage-hero .hero-content{

    padding-bottom:25px;
}

.homepage-hero .hero-content::marker{

    display:none;
}


/* ==========================================================
   DECORATIVE BOTTOM WAVE
========================================================== */

.hero-wave{

    position:absolute;

    left:-5%;
    bottom:-5px;

    width:110%;
    height:150px;

    z-index:1;

    opacity:.9;
}

.hero-wave svg{

    width:100%;
    height:100%;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"]{

    width:245px !important;

    background:#F4F6F0;

    border-right:1px solid #DCE3D4;
}

section[data-testid="stSidebar"] > div{

    background:#F4F6F0;
}

section[data-testid="stSidebarContent"]{

    padding:18px 16px;
}


/* Hide default Streamlit multipage navigation */

[data-testid="stSidebarNav"]{

    display:none !important;
}

[data-testid="stSidebarNavItems"]{

    display:none !important;
}


/* ==========================================================
   SIDEBAR BRAND
========================================================== */

.sidebar-brand{

    display:flex;

    align-items:center;

    gap:11px;

    margin-bottom:28px;
}

.sidebar-brand-text{

    line-height:1.25;
}

.sidebar-brand-title{

    font-size:15px;

    font-weight:750;

    color:var(--olive-dark);
}

.sidebar-brand-subtitle{

    font-size:11px;

    color:#78806F;

    margin-top:3px;
}


/* ==========================================================
   SIDEBAR BRAND LOGO
========================================================== */

.sidebar-brand-logo{

    width:52px;
    height:52px;

    border-radius:14px;

    display:flex;

    align-items:center;

    justify-content:center;

    background:

        linear-gradient(
            135deg,
            var(--olive),
            var(--forest)
        );

    color:#FFFFFF;

    font-size:22px;

    font-weight:800;

    box-shadow:

        0 8px 18px
        rgba(86,104,52,.18);
}


/* ==========================================================
   SIDEBAR TITLE
========================================================== */

.sidebar-title{

    font-size:15px;

    font-weight:700;

    color:var(--olive-dark);

    margin:0 0 10px 2px;
}


/* ==========================================================
   PAGE LINK
========================================================== */

div[data-testid="stPageLink"]{

    margin-bottom:4px;
}

div[data-testid="stPageLink"] a{

    display:flex;

    align-items:center;

    min-height:40px;

    padding:8px 12px;

    border-radius:11px;

    font-size:13px;

    font-weight:500;

    color:#4E584A;

    transition:

        background .18s ease,
        color .18s ease;
}


div[data-testid="stPageLink"] a:hover{

    background:#E7EDDC;

    color:var(--forest);
}


div[data-testid="stPageLink"][aria-current="page"] a{

    background:#E0E8D3;

    color:var(--forest);

    font-weight:650;
}


/* ==========================================================
   PAGE LINK BULLET
========================================================== */

div[data-testid="stPageLink"] a::before{

    content:"•";

    color:#9AA38F;

    margin-right:9px;

    font-size:11px;
}


div[data-testid="stPageLink"][aria-current="page"] a::before{

    color:var(--gold-dark);
}


/* ==========================================================
   SIDEBAR DIVIDER
========================================================== */

.sidebar-divider{

    border-top:1px solid #D8E0D0;

    margin:20px 0 14px;
}


/* ==========================================================
   SIDEBAR FOOTER
========================================================== */

section[data-testid="stSidebar"] .stCaption{

    font-size:11px !important;

    color:#858D7E !important;
}


/* ==========================================================
   PAGE HEADER
========================================================== */

.page-breadcrumb{

    font-size:11px;

    color:#929A88;

    margin-bottom:5px;
}


.page-title{

    font-size:32px;

    font-weight:800;

    letter-spacing:-.5px;

    color:var(--olive-dark);

    margin:0;
}


.page-caption{

    font-size:13px;

    color:#747C6D;

    margin-top:5px;
}


/* ==========================================================
   CARD
========================================================== */

.card{

    background:#FFFFFF;

    border:1px solid var(--border);

    border-radius:16px;

    padding:22px;

    box-shadow:

        0 3px 12px
        rgba(86,104,52,.035);
}


.stMarkdown .card h3{

    font-size:18px !important;

    font-weight:700 !important;

    margin:0 0 10px !important;

    color:var(--olive-dark) !important;
}


.stMarkdown .card p{

    font-size:13px !important;

    line-height:1.7 !important;

    color:#626B5D !important;
}


/* ==========================================================
   STREAMLIT CONTAINER
========================================================== */

div[data-testid="stVerticalBlockBorderWrapper"]{

    border:1px solid var(--border);

    border-radius:16px;

    background:#FFFFFF;
}


/* ==========================================================
   INFO / WARNING / ERROR / SUCCESS
========================================================== */

div[data-testid="stInfo"],
div[data-testid="stWarning"],
div[data-testid="stError"],
div[data-testid="stSuccess"]{

    border-radius:12px;

    border:1px solid var(--border);

    padding:.75rem 1rem;

    font-size:13px;
}


/* ==========================================================
   METRIC
========================================================== */

[data-testid="stMetric"]{

    background:#FFFFFF;

    border:1px solid var(--border);

    border-radius:14px;

    padding:16px;

    box-shadow:

        0 2px 8px
        rgba(86,104,52,.025);
}


[data-testid="stMetricLabel"]{

    font-size:12px !important;

    color:#777F6E !important;
}


[data-testid="stMetricValue"]{

    font-size:25px !important;

    font-weight:750 !important;

    color:var(--olive-dark) !important;
}


[data-testid="stMetricDelta"]{

    font-size:12px !important;
}


[data-testid="stMetricDelta"] svg{

    display:none;
}


/* ==========================================================
   BUTTON
========================================================== */

.stButton button{

    border-radius:10px;

    border:1px solid #D5DDCC;

    font-size:13px;

    font-weight:600;

    padding:.48rem 1rem;

    color:#4B5646;

    transition:.18s ease;
}


.stButton button:hover{

    border-color:var(--olive-light);

    color:var(--forest);

    background:#F1F5E9;
}


/* ==========================================================
   PRIMARY BUTTON
========================================================== */

.stButton button[kind="primary"]{

    background:var(--forest);

    border-color:var(--forest);

    color:#FFFFFF;

    box-shadow:

        0 4px 10px
        rgba(0,103,56,.12);
}


.stButton button[kind="primary"]:hover{

    background:var(--forest-dark);

    border-color:var(--forest-dark);

    color:#FFFFFF;
}


/* ==========================================================
   INPUT
========================================================== */

.stTextInput label,
.stSelectbox label,
.stNumberInput label,
.stDateInput label,
.stRadio label,
.stCheckbox label{

    font-size:13px;

    font-weight:600;

    color:#454F42;
}


.stTextInput input,
.stNumberInput input{

    font-size:13px;

    border-radius:9px;
}


.stSelectbox div[data-baseweb="select"]{

    font-size:13px;
}


/* ==========================================================
   INPUT FOCUS
========================================================== */

.stTextInput input:focus,
.stNumberInput input:focus{

    border-color:var(--olive-light) !important;

    box-shadow:

        0 0 0 1px
        var(--olive-light) !important;
}


/* ==========================================================
   DATAFRAME
========================================================== */

[data-testid="stDataFrame"]{

    font-size:13px;
}


/* ==========================================================
   TABS
========================================================== */

button[data-baseweb="tab"]{

    font-size:13px;

    padding:9px 16px;
}


button[data-baseweb="tab"][aria-selected="true"]{

    color:var(--forest) !important;
}


/* ==========================================================
   SLIDER
========================================================== */

div[data-baseweb="slider"] div[role="slider"]{

    background:var(--forest) !important;
}


div[data-baseweb="slider"] > div > div{

    background:#DCE6D0 !important;
}


/* ==========================================================
   CHECKBOX
========================================================== */

[data-testid="stCheckbox"] label{

    color:#454F42;
}


/* ==========================================================
   RADIO
========================================================== */

div[role="radiogroup"] label{

    color:#454F42;
}


/* ==========================================================
   SCROLLBAR
========================================================== */

::-webkit-scrollbar{

    width:6px;

    height:6px;
}

::-webkit-scrollbar-track{

    background:transparent;
}

::-webkit-scrollbar-thumb{

    background:#CBD4C1;

    border-radius:10px;
}

::-webkit-scrollbar-thumb:hover{

    background:#AEBBA1;
}


/* ==========================================================
   HIDE STREAMLIT COMMUNITY CLOUD BOTTOM-RIGHT PROFILE
========================================================== */

[data-testid="stDecoration"],
button[data-testid="baseButton-stDecoration"],
button[aria-label="View profile"],
div[data-testid="stDecoration"],
span[data-testid="stDecoration"],
a[data-testid="stDecoration"],
button[data-testid="stDecoration"]{

    display:none !important;

    visibility:hidden !important;

    opacity:0 !important;

    pointer-events:none !important;
}


div.css-1cpxqw2,
div.css-1v0mbdj,
div[data-testid="stBottomRightContainer"]{

    display:none !important;
}


body > div[data-testid="stDecoration"]{

    display:none !important;
}


div[data-testid="stDecoration"] button,
div[data-testid="stDecoration"] a,
div[data-testid="stDecoration"] span{

    display:none !important;

    visibility:hidden !important;

    opacity:0 !important;

    pointer-events:none !important;
}


/* ==========================================================
   SMALL SCREEN
========================================================== */

@media(max-width:900px){

    .homepage-hero{

        padding:50px 30px 35px;

        min-height:620px;
    }

    .hero-title{

        font-size:31px;
    }

    .hero-logo{

        width:95px;
        height:95px;

        border-radius:24px;
    }

    .hero-logo-image{

        width:95px;
        height:95px;
    }

    .hero-authors{

        font-size:13px;

        flex-wrap:wrap;
    }

    .sidebar-brand-title{

        font-size:14px;
    }

    .hero-content::before,
    .hero-content::after{

        display:none;
    }
}

</style>
"""


# path halaman & label navigasi (urutan sesuai konsep tampilan awal)
NAV_ITEMS = [
    ("homepage.py", "Homepage"),
    ("pages/input_data.py", "Input Dataset"),
    ("pages/analisis_desk.py", "Analisis Deskriptif"),
    ("pages/input_params.py", "Input Parameter"),
    ("pages/output.py", "Output"),
]


def inject_custom_css():
    """Suntikkan CSS global (card, sidebar, page link, info box, container)."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar KomoditasAI."""

    with st.sidebar:
        # ======================================================
        # LOGO
        # ======================================================
        logo_path = Path("logo.png")

        logo_base64 = base64.b64encode(
            logo_path.read_bytes()
        ).decode()

        st.html(f"""
        <div style="
            display:flex;
            align-items:center;
            gap:11px;
            margin-bottom:28px;
            padding:2px 2px 4px 2px;
        ">

            <!-- LOGO -->
            <div style="
                width:50px;
                height:50px;
                flex-shrink:0;

                display:flex;
                justify-content:center;
                align-items:center;

                background:transparent;

                border-radius:12px;
            ">

                <img
                    src="data:image/png;base64,{logo_base64}"
                    style="
                        width:60px;
                        height:60px;

                        object-fit:contain;
                        display:block;

                        background:transparent;

                        filter:
                            drop-shadow(
                                0 4px 7px
                                rgba(86,104,52,.12)
                            );
                    "
                >

            </div>


            <!-- BRAND TEXT -->
            <div style="
                line-height:1.25;
                min-width:0;
            ">

                <!-- TITLE -->
                <div style="
                    font-size:15px;
                    font-weight:750;
                    color:#46562A;

                    letter-spacing:-0.15px;

                    white-space:nowrap;
                ">
                    FinanceForecast
                </div>


                <!-- SUBTITLE -->
                <div style="
                    font-size:11px;
                    color:#737C68;

                    margin-top:3px;

                    line-height:1.4;

                    white-space:wrap;
                ">
                    Prediksi Risiko Harga Akurat
                </div>

            </div>

        </div>
        """)

        # ======================================================
        # NAVIGATION
        # ======================================================

        st.markdown(
            '<div class="sidebar-title">Navigasi</div>',
            unsafe_allow_html=True,
        )

        for path, label in NAV_ITEMS:
            st.page_link(path, label=label)

        # ======================================================
        # FOOTER
        # ======================================================

        st.markdown(
            '<div class="sidebar-divider"></div>',
            unsafe_allow_html=True,
        )

        st.caption("© 2026 • FinanceForecast Dashboard")

def page_header(breadcrumb: str, title: str, caption: str = ""):
    """Header konsisten untuk tiap halaman: breadcrumb, judul, dan sub-judul."""
    st.markdown(f"`{breadcrumb}`")
    st.title(title)
    if caption:
        st.caption(caption)


def setup_page(page_title: str, page_icon: str, breadcrumb: str, title: str, caption: str = ""):
    """
    Satu pemanggilan untuk seluruh boilerplate awal sebuah halaman:
    page_config -> session_state -> CSS -> sidebar -> header.
    Dipanggil paling atas, tepat setelah import.
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    init_session_state()
    inject_custom_css()
    render_sidebar()
    page_header(breadcrumb, title, caption)


# ======================================================================
# GUARD / VALIDASI ALUR HALAMAN
# ======================================================================

def require_dataset():
    """
    Pastikan dataset & pemetaan kolom (tanggal, harga) sudah tersedia.
    Menghentikan halaman dengan pesan yang konsisten jika belum siap.
    """
    if st.session_state.get("df") is None or len(st.session_state.df) == 0:
        st.warning("Silakan unggah dan pilih dataset terlebih dahulu pada halaman **Input Dataset**.")
        st.stop()
    if st.session_state.get("date_column") is None:
        st.error("Kolom tanggal belum ditentukan pada halaman Input Dataset.")
        st.stop()
    if st.session_state.get("commodity_column") is None:
        st.error("Kolom harga belum ditentukan pada halaman Input Dataset.")
        st.stop()

    return (
        st.session_state.df.copy(),
        st.session_state.date_column,
        st.session_state.commodity_column,
    )


def require_trained_model():
    """Pastikan proses training (Input Parameter) sudah pernah dijalankan."""
    if st.session_state.get("model_result") is None:
        st.warning("Silakan jalankan proses training terlebih dahulu pada halaman **Input Parameter**.")
        st.stop()

    return (
        st.session_state.model_result,
        st.session_state.model_data,
        st.session_state.model_params,
    )


# ======================================================================
# PEMBERSIHAN DATA HARGA KOMODITAS
# ======================================================================

def clean_commodity_series(df: pd.DataFrame, commodity_column: str) -> pd.Series:
    """
    Bersihkan kolom harga komoditas dari format Rupiah (mis. "Rp12.345,67")
    menjadi nilai numerik (float), dan ubah placeholder ("-", "nan", dst) menjadi NaN.
    """
    cleaned = (
        df[commodity_column]
        .astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace(["-", "", "nan", "None"], np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ======================================================================
# FORMATTING ANGKA (GAYA INDONESIA)
# ======================================================================

def format_id(value, decimal: int = 0) -> str:
    """Format angka dengan pemisah ribuan '.' dan desimal ',' (gaya Indonesia)."""
    return f"{value:,.{decimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_rupiah(value) -> str:
    """Format angka menjadi teks Rupiah, mis. 12345.6 -> 'Rp12.345,6'."""
    text = format_id(value, decimal=2).rstrip("0").rstrip(",")
    return f"Rp{text}"


# ======================================================================
# METRIK EVALUASI MODEL (dipakai di Input Parameter & Output)
# ======================================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def mase(y_true, y_pred, insample) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)
    scale = np.mean(np.abs(np.diff(insample)))
    if scale <= 1e-12:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_prediction(y_true, y_pred, insample, model_name: str) -> dict:
    """Ringkasan metrik evaluasi (RMSE, MAE, MAPE, sMAPE, MASE, R2, Bias) untuk satu model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": mape_safe(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, insample),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }