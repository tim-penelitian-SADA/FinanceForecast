import streamlit as st
import base64
import os

from utils import (
    init_session_state,
    inject_custom_css,
    render_sidebar,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinanceForecast Dashboard",
    page_icon="🌾",
    layout="wide",
)

# ============================================================
# INITIALIZATION
# ============================================================

init_session_state()
inject_custom_css()
render_sidebar()

# ============================================================
# LOGO
# ============================================================

logo_path = os.path.join(
    os.path.dirname(__file__),
    "logo.png",
)

with open(
    logo_path,
    "rb",
) as image_file:

    logo_base64 = base64.b64encode(
        image_file.read()
    ).decode()


# ============================================================
# BREADCRUMB
# ============================================================

st.markdown(
    '<div style="font-size:12px;color:#8A8D98;margin-bottom:12px;">app / homepage</div>',
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================

st.html(f"""
<div class="homepage-hero">

    <div class="hero-content">

        <div class="hero-logo">
            <img
                src="data:image/png;base64,{logo_base64}"
                class="hero-logo-image"
            >
        </div>

        <div class="hero-kicker">
            FinanceForecast • Prediksi Risiko Harga  Akurat
        </div>

        <div class="hero-title">
            Support Vector Regression Model untuk Prediksi 
            Harga dan Risiko Akurat
        </div>

        <div class="hero-accent"></div>

        <div class="hero-authors">

            <div class="hero-author-icon">
                👥
            </div>

            <div>
                Disusun oleh:
                <b>
                    Mohammad Idhom, Trimono, Ajeng Puspa, Shafira Amanda
                </b>
            </div>

        </div>

    </div>

</div>
""")