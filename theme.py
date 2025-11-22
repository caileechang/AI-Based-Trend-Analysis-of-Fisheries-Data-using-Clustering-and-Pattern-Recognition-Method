import streamlit as st

def inject_premium_theme():
    st.markdown("""
    <style>

    /* --- GLOBAL --- */
    .stApp {
        background-color: #0d1117 !important;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    h2 { font-size: 32px !important; }
    h3 { font-size: 22px !important; }

    p {
        color: #c9d1d9 !important;
        font-size: 15px !important;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.4rem 0 1rem 0;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] * {
        font-size: 15px !important;
    }

    /* --- CLEAN CHIPS (radio buttons) --- */
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 6px !important;
    }
    div[data-testid="stRadio"] label > div {
        background-color: #161b22 !important;
        padding: 6px 14px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        transition: 0.2s;
    }
    div[data-testid="stRadio"] label > div:hover {
        border-color: rgba(255,255,255,0.18) !important;
    }

    /* --- METRIC CARDS --- */
    .metric-card {
        background-color: #161b22 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        padding: 20px 24px !important;
        transition: 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(255,255,255,0.15) !important;
        transform: translateY(-2px);
    }
    .metric-value {
        color: white !important;
        font-size: 36px !important;
        font-weight: 700 !important;
    }
    .metric-label {
        color: #8b949e !important;
        font-size: 14px !important;
    }
    .metric-delta {
        margin-top: 4px;
        font-size: 13px !important;
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background-color: #238636 !important;
        border-radius: 6px !important;
        color: white !important;
        padding: 8px 18px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #2ea043 !important;
    }

    /* --- TABLES --- */
    .stDataFrame, .stTable {
        background-color: #161b22 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
    }

    </style>
    """, unsafe_allow_html=True)



def section_header(title: str, subtitle: str | None = None, icon: str = "🎣"):
    st.markdown(f"<h2>{icon} {title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f"<p style='margin-top:-6px; margin-bottom:8px;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr>", unsafe_allow_html=True)


def metric_card(title, value, unit="tonnes", delta_html=None, accent="#4dabf7"):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-card-inner">
            <div class="metric-label">
                <span class="metric-badge-dot" style="background:{accent};"></span>
                {title}
            </div>
            <div style="display:flex; align-items:flex-end; gap:6px; margin-top:4px;">
                <span class="metric-value">{value:,.0f}</span>
                <span class="metric-unit">{unit}</span>
            </div>
            {f"<div class='metric-delta'>{delta_html}</div>" if delta_html else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def growth_html(curr, prev):
    if prev == 0:
        return "<span style='color:#888;'>No comparison</span>"
    ratio = curr / prev
    if ratio >= 1:
        return f"<span style='color:#69db7c;'>↑ {ratio:.2f}x vs previous</span>"
    else:
        return f"<span style='color:#ff6b6b;'>↓ {ratio:.2f}x vs previous</span>"
