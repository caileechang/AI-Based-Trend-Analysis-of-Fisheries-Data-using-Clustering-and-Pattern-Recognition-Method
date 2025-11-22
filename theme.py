import streamlit as st

def inject_premium_theme():
    st.markdown("""
    <style>
    :root {
      --bg-main: #05070b;
      --bg-elevated: #111318;
      --bg-elevated-soft: rgba(255,255,255,0.03);
      --accent: #4dabf7;
      --accent-soft: rgba(77,171,247,0.15);
      --text-main: #f5f5f7;
      --text-muted: #a0a4b8;
      --border-subtle: rgba(255,255,255,0.08);
      --radius-lg: 18px;
      --shadow-soft: 0 18px 45px rgba(0,0,0,0.55);
    }

    /* App background + font */
    .stApp {
      background: radial-gradient(circle at top left, #182233 0, #05070b 45%);
      color: var(--text-main);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
    }

    .block-container {
      padding-top: 1.8rem;
      padding-bottom: 3rem;
      max-width: 1250px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0e1117 0, #05070b 60%);
      border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] * {
      color: var(--text-main);
      font-size: 15px;
    }

    /* Headings + body text */
    h1, h2, h3 {
      font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      letter-spacing: 0.02em;
    }
    h2 {
      font-size: 34px !important;
      font-weight: 700 !important;
      margin-bottom: 0.35rem;
    }
    h3 {
      font-size: 24px !important;
      font-weight: 600 !important;
      margin-bottom: 0.25rem;
    }
    p {
      font-size: 15px;
      color: var(--text-muted);
    }

    hr {
      border: none;
      border-top: 1px solid rgba(255,255,255,0.06);
      margin: 1.4rem 0 1rem 0;
    }

    /* Metric cards */
    .metric-card {
      background: var(--bg-elevated-soft);
      border-radius: var(--radius-lg);
      border: 1px solid var(--border-subtle);
      padding: 20px 22px 18px 22px;
      box-shadow: var(--shadow-soft);
      position: relative;
      overflow: hidden;
      transition: all 0.22s ease-out;
    }
    .metric-card::before {
      content: "";
      position: absolute;
      inset: -40%;
      background: radial-gradient(circle at top left, var(--accent-soft), transparent 60%);
      opacity: 0.9;
      pointer-events: none;
    }
    .metric-card-inner {
      position: relative;
      z-index: 2;
    }
    .metric-label {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-muted);
      margin-bottom: 2px;
    }
    .metric-value {
      font-size: 38px;
      font-weight: 700;
      color: var(--text-main);
      line-height: 1.1;
    }
    .metric-unit {
      font-size: 16px;
      color: var(--text-muted);
      margin-left: 6px;
    }
    .metric-delta {
      margin-top: 6px;
      font-size: 14px;
    }
    .metric-badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      display: inline-block;
      margin-right: 6px;
      background: var(--accent);
    }
    .metric-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 24px 55px rgba(0,0,0,0.70);
      border-color: rgba(148, 187, 233, 0.55);
    }

    /* Radio buttons as chips */
    div[data-testid="stRadio"] > label {
      font-weight: 600;
      margin-bottom: 4px;
    }
    div[data-testid="stRadio"] > div {
      flex-direction: row;
      gap: 0.5rem;
    }
    div[data-testid="stRadio"] label > div {
      background: #151821;
      padding: 6px 14px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.1);
    }

    /* Buttons */
    .stButton>button {
      border-radius: 999px;
      background: linear-gradient(135deg, #4dabf7, #9775fa);
      color: white;
      border: none;
      padding: 0.45rem 1.2rem;
      font-weight: 600;
    }
    .stButton>button:hover {
      filter: brightness(1.07);
      transform: translateY(-1px);
      box-shadow: 0 10px 25px rgba(0,0,0,0.45);
    }

    /* Tables */
    .stDataFrame, .stTable {
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
      background: var(--bg-elevated);
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
