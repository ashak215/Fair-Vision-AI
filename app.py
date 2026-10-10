from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.analytics import (
    build_detailed_gender_analytics,
    build_gender_summary,
    filter_employees,
    get_promotion_summary,
    get_salary_summary,
)
from utils.data_loader import DATA_PATH, load_employee_data
from utils.ml_model import train_promotion_model
from utils.reports import create_ml_report, create_summary_report, serialize_report_csv


st.set_page_config(
    page_title="FairVision AI | AI-Powered Gender Equality Analytics",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PURPLE = "#6C4CF1"
CYAN = "#38C7D9"
NAVY = "#11162D"
FEMALE_COLOR = "#9B7BFF"
MALE_COLOR = "#35BCD2"
OTHER_COLOR = "#A7ADC0"
LOGO_SVG = (Path(__file__).resolve().parent / "assets" / "equality-mark.svg").read_text(
    encoding="utf-8"
)
HERO_SVG = (Path(__file__).resolve().parent / "assets" / "hero-equality.svg").read_text(
    encoding="utf-8"
)
PAGES = ["Home", "Dashboard", "Gender Analytics", "AI Insights", "Reports"]

st.markdown(
    """
    <style>
    :root { --fv-purple:#7047eb; --fv-navy:#11152f; --fv-cyan:#38c7d9; }
    html, body, [class*="css"] { font-family:'Segoe UI',Arial,sans-serif; }
    .stApp { background:linear-gradient(135deg,#fbfaff 0%,#f7f6fc 58%,#f3f7fc 100%); color:#20243b; }
    [data-testid="stHeader"] { background:transparent; }
    body:has(.st-key-home_hero) [data-testid="stToolbar"] { display:none !important; }
    body:has(.st-key-home_hero) [data-testid="stSidebar"],
    body:has(.st-key-home_hero) [data-testid="stSidebarCollapsedControl"] { display:none !important; }
    body:has(.st-key-home_hero) [data-testid="stMain"] { margin-left:0 !important; }
    body:has(.st-key-home_hero) [data-testid="stMainBlockContainer"] { max-width:1320px !important; }
    body:has(.st-key-home_hero) .block-container { padding-top:.65rem !important; }
    [data-testid="stSidebar"] { background:linear-gradient(175deg,#11162d 0%,#211b48 72%,#30215b 100%); }
    [data-testid="stSidebar"] * { color:#f4f2ff; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:#ffffff12; }
    h1,h2,h3,h4 { font-family:'Segoe UI',Arial,sans-serif !important; color:#1c2038; letter-spacing:-.035em; }
    h1 { font-size:2.1rem !important; font-weight:800 !important; }
    h2 { font-size:1.48rem !important; font-weight:750 !important; }
    h3 { font-size:1.1rem !important; font-weight:700 !important; }
    p, li { color:#62677e; }
    .block-container { max-width:1440px; padding-top:4rem; padding-bottom:3.5rem; container-type:inline-size; }
    [data-testid="stMetric"] { border:1px solid #e9e6f2; background:#fff; padding:1rem 1.1rem; border-radius:17px; box-shadow:0 7px 25px #2723450a; min-height:118px; }
    [data-testid="stMetricLabel"] { color:#747990; font-weight:600; }
    [data-testid="stMetricValue"] { color:#262443; font-family:'Segoe UI',Arial,sans-serif; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color:#e8e5f0 !important; border-radius:18px; background:#fff; box-shadow:0 8px 28px #24204408; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; color:#7963df; font-size:.76rem; font-weight:800; }
    .subtle { color:#72768b; font-size:.98rem; }
    .fv-topnav { position:relative; z-index:5; min-height:60px; width:100%; display:flex; align-items:center; gap:clamp(.5rem,2vw,1.65rem); padding:.45rem clamp(.8rem,2vw,1.6rem); margin-bottom:1rem; background:#11152f; border:1px solid #ffffff13; border-radius:15px; box-shadow:0 8px 26px #11152f1c; }
    .fv-brand { display:flex; flex:1 0 205px; align-items:center; gap:.65rem; color:#fff !important; text-decoration:none !important; font-weight:800; font-size:1.04rem; white-space:nowrap; }
    .fv-brand svg { width:38px; height:38px; flex:none; }
    .fv-links { display:flex; align-items:center; justify-content:center; gap:clamp(.3rem,1.3vw,1.2rem); }
    .fv-links a { color:#d8d8eb !important; text-decoration:none !important; font-size:.88rem; font-weight:600; white-space:nowrap; padding:.5rem .1rem; border-bottom:2px solid transparent; transition:color .16s,border-color .16s; }
    .fv-links a:hover,.fv-links a[aria-current="page"] { color:#fff !important; border-bottom-color:#a78bfa; }
    .fv-get-started { margin-left:auto; flex:none; background:linear-gradient(115deg,#7047eb,#9066f5); color:white !important; border-radius:10px; padding:.66rem 1.05rem; font-size:.88rem; font-weight:750; text-decoration:none !important; box-shadow:0 6px 18px #7047eb4a; white-space:nowrap; }
    .fv-get-started:hover,.fv-cta:hover { filter:brightness(1.08); transform:translateY(-1px); }
    .st-key-home_hero { min-height:440px; padding:clamp(2rem,5vw,4.4rem); color:white; background:radial-gradient(ellipse at 91% 50%,#713cbb75 0%,transparent 47%),linear-gradient(112deg,#11152f 0%,#17173a 55%,#29194d 100%); border:1px solid #35325f; border-radius:24px; box-shadow:0 22px 58px #1b163326; overflow:hidden; position:relative; }
    .st-key-home_hero:before { position:absolute; inset:0; pointer-events:none; content:""; background:radial-gradient(ellipse at 71% 12%,#4de7e21c,transparent 29%),radial-gradient(ellipse at 95% 87%,#ed50ba1c,transparent 33%); }
    .st-key-home_hero [data-testid="stHorizontalBlock"] { position:relative; z-index:1; align-items:center; }
    .st-key-home_hero [data-testid="column"] { min-width:0; }
    .fv-hero-copy,.fv-hero-art { position:relative; z-index:1; }
    .fv-hero-brand { display:flex; align-items:center; gap:.65rem; color:#f7f5ff; font-size:.95rem; font-weight:750; letter-spacing:.035em; }
    .fv-hero-brand svg { width:38px; height:38px; }
    .fv-kicker { display:inline-flex; margin-top:1.8rem; padding:.36rem .7rem; color:#d9d0ff; background:#ffffff0d; border:1px solid #ffffff21; border-radius:999px; font-size:.69rem; font-weight:800; letter-spacing:.12em; }
    .fv-hero-copy h1 { max-width:610px; margin:.95rem 0 .85rem; color:#fff !important; font-size:clamp(1.8rem,3.3vw,3.1rem) !important; font-weight:800 !important; line-height:1.08; letter-spacing:-.055em; }
    .fv-hero-copy h1 span { color:#c7b4ff; }
    .fv-hero-copy>p { max-width:490px; margin:0; color:#d5d4e7 !important; font-size:1.04rem; line-height:1.75; }
    .fv-cta { display:inline-flex; align-items:center; gap:.5rem; margin-top:1.55rem; padding:.82rem 1.15rem; color:#fff !important; background:linear-gradient(115deg,#7047eb,#9066f5); border:1px solid #ffffff1c; border-radius:11px; text-decoration:none !important; font-size:.91rem; font-weight:750; box-shadow:0 8px 22px #7047eb4b; transition:transform .16s,filter .16s; }
    .fv-hero-art { display:flex; align-items:center; justify-content:center; min-width:0; }
    .fv-hero-art svg { display:block; width:min(100%,510px); height:auto; overflow:visible; }
    .fv-home-content { padding:clamp(2rem,4vw,3.35rem) clamp(.15rem,1.5vw,1rem) .7rem; }
    .fv-about { display:grid; grid-template-columns:1.42fr .9fr; gap:1.15rem; align-items:stretch; }
    .fv-about-copy,.fv-sdg-card { min-height:184px; padding:1.55rem 1.7rem; border:1px solid #e9e6f2; border-radius:18px; box-shadow:0 9px 28px #2420440b; }
    .fv-about-copy { background:linear-gradient(135deg,#fff,#fbfaff); }
    .fv-about-copy .fv-eyebrow,.fv-features-heading { color:#7864dc; font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .fv-about-copy h2 { margin:.55rem 0 .55rem; font-size:1.48rem !important; }
    .fv-about-copy p { max-width:720px; margin:0; color:#666a7e; font-size:.94rem; line-height:1.75; }
    .fv-sdg-card { display:flex; align-items:center; gap:1rem; background:#fff; }
    .fv-sdg-tile { width:68px; height:76px; display:flex; flex:none; flex-direction:column; justify-content:center; padding:.5rem; color:#fff; background:linear-gradient(145deg,#bd2637,#e94c58); border-radius:12px; box-shadow:0 7px 18px #c6384630; }
    .fv-sdg-tile strong { font-size:1.55rem; line-height:1; }
    .fv-sdg-tile span { margin-top:.3rem; font-size:.53rem; font-weight:800; line-height:1.08; }
    .fv-sdg-copy span { color:#8266df; font-size:.73rem; font-weight:800; letter-spacing:.035em; }
    .fv-sdg-copy p { margin:.42rem 0 0; color:#565a6f; font-size:.9rem; line-height:1.55; }
    .fv-features-heading { margin:1.65rem 0 .75rem; }
    .fv-features { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; }
    .fv-feature { min-height:126px; padding:1rem 1.05rem; background:#fff; border:1px solid #ebe8f3; border-radius:15px; box-shadow:0 7px 20px #24204408; }
    .fv-feature-top { display:flex; align-items:center; gap:.65rem; margin-bottom:.6rem; }
    .fv-feature-icon { width:36px; height:36px; display:grid; place-items:center; flex:none; color:#6549d4; background:#efebff; border:1px solid #e5dfff; border-radius:50%; }
    .fv-feature-icon svg { width:19px; height:19px; }
    .fv-feature:nth-child(2) .fv-feature-icon { color:#3289b6; background:#e8f6fc; border-color:#daf0fa; }
    .fv-feature:nth-child(3) .fv-feature-icon { color:#6b5bd0; background:#f0edff; }
    .fv-feature:nth-child(4) .fv-feature-icon { color:#3a9eaa; background:#e8f8f8; border-color:#ddf1f0; }
    .fv-feature strong { color:#2c2c47; font-size:.88rem; }
    .fv-feature p { margin:0; color:#777b8e; font-size:.79rem; line-height:1.5; }
    .fv-data-note { margin-top:1rem; color:#7b7e90; font-size:.79rem; }
    .callout { border-left:4px solid #8c75fa; background:#f0edff; border-radius:0 12px 12px 0; padding:.9rem 1rem; color:#4e4c6b; margin:.7rem 0 1rem; }
    .footer { padding-top:1rem; border-top:1px solid #e8e5f0; color:#898da0; font-size:.83rem; }
    .stButton>button, .stDownloadButton>button { border-radius:12px; font-weight:700; border:1px solid #e5e1f0; transition:all .15s ease; }
    .stButton>button[kind="primary"] { background:linear-gradient(110deg,#6c4cf1,#8b6df7); border:0; box-shadow:0 7px 18px #6c4cf133; }
    .stButton>button:hover,.stDownloadButton>button:hover { border-color:#8b76ed; transform:translateY(-1px); }
    @media (max-width: 1050px) {
      .fv-about { grid-template-columns:1.2fr .9fr; }
    }
    @container (max-width: 850px) {
      .fv-topnav { min-height:unset; flex-wrap:wrap; justify-content:space-between; padding:.65rem .8rem; }
      .fv-brand { flex:1; }
      .fv-links { order:3; width:100%; justify-content:space-between; gap:.2rem; }
    }
    @media (min-width: 761px) and (max-width: 1000px) {
      .fv-hero-copy h1 { font-size:1.65rem !important; }
    }
    @media (max-width: 760px) {
      .block-container { padding:1.2rem 1rem 2rem; }
      h1 { font-size:1.75rem !important; }
      .fv-topnav { min-height:unset; flex-wrap:wrap; justify-content:space-between; padding:.65rem .8rem; }
      .fv-brand { flex:1; }
      .fv-links { order:3; width:100%; justify-content:space-between; gap:.2rem; overflow-x:auto; }
      .fv-links a { font-size:.75rem; }
      .fv-get-started { padding:.57rem .78rem; font-size:.8rem; }
      .st-key-home_hero { min-height:unset; padding:1.65rem 1.35rem 1.3rem; }
      .st-key-home_hero [data-testid="stHorizontalBlock"] { gap:1rem; }
      .fv-kicker { margin-top:1.25rem; font-size:.61rem; }
      .fv-hero-copy h1 { font-size:clamp(2.15rem,9vw,3.1rem) !important; }
      .fv-hero-copy>p { font-size:.95rem; }
      .fv-hero-art svg { width:min(92%,390px); }
      .fv-home-content { padding:1.75rem .05rem .5rem; }
      .fv-about { grid-template-columns:1fr; }
      .fv-about-copy,.fv-sdg-card { min-height:unset; padding:1.25rem; }
      .fv-features { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    @media (max-width: 430px) {
      .fv-links a { font-size:.68rem; }
      .fv-feature { padding:.8rem; }
      .fv-feature-top { align-items:flex-start; flex-direction:column; }
      .fv-feature strong { font-size:.82rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_heading(title: str, description: str | None = None) -> None:
    st.title(title)
    if description:
        st.markdown(f'<p class="subtle">{description}</p>', unsafe_allow_html=True)


def chart_layout(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#62677e"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#eceaf2")
    fig.update_yaxes(gridcolor="#efedf4", zeroline=False)
    return fig


def add_recent_report(name: str, content: bytes, mime: str) -> None:
    reports = st.session_state.setdefault("recent_reports", [])
    reports.insert(
        0,
        {
            "name": name,
            "content": content,
            "mime": mime,
            "created_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        },
    )
    st.session_state["recent_reports"] = reports[:10]


def load_page_data(
    uploaded_file: Any,
    use_synthetic_demo: bool,
) -> tuple[pd.DataFrame, dict[str, int], str]:
    """Reuse cleaned data within this user's session until its source changes."""
    if uploaded_file is not None:
        cache_key = ("upload", uploaded_file.file_id, uploaded_file.size)
        cached = st.session_state.get("_employee_data_cache")
        if cached is not None and cached["key"] == cache_key:
            return cached["data"]
        data = load_employee_data(uploaded_file=BytesIO(uploaded_file.getvalue()))
    else:
        try:
            stat = DATA_PATH.stat()
            file_version = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            file_version = (0, 0)
        cache_key = ("bundled", str(DATA_PATH), *file_version, use_synthetic_demo)
        cached = st.session_state.get("_employee_data_cache")
        if cached is not None and cached["key"] == cache_key:
            return cached["data"]
        data = load_employee_data(use_synthetic_demo=use_synthetic_demo)

    st.session_state["_employee_data_cache"] = {"key": cache_key, "data": data}
    return data


def render_top_nav(page: str) -> None:
    links = []
    for name in PAGES:
        current = ' aria-current="page"' if name == page else ""
        links.append(
            f'<a href="?page={name}" target="_self"{current}>{name}</a>'
        )
    st.markdown(
        f'<nav class="fv-topnav" aria-label="Main navigation">'
        f'<a class="fv-brand" href="?page=Home" target="_self" aria-label="FairVision AI home">'
        f'{LOGO_SVG}<span>FairVision AI</span></a>'
        f'<div class="fv-links">{"".join(links)}</div>'
        f'<a class="fv-get-started" href="?page=Dashboard" target="_self">Get Started</a>'
        f'</nav>',
        unsafe_allow_html=True,
    )


def render_home() -> None:
    with st.container(key="home_hero"):
        left, right = st.columns([1.18, 0.82], vertical_alignment="center")
        with left:
            st.markdown(
                f"""
                <div class="fv-hero-copy">
                  <div class="fv-hero-brand">{LOGO_SVG}<span>FairVision AI</span></div>
                  <div class="fv-kicker">UNITED NATIONS · SDG 5 · GENDER EQUALITY</div>
                  <h1>AI-Powered Gender<br><span>Equality Analytics</span></h1>
                  <p>Building a fairer future through data and AI.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                '<a class="fv-cta" href="?page=Dashboard" target="_self">'
                'Explore Dashboard <span aria-hidden="true">→</span></a>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f'<div class="fv-hero-art" role="img" aria-label="Glowing balance scale illustration with equality symbols">{HERO_SVG}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <section class="fv-home-content" aria-labelledby="about-title">
          <div class="fv-about">
            <article class="fv-about-copy">
              <span class="fv-eyebrow">OUR PURPOSE</span>
              <h2 id="about-title">About Our Project</h2>
              <p>FairVision AI analyzes workplace data to explore gender-related differences in salaries, promotions, and representation. Using data analysis and machine learning, it presents insights to support fairer and more inclusive workplaces.</p>
            </article>
            <article class="fv-sdg-card" aria-label="Aligned with United Nations Sustainable Development Goal 5">
              <div class="fv-sdg-tile"><strong>5</strong><span>GENDER<br>EQUALITY</span></div>
              <div class="fv-sdg-copy"><span>Aligned with UN SDG 5</span><p>Achieve gender equality and empower all women and girls.</p></div>
            </article>
          </div>
          <h2 class="fv-features-heading">Tools for a fairer workplace</h2>
          <div class="fv-features">
            <article class="fv-feature"><div class="fv-feature-top"><span class="fv-feature-icon"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 19V9m5 10V5m5 14v-7m5 7V3M2 19h20" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></span><strong>Data Analysis</strong></div><p>Explore workplace data.</p></article>
            <article class="fv-feature"><div class="fv-feature-top"><span class="fv-feature-icon"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.64 5.64l2.12 2.12m8.48 8.48 2.12 2.12m0-12.72-2.12 2.12m-8.48 8.48-2.12 2.12M9 15l6-6m-5.5 1A2.5 2.5 0 1 0 9.5 5a2.5 2.5 0 0 0 0 5Zm5 9a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span><strong>Machine Learning</strong></div><p>Explore patterns and model limitations.</p></article>
            <article class="fv-feature"><div class="fv-feature-top"><span class="fv-feature-icon"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5v-15ZM4 20.5A2.5 2.5 0 0 1 6.5 18H20M8 7h8m-8 4h6" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg></span><strong>Interactive Dashboard</strong></div><p>Visualize key insights.</p></article>
            <article class="fv-feature"><div class="fv-feature-top"><span class="fv-feature-icon"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3v12m0 0 4-4m-4 4-4-4M5 16v4h14v-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></span><strong>Actionable Reports</strong></div><p>Support informed decisions.</p></article>
          </div>
          <p class="fv-data-note">The bundled dataset is synthetic demonstration data. Its patterns are not real-world evidence.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_data_quality(quality: dict[str, Any], source: str) -> None:
    with st.expander("Data quality & provenance", expanded=False):
        st.markdown(f"**Source:** {source}")
        st.markdown(
            f"Rows loaded: **{quality['rows_loaded']}** · Usable rows: **{quality['rows_after_cleaning']}** · "
            f"Duplicates removed: **{quality['duplicates_removed']}**"
        )
        st.markdown(
            f"Missing values filled or retained as unknown: **{quality['missing_values']}** · "
            f"Invalid numeric values set to missing: **{quality['invalid_numeric_values']}** · "
            f"Unrecognized gender labels: **{quality['unexpected_gender_labels']}** · "
            f"Unrecognized promotion labels: **{quality['unexpected_promotion_labels']}**"
        )
        st.caption(
            "Required fields are checked at load time. Invalid numeric values remain missing and are excluded only from calculations that need them; "
            "unknown categories are not silently assigned a gender or outcome."
        )


def render_dashboard(df: pd.DataFrame, quality: dict[str, Any], source: str) -> None:
    section_heading(
        "Workforce dashboard",
        "A live snapshot of representation and outcomes in the selected records.",
    )
    departments = sorted(df["department"].dropna().astype(str).unique().tolist())
    min_exp = float(df["experience_years"].min()) if df["experience_years"].notna().any() else 0.0
    max_exp = float(df["experience_years"].max()) if df["experience_years"].notna().any() else 0.0
    with st.container(border=True):
        filter_cols = st.columns([1.5, 2.5])
        selected_departments = filter_cols[0].multiselect(
            "Department", departments, default=departments, key="dash_departments"
        )
        if min_exp < max_exp:
            selected_experience = filter_cols[1].slider(
                "Experience (years)",
                min_value=min_exp,
                max_value=max_exp,
                value=(min_exp, max_exp),
                step=0.5,
                key="dash_experience",
            )
        else:
            selected_experience = (min_exp, max_exp)
            filter_cols[1].caption(f"Experience: {min_exp:g} years")
    filtered = filter_employees(
        df,
        departments=selected_departments,
        experience_range=selected_experience,
    )
    if filtered.empty:
        st.warning("No employees match the selected filters. Adjust the filters to view metrics.")
        render_data_quality(quality, source)
        return

    total = len(filtered)
    known_gender = filtered[filtered["gender"].isin(["Female", "Male"])]
    female_count = int((filtered["gender"] == "Female").sum())
    male_count = int((filtered["gender"] == "Male").sum())
    valid_promotions = filtered[filtered["promotion_status"].isin(["Yes", "No"])]
    promotion_rate = (
        float((valid_promotions["promotion_status"] == "Yes").mean() * 100)
        if not valid_promotions.empty
        else None
    )
    metrics = st.columns(5)
    metrics[0].metric("Total employees", f"{total:,}")
    metrics[1].metric("Female employees", f"{female_count:,}", f"{female_count / total:.1%} of filtered")
    metrics[2].metric("Male employees", f"{male_count:,}", f"{male_count / total:.1%} of filtered")
    metrics[3].metric("Other / unknown", f"{total - len(known_gender):,}")
    metrics[4].metric("Promotion rate", f"{promotion_rate:.1f}%" if promotion_rate is not None else "N/A")
    st.caption(f"Showing {len(filtered):,} of {len(df):,} usable rows · Dataset: **{source}**")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Gender representation by department")
        representation = filtered.groupby(["department", "gender"], observed=True).size().reset_index(name="Employees")
        if representation.empty:
            st.info("There is no gender or department data to chart.")
        else:
            fig = px.bar(
                representation,
                x="department",
                y="Employees",
                color="gender",
                barmode="group",
                color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR, "Other/Unknown": OTHER_COLOR},
                labels={"department": "Department"},
            )
            st.plotly_chart(chart_layout(fig, 370), use_container_width=True)
    with right:
        st.markdown("#### Promotion outcomes by gender")
        promotion = get_promotion_summary(filtered)
        promotion = promotion[promotion["gender"].isin(["Female", "Male", "Other/Unknown"])]
        if promotion.empty:
            st.info("No recognized promotion outcomes are available in this filtered data.")
        else:
            fig = px.bar(
                promotion,
                x="gender",
                y="promotion_rate",
                color="gender",
                text=promotion["promotion_rate"].map(lambda value: f"{value:.1f}%"),
                color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR, "Other/Unknown": OTHER_COLOR},
                labels={"gender": "Gender", "promotion_rate": "Promotion rate (%)"},
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(chart_layout(fig, 370), use_container_width=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Average salary by gender")
        salary = get_salary_summary(filtered)
        salary = salary[salary["gender"].isin(["Female", "Male", "Other/Unknown"])]
        if salary.empty:
            st.info("No valid salary values are available in this filtered data.")
        else:
            fig = px.bar(
                salary,
                x="gender",
                y="average_salary",
                color="gender",
                color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR, "Other/Unknown": OTHER_COLOR},
                labels={"gender": "Gender", "average_salary": "Average salary"},
            )
            fig.update_yaxes(tickprefix="$", separatethousands=True)
            st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    with right:
        st.markdown("#### Employee distribution by department")
        distribution = filtered.groupby("department", observed=True).size().reset_index(name="Employees")
        fig = px.pie(
            distribution,
            names="department",
            values="Employees",
            hole=0.62,
            color_discrete_sequence=[PURPLE, CYAN, "#A58BFA", "#69D5D1", "#4E5C9B"],
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    st.markdown(
        '<div class="callout"><strong>Interpret with care.</strong> These are descriptive comparisons in the selected records. '
        "Differences do not by themselves establish discrimination; job level, tenure, hours, location, sample size, and other context may matter.</div>",
        unsafe_allow_html=True,
    )
    render_data_quality(quality, source)


def render_gender_analytics(df: pd.DataFrame, source: str) -> None:
    section_heading(
        "Gender analytics",
        "Explore pay, promotions, roles, and experience. All comparisons are descriptive and depend on the data provided.",
    )
    departments = sorted(df["department"].dropna().astype(str).unique().tolist())
    selected = st.multiselect(
        "Filter departments",
        departments,
        default=departments,
        key="analytics_departments",
    )
    filtered = filter_employees(df, departments=selected)
    if filtered.empty:
        st.info("Select at least one department with data.")
        return
    salary_tab, promotion_tab, role_tab, experience_tab = st.tabs(
        ["Salary comparison", "Promotion outcomes", "Job role distribution", "Experience analysis"]
    )
    with salary_tab:
        st.markdown("#### Salary comparison")
        salary_summary = get_salary_summary(filtered)
        if salary_summary.empty:
            st.info("No valid salary data is available.")
        else:
            st.dataframe(salary_summary, use_container_width=True, hide_index=True)
            female = salary_summary.loc[salary_summary["gender"] == "Female", "average_salary"]
            male = salary_summary.loc[salary_summary["gender"] == "Male", "average_salary"]
            if not female.empty and not male.empty:
                female_avg = float(female.iloc[0])
                male_avg = float(male.iloc[0])
                absolute = male_avg - female_avg
                percent = (absolute / male_avg * 100) if male_avg else None
                st.markdown(
                    f"**Average salary difference (Male − Female):** ${absolute:,.2f} "
                    f"({percent:.2f}% of the male average)." if percent is not None else
                    f"**Average salary difference (Male − Female):** ${absolute:,.2f} (percentage unavailable because male average is zero)."
                )
                st.caption(
                    "The percentage formula is (male average − female average) ÷ male average × 100. "
                    "Median values are also provided to reduce sensitivity to outliers."
                )
            fig = px.bar(
                salary_summary,
                x="gender",
                y=["average_salary", "median_salary"],
                barmode="group",
                color_discrete_sequence=[PURPLE, CYAN],
                labels={"value": "Salary", "gender": "Gender", "variable": "Measure"},
            )
            fig.update_yaxes(tickprefix="$", separatethousands=True)
            st.plotly_chart(chart_layout(fig), use_container_width=True)
            st.download_button(
                "Download salary summary CSV",
                serialize_report_csv(
                    salary_summary,
                    source,
                    {"departments": selected, "experience": ("all", "all")},
                ),
                "fairvision_salary_summary.csv",
                "text/csv",
                key="salary_csv",
            )
    with promotion_tab:
        st.markdown("#### Promotion outcomes")
        promotion_summary = get_promotion_summary(filtered)
        if promotion_summary.empty:
            st.info("There are no recognized promotion outcomes to compare.")
        else:
            comparable = promotion_summary[promotion_summary["gender"].isin(["Female", "Male"])]
            group_sizes = comparable.set_index("gender")["employees_with_recorded_outcome"].to_dict()
            sufficient_data = all(group_sizes.get(gender, 0) >= 5 for gender in ("Female", "Male"))
            if sufficient_data:
                st.dataframe(promotion_summary, use_container_width=True, hide_index=True)
                fig = px.bar(
                    promotion_summary,
                    x="gender",
                    y="promotion_rate",
                    color="gender",
                    color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR, "Other/Unknown": OTHER_COLOR},
                    labels={"gender": "Gender", "promotion_rate": "Promotion rate (%)"},
                    text=promotion_summary["promotion_rate"].map(lambda value: f"{value:.1f}%"),
                )
                fig.update_traces(textposition="outside")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(chart_layout(fig), use_container_width=True)
                promotion_export = promotion_summary
            else:
                st.warning(
                    "Promotion-rate comparisons are hidden unless both Female and Male groups have at least "
                    "5 recorded Yes/No outcomes. The available counts are shown without rates."
                )
                promotion_export = promotion_summary[
                    ["gender", "employees_with_recorded_outcome"]
                ]
                st.dataframe(promotion_export, use_container_width=True, hide_index=True)
            st.download_button(
                "Download promotion summary CSV",
                serialize_report_csv(
                    promotion_export,
                    source,
                    {"departments": selected, "experience": ("all", "all")},
                ),
                "fairvision_promotion_summary.csv",
                "text/csv",
                key="promotion_csv",
            )
            st.caption("Rate = employees with a recorded 'Yes' outcome ÷ employees with a recognized Yes/No outcome.")
    with role_tab:
        st.markdown("#### Gender representation by department")
        role_summary = build_gender_summary(filtered)
        st.dataframe(role_summary, use_container_width=True, hide_index=True)
        fig = px.bar(
            role_summary,
            x="department",
            y="employees",
            color="gender",
            barmode="group",
            color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR, "Other/Unknown": OTHER_COLOR},
            labels={"department": "Department", "employees": "Employees"},
        )
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
        st.download_button(
            "Download representation summary CSV",
            serialize_report_csv(
                role_summary,
                source,
                {"departments": selected, "experience": ("all", "all")},
            ),
            "fairvision_representation_summary.csv",
            "text/csv",
            key="representation_csv",
        )
    with experience_tab:
        st.markdown("#### Experience-level comparisons")
        experience_data = filtered.dropna(subset=["experience_years"]).copy()
        if experience_data.empty:
            st.info("No valid experience values are available.")
        else:
            experience_data["experience_band"] = pd.cut(
                experience_data["experience_years"],
                bins=[-0.01, 5, 10, 15, float("inf")],
                labels=["0–5 years", ">5–10 years", ">10–15 years", ">15 years"],
            )
            experience_summary = (
                experience_data.groupby(["experience_band", "gender"], observed=True)
                .agg(employees=("gender", "size"), average_salary=("salary", "mean"))
                .reset_index()
            )
            st.dataframe(experience_summary, use_container_width=True, hide_index=True)
            fig = px.bar(
                experience_summary,
                x="experience_band",
                y="employees",
                color="gender",
                barmode="group",
                color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR, "Other/Unknown": OTHER_COLOR},
                labels={"experience_band": "Experience", "employees": "Employees"},
            )
            st.plotly_chart(chart_layout(fig), use_container_width=True)
            st.download_button(
                "Download experience summary CSV",
                serialize_report_csv(
                    experience_summary,
                    source,
                    {"departments": selected, "experience": ("all", "all")},
                ),
                "fairvision_experience_summary.csv",
                "text/csv",
                key="experience_csv",
            )
    st.markdown(
        '<div class="callout"><strong>Important:</strong> A difference in this dataset does not automatically prove discrimination. '
        "Results are descriptive, may be affected by data quality or confounding factors, and should be reviewed by people with appropriate organizational context.</div>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_model(df: pd.DataFrame) -> dict[str, Any]:
    return train_promotion_model(df)


def render_ai_insights(df: pd.DataFrame, source: str) -> dict[str, Any]:
    section_heading(
        "AI insights",
        "An experimental model for recorded promotion outcomes, with a held-out evaluation and explicit limitations.",
    )
    st.markdown(
        '<div class="callout"><strong>Responsible use.</strong> This demonstration is not a hiring or promotion recommendation. '
        "It must not be used to make decisions about individuals. Predictions require human review and are not evidence of causality.</div>",
        unsafe_allow_html=True,
    )
    with st.spinner("Checking whether the current data supports an evaluation…"):
        result = cached_model(df)
    if not result["available"]:
        st.warning(result["reason"])
        st.markdown("### Descriptive insights")
        promotion = get_promotion_summary(df)
        if promotion.empty:
            st.info("No valid promotion outcomes are available for a descriptive comparison.")
        else:
            st.dataframe(promotion, use_container_width=True, hide_index=True)
        st.caption(f"Data source: {source}. No model was trained for this dataset.")
        return result

    st.success(f"Holdout evaluation completed on {result['test_size']} records · Data source: {source}")
    metrics = st.columns(4)
    metrics[0].metric("Accuracy", f"{result['accuracy']:.1%}")
    metrics[1].metric("Precision", f"{result['precision']:.1%}")
    metrics[2].metric("Recall", f"{result['recall']:.1%}")
    metrics[3].metric("F1 score", f"{result['f1']:.1%}")
    st.caption(
        f"Train/test split: {result['train_size']}/{result['test_size']} records · Stratified split · "
        "Gender excluded from model inputs · Random forest, fixed random seed."
    )
    matrix = pd.DataFrame(
        result["confusion_matrix"],
        index=[f"Actual {label}" for label in result["labels"]],
        columns=[f"Predicted {label}" for label in result["labels"]],
    )
    col_a, col_b = st.columns([1, 1.25], gap="large")
    with col_a:
        st.markdown("#### Confusion matrix")
        st.dataframe(matrix, use_container_width=True)
    with col_b:
        st.markdown("#### Model feature importance")
        importance = result["feature_importance"]
        fig = px.bar(
            importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            color_discrete_sequence=[PURPLE],
            labels={"importance": "Importance", "feature": "Encoded feature"},
        )
        st.plotly_chart(chart_layout(fig, max(300, min(520, 28 * len(importance)))), use_container_width=True)
        st.caption("Random-forest impurity-based importance is descriptive, can be biased, and does not imply causation.")

    st.markdown("### Experimental prediction interface")
    st.caption(
        "Try a hypothetical record using eligible inputs only. Gender is excluded from this individual prediction model."
    )
    pipeline = result["pipeline"]
    feature_cols = result["feature_columns"]
    defaults = result["defaults"]
    inputs: dict[str, Any] = {}
    input_columns = st.columns(min(3, max(1, len(feature_cols))))
    for index, name in enumerate(feature_cols):
        spec = result["feature_specs"][name]
        target_column = input_columns[index % len(input_columns)]
        with target_column:
            if spec["kind"] == "numeric":
                inputs[name] = st.number_input(
                    name.replace("_", " ").title(),
                    value=float(defaults[name]),
                    key=f"prediction_{name}",
                )
            else:
                options = spec["options"]
                default_value = defaults[name]
                if default_value not in options:
                    options = [default_value] + options
                inputs[name] = st.selectbox(
                    name.replace("_", " ").title(),
                    options=options,
                    index=options.index(default_value),
                    key=f"prediction_{name}",
                )
    if st.button("Run experimental prediction", type="primary"):
        try:
            predicted = pipeline.predict(pd.DataFrame([inputs], columns=feature_cols))[0]
            probability = pipeline.predict_proba(pd.DataFrame([inputs], columns=feature_cols))[0]
            classes = list(pipeline.named_steps["classifier"].classes_)
            yes_probability = float(probability[classes.index("Yes")]) if "Yes" in classes else None
            st.info(
                f"Experimental model output: **{predicted}**"
                + (f" · Estimated Yes probability: **{yes_probability:.1%}**" if yes_probability is not None else "")
            )
        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Could not generate this experimental prediction: {error}")
    st.caption(
        "Metrics describe only this held-out sample and may be unstable for small or synthetic data. "
        "A high score does not establish fairness, validity, or generalization."
    )
    return result


def render_reports(
    df: pd.DataFrame,
    source: str,
    ml_result: dict[str, Any] | None,
    dashboard_filters: dict[str, Any],
    model_df: pd.DataFrame,
) -> None:
    section_heading(
        "Reports",
        "Generate downloadable files from the current dataset and dashboard filters.",
    )
    st.markdown(
        f"**Current data source:** {source}  \n"
        f"**Dashboard filters:** departments: {', '.join(dashboard_filters['departments']) or 'none'}; "
        f"experience: {dashboard_filters['experience'][0]}–{dashboard_filters['experience'][1]} years."
    )
    st.caption(
        "Every generated report includes a generation date, data-source label, applied filters, and limitations. "
        "The bundled file is synthetic demonstration data; uploaded data is not independently verified. "
        "The ML evaluation uses all usable rows and is not filtered by the dashboard selections."
    )
    if st.button("Generate current report pack", type="primary"):
        report_date = datetime.now()
        if ml_result is None and model_df["promotion_status"].isin(["Yes", "No"]).any():
            with st.spinner("Evaluating the optional promotion model for its report…"):
                ml_result = cached_model(model_df)
        summary_pdf = create_summary_report(
            df,
            source,
            dashboard_filters,
            report_date=report_date,
        )
        detailed = build_detailed_gender_analytics(df)
        reports = [
            ("fairvision_summary_report.pdf", summary_pdf, "application/pdf"),
            (
                "fairvision_detailed_gender_analytics.csv",
                serialize_report_csv(detailed, source, dashboard_filters, report_date),
                "text/csv",
            ),
            (
                "fairvision_filtered_dataset.csv",
                serialize_report_csv(df, source, dashboard_filters, report_date),
                "text/csv",
            ),
        ]
        if ml_result and ml_result.get("available"):
            reports.append(
                (
                    "fairvision_ml_evaluation_report.pdf",
                    create_ml_report(
                        ml_result,
                        source,
                        filters={"departments": [], "experience": ("all", "all")},
                        report_date=report_date,
                    ),
                    "application/pdf",
                )
            )
        for name, content, mime in reports:
            add_recent_report(name, content, mime)
        st.success(f"Generated {len(reports)} downloadable report(s).")

    st.markdown("### Recent reports")
    available_reports = st.session_state.get("recent_reports", [])
    if available_reports:
        for index, report in enumerate(available_reports):
            col_name, col_time, col_button = st.columns([2, 1.4, 1])
            col_name.markdown(f"**{report['name']}**")
            col_time.caption(report["created_at"])
            col_button.download_button(
                "Download",
                report["content"],
                report["name"],
                report["mime"],
                key=f"recent_download_{index}_{report['name']}",
            )
    else:
        st.info("Generate a report pack to create the PDF and CSV files for download.")
    st.markdown("### Report contents")
    st.markdown(
        "- **Summary PDF:** headline workforce metrics, salary and promotion summaries, data provenance, filters, and limitations.\n"
        "- **Detailed analytics CSV:** department representation plus gender-level salary and promotion summaries.\n"
        "- **Filtered dataset CSV:** rows selected by the dashboard department and experience filters.\n"
        "- **ML evaluation PDF:** holdout scores, confusion matrix, feature importance, and model limitations (only when model evaluation is available)."
    )


def main() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;gap:.65rem;align-items:center;padding:.35rem 0 1.15rem">'
            f'{LOGO_SVG}<div><div style="font-family:Segoe UI;font-weight:800;font-size:1.12rem">FairVision AI</div>'
            '<div style="font-size:.73rem;color:#c4bee4">Building a fairer future</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        uploaded = st.file_uploader("Upload an alternative CSV", type=["csv"])
        st.caption("Expected fields: gender, age, department, experience_years, salary, promotion_status.")
        use_demo = False
        if not DATA_PATH.exists() and uploaded is None:
            use_demo = st.checkbox("Use a clearly labeled synthetic demo dataset", value=True)
        st.markdown("---")
        st.markdown(
            '<div style="font-size:.8rem;color:#d3cdec">UNITED NATIONS<br><strong>SDG 5 · GENDER EQUALITY</strong></div>',
            unsafe_allow_html=True,
        )

    requested_page = st.query_params.get("page", "Home")
    page = requested_page if requested_page in PAGES else "Home"
    render_top_nav(page)

    if not DATA_PATH.exists() and uploaded is None and not use_demo:
        st.title("FairVision AI")
        st.error("The bundled CSV was not found. Upload a CSV in the sidebar, or enable the synthetic demo option.")
        st.info(f"Expected file location: `{DATA_PATH}`")
        return
    try:
        df, quality, source = load_page_data(uploaded, use_demo)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        st.title("FairVision AI")
        st.error(f"Unable to load the employee dataset: {error}")
        st.info("Upload a readable CSV with all required columns in the sidebar.")
        return
    if df.empty:
        st.title("FairVision AI")
        st.warning("The CSV loaded, but no rows remain after duplicate removal and validation.")
        render_data_quality(quality, source)
        return

    available_departments = set(df["department"].dropna().astype(str).unique())
    if "dash_departments" in st.session_state:
        current_departments = st.session_state["dash_departments"]
        if not set(current_departments).issubset(available_departments):
            valid_departments = [
                department
                for department in current_departments
                if department in available_departments
            ]
            st.session_state["dash_departments"] = valid_departments or sorted(
                available_departments
            )
    experience_values = df["experience_years"].dropna()
    if not experience_values.empty and "dash_experience" in st.session_state:
        minimum = float(experience_values.min())
        maximum = float(experience_values.max())
        old_low, old_high = st.session_state["dash_experience"]
        low = max(minimum, min(maximum, float(old_low)))
        high = max(minimum, min(maximum, float(old_high)))
        st.session_state["dash_experience"] = (
            (minimum, maximum) if low > high else (low, high)
        )
    elif experience_values.empty and "dash_experience" in st.session_state:
        st.session_state["dash_experience"] = (0.0, 0.0)

    if page == "Home":
        render_home()
    elif page == "Dashboard":
        render_dashboard(df, quality, source)
    elif page == "Gender Analytics":
        render_gender_analytics(df, source)
    elif page == "AI Insights":
        render_ai_insights(df, source)
    elif page == "Reports":
        report_df = df
        exp_values = df["experience_years"].dropna()
        exp_range = (
            (float(exp_values.min()), float(exp_values.max()))
            if not exp_values.empty
            else (0, 0)
        )
        selected_departments = sorted(df["department"].dropna().astype(str).unique().tolist())
        if "dash_departments" in st.session_state:
            selected_departments = st.session_state["dash_departments"]
        if "dash_experience" in st.session_state:
            exp_range = st.session_state["dash_experience"]
        report_df = filter_employees(
            df,
            departments=selected_departments,
            experience_range=exp_range,
        )
        render_reports(
            report_df,
            source,
            None,
            {"departments": selected_departments, "experience": exp_range},
            df,
        )

    st.markdown(
        '<div class="footer">FairVision AI · SDG 5 Gender Equality · Descriptive analytics only; not a substitute for human review.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
