from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------
# AquaPalladium | Desalination Analytics
# Local-first portfolio prototype. No live plant connection is assumed.
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
ANOMALY_PATH = (
    ROOT
    / "outputs"
    / "stable_anomaly_investigation"
    / "all_investigation_results.csv"
)
RULE_PATH = ROOT / "outputs" / "anomaly_detection" / "all_anomaly_results.csv"
QUALITY_PATH = ROOT / "outputs" / "data_quality" / "data_quality_report.csv"

st.set_page_config(
    page_title="AquaPalladium | Desalination Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

      :root {
        --canvas: #f5f5f7; --surface: #ffffff; --ink: #1d1d1f;
        --muted: #6e6e73; --line: #e8e8ed; --blue: #0071e3;
        --teal: #007f86; --soft: #eef7f7;
      }
      html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
      .stApp { background: var(--canvas); color: var(--ink); }
      /* Explicit contrast rules: Streamlit dark-mode preferences must not make light text sit on light cards. */
      .stApp, .stApp p, .stApp label, .stApp li,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] span,
      .stApp [data-testid="stWidgetLabel"] p,
      .stApp [data-testid="stCaptionContainer"] {
        color: #1d1d1f;
      }
      .stApp [data-testid="stCaptionContainer"] { color: #6e6e73 !important; }
      .stApp [data-testid="stAlert"] p,
      .stApp [data-testid="stAlert"] span { color: #3d3215 !important; }
      .stApp [data-testid="stAlert"] a { color: #005bb5 !important; }
      .stApp input,
      .stApp textarea {
        color: #00000 !important;
      }

      /* Use a light select control so its value and options stay readable. */
      .stApp [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1d1d1f !important;
        border-color: #d2d2d7 !important;
      }
      .stApp [data-baseweb="select"] input,
      .stApp [data-baseweb="select"] [role="combobox"],
      .stApp [data-baseweb="select"] [data-testid="stMarkdownContainer"],
      .stApp [data-baseweb="select"] span {
        color: #1d1d1f !important;
        -webkit-text-fill-color: #1d1d1f !important;
      }
      .stApp [data-baseweb="popover"],
      .stApp [data-baseweb="menu"],
      .stApp [role="listbox"] {
        background-color: #ffffff !important;
        color: #1d1d1f !important;
      }
      .stApp [data-baseweb="popover"] *,
      .stApp [data-baseweb="menu"] *,
      .stApp [role="listbox"] * {
        color: #1d1d1f !important;
        -webkit-text-fill-color: #000000 !important;
      }
      .stApp [data-testid="stDataFrame"] * { color: #1d1d1f; }
      [data-testid="stHeader"] { background: rgba(245,245,247,.88); backdrop-filter: blur(16px); }
      [data-testid="stSidebar"] {
        background: #fbfbfd; border-right: 1px solid #e8e8ed;
      }
      [data-testid="stSidebar"] * { color: #1d1d1f !important; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #6e6e73 !important; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #6e6e73 !important; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 9px; padding: .3rem .5rem; transition: background .15s ease;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #ededf0; }
      .block-container {
        width: 100%; max-width: 1500px; padding: 1.6rem clamp(1rem, 3vw, 2.5rem) 3rem;
        overflow-wrap: anywhere;
      }
      [data-testid="stMainBlockContainer"], [data-testid="stVerticalBlock"] { min-width: 0; }
      [data-testid="stMarkdownContainer"], [data-testid="stAlert"] { min-width: 0; }
      [data-testid="stAlert"] p { white-space: normal !important; overflow-wrap: anywhere; line-height: 1.55; }
      [data-testid="stPlotlyChart"] { min-width: 0; }
      .stDataFrame, [data-testid="stDataFrame"] { max-width: 100%; }
      h1, h2, h3, h4 {
        font-family: Manrope, Inter, -apple-system, sans-serif !important;
        color: var(--ink) !important; letter-spacing: -.035em;
      }
      h1 { font-size: 2.15rem !important; font-weight: 800 !important; }
      h2 { font-size: 1.45rem !important; font-weight: 700 !important; }
      h3 { font-size: 1.08rem !important; font-weight: 700 !important; letter-spacing: -.02em; }
      p, label, span, li { -webkit-font-smoothing: antialiased; }
      .hero {
        position: relative; isolation: isolate; overflow: hidden; padding: 2rem clamp(1.2rem, 3vw, 2.2rem);
        border-radius: 24px; background: #ffffff; color: #1d1d1f;
        border: 1px solid rgba(0,0,0,.045);
        box-shadow: 0 8px 32px rgba(0,0,0,.035); margin-bottom: 1.25rem;
        width: 100%; box-sizing: border-box;
      }
      .hero:after {
        content: ""; position: absolute; z-index: -1; width: 240px; height: 240px; right: -70px; top: -105px;
        border-radius: 50%; background: radial-gradient(circle, rgba(0,127,134,.10), rgba(0,113,227,.025) 58%, transparent 72%);
        pointer-events: none;
      }
      .hero h1 { color: #1d1d1f !important; margin: 0; font-size: 2.45rem !important; }
      .hero p { color: #515154 !important; margin: .6rem 0 0; font-size: 1rem; line-height: 1.6; max-width: 62ch; overflow-wrap: anywhere; }
      .hero-kicker { text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; color: var(--teal); font-weight: 700; margin-bottom: .65rem; }
      .metric-card {
        background: var(--surface); border: 1px solid rgba(0,0,0,.045); border-radius: 18px;
        padding: 1.15rem 1.2rem; min-height: 120px;
        box-shadow: 0 5px 22px rgba(0,0,0,.025);
      }
      .metric-label { color: var(--muted); font-size: .8rem; font-weight: 500; }
      .metric-value { color: var(--ink); font-family: Manrope, Inter, sans-serif; font-size: 1.9rem; font-weight: 800; margin-top: .45rem; line-height: 1.1; letter-spacing: -.04em; }
      .metric-help { color: #86868b; font-size: .73rem; margin-top: .45rem; }
      .section-note { color: var(--muted); font-size: .92rem; line-height: 1.65; margin-top: -.35rem; margin-bottom: 1.25rem; }
      div[data-testid="stMetric"] {
        background: var(--surface); padding: 18px 20px; border: 1px solid rgba(0,0,0,.045);
        border-radius: 18px; box-shadow: 0 5px 22px rgba(0,0,0,.025);
      }
      div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
      div[data-testid="stMetricValue"] { color: var(--ink); font-family: Manrope, Inter, sans-serif; letter-spacing: -.035em; }
      div[data-testid="stPlotlyChart"] {
        background: var(--surface); border: 1px solid rgba(0,0,0,.045); border-radius: 18px;
        padding: .35rem .4rem .1rem; box-shadow: 0 5px 22px rgba(0,0,0,.025);
      }
      div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 13px; overflow: hidden; }
      div[data-testid="stAlert"] {
        border-radius: 14px; border: 1px solid #eadca4;
        background: #fff8dc !important; color: #3d3215 !important;
      }
      .stButton button, .stDownloadButton button {
        border-radius: 999px; font-weight: 600; min-height: 2.6rem;
      }
      .stButton button[kind="primary"] { background: var(--blue); border-color: var(--blue); }
      [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        border-radius: 11px !important; border-color: #d2d2d7 !important;
      }
      [data-testid="stCaptionContainer"] { color: var(--muted); }
      hr { border-color: var(--line); }
      @media (max-width: 1000px) {
        .block-container { padding: 1rem .9rem 2rem; }
        .hero { padding: 1.35rem; border-radius: 18px; }
        .hero h1 { font-size: clamp(1.6rem, 5vw, 2.05rem) !important; overflow-wrap: anywhere; }
        .hero p { font-size: .95rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .75rem !important; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
          min-width: min(100%, 260px) !important; flex: 1 1 260px !important;
        }
        div[data-testid="stPlotlyChart"] { padding: .15rem; }
        div[data-testid="stPlotlyChart"] > div { max-width: 100%; }
      }
      @media (max-width: 520px) {
        .block-container { padding-left: .65rem; padding-right: .65rem; }
        .hero { padding: 1.1rem; }
        .hero h1 { font-size: 1.55rem !important; }
        div[data-testid="stMetric"] { padding: 12px; }
      }


/* Make the sidebar expand/collapse control visible */
[data-testid="stSidebarCollapsedControl"] {
    background-color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    z-index: 1000 !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    background-color: #007f86 !important;
    color: #ffffff !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapsedControl"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
    stroke: #ffffff !important;
}


/* Keep the sidebar visible and hide its collapse control */
[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 16rem !important;
    visibility: visible !important;
    margin-left: 0 !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

[data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
}


    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading desalination dataset…")
def load_data(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


@st.cache_data(show_spinner="Loading anomaly analysis…")
def load_optional_csv(path_string: str) -> pd.DataFrame | None:
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return None


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_number(value, digits=2):
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def main():
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Expected `data/raw/desalination_sensor_data.csv` "
            "relative to `app.py`. Check that the dataset exists locally."
        )
        st.stop()

    df = load_data(str(DATA_PATH))
    stable = load_optional_csv(str(ANOMALY_PATH))
    rule_results = load_optional_csv(str(RULE_PATH))
    quality_report = load_optional_csv(str(QUALITY_PATH))

    st.sidebar.markdown("## 💧 AquaPalladium")
    st.sidebar.caption("DESALINATION INTELLIGENCE")
    st.sidebar.divider()
    page = st.sidebar.radio(
        "WORKSPACE",
        ["Executive overview", "Operating explorer", "Cleaning intelligence", "Anomaly intelligence", "Data quality"],
        label_visibility="visible",
    )
    st.sidebar.divider()
    st.sidebar.caption("Portfolio prototype • Offline dataset")
    st.sidebar.caption("Flags are exploratory, not confirmed faults.")

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Water intelligence platform</div>
          <h1>AquaPalladium<span style="color:#007f86">.</span></h1>
          <p>Understand performance. Explore anomalies. Make better operational decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Portfolio prototype · Dataset provenance is unverified · Not connected to a live plant · Not an engineering alarm system.",
        icon="⚠️",
    )

    # ---------------------------- Overview ----------------------------
    if page == "Executive overview":
        st.subheader("Executive overview")
        st.markdown(
            '<p class="section-note">A quick read of the available dataset and the current analysis outputs.</p>',
            unsafe_allow_html=True,
        )

        rows, cols = df.shape
        missing_pct = df.isna().sum().sum() / max(df.size, 1) * 100
        unique_total = int(df.nunique(dropna=True).sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Observations", f"{rows:,}", "Rows in source dataset")
        with c2:
            metric_card("Parameters", f"{cols}", "Columns available")
        with c3:
            metric_card("Missing cells", f"{missing_pct:.2f}%", "Of all dataset cells")
        with c4:
            metric_card("Unique-value total", f"{unique_total:,}", "Sum across columns")

        st.write("")
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("### Operating parameter distributions")
            available = [
                c for c in [
                    "Feed_Salinity_mgL", "Feed_Pressure_bar",
                    "Recovery_Pct", "Membrane_Age_months",
                    "Feed_Flow_m3h", "System_Downtime_hr_per_month",
                    "Crossflow_Velocity_m_s",
                ] if c in df.columns
            ]
            chosen = st.selectbox("Select parameter", available, key="overview_parameter")
            values = safe_numeric(df, chosen).dropna()
            fig = px.histogram(
                values.to_frame(name=chosen), x=chosen, nbins=45,
                template="plotly_white", color_discrete_sequence=["#007f86"],
            )
            fig.update_layout(
                font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                margin=dict(l=10, r=18, t=24, b=12), height=380,
                xaxis_title=chosen.replace("_", " "), yaxis_title="Observations",
            )
            fig.update_xaxes(
                tickfont=dict(color="#424245", size=11),
                title_font=dict(color="#1d1d1f", size=12),
                color="#424245", automargin=True,
            )
            fig.update_yaxes(
                tickfont=dict(color="#424245", size=11),
                title_font=dict(color="#1d1d1f", size=12),
                color="#424245", automargin=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "aquapalladium-chart"}})

        with right:
            st.markdown("### Data completeness")
            completeness = (
                df.isna().mean().mul(100).sort_values(ascending=True).tail(10)
                .rename("Missing (%)").reset_index()
            )
            completeness.columns = ["Parameter", "Missing (%)"]
            fig = px.bar(
                completeness, x="Missing (%)", y="Parameter", orientation="h",
                template="plotly_white", color_discrete_sequence=["#0071e3"],
            )
            fig.update_layout(
                font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                margin=dict(l=10, r=18, t=24, b=12), height=380,
                yaxis_title="", xaxis_title="Missing values (%)",
            )
            fig.update_xaxes(
                tickfont=dict(color="#424245", size=11),
                title_font=dict(color="#1d1d1f", size=12),
                color="#424245", automargin=True,
            )
            fig.update_yaxes(
                tickfont=dict(color="#424245", size=11),
                title_font=dict(color="#1d1d1f", size=12),
                color="#424245", automargin=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "aquapalladium-chart"}})

        st.markdown("### Key data caveats")
        caveat_cols = st.columns(3)
        with caveat_cols[0]:
            st.info(
                "**Energy feature**\n\nSpecific energy is constant at 5.0 "
                "for non-missing rows in the current dataset, so it is not "
                "a meaningful performance KPI yet."
            )
        with caveat_cols[1]:
            st.info(
                "**Anomaly flags**\n\nIsolation Forest flags unusual feature "
                "combinations; the flagged percentage is not the true fault rate."
            )
        with caveat_cols[2]:
            st.info(
                "**Provenance**\n\nThe dataset's source and relationship to any "
                "specific plant have not been verified."
            )

    # ------------------------ Operating explorer -----------------------
    elif page == "Operating explorer":
        st.subheader("Operating explorer")
        st.markdown(
            '<p class="section-note">Explore relationships between selected process variables. '
            'These are descriptive associations, not causal conclusions.</p>',
            unsafe_allow_html=True,
        )

        numeric = df.select_dtypes(include="number").columns.tolist()
        preferred = [
            "Feed_Salinity_mgL", "Feed_Pressure_bar", "Recovery_Pct",
            "Membrane_Age_months", "Feed_Flow_m3h",
            "Crossflow_Velocity_m_s", "System_Downtime_hr_per_month",
            "Antiscalant_Dose_mgL", "SDI15", "pH",
        ]
        defaults = [x for x in preferred if x in numeric]
        controls = st.columns(3)
        with controls[0]:
            x_col = st.selectbox("X-axis", numeric, index=numeric.index(defaults[0]) if defaults else 0)
        with controls[1]:
            default_y = defaults[1] if len(defaults) > 1 else numeric[min(1, len(numeric)-1)]
            y_col = st.selectbox("Y-axis", numeric, index=numeric.index(default_y))
        with controls[2]:
            sample_size = st.select_slider(
                "Plot sample size", options=[1000, 3000, 5000, 10000, 25000],
                value=5000,
            )

        plot_df = df[[x_col, y_col]].dropna()
        if len(plot_df) > sample_size:
            plot_df = plot_df.sample(sample_size, random_state=42)
        fig = px.scatter(
            plot_df, x=x_col, y=y_col, opacity=0.55,
            template="plotly_white", render_mode="webgl",
            color_discrete_sequence=["#0b7285"],
        )
        fig.update_layout(
            height=560, font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            margin=dict(l=24, r=24, t=30, b=30),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f2", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f2", zeroline=False),
        )
        fig.update_xaxes(
            tickfont=dict(color="#424245", size=11),
            title_font=dict(color="#1d1d1f", size=12),
            color="#424245", automargin=True,
        )
        fig.update_yaxes(
            tickfont=dict(color="#424245", size=11),
            title_font=dict(color="#1d1d1f", size=12),
            color="#424245", automargin=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "aquapalladium-chart"}})

        with st.expander("View descriptive statistics"):
            stats = df[numeric].describe().T
            stats["missing_pct"] = df[numeric].isna().mean() * 100
            st.dataframe(stats.round(3), use_container_width=True)

        with st.expander("View Spearman correlations"):
            corr_cols = st.multiselect(
                "Choose up to 10 parameters",
                numeric,
                default=[c for c in defaults[:6] if c in numeric],
                max_selections=10,
            )
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr(method="spearman")
                fig = px.imshow(
                    corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    template="plotly_white",
                )
                fig.update_layout(
                    height=560, font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
                    margin=dict(l=20, r=20, t=28, b=20),
                )
                fig.update_xaxes(
                    tickfont=dict(color="#424245", size=11),
                    title_font=dict(color="#1d1d1f", size=12),
                    color="#424245", automargin=True,
                )
                fig.update_yaxes(
                    tickfont=dict(color="#424245", size=11),
                    title_font=dict(color="#1d1d1f", size=12),
                    color="#424245", automargin=True,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "aquapalladium-chart"}})
            else:
                st.caption("Choose at least two parameters to show correlations.")

    # ----------------------- Cleaning intelligence ---------------------
    elif page == "Cleaning intelligence":
        st.subheader("Cleaning intelligence")
        st.markdown(
            '<p class="section-note">Explore historical cleaning patterns and estimate annual cleaning frequency '
            'from operating conditions. This is an exploratory model, not a maintenance recommendation.</p>',
            unsafe_allow_html=True,
        )

        target = "Cleaning_Frequency_per_year"
        predictors = [
            "Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
            "Recovery_Pct", "Membrane_Age_months", "SDI15",
            "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
            "System_Downtime_hr_per_month", "Feed_Flow_m3h",
        ]
        predictors = [c for c in predictors if c in df.columns]
        model_data = df[predictors + [target]].copy()
        model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
        model_data = model_data.dropna(subset=[target])
        model_data = model_data[model_data[target].between(0, 30)]

        if model_data[target].nunique() < 2 or len(model_data) < 100:
            st.error("There is not enough target variation to train a useful exploratory model.")
        else:
            @st.cache_data(show_spinner="Training a baseline cleaning-frequency model…")
            def train_cleaning_model(csv_path: str):
                raw = pd.read_csv(csv_path)
                target_col = "Cleaning_Frequency_per_year"
                feature_cols = [
                    "Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
                    "Recovery_Pct", "Membrane_Age_months", "SDI15",
                    "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
                    "System_Downtime_hr_per_month", "Feed_Flow_m3h",
                ]
                feature_cols = [c for c in feature_cols if c in raw.columns]
                raw[target_col] = pd.to_numeric(raw[target_col], errors="coerce")
                raw = raw.dropna(subset=[target_col])
                raw = raw[raw[target_col].between(0, 30)]
                X = raw[feature_cols]
                y = raw[target_col]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(
                        n_estimators=120, min_samples_leaf=3,
                        random_state=42, n_jobs=-1
                    )),
                ])
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                baseline = np.full(len(y_test), y_train.median())
                metrics = {
                    "mae": float(mean_absolute_error(y_test, predictions)),
                    "baseline_mae": float(mean_absolute_error(y_test, baseline)),
                    "r2": float(r2_score(y_test, predictions)),
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "target_mean": float(y.mean()),
                    "target_median": float(y.median()),
                    "features": feature_cols,
                }
                importances = pd.DataFrame({
                    "Parameter": feature_cols,
                    "Importance": pipeline.named_steps["model"].feature_importances_,
                }).sort_values("Importance", ascending=False)
                return pipeline, metrics, importances

            with st.spinner("Preparing the cleaning-frequency model for this session…"):
                model, metrics, importances = train_cleaning_model(str(DATA_PATH))
            st.markdown("### Model evaluation")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test MAE", f"{metrics['mae']:.2f} cleanings/year")
            m2.metric("Median baseline MAE", f"{metrics['baseline_mae']:.2f}")
            m3.metric("Test R²", f"{metrics['r2']:.3f}")
            m4.metric("Test observations", f"{metrics['test_rows']:,}")

            if metrics["mae"] < metrics["baseline_mae"]:
                st.success(
                    f"The model beats the median baseline by "
                    f"{metrics['baseline_mae'] - metrics['mae']:.2f} cleanings/year MAE on this held-out split."
                )
            else:
                st.warning(
                    "The model does not beat the simple median baseline on this test split. "
                    "Treat its predictions as a prototype, not a dependable forecast."
                )
            st.caption(
                f"Trained on {metrics['train_rows']:,} rows; tested on {metrics['test_rows']:,}. "
                "Random split, no temporal validation. Dataset provenance is unverified."
            )

            left, right = st.columns([1.05, 1])
            with left:
                st.markdown("### Which parameters influence the model?")
                fig = px.bar(
                    importances.sort_values("Importance", ascending=True),
                    x="Importance", y="Parameter", orientation="h",
                    template="plotly_white", color_discrete_sequence=["#007f86"],
                )
                fig.update_layout(
                    height=430,
                    font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
                    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                    margin=dict(l=18, r=18, t=25, b=20),
                    xaxis=dict(showgrid=True, gridcolor="#f0f0f2", zeroline=False),
                )
                fig.update_xaxes(
                    tickfont=dict(color="#424245", size=11),
                    title_font=dict(color="#1d1d1f", size=12),
                    color="#424245", automargin=True,
                )
                fig.update_yaxes(
                    tickfont=dict(color="#424245", size=11),
                    title_font=dict(color="#1d1d1f", size=12),
                    color="#424245", automargin=True,
                )
                st.plotly_chart(
                    fig, use_container_width=True,
                    config={"displaylogo": False, "responsive": True,
                            "toImageButtonOptions": {"format": "png", "scale": 4,
                                                     "filename": "aquapalladium-cleaning-feature-importance"}},
                )
            with right:
                st.markdown("### Historical cleaning frequency")
                hist = df[target].dropna()
                counts = hist.value_counts().sort_index().rename_axis("Cleanings/year").reset_index(name="Observations")
                fig = px.bar(
                    counts, x="Cleanings/year", y="Observations",
                    template="plotly_white", color_discrete_sequence=["#0071e3"],
                )
                fig.update_layout(
                    height=430,
                    font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
                    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                    margin=dict(l=18, r=18, t=25, b=20),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f2", zeroline=False),
                )
                fig.update_xaxes(
                    tickfont=dict(color="#424245", size=11),
                    title_font=dict(color="#1d1d1f", size=12),
                    color="#424245", automargin=True,
                )
                fig.update_yaxes(
                    tickfont=dict(color="#424245", size=11),
                    title_font=dict(color="#1d1d1f", size=12),
                    color="#424245", automargin=True,
                )
                st.plotly_chart(
                    fig, use_container_width=True,
                    config={"displaylogo": False, "responsive": True,
                            "toImageButtonOptions": {"format": "png", "scale": 4,
                                                     "filename": "aquapalladium-cleaning-frequency"}},
                )

            st.markdown("### Try an operating scenario")
            st.caption(
                "Adjust the values below to see the model's estimated cleaning frequency. "
                "Defaults use dataset medians; this is a what-if demonstration, not a scheduling instruction."
            )
            medians = df[predictors].median(numeric_only=True)
            qlow = df[predictors].quantile(0.01, numeric_only=True)
            qhigh = df[predictors].quantile(0.99, numeric_only=True)
            with st.form("cleaning_scenario_form"):
                input_values = {}
                grid = st.columns(3)
                labels = {
                    "Feed_Salinity_mgL": "Feed salinity (mg/L)",
                    "Feed_Temperature_C": "Feed temperature (°C)",
                    "Feed_Pressure_bar": "Feed pressure (bar)",
                    "Recovery_Pct": "Recovery (%)",
                    "Membrane_Age_months": "Membrane age (months)",
                    "SDI15": "SDI15",
                    "Crossflow_Velocity_m_s": "Crossflow velocity (m/s)",
                    "pH": "pH",
                    "Antiscalant_Dose_mgL": "Antiscalant dose (mg/L)",
                    "System_Downtime_hr_per_month": "Downtime (hours/month)",
                    "Feed_Flow_m3h": "Feed flow (m³/h)",
                }
                for i, feature in enumerate(predictors):
                    low = float(qlow.get(feature, 0))
                    high = float(qhigh.get(feature, 100))
                    if not np.isfinite(low):
                        low = float(df[feature].min())
                    if not np.isfinite(high) or high <= low:
                        high = max(low + 1.0, float(df[feature].max()))
                    default = float(medians.get(feature, (low + high) / 2))
                    default = min(max(default, low), high)
                    with grid[i % 3]:
                        input_values[feature] = st.number_input(
                            labels.get(feature, feature.replace("_", " ")),
                            min_value=low, max_value=high, value=default,
                            key=f"cleaning_{feature}", format="%.3f",
                        )
                submitted = st.form_submit_button("Estimate cleaning frequency", type="primary")
            if submitted:
                scenario = pd.DataFrame([input_values], columns=predictors)
                estimate = float(model.predict(scenario)[0])
                st.markdown("### Scenario estimate")
                st.metric("Estimated cleanings per year", f"{max(0, estimate):.1f}")
                st.caption(
                    "The model output is continuous and rounded for display. "
                    "It is learned from historical patterns in this dataset, not from validated plant-cleaning rules."
                )

            with st.expander("Model details and limitations"):
                st.markdown(
                    """
                    - **Model:** Random Forest Regressor with median imputation.
                    - **Target:** `Cleaning_Frequency_per_year`.
                    - **Features:** selected operating parameters; production output and constant specific-energy feature are excluded.
                    - **Evaluation:** held-out random 20% test split; MAE compared with a median-only baseline.
                    - **Important:** feature importance is not causation. A strong score would not prove the model generalizes to a real plant. Verify dataset provenance and validate against real maintenance records before operational use.
                    """
                )

    # ----------------------- Anomaly intelligence ----------------------
    elif page == "Anomaly intelligence":
        st.subheader("Anomaly intelligence")
        st.markdown(
            '<p class="section-note">Investigate unusual observations and see which operating values stand out against the dataset baseline.</p>',
            unsafe_allow_html=True,
        )
        st.info(
            "Interpretation note: this page describes unusual feature values; it does not prove a fault or provide causal explanations for the Isolation Forest decision.",
            icon="ℹ️",
        )

        if stable is not None and "stable_anomaly" in stable.columns:
            stable = stable.copy()
            stable["stable_anomaly"] = (
                stable["stable_anomaly"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
            )
            stable_count = int(stable["stable_anomaly"].sum())
            a, b, c = st.columns(3)
            a.metric("Stable anomalies", f"{stable_count:,}")
            b.metric("Stable share", f"{stable_count / max(len(stable), 1) * 100:.2f}%")
            c.metric("Rows analysed", f"{len(stable):,}")

            stable_only = stable[stable["stable_anomaly"]]
            if stable_only.empty:
                st.warning("No stable anomaly rows were found in the loaded output.")
            else:
                st.markdown("### Investigate one observation")
                # Use a stable table index as the UI key; show any existing score/ID fields if available.
                score_cols = [
                    col for col in stable_only.columns
                    if any(token in col.lower() for token in ["score", "decision", "anomaly_id", "record_id", "row_id"])
                ]
                feature_candidates = [
                    "Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
                    "Recovery_Pct", "Membrane_Age_months", "SDI15",
                    "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
                    "System_Downtime_hr_per_month", "Cleaning_Frequency_per_year",
                    "Feed_Flow_m3h",
                ]
                feature_cols = [
                    col for col in feature_candidates
                    if col in stable_only.columns and pd.api.types.is_numeric_dtype(stable_only[col])
                ]


                picker_options = stable_only.index.tolist()
                picker_labels = {}

                # Calculate reference statistics once, using the full dataset.
                reference_stats = {}
                for col in feature_cols:
                    if col not in df.columns:
                        continue

                    baseline = pd.to_numeric(
                        df[col], errors="coerce"
                    ).dropna()

                    if baseline.empty:
                        continue

                    median = float(baseline.median())
                    q1 = float(baseline.quantile(0.25))
                    q3 = float(baseline.quantile(0.75))
                    iqr = q3 - q1

                    if iqr > 0:
                        reference_stats[col] = (median, iqr)

                for idx in picker_options:
                    row = stable_only.loc[idx]
                    unusual_feature = None
                    unusual_value = None
                    largest_distance = -1.0

                    for col, (median, iqr) in reference_stats.items():
                        observed = pd.to_numeric(
                            pd.Series([row[col]]), errors="coerce"
                        ).iloc[0]

                        if pd.isna(observed):
                            continue

                        distance = abs(float(observed) - median) / iqr

                        if distance > largest_distance:
                            largest_distance = distance
                            unusual_feature = col
                            unusual_value = float(observed)

                    if unusual_feature is not None:
                        parameter_label = (
                            f"{unusual_feature.replace('_', ' ')}: "
                            f"{unusual_value:,.3g}"
                        )
                        picker_labels[idx] = (
                            f"Record {idx} · {parameter_label}"
                        )
                    else:
                        picker_labels[idx] = f"Record {idx}"


                selected_idx = st.selectbox(
                    "Select a stable anomaly",
                    options=picker_options,
                    format_func=lambda idx: picker_labels[idx],
                    key="selected_anomaly_record",
                )
                selected_row = stable_only.loc[selected_idx]

                left, right = st.columns([1, 1.2])
                with left:
                    st.markdown("#### Selected record")
                    shown_fields = [
                        c for c in [
                            "Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
                            "Recovery_Pct", "Membrane_Age_months", "SDI15",
                            "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
                            "System_Downtime_hr_per_month", "Cleaning_Frequency_per_year",
                            "Feed_Flow_m3h",
                        ] if c in stable_only.columns
                    ]
                    record_table = pd.DataFrame({
                        "Parameter": shown_fields,
                        "Observed value": [
                            pd.to_numeric(pd.Series([selected_row[c]]), errors="coerce").iloc[0]
                            for c in shown_fields
                        ],
                    })
                    st.dataframe(record_table, hide_index=True, use_container_width=True, height=420)

                with right:
                    st.markdown("#### What looks unusual?")
                    profile_rows = []
                    for col in feature_cols:
                        observed = pd.to_numeric(pd.Series([selected_row[col]]), errors="coerce").iloc[0]
                        baseline = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
                        if pd.isna(observed) or baseline.empty:
                            continue
                        median = float(baseline.median())
                        q1 = float(baseline.quantile(0.25))
                        q3 = float(baseline.quantile(0.75))
                        iqr = q3 - q1
                        pct = float((baseline <= observed).mean() * 100)
                        if iqr > 0:
                            iqr_distance = (observed - median) / iqr
                        else:
                            iqr_distance = np.nan
                        if pct <= 5 or pct >= 95:
                            note = "Extreme tail (≤5th or ≥95th percentile)"
                        elif pct <= 10 or pct >= 90:
                            note = "Uncommon tail (≤10th or ≥90th percentile)"
                        else:
                            note = "Within central 80% of observations"
                        profile_rows.append({
                            "Parameter": col,
                            "Observed": observed,
                            "Dataset median": median,
                            "Percentile": pct,
                            "Distance from median (IQRs)": iqr_distance,
                            "Screening note": note,
                        })
                    if profile_rows:
                        profile = pd.DataFrame(profile_rows)
                        profile = profile.sort_values(
                            "Percentile",
                            key=lambda series: np.minimum(series, 100 - series),
                        )
                        st.dataframe(
                            profile.round({
                                "Observed": 3, "Dataset median": 3,
                                "Percentile": 1, "Distance from median (IQRs)": 2,
                            }),
                            hide_index=True, use_container_width=True, height=420,
                        )
                        st.caption(
                            "Percentiles and IQR distances are descriptive comparisons with the full dataset, not model feature attributions."
                        )
                    else:
                        st.caption("No comparable numeric operating features were available for this record.")

                st.markdown("### Compare anomaly group with other observations")
                available = [
                    c for c in [
                        "System_Downtime_hr_per_month", "Membrane_Age_months",
                        "Crossflow_Velocity_m_s", "Feed_Flow_m3h",
                        "Recovery_Pct", "Feed_Pressure_bar",
                    ] if c in stable.columns
                ]
                if available:
                    feature = st.selectbox("Parameter to compare", available, key="anomaly_compare_feature")
                    comparison = stable[[feature, "stable_anomaly"]].copy()
                    comparison[feature] = pd.to_numeric(comparison[feature], errors="coerce")
                    comparison = comparison.dropna()
                    comparison["Group"] = comparison["stable_anomaly"].map({
                        True: "Stable anomaly", False: "Other observation"
                    })
                    fig = px.box(
                        comparison, x="Group", y=feature, color="Group",
                        points=False, template="plotly_white",
                        color_discrete_map={
                            "Stable anomaly": "#d28b26",
                            "Other observation": "#007f86",
                        },
                    )
                    fig.update_layout(
                        showlegend=False, height=430,
                        font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
                        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                        margin=dict(l=20, r=20, t=28, b=24),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="#f0f0f2", zeroline=False),
                    )
                    fig.update_xaxes(
                        tickfont=dict(color="#424245", size=11),
                        title_font=dict(color="#1d1d1f", size=12),
                        color="#424245", automargin=True,
                    )
                    fig.update_yaxes(
                        tickfont=dict(color="#424245", size=11),
                        title_font=dict(color="#1d1d1f", size=12),
                        color="#424245", automargin=True,
                    )
                    st.plotly_chart(
                        fig, use_container_width=True,
                        config={"displaylogo": False, "responsive": True,
                                "toImageButtonOptions": {"format": "png", "scale": 4,
                                                         "filename": "aquapalladium-anomaly-comparison"}},
                    )

                st.markdown("### Browse and export records")
                filter_cols = st.columns(3)
                with filter_cols[0]:
                    only_stable = st.checkbox("Stable anomalies only", value=True, key="browse_only_stable")
                with filter_cols[1]:
                    min_age = st.number_input("Minimum membrane age (months)", min_value=0.0, value=0.0, key="browse_min_age")
                with filter_cols[2]:
                    max_rows = st.selectbox("Maximum rows shown", [100, 500, 1000, 5000], index=1, key="browse_max_rows")

                view = stable.copy()
                if only_stable:
                    view = view[view["stable_anomaly"]]
                if "Membrane_Age_months" in view.columns:
                    age = pd.to_numeric(view["Membrane_Age_months"], errors="coerce")
                    view = view[age.fillna(-1) >= min_age]

                st.caption(f"{len(view):,} matching observations")
                st.dataframe(view.head(max_rows), use_container_width=True, height=360)
                st.download_button(
                    "Download filtered records",
                    data=view.to_csv(index=False).encode("utf-8"),
                    file_name="aquapalladium_filtered_anomalies.csv",
                    mime="text/csv",
                )
        else:
            st.warning(
                "Stable anomaly results not found. Run "
                "`python -m src.investigate_stable_anomalies` first."
            )

        if rule_results is not None:
            st.markdown("### Rule-based analysis output")
            st.caption(
                "Formula-consistency flags are separate from Isolation Forest. A discrepancy is a candidate for review, not a confirmed fault."
            )
            st.write(f"Rule-based result file loaded: **{len(rule_results):,} rows**")
            st.download_button(
                "Download rule-based results",
                data=rule_results.to_csv(index=False).encode("utf-8"),
                file_name="aquapalladium_rule_results.csv",
                mime="text/csv",
            )

    # ---------------------------- Data quality -------------------------
    elif page == "Data quality":
        st.subheader("Data quality")
        st.markdown(
            '<p class="section-note">Missingness, feature variability and dataset diagnostics.</p>',
            unsafe_allow_html=True,
        )

        quality = pd.DataFrame({
            "Parameter": df.columns,
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Missing count": [int(df[c].isna().sum()) for c in df.columns],
            "Missing (%)": [df[c].isna().mean() * 100 for c in df.columns],
            "Unique non-missing": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }).sort_values("Missing (%)", ascending=False)

        q1, q2, q3 = st.columns(3)
        q1.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        q2.metric("Features", f"{len(df.columns)}")
        q3.metric("Constant features", f"{sum(df[c].nunique(dropna=True) <= 1 for c in df.columns)}")

        st.dataframe(quality.round(3), use_container_width=True)

        fig = px.bar(
            quality.sort_values("Missing (%)", ascending=True),
            x="Missing (%)", y="Parameter", orientation="h",
            template="plotly_white", color_discrete_sequence=["#007f86"],
        )
        fig.update_layout(
            height=560, font=dict(family="Inter, -apple-system, sans-serif", color="#424245", size=12),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            margin=dict(l=20, r=20, t=28, b=20),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f2", zeroline=False),
        )
        fig.update_xaxes(
            tickfont=dict(color="#424245", size=11),
            title_font=dict(color="#1d1d1f", size=12),
            color="#424245", automargin=True,
        )
        fig.update_yaxes(
            tickfont=dict(color="#424245", size=11),
            title_font=dict(color="#1d1d1f", size=12),
            color="#424245", automargin=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "aquapalladium-chart"}})

        constants = quality[quality["Unique non-missing"] <= 1]
        if not constants.empty:
            st.warning(
                "Constant or all-missing features need investigation before "
                "being used for modelling or performance KPIs."
            )
            st.dataframe(constants, use_container_width=True)

        if quality_report is not None:
            with st.expander("Open saved data-quality report"):
                st.dataframe(quality_report, use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption("AquaPalladium · Analytics prototype")


if __name__ == "__main__":
    main()
