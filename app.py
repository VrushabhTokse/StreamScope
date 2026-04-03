import os
import sys
import importlib
import base64
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup (Robust absolute paths for deployment) ─────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# Local modules
from preprocessing import get_clean_data
import eda
import modeling

# Force reload of modeling module to pick up changes in submodules
importlib.reload(modeling)

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StreamScope · Netflix Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Theme ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at top left, #0d0d0d 0%, #111 40%, #0a0a14 100%);
    color: #f0f0f0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #e50914; border-radius: 3px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f0f 0%, #1a0708 60%, #0f0f0f 100%);
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e1e1e !important;
    border: 1px solid #333 !important;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #191919 0%, #230d0d 100%);
    border: 1px solid rgba(229,9,20,0.5);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(229,9,20,0.1);
    transition: all 0.25s ease;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #e50914, #ff6f61);
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 35px rgba(229,9,20,0.3);
    border-color: #e50914;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #e50914, #ff6f61);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.75rem;
    color: #888;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 4px; display: block; }

/* Section headers */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #fff;
    border-left: 4px solid #e50914;
    padding: 6px 0 6px 14px;
    margin: 24px 0 14px 0;
    background: linear-gradient(90deg, rgba(229,9,20,0.07) 0%, transparent 100%);
    border-radius: 0 6px 6px 0;
}

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #1a1a1a, #1f0a0a);
    border: 1px solid #2a2a2a;
    border-left: 3px solid #e50914;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ccc;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(20,20,20,0.9);
    border-radius: 10px;
    gap: 3px;
    padding: 4px;
    border: 1px solid #2a2a2a;
}
.stTabs [data-baseweb="tab"] {
    color: #888;
    border-radius: 7px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #e50914, #c10812) !important;
    color: white !important;
    box-shadow: 0 3px 12px rgba(229,9,20,0.4);
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(229,9,20,0.1) !important;
    color: #e50914 !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #e50914 !important; }

/* Info / warning boxes */
[data-testid="stAlert"] { border-radius: 8px; }

hr { border-color: #222; }
label { color: #bbb !important; font-size: 0.83rem !important; }
[data-testid="stDataFrame"] { border-radius: 10px; }

/* Recommendation card */
.rec-card {
    background: linear-gradient(135deg, #1a1a1a, #1f1020);
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    transition: all 0.2s;
}
.rec-card:hover { border-color: #e50914; transform: translateX(4px); }
.rec-title { font-weight: 700; color: #fff; font-size: 0.95rem; }
.rec-meta { color: #888; font-size: 0.78rem; margin-top: 3px; }
.rec-score {
    display: inline-block;
    background: linear-gradient(135deg, #e50914, #c10812);
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    float: right;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,18,18,0.95)",
    font=dict(color="#d0d0d0", family="Inter", size=12),
    margin=dict(l=20, r=20, t=42, b=20),
)
NC = ["#e50914", "#ff6f61", "#ff9f43", "#ffd32a", "#0be881",
      "#4bcffa", "#a29bfe", "#fd9644", "#26de81", "#45aaf2",
      "#ff5f57", "#febc2e", "#28c840", "#007aff", "#af52de"]

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Data loading (cached) ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_titles.csv")

@st.cache_data(show_spinner=False)
def load_dataset():
    return get_clean_data(DATA_PATH)

@st.cache_data(show_spinner=False)
def run_ml_models_v4_lr(_df):
    clustered, inertia, pca_exp = modeling.run_clustering(_df, n_clusters=3)
    acc, cm, report, imp_df, class_names, cv_scores, auc, clf, encoders = modeling.run_classification(_df)
    elbow = modeling.elbow_inertias(_df, k_range=range(2, 11))
    comparison = modeling.run_model_comparison(_df)
    return clustered, inertia, pca_exp, acc, cm, report, imp_df, class_names, cv_scores, auc, elbow, comparison, clf, encoders

@st.cache_data(show_spinner=False)
def build_recs(_df):
    return modeling.build_recommendation_engine(_df)

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(
        "📂 **Dataset not found!**\n\n"
        "Download `netflix_titles.csv` from "
        "[Kaggle](https://www.kaggle.com/datasets/ranaghulamnabi/netflix-movies-and-tv-shows-dataset) "
        "and place it in the `data/` folder."
    )
    st.stop()

with st.spinner("🎬 Loading Netflix dataset…"):
    df_full = load_dataset()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 4px 0;'>
        <span style='font-size:1.8rem;font-weight:900;
        background:linear-gradient(90deg,#e50914,#ff6f61);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🎬 StreamScope
        </span><br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 🔧 Global Filters")

    # Year range
    valid_years = sorted(df_full[df_full["year_added"] > 2000]["year_added"].unique())
    if not valid_years:
        # Fallback if date parsing failed or no titles after 2000
        year_min, year_max = 2000, 2021
    else:
        year_min, year_max = int(min(valid_years)), int(max(valid_years))
        
    year_range = st.slider("📅 Year Added", year_min, year_max, (year_min, year_max))

    type_options = ["All"] + sorted(df_full["type"].unique().tolist())
    selected_type = st.selectbox("📺 Content Type", type_options)

    all_genres = sorted(set(g for gl in df_full["genres_list"] for g in gl if g != "Unknown"))
    selected_genre = st.selectbox("🎭 Genre", ["All"] + all_genres)

    top_countries = (
        ["All"] +
        df_full[df_full["primary_country"] != "Unknown"]["primary_country"]
        .value_counts().head(50).index.tolist()
    )
    selected_country = st.selectbox("🌍 Country", top_countries)

    all_ratings = ["All"] + sorted(
        df_full[df_full["rating"] != "Unknown"]["rating"].unique().tolist()
    )
    selected_rating = st.selectbox("🔞 Rating", all_ratings)

    st.markdown("---")
    st.markdown("#### 📊 Dataset Stats")
    total_titles = len(df_full)
    total_movies = len(df_full[df_full["type"] == "Movie"])
    total_shows = len(df_full[df_full["type"] == "TV Show"])
    st.markdown(f"""
    <div style='color:#888;font-size:0.82rem;line-height:2;'>
    🎬 <b style='color:#e50914'>{total_movies:,}</b> Movies<br>
    📺 <b style='color:#4bcffa'>{total_shows:,}</b> TV Shows<br>
    📁 <b style='color:#ffd32a'>{total_titles:,}</b> Total Titles<br>
    🌍 <b style='color:#0be881'>{df_full["primary_country"].nunique()-1}</b> Countries<br>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Source: [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/ranaghulamnabi/netflix-movies-and-tv-shows-dataset)")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full[
    (df_full["year_added"] >= year_range[0]) &
    (df_full["year_added"] <= year_range[1])
].copy()
if selected_type != "All":
    df = df[df["type"] == selected_type]
if selected_genre != "All":
    df = df[df["genres_list"].apply(lambda g: selected_genre in g)]
if selected_country != "All":
    df = df[df["primary_country"] == selected_country]
if selected_rating != "All":
    df = df[df["rating"] == selected_rating]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:24px 0 8px 0;'>
<span style='font-size:3.2rem;font-weight:900;
    background:linear-gradient(90deg,#e50914 10%,#ff6f61 60%,#ff9f43 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    🎬 StreamScope
</span><br>
<span style='color:#555;font-size:0.95rem;letter-spacing:3px;font-weight:600;'>
    NETFLIX CONTENT STRATEGY ANALYZER
</span>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
total     = len(df)
movies    = len(df[df["type"] == "Movie"])
shows     = len(df[df["type"] == "TV Show"])
countries = df[df["primary_country"] != "Unknown"]["primary_country"].nunique()
genres_n  = len(set(g for gl in df["genres_list"] for g in gl if g != "Unknown"))
avg_dur   = df[df["type"] == "Movie"]["duration_value"].mean()
avg_dur_s = f"{avg_dur:.0f} min" if not np.isnan(avg_dur) else "—"

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi_data = [
    (c1, "🎥", f"{total:,}",   "Total Titles"),
    (c2, "🎬", f"{movies:,}",  "Movies"),
    (c3, "📺", f"{shows:,}",   "TV Shows"),
    (c4, "🌍", f"{countries}", "Countries"),
    (c5, "🎭", f"{genres_n}",  "Genres"),
    (c6, "⏱️", avg_dur_s,      "Avg. Movie Length"),
]
for col, icon, val, lbl in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
(tab1, tab2, tab3, tab4,
 tab5, tab6, tab7, tab8) = st.tabs([
    "📈 Growth & Trends",
    "🎭 Genre Deep Dive",
    "🌍 Global Reach",
    "🎬 Directors & Cast",
    "⏱️ Duration Analysis",
    "🤖 ML Models",
    "🔍 Recommendations",
    "📋 Data Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Growth & Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c_a, c_b = st.columns([2, 1])

    with c_a:
        st.markdown('<div class="section-header">Netflix Content Growth Over Time</div>', unsafe_allow_html=True)
        growth = eda.content_growth_over_time(df)
        if not growth.empty:
            fig_g = px.area(growth, x="year_added", y="count", color="type",
                title="Titles Added to Netflix Per Year",
                labels={"year_added":"Year","count":"Titles","type":"Type"},
                color_discrete_sequence=["#e50914","#4bcffa"],
                template="plotly_dark")
            fig_g.update_layout(**PL)
            fig_g.update_traces(line_width=2.5, opacity=0.85)
            fig_g.update_xaxes(showgrid=False)
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.info("No data with current filters.")

    with c_b:
        st.markdown('<div class="section-header">Type Split</div>', unsafe_allow_html=True)
        tc = eda.content_type_split(df)
        if tc:
            fig_d = go.Figure(go.Pie(
                labels=list(tc.keys()), values=list(tc.values()),
                hole=0.6,
                marker=dict(colors=["#e50914","#4bcffa"],
                            line=dict(color="#111",width=2)),
                textfont=dict(color="white"),
            ))
            fig_d.update_layout(title="Movies vs TV Shows",
                showlegend=True, legend=dict(orientation="h",y=-0.1), **PL)
            st.plotly_chart(fig_d, use_container_width=True)

    # Monthly heatmap
    st.markdown('<div class="section-header">📅 Monthly Content Addition Heatmap</div>', unsafe_allow_html=True)
    try:
        monthly = eda.monthly_additions_heatmap(df)
        if not monthly.empty:
            monthly.index = [MONTH_NAMES[i-1] for i in monthly.index]
            fig_heat = px.imshow(
                monthly,
                color_continuous_scale="Reds",
                aspect="auto",
                title="Number of Titles Added Per Month × Year",
                labels={"x":"Year","y":"Month","color":"Titles"},
                template="plotly_dark",
            )
            fig_heat.update_layout(**PL, height=320)
            fig_heat.update_coloraxes(colorbar=dict(
                bgcolor="#1a1a1a", tickfont=dict(color="#ccc")
            ))
            st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        pass

    # Content addition lag
    st.markdown('<div class="section-header">⏳ Content Addition Lag (Release → Netflix)</div>', unsafe_allow_html=True)
    lag = eda.content_addition_lag(df)
    if not lag.empty:
        fig_lag = px.bar(lag, x="lag_years", y="count",
            color="count", color_continuous_scale="Reds",
            template="plotly_dark",
            title="How Many Years After Release Do Titles Appear on Netflix?",
            labels={"lag_years":"Years After Release","count":"Number of Titles"})
        fig_lag.update_layout(**PL)
        fig_lag.update_coloraxes(showscale=False)
        st.plotly_chart(fig_lag, use_container_width=True)

        median_lag = lag.loc[lag["count"].idxmax(), "lag_years"]
        st.markdown(f"""<div class="insight-box">
        💡 <b>Insight:</b> The most common lag is <b>{median_lag} year(s)</b> between a title's 
        original release and its appearance on Netflix. A high volume at lag = 0 indicates 
        Netflix Originals or same-year acquisitions.
        </div>""", unsafe_allow_html=True)

    # Decade
    st.markdown('<div class="section-header">🕰️ Release Decade Distribution</div>', unsafe_allow_html=True)
    dec = eda.decade_distribution(df)
    fig_dec = px.bar(dec, x="decade", y="count", color="count",
        color_continuous_scale="Reds", template="plotly_dark",
        title="Netflix Library by Release Decade",
        labels={"decade":"Decade","count":"Titles"})
    fig_dec.update_layout(**PL)
    fig_dec.update_coloraxes(showscale=False)
    st.plotly_chart(fig_dec, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Genre Deep Dive
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Top 15 Genres</div>', unsafe_allow_html=True)
        gd = eda.genre_distribution(df, top_n=15)
        fig_genre = px.bar(gd, x="count", y="genre", orientation="h",
            color="count", color_continuous_scale="Reds", template="plotly_dark",
            title="Top 15 Genres by Title Count",
            labels={"genre":"","count":"Titles"})
        fig_genre.update_layout(**PL, yaxis=dict(autorange="reversed"))
        fig_genre.update_coloraxes(showscale=False)
        st.plotly_chart(fig_genre, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Ratings Distribution</div>', unsafe_allow_html=True)
        rd = eda.rating_distribution(df)
        fig_rat = px.bar(rd, x="rating", y="count",
            color="count", color_continuous_scale="Oranges",
            template="plotly_dark", title="Content Ratings Breakdown",
            labels={"rating":"Rating","count":"Count"})
        fig_rat.update_layout(**PL)
        fig_rat.update_coloraxes(showscale=False)
        st.plotly_chart(fig_rat, use_container_width=True)

    # Genre co-occurrence heatmap
    st.markdown('<div class="section-header">🔗 Genre Co-Occurrence Matrix</div>', unsafe_allow_html=True)
    try:
        co_matrix = eda.genre_cooccurrence_matrix(df, top_n=12)
        if not co_matrix.empty:
            fig_co = px.imshow(
                co_matrix,
                color_continuous_scale="Reds",
                aspect="auto",
                template="plotly_dark",
                title="How Often Genre Pairs Appear Together (Darker = More Co-occurrence)",
            )
            fig_co.update_layout(**PL, height=480)
            st.plotly_chart(fig_co, use_container_width=True)
            st.markdown("""<div class="insight-box">
            💡 <b>Insight:</b> Darker cells indicate genre pairs that frequently appear on the same title.
            Strong co-occurrences reveal Netflix's genre bundling strategy 
            (e.g. <i>International + Drama</i>, <i>Comedy + Romance</i>).
            </div>""", unsafe_allow_html=True)
    except Exception:
        pass

    # Country × Genre heatmap
    st.markdown('<div class="section-header">🌍 Country × Genre Heatmap</div>', unsafe_allow_html=True)
    try:
        cg_matrix = eda.country_genre_heatmap(df, top_countries=12, top_genres=10)
        if not cg_matrix.empty:
            fig_cg = px.imshow(
                cg_matrix,
                color_continuous_scale="Inferno",
                aspect="auto",
                template="plotly_dark",
                title="Genre Focus Per Country (Top 12 Countries × Top 10 Genres)",
            )
            fig_cg.update_layout(**PL, height=420)
            st.plotly_chart(fig_cg, use_container_width=True)
    except Exception:
        pass

    # Audience split + Top genres per year
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">👥 Audience Groups</div>', unsafe_allow_html=True)
        rg = eda.rating_group_distribution(df)
        fig_rg = px.pie(rg, names="rating_group", values="count", hole=0.5,
            color_discrete_sequence=NC, template="plotly_dark",
            title="Family / Teen / Mature / Other")
        fig_rg.update_layout(**PL)
        st.plotly_chart(fig_rg, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">📊 Top 3 Genres Per Year</div>', unsafe_allow_html=True)
        tgy = eda.top_genres_per_year(df, top_n=3)
        if not tgy.empty:
            fig_tgy = px.bar(tgy, x="year_added", y="count", color="genres_list",
                barmode="stack", template="plotly_dark",
                color_discrete_sequence=NC,
                labels={"year_added":"Year","count":"Count","genres_list":"Genre"})
            fig_tgy.update_layout(**PL)
            st.plotly_chart(fig_tgy, use_container_width=True)

    # Word frequency
    st.markdown('<div class="section-header">🔤 Most Frequent Words in Descriptions</div>', unsafe_allow_html=True)
    try:
        words = eda.description_word_frequency(df, top_n=30)
        fig_words = px.bar(words, x="count", y="word", orientation="h",
            color="count", color_continuous_scale="Reds",
            template="plotly_dark",
            title="Top 30 Words in Netflix Title Descriptions",
            labels={"word":"","count":"Frequency"})
        fig_words.update_layout(**PL, height=520, yaxis=dict(autorange="reversed"))
        fig_words.update_coloraxes(showscale=False)
        st.plotly_chart(fig_words, use_container_width=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Global Reach
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🗺️ Global Content Distribution</div>', unsafe_allow_html=True)
    country_data = eda.country_content_counts(df)

    fig_map = px.choropleth(
        country_data,
        locations="country", locationmode="country names",
        color="count", color_continuous_scale="Reds",
        template="plotly_dark",
        title="Netflix Content by Country of Origin",
        labels={"count":"Titles"}, hover_name="country",
    )
    fig_map.update_geos(
        bgcolor="rgba(0,0,0,0)", landcolor="#1e1e2a",
        oceancolor="#060612", showocean=True,
        framecolor="#1a1a2a", showframe=True,
        coastlinecolor="#2a2a3a", showcoastlines=True,
        showlakes=False,
    )
    fig_map.update_layout(
        **PL, height=530,
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        coloraxis_colorbar=dict(
            bgcolor="#1a1a1a", thickness=12,
            tickfont=dict(color="#ccc"),
            title=dict(text="Titles", font=dict(color="#ccc")),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Top 20 Countries</div>', unsafe_allow_html=True)
        top20 = country_data.head(20)
        fig_c = px.bar(top20, x="count", y="country", orientation="h",
            color="count", color_continuous_scale="Reds", template="plotly_dark",
            title="Top 20 Content-Producing Countries",
            labels={"country":"","count":"Titles"})
        fig_c.update_layout(**PL, yaxis=dict(autorange="reversed"))
        fig_c.update_coloraxes(showscale=False)
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Movies vs Shows – By Country</div>', unsafe_allow_html=True)
        top10_countries = country_data.head(10)["country"].tolist()
        df_top_c = df[df["primary_country"].isin(top10_countries)]
        pivot_c = df_top_c.groupby(["primary_country","type"]).size().reset_index(name="count")
        fig_cc = px.bar(pivot_c, x="primary_country", y="count", color="type",
            barmode="group", template="plotly_dark",
            color_discrete_sequence=["#e50914","#4bcffa"],
            title="Movie vs TV Show Split – Top 10 Countries",
            labels={"primary_country":"","count":"Titles","type":"Type"})
        fig_cc.update_layout(**PL, xaxis_tickangle=-30)
        st.plotly_chart(fig_cc, use_container_width=True)

    # Content share treemap
    st.markdown('<div class="section-header">🌐 Content Share Treemap</div>', unsafe_allow_html=True)
    treemap_data = country_data.head(30).copy()
    fig_tree = px.treemap(treemap_data, path=["country"], values="count",
        color="count", color_continuous_scale="Reds",
        template="plotly_dark",
        title="Top 30 Countries — Content Share Treemap")
    fig_tree.update_layout(**PL, height=380)
    fig_tree.update_traces(textfont=dict(color="white"))
    st.plotly_chart(fig_tree, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Directors & Cast
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🎬 Top 20 Directors</div>', unsafe_allow_html=True)
        dir_df = eda.director_leaderboard(df, top_n=20)
        fig_dir = px.bar(dir_df, x="count", y="director", orientation="h",
            color="count", color_continuous_scale="Reds", template="plotly_dark",
            title="Most Prolific Directors on Netflix",
            labels={"director":"","count":"Titles"})
        fig_dir.update_layout(**PL, height=580, yaxis=dict(autorange="reversed"))
        fig_dir.update_coloraxes(showscale=False)
        st.plotly_chart(fig_dir, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">⭐ Top 20 Actors / Cast</div>', unsafe_allow_html=True)
        act_df = eda.actor_leaderboard(df, top_n=20)
        fig_act = px.bar(act_df, x="count", y="actor", orientation="h",
            color="count", color_continuous_scale="Blues", template="plotly_dark",
            title="Most Featured Actors on Netflix",
            labels={"actor":"","count":"Titles"})
        fig_act.update_layout(**PL, height=580, yaxis=dict(autorange="reversed"))
        fig_act.update_coloraxes(showscale=False)
        st.plotly_chart(fig_act, use_container_width=True)

    # Director search
    st.markdown('<div class="section-header">🔎 Director Deep Dive</div>', unsafe_allow_html=True)
    director_search = st.text_input("Search for a director…", placeholder="e.g. Steven Spielberg")
    if director_search:
        director_titles = df[df["director"].str.contains(director_search, case=False, na=False)]
        if not director_titles.empty:
            st.success(f"Found **{len(director_titles)}** titles for **{director_search}**")
            cols_to_show = ["title", "type", "primary_genre", "release_year", "rating", "duration", "primary_country"]
            st.dataframe(director_titles[[c for c in cols_to_show if c in director_titles.columns]]
                         .reset_index(drop=True), use_container_width=True)

            genre_split = director_titles.explode("genres_list")["genres_list"].value_counts().reset_index()
            genre_split.columns = ["genre", "count"]
            fig_dg = px.pie(genre_split, names="genre", values="count",
                hole=0.5, title=f"Genre Mix — {director_search}",
                color_discrete_sequence=NC, template="plotly_dark")
            fig_dg.update_layout(**PL)
            st.plotly_chart(fig_dg, use_container_width=True)
        else:
            st.warning("No titles found. Try a different name.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Duration Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🎬 Movie Duration Distribution</div>', unsafe_allow_html=True)
        movie_dur = eda.movie_duration_distribution(df)
        if not movie_dur.empty:
            fig_md = px.histogram(movie_dur, x="duration_value", nbins=40,
                color_discrete_sequence=["#e50914"], template="plotly_dark",
                title="Distribution of Movie Runtimes (minutes)",
                labels={"duration_value":"Runtime (min)","count":"Count"})
            fig_md.update_layout(**PL)
            fig_md.update_traces(marker_line_color="#111", marker_line_width=1)
            st.plotly_chart(fig_md, use_container_width=True)

            med = movie_dur["duration_value"].median()
            mean = movie_dur["duration_value"].mean()
            st.markdown(f"""<div class="insight-box">
            📊 <b>Movie Runtime Stats:</b><br>
            Median: <b>{med:.0f} min</b> &nbsp;|&nbsp; 
            Mean: <b>{mean:.0f} min</b> &nbsp;|&nbsp; 
            Shortest: <b>{movie_dur["duration_value"].min():.0f} min</b> &nbsp;|&nbsp;
            Longest: <b>{movie_dur["duration_value"].max():.0f} min</b>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">📺 TV Show Seasons Distribution</div>', unsafe_allow_html=True)
        tv_seas = eda.tv_seasons_distribution(df)
        if not tv_seas.empty:
            fig_tv = px.bar(tv_seas, x="seasons", y="count",
                color="count", color_continuous_scale="Blues",
                template="plotly_dark",
                title="Number of Seasons — TV Show Distribution",
                labels={"seasons":"Seasons","count":"Shows"})
            fig_tv.update_layout(**PL)
            fig_tv.update_coloraxes(showscale=False)
            st.plotly_chart(fig_tv, use_container_width=True)

    # Duration by genre
    st.markdown('<div class="section-header">⏱️ Average Movie Runtime by Genre</div>', unsafe_allow_html=True)
    movie_dur_genre = df[(df["type"] == "Movie") & df["duration_value"].notna()].copy()
    if not movie_dur_genre.empty:
        dur_by_genre = (
            movie_dur_genre.groupby("primary_genre")["duration_value"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_duration", "count": "num_titles"})
            .sort_values("avg_duration", ascending=False)
            .head(20)
        )
        fig_dg2 = px.bar(dur_by_genre, x="avg_duration", y="primary_genre",
            orientation="h", color="avg_duration",
            color_continuous_scale="Reds", template="plotly_dark",
            title="Average Movie Runtime by Primary Genre (Top 20)",
            labels={"primary_genre":"","avg_duration":"Avg Duration (min)"},
            hover_data={"num_titles": True})
        fig_dg2.update_layout(**PL, height=560, yaxis=dict(autorange="reversed"))
        fig_dg2.update_coloraxes(showscale=False)
        st.plotly_chart(fig_dg2, use_container_width=True)

    # Content length breakdown
    st.markdown('<div class="section-header">📏 Content Length Categories</div>', unsafe_allow_html=True)
    cl = eda.content_length_breakdown(df)
    cl = cl[cl["content_length_category"] != "Unknown"]
    fig_cl = px.bar(cl, x="content_length_category", y="count", color="type",
        barmode="group", template="plotly_dark",
        color_discrete_sequence=["#e50914","#4bcffa"],
        title="Short / Medium / Long Content",
        labels={"content_length_category":"Category","count":"Count","type":"Type"})
    fig_cl.update_layout(**PL)
    st.plotly_chart(fig_cl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML Models
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""<div class="insight-box">
    ⚙️ <b>Note:</b> All ML models train on the <i>full unfiltered dataset</i> 
    for statistical validity. Models are cached after first run (~15–30s).
    </div>""", unsafe_allow_html=True)

    with st.spinner("🤖 Training models (cached after first run)…"):
        ml_data = None
        try:
            ml_data = run_ml_models_v4_lr(df_full)
            (clustered, inertia, pca_exp,
             acc, cm, report, imp_df,
             class_names, cv_scores, auc,
             elbow, comparison, clf, encoders) = ml_data
        except Exception as e:
            st.error(f"⚠️ ML Training Error: {e}")
            st.info("The Machine Learning tab is currently unavailable due to this error, but other tabs are still functional.")
            ml_data = None

    # Only show ML content if data was successfully generated
    if ml_data:
        # ── Clustering section ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔵 KMeans Clustering (PCA 2D Projection)</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_pca = px.scatter(clustered, x="pca_x", y="pca_y",
                color="cluster",
                hover_data=["title", "primary_genre", "type", "rating", "release_year"],
                title=f"KMeans Clusters — PCA 2D Representation",
                template="plotly_dark",
                color_discrete_sequence=NC,
                labels={"pca_x":"PC1","pca_y":"PC2"},
                opacity=0.7)
            fig_pca.update_layout(**PL, height=480)
            fig_pca.update_traces(marker=dict(size=6))
            st.plotly_chart(fig_pca, use_container_width=True)

        with c2:
            st.markdown("##### 📁 Cluster Summary")
            cluster_summary = (
                clustered.groupby("cluster")
                .agg(
                    Titles=("title","count"),
                    Top_Genre=("primary_genre", lambda x: x.mode().iloc[0] if not x.empty else "—"),
                )
                .reset_index().rename(columns={"cluster":"#"})
            )
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

            st.markdown("---")
            
            # 🔮 Live AI Predictor (Simplified implementation)
            st.markdown("##### 🔮 Live AI Content Predictor")
            st.write("Predict content type based on features:")
            
            p_gen = st.selectbox("Genre", sorted(df_full["primary_genre"].unique()))
            p_rat = st.selectbox("Rating", sorted(df_full["rating"].unique()), index=5)
            p_yr  = st.slider("Year", int(df_full["release_year"].min()), int(df_full["release_year"].max()), 2021)
            p_len = st.radio("Length", ["Short", "Medium", "Long"], index=1, horizontal=True)

            if st.button("🚀 Run AI Prediction", use_container_width=True):
                user_f = {"primary_genre":p_gen, "rating":p_rat, "release_year":p_yr, "content_length_category":p_len}
                res, conf, prob_dict = modeling.get_prediction(clf, encoders, class_names, user_f)
                
                clr = "#e50914" if res == "Movie" else "#4bcffa"
                st.markdown(f"""
                <div style="background:{clr}22; border:1px solid {clr}; border-radius:10px; padding:15px; text-align:center; margin-bottom:15px;">
                    <span style="color:#aaa; font-size:0.8rem; text-transform:uppercase;">AI Prediction</span><br>
                    <span style="color:{clr}; font-size:2.2rem; font-weight:900;">{res}</span><br>
                    <span style="color:{clr}aa; font-size:0.9rem;">{conf*100:.1f}% Confidence</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Show mini probability bar
                for cname, pval in prob_dict.items():
                    st.caption(f"{cname}: {pval*100:.1f}%")
                    st.progress(pval)
            
            st.markdown("---")
            st.markdown("##### 🎯 Model Performance")
            st.metric("Test Accuracy", f"{acc*100:.1f}%")
            st.metric("ROC AUC Score", f"{auc:.3f}" if auc else "N/A")

        # Row 2: Basic Insights
        st.markdown('<div class="section-header">🤖 Insights & Analysis — What drives the predictions?</div>', unsafe_allow_html=True)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Reds",
            template="plotly_dark", title="Global Feature Importance",
            labels={"feature":"","importance":"Score"})
        fig_imp.update_layout(**PL, height=340, yaxis=dict(autorange="reversed"))
        fig_imp.update_coloraxes(showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        # ⚖️ Model Comparison Section
        st.markdown('<div class="section-header">⚖️ Model Accuracy Comparison — Proof of Performance</div>', unsafe_allow_html=True)
        fig_comp = px.bar(comparison, x="Model", y="Mean Accuracy",
            error_y="Std", color="Model",
            color_discrete_sequence=["#e50914", "#4bcffa", "#ffd32a"],
            template="plotly_dark",
            title="5-Fold Cross-Validation Accuracy Comparison",
            labels={"Mean Accuracy":"Accuracy","Model":""},
            text=comparison["Mean Accuracy"].apply(lambda x: f"{x*100:.1f}%"))
        fig_comp.update_layout(**PL, showlegend=True, height=450)
        fig_comp.update_traces(textposition="outside")
        fig_comp.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig_comp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Recommendation Engine
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("""<div class="insight-box">
    🔍 <b>Content-Based Recommendation Engine</b><br>
    Powered by TF-IDF vectorization on Netflix title descriptions, genres, directors, 
    and cast. Enter any title to find the 8 most similar content pieces.
    </div>""", unsafe_allow_html=True)

    with st.spinner("🔧 Building recommendation engine…"):
        rec_df, tfidf_matrix = build_recs(df_full)

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        rec_input = st.text_input(
            "🎬 Enter a Netflix title to get recommendations:",
            placeholder="e.g. Breaking Bad, The Crown, Inception…"
        )
        top_n_recs = st.slider("Number of recommendations", 4, 12, 8)

    with col_r2:
        st.markdown("**💡 Try these titles:**")
        for t in ["Breaking Bad", "The Crown", "Bird Box", "Narcos", "Roma", "Squid Game"]:
            st.markdown(f"• {t}")

    if rec_input:
        recs = modeling.get_recommendations(rec_input, rec_df, tfidf_matrix, top_n=top_n_recs)
        if recs.empty:
            st.warning(f"No title found matching **'{rec_input}'**. Try a different spelling.")
        else:
            st.markdown(f"### 🎯 Recommendations for: *{rec_input}*")
            for _, row in recs.iterrows():
                sim_color = "#e50914" if row["similarity"] >= 50 else ("#ff9f43" if row["similarity"] >= 25 else "#888")
                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-score" style="background:{sim_color};">{row['similarity']:.0f}% match</span>
                    <div class="rec-title">{row['title']}</div>
                    <div class="rec-meta">
                        📺 {row['type']} &nbsp;·&nbsp; 
                        🎭 {str(row['listed_in'])[:60]} &nbsp;·&nbsp;
                        🎬 {str(row['director'])[:40]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Similarity chart
            fig_rec = px.bar(recs.sort_values("similarity"), x="similarity", y="title",
                orientation="h", color="similarity",
                color_continuous_scale="Reds",
                template="plotly_dark",
                title=f"Similarity Scores — Recommendations for '{rec_input}'",
                labels={"title":"","similarity":"Similarity Score (%)"})
            fig_rec.update_layout(**PL, height=350)
            fig_rec.update_coloraxes(showscale=False)
            st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:40px;color:#444;'>
            <div style='font-size:4rem;'>🎬</div>
            <div style='font-size:1.1rem;margin-top:10px;'>
                Type a Netflix title above to get personalized recommendations
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-header">🔎 Data Explorer</div>', unsafe_allow_html=True)

    display_cols = ["title", "type", "primary_genre", "listed_in", "primary_country",
                    "release_year", "year_added", "rating", "rating_group",
                    "duration", "content_length_category", "director", "cast"]
    available_cols = [c for c in display_cols if c in df.columns]

    sch1, sch2 = st.columns([3, 1])
    with sch1:
        search = st.text_input("🔍 Search by title, director, actor, or genre…", "")
    with sch2:
        sort_col = st.selectbox("Sort by", ["release_year", "year_added", "title", "rating"])
        sort_asc = st.checkbox("Ascending", value=False)

    display_df = df[available_cols].copy()
    if search:
        mask = (
            df["title"].str.contains(search, case=False, na=False) |
            df["director"].str.contains(search, case=False, na=False) |
            df["cast"].str.contains(search, case=False, na=False) |
            df["listed_in"].str.contains(search, case=False, na=False)
        )
        display_df = df[mask][available_cols]

    if sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=sort_asc)

    st.caption(f"Showing **{len(display_df):,}** of {len(df_full):,} titles")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=480)

    d1, d2 = st.columns(2)
    with d1:
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered CSV", csv_data,
            "streamscope_filtered.csv", "text/csv")
    with d2:
        st.markdown(f"""
        <div style='color:#666;font-size:0.82rem;padding:8px 0;'>
        🎬 Movies: {len(display_df[display_df['type']=='Movie']):,} &nbsp;|&nbsp;
        📺 TV Shows: {len(display_df[display_df['type']=='TV Show']):,}
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)