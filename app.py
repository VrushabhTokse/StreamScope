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

/* Section dividers for merged tabs */
.section-divider {
    font-size: 1.6rem;
    font-weight: 900;
    color: #fff;
    padding: 12px 0;
    margin: 40px 0 20px 0;
    border-bottom: 2px solid #e50914;
    text-transform: uppercase;
    letter-spacing: 2px;
    background: linear-gradient(90deg, rgba(229,9,20,0.15) 0%, transparent 100%);
    border-radius: 4px;
    display: block;
    text-align: center;
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
(tab1, tab2, tab3) = st.tabs([
    "📊 Dashboard Insights",
    "🤖 AI Intelligence Core",
    "🔍 Content Discovery Hub",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard Insights (Growth, Genre, Global)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-divider">📈 Netflix Growth & Trends</div>', unsafe_allow_html=True)
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



    st.markdown("---")
    st.markdown('<div class="section-divider">🎭 Genre & Ratings Depth</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Top 15 Genres</div>', unsafe_allow_html=True)
    gd = eda.genre_distribution(df, top_n=15)
    fig_genre = px.bar(gd, x="count", y="genre", orientation="h",
        color="count", color_continuous_scale="Reds", template="plotly_dark",
        title="Top 15 Genres by Title Count",
        labels={"genre":"","count":"Titles"})
    fig_genre.update_layout(**PL, yaxis=dict(autorange="reversed"))
    fig_genre.update_coloraxes(showscale=False)
    st.plotly_chart(fig_genre, use_container_width=True)


    # Top genres per year
    st.markdown('<div class="section-header">📊 Top 3 Genres Per Year</div>', unsafe_allow_html=True)
    tgy = eda.top_genres_per_year(df, top_n=3)
    if not tgy.empty:
        fig_tgy = px.bar(tgy, x="year_added", y="count", color="genres_list",
            barmode="stack", template="plotly_dark",
            color_discrete_sequence=NC,
            labels={"year_added":"Year","count":"Count","genres_list":"Genre"})
        fig_tgy.update_layout(**PL)
        st.plotly_chart(fig_tgy, use_container_width=True)



    st.markdown("---")
    st.markdown('<div class="section-divider">🌍 Global Footprint</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-header">Top 20 Countries</div>', unsafe_allow_html=True)
    top20 = country_data.head(20)
    fig_c = px.bar(top20, x="count", y="country", orientation="h",
        color="count", color_continuous_scale="Reds", template="plotly_dark",
        title="Top 20 Content-Producing Countries",
        labels={"country":"","count":"Titles"})
    fig_c.update_layout(**PL, yaxis=dict(autorange="reversed"))
    fig_c.update_coloraxes(showscale=False)
    st.plotly_chart(fig_c, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI Intelligence Core (Machine Learning)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
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
# TAB 3 — Content Discovery Hub (Smart Search & Recommendations)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-divider">🔍 AI Recommendation Engine</div>', unsafe_allow_html=True)
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


    st.markdown("---")
    st.markdown('<div class="section-divider">📋 Raw Data Explorer</div>', unsafe_allow_html=True)
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