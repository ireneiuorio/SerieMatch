"""
=============================================================
  SerieMatch — streamlit_demo.py
  Demo web con Streamlit
  Esegui: python3 -m streamlit run demo.py
=============================================================

"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="SerieMatch",
    page_icon="SM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #fafafa; }
    .block-container { padding: 3rem 4rem !important; max-width: 1300px !important; }

    /* Header */
    .header { margin-bottom: 2.5rem; }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .header-sub {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Search */
    .search-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .stTextInput > div > div > input {
        background: white !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 8px !important;
        color: #111827 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1rem !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #111827 !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: #d1d5db !important; }

    /* Buttons */
    .stButton > button {
        background: white !important;
        color: #374151 !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: #111827 !important;
        color: white !important;
        border-color: #111827 !important;
    }

    /* Quick label */
    .quick-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        font-weight: 600;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }

    /* Badge cluster */
    .cluster-badge {
        display: inline-block;
        background: #111827;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* Serie panel */
    .serie-panel {
        background: white;
        border-radius: 12px;
        padding: 1.75rem;
        border: 1px solid #f3f4f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .serie-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin: 0 0 0.2rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .serie-genres {
        color: #9ca3af;
        font-size: 0.82rem;
        margin-bottom: 1.25rem;
    }
    .info-row { display: flex; gap: 0.75rem; margin-bottom: 1.25rem; }
    .info-box {
        background: #f9fafb;
        border-radius: 8px;
        padding: 0.75rem;
        flex: 1;
        text-align: center;
        border: 1px solid #f3f4f6;
    }
    .info-box-label {
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        font-weight: 600;
    }
    .info-box-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-top: 2px;
    }
    .meta-item {
        font-size: 0.83rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }
    .meta-item strong { color: #374151; font-weight: 600; }

    /* Raccomandazioni */
    .rec-header {
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
    }
    .rec-caption {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 1rem;
    }
    .rec-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
        border: 1px solid #f3f4f6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: box-shadow 0.15s;
    }
    .rec-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .rec-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
    }
    .rec-num { font-size: 0.68rem; color: #d1d5db; font-weight: 600; margin-bottom: 2px; }
    .rec-name { font-size: 0.95rem; font-weight: 600; color: #111827; }
    .rec-sim { font-size: 0.9rem; font-weight: 700; color: #111827; }
    .rec-meta { font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.5rem; }
    .rec-bar-bg {
        background: #f3f4f6;
        border-radius: 4px;
        height: 3px;
        overflow: hidden;
    }
    .rec-bar { height: 3px; border-radius: 4px; background: #111827; }

    /* Divider */
    hr { border-color: #f3f4f6 !important; margin: 2rem 0 !important; }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.82rem !important;
        color: #6b7280 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────

def parse_genres(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [g.strip() for g in str(val).split(",") if g.strip()]

def parse_networks(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [n.strip() for n in str(val).split(",") if n.strip()]

@st.cache_resource(show_spinner="Caricamento dataset...")
def load():
    df = pd.read_csv(os.path.join(DATA_DIR, "TMDB_tv_dataset_v3.csv"), low_memory=False)
    df["genres_list"]   = df["genres"].apply(parse_genres)
    df["networks_list"] = df["networks"].apply(parse_networks)

    df = df[df["vote_count"] >= 20].copy()
    valid = ["Ended", "Returning Series", "Canceled", "In Production"]
    df = df[df["status"].isin(valid)].copy()
    df = df.dropna(subset=["name", "genres"])
    df = df[df["genres"].str.strip() != ""]

    num_cols = ["vote_average", "vote_count", "popularity",
                "number_of_seasons", "number_of_episodes"]
    df["popularity"] = df["popularity"].replace(0, np.nan)
    imp = SimpleImputer(strategy="median")
    df[num_cols] = imp.fit_transform(df[num_cols])

    for col, lo_q in [("popularity", 0.01), ("number_of_episodes", 0.0)]:
        lo = df[col].quantile(lo_q)
        hi = df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    mlb = MultiLabelBinarizer()
    ge  = mlb.fit_transform(df["genres_list"])
    gdf = pd.DataFrame(ge, columns=[f"g_{g}" for g in mlb.classes_], index=df.index)
    low = gdf.columns[gdf.var() < 0.005].tolist()
    gdf = gdf.drop(columns=low)
    X_genres = gdf.values

    le = LabelEncoder()
    df["status_enc"] = le.fit_transform(df["status"].fillna("Unknown"))

    all_nets = [n for s in df["networks_list"] for n in s]
    top_nets = pd.Series(all_nets).value_counts().head(20).index.tolist()
    net_cols = []
    for net in top_nets:
        col = f"net_{net[:15]}"
        df[col] = df["networks_list"].apply(lambda x: 1 if net in x else 0)
        net_cols.append(col)

    feat = pd.concat([
        df[num_cols + ["status_enc"]].reset_index(drop=True),
        gdf.reset_index(drop=True),
        df[net_cols].reset_index(drop=True)
    ], axis=1)
    X_scaled = StandardScaler().fit_transform(feat)

    km = KMeans(n_clusters=13, random_state=42, n_init=10)
    df = df.reset_index(drop=True)
    df["cluster"] = km.fit_predict(X_genres)

    gdf_r = gdf.reset_index(drop=True)
    for col in gdf_r.columns:
        df[col] = gdf_r[col].values

    return df, X_scaled

def recommend(title, df, X_scaled, top_n=5):
    m = df[df["name"].str.lower() == title.strip().lower()]
    if m.empty:
        m = df[df["name"].str.lower().str.contains(title.strip().lower(), na=False)]
    if m.empty:
        return None, None
    idx = m.index[0]
    vec = X_scaled[idx]
    cdf = df[df["cluster"] == df.loc[idx, "cluster"]].copy()
    cdf["sim"] = [1 - cosine(vec, X_scaled[i]) for i in cdf.index]
    res = cdf[cdf["name"].str.lower() != title.strip().lower()] \
            .sort_values("sim", ascending=False).head(top_n)
    return res, df.loc[idx]


# ── APP ───────────────────────────────────────────────────
df, X_scaled = load()

# Header
st.markdown("""
<div class="header">
    <div class="header-title">SerieMatch</div>
    <div class="header-sub">Sistema di raccomandazione basato su K-Means, Random Forest e Cosine Similarity</div>
</div>
""", unsafe_allow_html=True)

# Ricerca
st.markdown('<div class="search-label">Cerca una serie</div>', unsafe_allow_html=True)
col_search, col_btn = st.columns([5, 1])
with col_search:
    query = st.text_input("", placeholder="Breaking Bad, Stranger Things, Dark...",
                          label_visibility="collapsed")
with col_btn:
    cerca = st.button("Cerca", use_container_width=True)

st.markdown('<div class="quick-label">Suggeriti</div>', unsafe_allow_html=True)
quick = ["Breaking Bad", "Stranger Things", "Gomorra", "Dark", "Narcos", "Black Mirror", "The Crown"]
cols = st.columns(len(quick))
for i, nome in enumerate(quick):
    if cols[i].button(nome, key=f"q{i}"):
        query = nome

# Risultati
if query:
    results, serie = recommend(query, df, X_scaled)

    if results is None:
        st.error(f"'{query}' non trovata. Prova con un altro titolo.")
    else:
        st.markdown("---")
        col_sx, col_dx = st.columns([1, 1.6], gap="large")

        with col_sx:
            genres_str = ", ".join(serie["genres_list"][:4]) if serie["genres_list"] else "N/A"
            nets_str   = ", ".join(serie["networks_list"][:3]) if serie.get("networks_list") else "N/A"
            status     = serie.get("status", "N/A")
            voto       = f"{serie['vote_average']:.1f}"
            stagioni   = int(serie["number_of_seasons"])
            episodi    = int(serie["number_of_episodes"])
            cluster_id = int(serie["cluster"])

            st.markdown(f"""
            <div class="serie-panel">
                <div class="cluster-badge">Cluster {cluster_id}</div>
                <div class="serie-title">{serie['name']}</div>
                <div class="serie-genres">{genres_str}</div>
                <div class="info-row">
                    <div class="info-box">
                        <div class="info-box-label">Voto</div>
                        <div class="info-box-value">{voto}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-box-label">Stagioni</div>
                        <div class="info-box-value">{stagioni}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-box-label">Episodi</div>
                        <div class="info-box-value">{episodi}</div>
                    </div>
                </div>
                <div class="meta-item"><strong>Network</strong> &nbsp; {nets_str}</div>
                <div class="meta-item"><strong>Status</strong> &nbsp; {status}</div>
            </div>
            """, unsafe_allow_html=True)

            overview = str(serie.get("overview", "")).strip()
            if overview and overview != "nan":
                with st.expander("Trama"):
                    st.write(overview)

        with col_dx:
            st.markdown('<div class="rec-header">Serie consigliate</div>', unsafe_allow_html=True)
            st.markdown('<div class="rec-caption">Similarità coseno nello stesso cluster</div>',
                        unsafe_allow_html=True)

            for i, (_, row) in enumerate(results.iterrows()):
                sim      = float(row["sim"])
                genres_r = ", ".join(row["genres_list"][:3]) if row["genres_list"] else ""
                voto_r   = f"{row['vote_average']:.1f}"
                sim_pct  = int(sim * 100)

                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-card-top">
                        <div>
                            <div class="rec-num">#{i+1}</div>
                            <div class="rec-name">{row['name']}</div>
                        </div>
                        <div class="rec-sim">{sim:.3f}</div>
                    </div>
                    <div class="rec-meta">{voto_r}/10 &nbsp;&middot;&nbsp; {genres_r}</div>
                    <div class="rec-bar-bg">
                        <div class="rec-bar" style="width:{sim_pct}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#d1d5db; font-size:0.72rem; letter-spacing:0.05em;">'
    'SerieMatch &nbsp;&middot;&nbsp; Iuorio Irene &nbsp;&middot;&nbsp; TMDB 168K'
    '</div>',
    unsafe_allow_html=True
)