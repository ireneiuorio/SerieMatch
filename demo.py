"""
=============================================================
  SerieMatch — streamlit_demo.py
  Demo web con Streamlit
  Esegui: streamlit run streamlit_demo.py

  python3 -m streamlit run demo.py
=============================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import ast
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── PAGINA ────────────────────────────────────────────────
st.set_page_config(
    page_title="SerieMatch",
    page_icon="TV",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .title { font-size: 2.5rem; font-weight: 800; color: #1e40af; }
    .subtitle { color: #64748b; font-size: 1rem; margin-top: -10px; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    .card:hover { border-left-color: #1d4ed8; box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
    .rec-title { font-size: 1.1rem; font-weight: 700; color: #1e293b; }
    .rec-genre { color: #64748b; font-size: 0.85rem; margin: 2px 0; }
    .badge {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .sim-bar-bg {
        background: #e2e8f0;
        border-radius: 6px;
        height: 8px;
        margin: 4px 0;
    }
    .info-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .info-value { font-weight: 600; color: #1e293b; }
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #cbd5e1 !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
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

@st.cache_resource(show_spinner="Caricamento dataset e modello ML...")
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

    for col in ["popularity", "number_of_episodes"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    mlb = MultiLabelBinarizer()
    ge  = mlb.fit_transform(df["genres_list"])
    gdf = pd.DataFrame(ge, columns=[f"g_{g}" for g in mlb.classes_], index=df.index)
    low = gdf.columns[gdf.var() < 0.005].tolist()
    gdf = gdf.drop(columns=low)
    X_genres = gdf.values

    le = LabelEncoder()
    df["status_enc"] = le.fit_transform(df["status"].fillna("Unknown"))

    all_nets  = [n for s in df["networks_list"] for n in s]
    top_nets  = pd.Series(all_nets).value_counts().head(20).index.tolist()
    net_cols  = []
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
    cl  = df.loc[idx, "cluster"]
    vec = X_scaled[idx]
    cdf = df[df["cluster"] == cl].copy()
    cdf["sim"] = [1 - cosine(vec, X_scaled[i]) for i in cdf.index]
    res = cdf[cdf["name"].str.lower() != title.strip().lower()] \
            .sort_values("sim", ascending=False).head(top_n)
    return res, df.loc[idx]


# ── APP ───────────────────────────────────────────────────

df, X_scaled = load()

# Header
st.markdown('<div class="title"> SerieMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Progetto di ML di Iuorio Irene</div>',
            unsafe_allow_html=True)
st.markdown("---")

# Barra di ricerca
col_search, col_btn = st.columns([5, 1])
with col_search:
    query = st.text_input("", placeholder="Cerca una serie... es. Breaking Bad, Stranger Things, Gomorra",
                          label_visibility="collapsed")
with col_btn:
    cerca = st.button("Cerca", use_container_width=True, type="primary")

# Serie suggerite come bottoni rapidi
st.markdown("**Prova subito:**")
quick = ["Breaking Bad", "Stranger Things", "Gomorra",
         "Dark", "Narcos", "Black Mirror", "Gomorra"]
cols = st.columns(len(quick))
for i, nome in enumerate(quick):
    if cols[i].button(nome, key=f"q{i}"):
        query = nome
        cerca = True

# Esegui ricerca
if query and (cerca or query):
    results, serie = recommend(query, df, X_scaled)

    if results is None:
        st.error(f"Serie '{query}' non trovata nel dataset. Prova con un altro titolo.")
    else:
        st.markdown("---")
        col_sx, col_dx = st.columns([1, 1.6])

        # ── PANNELLO SINISTRO ──
        with col_sx:
            genres  = ", ".join(serie["genres_list"][:4]) if serie["genres_list"] else "N/A"
            nets    = ", ".join(serie["networks_list"][:3]) if serie.get("networks_list") else "N/A"
            status  = serie.get("status", "N/A")
            status_color = "" if "Return" in str(status) else "" if "Cancel" in str(status) else ""

            st.markdown(f'<span class="badge">Cluster {int(serie["cluster"])}</span>',
                        unsafe_allow_html=True)
            st.markdown(f"## {serie['name']}")
            st.markdown(f"*{genres}*")

            c1, c2, c3 = st.columns(3)
            c1.metric("Voto", f"{serie['vote_average']:.1f} / 10")
            c2.metric("Stagioni", int(serie["number_of_seasons"]))
            c3.metric("Episodi", int(serie["number_of_episodes"]))

            st.markdown(f"**Network:** {nets}")
            st.markdown(f"**Status:** {status_color} {status}")

            if pd.notna(serie.get("overview", "")) and str(serie.get("overview", "")).strip():
                with st.expander("Trama"):
                    st.write(str(serie["overview"]))

        # ── PANNELLO DESTRO ──
        with col_dx:
            st.markdown("### Serie Consigliate")
            st.caption("Basate su genere e similarità coseno")

            for i, (_, row) in enumerate(results.iterrows()):
                sim     = float(row["sim"])
                genres_r = ", ".join(row["genres_list"][:3]) if row["genres_list"] else ""
                sim_pct  = int(sim * 100)
                bar_color = "#22c55e" if sim > 0.9 else "#3b82f6" if sim > 0.75 else "#94a3b8"

                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <div>
                            <span style="color:#94a3b8; font-size:0.8rem">#{i+1}</span>
                            <span class="rec-title"> {row['name']}</span>
                        </div>
                        <span style="font-weight:700; color:{bar_color}">{sim:.3f}</span>
                    </div>
                    <div class="rec-genre">{genres_r}</div>
                    <div style="color:#f59e0b; font-size:0.85rem">{row['vote_average']:.1f}</div>
                    <div class="sim-bar-bg">
                        <div style="width:{sim_pct}%; background:{bar_color}; height:8px; border-radius:6px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("SerieMatch · K-Means + Random Forest + Cosine Similarity · TMDB Dataset 150k")