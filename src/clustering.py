
#K-Means e DBSCAN con valutazione e visualizzazione
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def run(df, X_genres):

    # PCA 2D per visualizzazione: Riduce dimensionalità mantenendo più varianza possibile
    pca_vis = PCA(n_components=2, random_state=42)
    X_vis   = pca_vis.fit_transform(X_genres)
    print(f"  PCA 2D: varianza spiegata {pca_vis.explained_variance_ratio_.sum():.2%}")

    #Elbow Method:cerco il gomito
    K_range = range(2, 14)
    #Inertia = somma delle distanze quadratiche tra ogni punto e il suo centroide.
    inertia = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_genres)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(K_range), inertia, "bo-", linewidth=2)
    plt.xlabel("Numero di Cluster K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method — K-Means")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "clustering_elbow.png"), dpi=150)
    plt.close()

    # ── 3. Silhouette + Davies-Bouldin per ogni K ──
    sil_scores = []
    db_scores  = []
    for k in K_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_genres)
        sil_scores.append(silhouette_score(X_genres, labels))
        db_scores.append(davies_bouldin_score(X_genres, labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(list(K_range), sil_scores, "rs-", linewidth=2)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette Score (più alto = meglio)")

    axes[1].plot(list(K_range), db_scores, "gs-", linewidth=2)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Davies-Bouldin Score")
    axes[1].set_title("Davies-Bouldin Score (più basso = meglio)")

    plt.suptitle("Metriche di Valutazione Clustering — SerieMatch")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "clustering_metrics.png"), dpi=150)
    plt.close()

    best_k = list(K_range)[np.argmax(sil_scores)]
    print(f"  Miglior K (silhouette): {best_k}")

    #K-Means
    kmeans     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_genres)

    sil_final = silhouette_score(X_genres, df["cluster"])
    db_final  = davies_bouldin_score(X_genres, df["cluster"])
    print(f"  K-Means — Silhouette: {sil_final:.4f} | Davies-Bouldin: {db_final:.4f}")

    # DBSCAN confronto
    dbscan   = DBSCAN(eps=1.5, min_samples=5)
    db_labels = dbscan.fit_predict(X_genres)
    n_db     = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise  = list(db_labels).count(-1)
    print(f"  DBSCAN — Cluster: {n_db} | Noise: {n_noise}")

    #Dimensione cluste
    sizes = df["cluster"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    sizes.plot(kind="bar", color="steelblue", edgecolor="white")
    plt.title("Numero di Serie per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Numero di serie")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "clustering_sizes.png"), dpi=150)
    plt.close()

    # Visualizzazione PCA 2D
    sample_n = min(2000, len(X_genres))
    idx_s    = np.random.choice(len(X_genres), sample_n, replace=False)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_vis[idx_s, 0], X_vis[idx_s, 1],
                          c=df["cluster"].values[idx_s],
                          cmap="tab10", alpha=0.5, s=12)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Visualizzazione Cluster — PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "clustering_pca2d.png"), dpi=150)
    plt.close()

    #Profilo cluster
    print("\n  --- Caratteristiche medie per cluster ---")
    profile_cols = [c for c in ["vote_average", "popularity", "number_of_seasons",
                                 "number_of_episodes"] if c in df.columns]
    print(df.groupby("cluster")[profile_cols].mean().round(2))

    #Top 3 generi per cluster
    print("\n  --- Top 3 generi per cluster ---")
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    if genre_cols:
        for c in sorted(df["cluster"].unique()):
            sub  = df[df["cluster"] == c]
            top  = sub[genre_cols].mean().sort_values(ascending=False).head(3)
            names = [g.replace("genre_", "") for g in top.index]
            print(f"  Cluster {c:2d}: {', '.join(names)}")

    return df, X_vis