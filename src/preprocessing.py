#Pulizia dati, encoding, normalizzazione
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.impute import SimpleImputer
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


#Rimuove le righe con valori al di fuori dell'intervallo [lo, hi] percentile.
#Default bilaterale: taglia l'1% inferiore e l'1% superiore (lo=0.01, hi=0.99).
#Con lo=0 il taglio è solo superiore (nessuna rimozione dal basso).
def remove_outliers_percentile(df, col, lo=0.01, hi=0.99):
    lower = df[col].quantile(lo)
    upper = df[col].quantile(hi)
    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"  Outlier rimossi in '{col}': {before - len(df)} righe")
    return df


def run(df):
    print(f"  Shape iniziale: {df.shape}")

    # Filtra solo serie con abbastanza voti
    min_votes = 20
    df = df[df["vote_count"] >= min_votes].copy()
    print(f"  Shape dopo filtro voti >= {min_votes}: {df.shape}")

    #Filtra solo status significativi
    if "status" in df.columns:
        valid_status = ["Ended", "Returning Series", "Canceled", "In Production"]
        df = df[df["status"].isin(valid_status)].copy()
        print(f"  Shape dopo filtro status: {df.shape}")

    # Rimuovi righe senza titolo o genere
    df = df.dropna(subset=["name", "genres"])
    df = df[df["genres"].str.strip() != ""]
    print(f"  Shape dopo drop righe critiche: {df.shape}")

    # Gestione colonne numeriche
    num_cols = ["vote_average", "vote_count", "popularity"]
    if "number_of_seasons" in df.columns:
        num_cols.append("number_of_seasons")
    if "number_of_episodes" in df.columns:
        num_cols.append("number_of_episodes")

    #Sostituisci zeri impossibili con NaN
    for col in ["popularity"]:
        if col in df.columns:
            zeros = (df[col] == 0).sum()
            df[col] = df[col].replace(0, np.nan)
            if zeros > 0:
                print(f"  Zeri sostituiti con NaN in '{col}': {zeros}")

    #Imputazione con mediana
    imp = SimpleImputer(strategy="median")
    df[num_cols] = imp.fit_transform(df[num_cols])
    print(f"  Imputazione mediana su: {num_cols}")

    # Rimozione Outlier ──
    df = remove_outliers_percentile(df, "popularity")
    if "number_of_episodes" in df.columns:
        df = remove_outliers_percentile(df, "number_of_episodes", lo=0, hi=0.99)
    print(f"  Shape dopo rimozione outlier: {df.shape}")

    # Encoding Generi (MultiLabelBinarizer)
    mlb = MultiLabelBinarizer()
    genres_enc = mlb.fit_transform(df["genres_list"])
    genres_df  = pd.DataFrame(genres_enc,
                               columns=[f"genre_{g}" for g in mlb.classes_],
                               index=df.index)

    # Rimuovi generi con varianza quasi zero
    low_var = genres_df.columns[genres_df.var() < 0.005].tolist()
    if low_var:
        genres_df = genres_df.drop(columns=low_var)
        print(f"  Generi a bassa varianza rimossi: {[g.replace('genre_','') for g in low_var]}")

    print(f"  Generi codificati: {genres_df.shape[1]} generi unici")

    #Encoding Network (top 20 + "Other")
    top_networks = []
    if "networks_list" in df.columns:
        all_nets = [n for sublist in df["networks_list"] for n in sublist]
        top_networks = pd.Series(all_nets).value_counts().head(20).index.tolist()

        for net in top_networks:
            col_name = f"net_{net.replace(' ', '_').replace('-', '_')[:20]}"
            df[col_name] = df["networks_list"].apply(lambda x: 1 if net in x else 0)

        net_cols = [c for c in df.columns if c.startswith("net_")]
        print(f"  Network codificati: {len(net_cols)} top network")

    #Encoding Status
    if "status" in df.columns:
        le = LabelEncoder()
        df["status_enc"] = le.fit_transform(df["status"].fillna("Unknown"))

    # Visualizzazione distribuzione DOPO preprocessing
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, col in enumerate(["vote_average", "popularity", "vote_count"]):
        axes[i].hist(df[col], bins=30, color="mediumseagreen", edgecolor="white")
        axes[i].set_title(f"{col}")
    plt.suptitle("Feature Numeriche — Dopo Preprocessing")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "preprocessing_after.png"), dpi=150)
    plt.close()

    # Feature per clustering (solo generi)
    X_genres = genres_df.values

    # Feature complete per classificatore
    feat_cols = num_cols.copy()
    if "status_enc" in df.columns:
        feat_cols.append("status_enc")

    net_cols = [c for c in df.columns if c.startswith("net_")]
    features_full = pd.concat([
        df[feat_cols].reset_index(drop=True),
        genres_df.reset_index(drop=True),
        df[net_cols].reset_index(drop=True) if net_cols else pd.DataFrame()
    ], axis=1)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features_full)

    df = df.reset_index(drop=True)

    print(f"  Feature clustering (generi): {X_genres.shape}")
    print(f"  Feature matrix completa: {X_scaled.shape}")
    print(f"  Preprocessing completato.")

    # Aggiungi genre_cols al df per clustering o analisi successive
    genres_df_reset = genres_df.reset_index(drop=True)
    for col in genres_df_reset.columns:
        df[col] = genres_df_reset[col].values

    # Nomi delle feature per il classificatore (usati nel grafico Feature Importance)
    feature_names = list(features_full.columns)

    return df, X_genres, X_scaled, feature_names