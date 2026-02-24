"""
=============================================================
  SerieMatch — eda.py
  Analisi Esplorativa del Dataset (EDA)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


#Caricamento dati
def load_data():
    path = os.path.join(DATA_DIR, "TMDB_tv_dataset_v3.csv")
    #Forza lettura unica
    df = pd.read_csv(path, low_memory=False)
    #Restituisce numero righe e numero colonne
    print(f"  Dataset caricato: {df.shape[0]:,} serie, {df.shape[1]} colonne")
    return df

#Converte la stringa generi in lista
#Divide per virgola, rimuove gli spazi, elimina stringhe vuote
def parse_genres(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [g.strip() for g in str(val).split(",") if g.strip()]

#Estrae i network
def parse_networks(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [n.strip() for n in str(val).split(",") if n.strip()]


def run():
    df = load_data()

    # Applica la funzione riga per riga
    df["genres_list"]   = df["genres"].apply(parse_genres)
    df["networks_list"] = df["networks"].apply(parse_networks)

    print(f"\n  Colonne disponibili: {df.columns.tolist()}")

    #Missing Value
    #Crea una matrice booleana, True se il valore è NaN, false altrimenti
    print("\n  --- Missing Values ---")
    missing     = df.isnull().sum()
    #Calcolo la percentuale
    missing_pct = (missing / len(df) * 100).round(2)
    mv_df = pd.DataFrame({"missing": missing, "%": missing_pct})
    #Filtra solo colonne con almeno un NaN.
    mv_df = mv_df[mv_df["missing"] > 0].sort_values("%", ascending=False).head(15)
    print(mv_df)

    plt.figure(figsize=(10, 5))
    #Grafico
    mv_df["%"].plot(kind="bar", color="salmon", edgecolor="white")
    plt.title("Percentuale di Missing Values per Colonna")
    plt.ylabel("%")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_missing_values.png"), dpi=150)
    plt.close()

    # Distribuzione Voto Medio
    plt.figure(figsize=(8, 4))
    #Divide i valori in 30 intervalli, conta quante osservazioni per intervallo
    #KDE:curva di densità stimata
    sns.histplot(df["vote_average"].dropna(), bins=30, kde=True, color="steelblue")
    plt.title("Distribuzione del Voto Medio")
    plt.xlabel("Voto medio")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_vote_distribution.png"), dpi=150)
    plt.close()

    #Distribuzione Popolarità
    plt.figure(figsize=(8, 4))
    sns.histplot(df["popularity"].dropna(), bins=40, kde=True, color="coral")
    plt.title("Distribuzione della Popolarità")
    plt.xlabel("Popolarità")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_popularity.png"), dpi=150)
    plt.close()

    # Top Generi
    all_genres    = [g for sublist in df["genres_list"] for g in sublist]
    genre_counts  = pd.Series(all_genres).value_counts().head(15)

    plt.figure(figsize=(10, 5))
    genre_counts.plot(kind="bar", color="mediumseagreen", edgecolor="white")
    plt.title("Top 15 Generi — Serie TV")
    plt.ylabel("Numero di serie")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_genres.png"), dpi=150)
    plt.close()

    # Top Network
    all_networks   = [n for sublist in df["networks_list"] for n in sublist]
    network_counts = pd.Series(all_networks).value_counts().head(15)

    plt.figure(figsize=(10, 5))
    network_counts.plot(kind="bar", color="slateblue", edgecolor="white")
    plt.title("Top 15 Network / Piattaforme")
    plt.ylabel("Numero di serie")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_networks.png"), dpi=150)
    plt.close()

    # Conta gli Status (In corso / Conclusa / Cancellata)
    if "status" in df.columns:
        status_counts = df["status"].value_counts().head(8)
        plt.figure(figsize=(8, 4))
        status_counts.plot(kind="bar", color="darkorange", edgecolor="white")
        plt.title("Distribuzione Status delle Serie")
        plt.ylabel("Numero di serie")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "eda_status.png"), dpi=150)
        plt.close()

    #Serie per Anno ──
    if "first_air_date" in df.columns:
        df["first_year"] = pd.to_datetime(df["first_air_date"], errors="coerce").dt.year
        year_counts = df["first_year"].value_counts().sort_index()
        year_counts = year_counts[(year_counts.index >= 1980) & (year_counts.index <= 2024)]

        plt.figure(figsize=(14, 4))
        year_counts.plot(kind="bar", color="teal", width=0.9)
        plt.title("Numero di Serie per Anno di Prima Messa in Onda")
        plt.xlabel("Anno")
        plt.ylabel("Numero di serie")
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "eda_series_per_year.png"), dpi=150)
        plt.close()

    #Heatmap Correlazioni:Correlazione di Pearson
    num_cols = [c for c in ["vote_average", "vote_count", "popularity",
                             "number_of_seasons", "number_of_episodes"]
                if c in df.columns]

    if len(num_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", square=True)
        plt.title("Heatmap Correlazioni — Feature Numeriche")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "eda_correlation_heatmap.png"), dpi=150)
        plt.close()

    #Lingue originali
    if "original_language" in df.columns:
        lang_counts = df["original_language"].value_counts().head(12)
        plt.figure(figsize=(10, 4))
        lang_counts.plot(kind="bar", color="mediumpurple", edgecolor="white")
        plt.title("Top 12 Lingue Originali")
        plt.ylabel("Numero di serie")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "eda_languages.png"), dpi=150)
        plt.close()

    print(f"  Grafici EDA salvati in output/plots/")
    return df