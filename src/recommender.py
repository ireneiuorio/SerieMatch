
#Sistema di raccomandazione
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


def recommend(title, df, X_scaled, top_n=5):
    """Dato il titolo di una serie, restituisce le top_n più simili."""
    # Ricerca esatta (case-insensitive)
    match = df[df["name"].str.lower() == title.strip().lower()]

    # Ricerca parziale
    if match.empty:
        match = df[df["name"].str.lower().str.contains(title.strip().lower(), na=False)]
        if match.empty:
            return None, None
        print(f"  (Titolo non trovato esattamente, uso: '{match.iloc[0]['name']}')")

#Recupero informazioni della serie
    idx     = match.index[0] #posizione della serie
    cluster = df.loc[idx, "cluster"] #cluster a cui appartiene
    vec     = X_scaled[idx] #vettore numerico completo della serie

    #Filtra serie dello stesso cluster
    cdf  = df[df["cluster"] == cluster].copy()
    sims = []
    for i in cdf.index:
        try:
            #Similarità coseno
            s = 1 - cosine(vec, X_scaled[i])
        except:
            s = 0.0
        sims.append(s)

#Ordinamento per similarità più alta
    cdf["similarity"] = sims
    result = (
        #Escludo la serie stessa
        cdf[cdf.index != idx]
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )
    #Ritorno serie consigliate e serie originale
    return result, df.loc[idx]


#Stampa
def print_recommendations(title, df, X_scaled, top_n=5):
    results, serie = recommend(title, df, X_scaled, top_n)
    if results is None:
        print(f"  Serie '{title}' non trovata.")
        return

    genres  = ", ".join(serie["genres_list"][:3]) if serie["genres_list"] else "N/A"
    network = ", ".join(serie["networks_list"][:2]) if serie.get("networks_list") else "N/A"

    print(f"\n  {'='*56}")
    print(f"   {serie['name']}  (Cluster {int(serie['cluster'])})")
    print(f"      Generi: {genres}  |  Network: {network}")
    print(f"       {serie['vote_average']:.1f}  |  Stagioni: {int(serie.get('number_of_seasons', 0))}")
    print(f"  {'='*56}")
    print(f"  {'#':<4} {'Titolo':<35} {'Voto':>5} {'Sim':>6}")
    print(f"  {'-'*54}")
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"  {i:<4} {row['name'][:34]:<35} {row['vote_average']:>5.1f} {row['similarity']:>6.3f}")
    print(f"  {'='*56}\n")


def run(df, X_scaled):


    # Lista di serie famose da cercare nel dataset
    candidates = [
        "Breaking Bad", "Stranger Things", "Game of Thrones",
        "The Crown", "Narcos", "Black Mirror",
        "Dark", "Money Heist", "Squid Game",
        "The Office", "Friends", "Sherlock",
        "Gomorra", "Il Commissario Montalbano",
        "Suburra", "Baby", "Skam Italia"
    ]

    # Trova quelle disponibili nel dataset
    available = []
    for serie in candidates:
        m = df[df["name"].str.lower() == serie.lower()]
        if not m.empty:
            available.append(serie)
        if len(available) >= 5:
            break

    if not available:
        # Fallback: prendi le 5 serie più votate
        available = df.sort_values("vote_count", ascending=False).head(5)["name"].tolist()

    print("\n  ── Demo Sistema di Raccomandazione Serie TV ──")
    for serie in available:
        print_recommendations(serie, df, X_scaled, top_n=5)