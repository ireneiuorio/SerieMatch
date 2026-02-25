"""
=============================================================
  SerieMatch — A Machine Learning Approach to TV Show Discovery
=============================================================
  Pipeline:
  1. EDA
  2. Preprocessing
  3. Clustering
  4. Classificazione
  5. Raccomandazione
=============================================================
"""

from src import eda, preprocessing, clustering, classification, recommender


def main():
    print("=" * 60)
    print("  SerieMatch — ML Pipeline")
    print("=" * 60)

    # Step 1 — EDA
    print("\n[1/5] Analisi Esplorativa...")
    df = eda.run()

    # Step 2 — Preprocessing
    print("\n[2/5] Preprocessing...")
    df_clean, X_genres, X_scaled, feature_names = preprocessing.run(df)

    # Step 3 — Clustering
    print("\n[3/5] Clustering...")
    df_clean, X_pca = clustering.run(df_clean, X_genres)

    # Step 4 — Classificazione
    print("\n[4/5] Classificazione...")
    classification.run(df_clean, X_scaled, feature_names)

    # Step 5 — Raccomandazione
    print("\n[5/5] Sistema di Raccomandazione...")
    recommender.run(df_clean, X_scaled)

    print("\n" + "=" * 60)
    print("  Pipeline completata! Grafici salvati in output/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()