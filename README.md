# SerieMatch

Sistema di raccomandazione di serie TV basato su Machine Learning.

Progetto per il corso di Machine Learning  A.A. 2025/26
Università degli Studi di Salerno

Iuorio Irene 

## Descrizione

SerieMatch suggerisce serie TV simili a quella che hai appena finito di guardare.
Dato un titolo in input, il sistema trova le serie più simili combinando tre tecniche:

- **K-Means** per raggruppare le serie in cluster semantici basati sui generi
- **Random Forest** per validare che i cluster trovati abbiano una struttura reale
- **Cosine Similarity** per ordinare le raccomandazioni all'interno del cluster

## Dataset

[Full TMDB TV Shows Dataset 2024](https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows)
- 168.639 serie TV originali
- 7.741 serie dopo il preprocessing
- 29 colonne (generi, network, voti, popolarità, stagioni, episodi...)

## Struttura del progetto
```
SerieMatch/
│
├── data/
│   └── TMDB_tv_dataset_v3.csv
│
├── src/
│   ├── preprocessing.py       # Pulizia e costruzione delle feature
│   ├── eda.py                 # Analisi esplorativa del dataset
│   ├── clustering.py          # K-Means, Elbow Method, Silhouette
│   ├── classification.py      # Random Forest, Grid Search, valutazione
│   └── recommendation.py      # Similarità coseno e raccomandazioni
│
├── demo.py                    # Demo interattiva
├── requirements.txt
└── README.md
```

## Come eseguire la demo
```bash
# Installa le dipendenze
pip install -r requirements.txt

# Avvia la demo
python3 -m streamlit run demo.py
```

## Risultati principali

| Metrica | Valore |
|--------|--------|
| Numero di cluster (K-Means) | 13 |
| Silhouette Score | 0.4395 |
| Davies-Bouldin Score | 1.1537 |
| F1-Score cross-validazione | 0.9959 ± 0.0030 |
| Accuratezza test set | 99.97% |
| Errori su 1.549 serie | 5 |

Va sottolineato che l'elevato F1-score non è sorprendente: 
la Random Forest viene addestrata su feature che includono 
gli stessi generi utilizzati dal K-Means per costruire i cluster. 
Il risultato non va quindi interpretato come una generalizazione 
su dati completamente nuovi, ma come una conferma che la struttura 
dei cluster è solida e apprendibile.

## Requisiti
```
pandas
numpy
scikit-learn
scipy
streamlit
matplotlib
seaborn
```

## Autore

Irene Iuorio — Università degli Studi di Salerno
