#Allena una Random Forest per predire il cluster di una serie TV
#Ottimizza gli iper-parametri con GridSearch, valuta il modello e salva i risultati

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os


#Risale alla root del progetto crea output/plots e output/models
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR  = os.path.join(BASE_DIR, "output", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")
#Se non esistono crea le cartelle
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


#df-> dataframe originale con colonna "cluster"
#X_scaled-> solo feature già scalate
#feature_names-> nomi reali delle colonne (passati da preprocessing.py)
def run(df, X_scaled, feature_names=None):

#La cosa che voglio prevedere è il cluster
#y è la risposta corretta che il modello deve imparare
    y = df["cluster"].values

    #Divido il dataset 80% training set, il restate test set
    #RandomState=stessi risultati ogni volta
    #Stratify mantiene proporzione nei cluster
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    #Stampo quante serie vanno in train e in test set: shape: numero di righe
    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    #Trova quali cluster esistono e quante serie ci sono per ciascuno
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 4))
    #Crea grafico a barre
    plt.bar(unique, counts, color="steelblue", edgecolor="white")
    plt.title("Distribuzione Classi (Cluster) — SerieMatch")
    plt.xlabel("Cluster") #Etichetta asse x
    plt.ylabel("Numero di serie") #Etichetta asse y
    plt.xticks(unique) #Visualizzare correttamente i numeri
    plt.tight_layout() #Sistema spazi
    #Salva l'immagine
    plt.savefig(os.path.join(PLOTS_DIR, "classification_class_distribution.png"), dpi=150)
    plt.close()

    #Grid Search
    print("  GridSearch in corso...")
    param_grid = {
        "n_estimators":      [100, 200], #Quanti alberi usare
        "max_depth":         [None, 10, 20], #profondità massima
        "min_samples_split": [2, 5] #minimo numero di dati per dividere un nodo
    }

    #Crea il modello base usando tutti i core del computer
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
   #Per ogni combinazione di parametri:
    #1 Divide il TRAIN in 5 parti
    #2 Allena su 4
    #3 Valuta su 1
    #Ripete 5 volte
    #Fa la media F1, poi confronta e sceglie la migliore
    gs = GridSearchCV(rf, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=0)

    #Il modello prova tutte e 12 le combinazioni
    gs.fit(X_train, y_train)

    #Salva il modello migliore
    best_rf = gs.best_estimator_
    print(f"  Migliori parametri: {gs.best_params_}")

    #Cross Validation:Prende il modello migliore e lo rivaluta 5 volte su tutto il dataset e restituisce 5 punteggi
    cv = cross_val_score(best_rf, X_scaled, y, cv=5, scoring="f1_weighted")
    #Stampo la media e la deviazione standard
    print(f"  Cross-Val F1 (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")

    #Test set
    y_pred = best_rf.predict(X_test)
    print("\n  --- Classification Report ---")
    #Il modello vede dati mai visti prima
    print(classification_report(y_test, y_pred))

    # Crea Confusion Matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    #Trasforma in un grafico blu
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Random Forest (SerieMatch)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "classification_confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature Importance
    importances = best_rf.feature_importances_
    top_n       = 15
    #Ordina le feature dalla più importante e prende le prime 15
    indices     = np.argsort(importances)[::-1][:top_n]

    # Usa i nomi reali delle feature se disponibili, altrimenti usa F1, F2...
    if feature_names is not None:
        def clean_name(n):
            n = n.replace("genre_", "").replace("net_", "Net: ")
            n = n.replace("status_enc", "Status")
            n = n.replace("vote_average", "Voto medio").replace("vote_count", "N. voti")
            n = n.replace("popularity", "Popolarità").replace("number_of_seasons", "Stagioni")
            n = n.replace("number_of_episodes", "Episodi")
            return n[:22]
        labels = [clean_name(feature_names[i]) for i in indices]
    else:
        labels = [f"F{i+1}" for i in indices]

    plt.figure(figsize=(12, 5))
    plt.bar(range(top_n), importances[indices], color="coral", edgecolor="white")
    plt.xticks(range(top_n), labels, rotation=45, ha="right", fontsize=10)
    plt.title(f"Top {top_n} Feature Importance — Random Forest")
    plt.ylabel("Importanza")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "classification_feature_importance.png"), dpi=150)
    plt.close()

    # Salva modello sul disco
    joblib.dump(best_rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    print(f"  Modello salvato in output/models/random_forest.pkl")

    return best_rf