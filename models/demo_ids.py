# demo_ids.py
# Démo clé en main – Détection d'attaques réseau (CICIDS2017)
# -----------------------------------------------------------
# 1) charge un CSV, 2) nettoie, 3) cible Attack vs Benign,
# 4) split train/test, 5) entraîne LogReg & RandomForest,
# 6) évalue (PR-AUC, rapports, confusion), 7) sauvegarde,
# 8) fait une prédiction d'exemple.

import os, sys, textwrap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score,
    precision_recall_curve, ConfusionMatrixDisplay
)

# ------------------------
# Paramètres utilisateur
# ------------------------
DATA_CSV = Path('data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
MODEL_DIR = Path('models'); MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("\n=== Démo : Détection d'attaques réseau (CICIDS2017) ===\n")

# 1) Chargement
if not DATA_CSV.exists():
    print(f"[ERREUR] Fichier introuvable : {DATA_CSV}. Placez le CSV puis relancez.")
    sys.exit(1)

print("[1/8] Chargement du CSV…")
df = pd.read_csv(DATA_CSV, low_memory=False)
print("  - Dimensions :", df.shape)
print("  - Colonnes (10 premières) :", list(df.columns[:10]))

# 2) Nettoyage minimal
print("[2/8] Nettoyage minimal…")
df.columns = [c.strip().replace(' ', '_').replace('/', '_') for c in df.columns]
df = df.replace([np.inf, -np.inf], np.nan)
num_cols = df.select_dtypes('number').columns
if len(num_cols) > 0:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
drop_empty = [c for c in df.columns if df[c].isna().all()]
if drop_empty:
    df = df.drop(columns=drop_empty)
    print("  - Colonnes vides supprimées :", drop_empty)
print("  - Données prêtes.")

# 3) Cible binaire
print("[3/8] Construction de la cible Attack vs Benign…")
if 'Label' not in df.columns:
    print("[ERREUR] Colonne 'Label' absente. Ce CSV doit provenir de CICIDS2017.")
    sys.exit(1)

y = (~df['Label'].str.contains('Benign', case=False)).astype(int)
X = df.drop(columns=['Label'])
print("  - Répartition (0=Benign,1=Attack) :\n", y.value_counts())

# 4) Split train/test (sans mélange complet du temps)
print("[4/8] Split train/test (80/20, shuffle=False)…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False)
print("  - Train:", X_train.shape, " Test:", X_test.shape)

# 5) Modèles
print("[5/8] Entraînement de deux modèles…")
# A) LogReg
logreg = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))
])
logreg.fit(X_train, y_train)

# B) RandomForest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)
print("  - Modèles entraînés : LogReg & RandomForest")

# 6) Évaluation (rapports + PR-AUC + courbe PR + matrice de confusion)
print("[6/8] Évaluation…")
results = []
for name, model in [("LogReg", logreg), ("RandomForest", rf)]:
    proba = model.predict_proba(X_test)[:,1]
    y_hat = (proba>=0.5).astype(int)
    ap = average_precision_score(y_test, proba)
    print(f"\n== {name} ==")
    print("AP (PR-AUC) :", round(ap, 4))
    print(classification_report(y_test, y_hat, digits=4))
    print("Matrice de confusion:\n", confusion_matrix(y_test, y_hat))
    results.append((ap, name, model, proba))

# Choisir le meilleur selon AP
results.sort(reverse=True)
best_ap, best_name, best_model, best_proba = results[0]
print(f"\n>> Modèle retenu : {best_name} (AP={best_ap:.4f})")

# Courbe PR
prec, rec, thr = precision_recall_curve(y_test, best_proba)
plt.figure()
plt.plot(rec, prec)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision-Recall ({best_name})')
plt.tight_layout(); plt.show()

# Matrice de confusion (seuil 0.5)
y_hat_best = (best_proba>=0.5).astype(int)
cm = confusion_matrix(y_test, y_hat_best)
Disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure()
Disp.plot()
plt.title(f'Confusion Matrix ({best_name})')
plt.tight_layout(); plt.show()

# 7) Sauvegarde + 1 prédiction d'exemple
print("[7/8] Sauvegarde du modèle…")
model_path = MODEL_DIR / f'ids_{best_name.lower()}.joblib'
joblib.dump(best_model, model_path)
print("  - Modèle enregistré :", model_path)

print("[8/8] Prédiction d'exemple (1 flow)…")
one = X_test.iloc[[0]]
proba_one = float(best_model.predict_proba(one)[:,1])
print(f"  - Proba attaque du premier flow de test : {proba_one:.3f}")

print("\n=== Démo terminée ✔ ===\n")
