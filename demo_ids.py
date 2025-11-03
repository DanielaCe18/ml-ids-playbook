# demo_ids.py
# Démo clé en main – Détection d'attaques réseau (CICIDS2017)
# Version corrigée : on ne conserve que les colonnes numériques pour le ML.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score,
    precision_recall_curve, ConfusionMatrixDisplay
)

# --- chemins robustes par rapport à ce fichier ---
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV  = BASE_DIR / 'Data' / 'raw' / 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("\n=== Démo : Détection d'attaques réseau (CICIDS2017) ===\n")
print("Répertoire du script :", BASE_DIR)
print("CSV attendu          :", DATA_CSV)

# 1) Chargement
if not DATA_CSV.exists():
    print(f"[ERREUR] Fichier introuvable : {DATA_CSV}\n"
          f"Placez le CSV ici ou ajustez DATA_CSV.")
    sys.exit(1)

print("[1/8] Chargement du CSV…")
df = pd.read_csv(DATA_CSV, low_memory=False)
print("  - Dimensions :", df.shape)
print("  - Colonnes (10 premières) :", list(df.columns[:10]))

# 2) Nettoyage minimal
print("[2/8] Nettoyage minimal…")
# Noms propres
df.columns = [c.strip().replace(' ', '_').replace('/', '_') for c in df.columns]
# Inf -> NaN
df = df.replace([np.inf, -np.inf], np.nan)

# 3) Cible binaire
print("[3/8] Construction de la cible Attack vs Benign…")
if 'Label' not in df.columns:
    print("[ERREUR] Colonne 'Label' absente. Vérifiez le CSV CICIDS2017.")
    sys.exit(1)
y = (~df['Label'].str.contains('Benign', case=False)).astype(int)

# ======= ne garder que les variables numériques =======
num_cols_all = df.select_dtypes(include=['number', 'bool']).columns.tolist()
if 'Label' in num_cols_all:
    num_cols_all.remove('Label')
X = df[num_cols_all].copy()

# Imputation médiane sur ces colonnes numériques
if len(num_cols_all) > 0:
    X[num_cols_all] = X[num_cols_all].fillna(X[num_cols_all].median())

# Sanity check
print(f"  - Colonnes numériques retenues : {len(num_cols_all)}")
if len(num_cols_all) < 10:
    print("    (Avertissement) Peu de colonnes numériques détectées — c'est normal si beaucoup de colonnes sont IP/ID/Timestamp.")
print("  - Répartition (0=Benign,1=Attack) :\n", y.value_counts())

# 4) Split train/test (sans mélange complet du temps)
print("[4/8] Split train/test (80/20, shuffle=False)…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False)
print("  - Train:", X_train.shape, " Test:", X_test.shape)

# 5) Modèles
print("[5/8] Entraînement de deux modèles…")
logreg = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))
])
logreg.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)
print("  - Modèles entraînés : LogReg & RandomForest")

# 6) Évaluation
print("[6/8] Évaluation…")
results = []
for name, model in [("LogReg", logreg), ("RandomForest", rf)]:
    proba = model.predict_proba(X_test)[:, 1]
    y_hat = (proba >= 0.5).astype(int)
    ap = average_precision_score(y_test, proba)
    print(f"\n== {name} ==")
    print("AP (PR-AUC) :", round(ap, 4))
    print(classification_report(y_test, y_hat, digits=4))
    print("Matrice de confusion:\n", confusion_matrix(y_test, y_hat))
    results.append((ap, name, model, proba))

results.sort(reverse=True)
best_ap, best_name, best_model, best_proba = results[0]
print(f"\n>> Modèle retenu : {best_name} (AP={best_ap:.4f})")

# Courbe PR
prec, rec, thr = precision_recall_curve(y_test, best_proba)
plt.figure(); plt.plot(rec, prec)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision-Recall ({best_name})')
plt.tight_layout(); plt.show()

# Matrice de confusion
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_hat_best = (best_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_hat_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(); disp.plot()
plt.title(f'Confusion Matrix ({best_name})')
plt.tight_layout(); plt.show()

# 7) Sauvegarde + 1 prédiction
print("[7/8] Sauvegarde du modèle…")
model_path = MODEL_DIR / f'ids_{best_name.lower()}.joblib'
joblib.dump(best_model, model_path)
print("  - Modèle enregistré :", model_path)

print("[8/8] Prédiction d'exemple (1 flow)…")
one = X_test.iloc[[0]]
proba_one = float(best_model.predict_proba(one)[:, 1])
print(f"  - Proba attaque du premier flow de test : {proba_one:.3f}")

print("\n=== Démo terminée ✔ ===\n")
