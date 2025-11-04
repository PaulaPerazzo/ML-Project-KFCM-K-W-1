import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score, classification_report
from scipy.optimize import linear_sum_assignment

# optional silhouette (may fail for some edge cases)
try:
    from sklearn.metrics import silhouette_score
except Exception:
    silhouette_score = None

# assistant: a file in common paths
def find_file(filename, start_dir=None):
    """Procura filename em start_dir, start_dir/results, start_dir/data e em current dir ancestors."""
    if start_dir is None:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(start_dir, filename),
        os.path.join(start_dir, "results", filename),
        os.path.join(start_dir, "data", filename),
        os.path.join(start_dir, "..", filename),
        os.path.join(start_dir, "..", "results", filename),
        os.path.join(start_dir, "..", "data", filename),
    ]
    # normalize and check direct candidates
    for c in candidates:
        cnorm = os.path.normpath(c)
        if os.path.exists(cnorm):
            return cnorm
    # fallback: search recursively a partir de start_dir (limit depth)
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

# find files 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_CSV = find_file("results_kfcm_yeast.csv", BASE_DIR)
if RESULTS_CSV is None:
    print("ERRO: Não foi possível localizar 'results_kfcm_yeast.csv'.")
    print("Procurei em:", BASE_DIR, "e subpastas.")
    sys.exit(1)

DATA_CSV = find_file("yeast.csv", BASE_DIR)
if DATA_CSV is None:
    print("ERRO: Não foi possível localizar 'yeast.csv' (dataset).")
    print("Coloque o arquivo 'yeast.csv' em data/ ou na pasta do script.")
    sys.exit(1)

print("Using results CSV:", RESULTS_CSV)
print("Using data CSV   :", DATA_CSV)

# load results and identify the best silhouette
results_df = pd.read_csv(RESULTS_CSV)
print("Columns in results CSV:", list(results_df.columns))

# automatic silhouette column detection
sil_cols = [c for c in results_df.columns if "sil" in c.lower()]
if not sil_cols:
    raise ValueError("Nenhuma coluna com 'sil' encontrada no results CSV.")
sil_col = sil_cols[0]
print(f"Using silhouette column: '{sil_col}'")

# retrieve the row with the best silhouette
best_idx = results_df[sil_col].idxmax()
best_row = results_df.loc[best_idx]

# retrieve fields with fallback
Kstar = int(best_row.get("K", best_row.get("k")))
best_run = int(best_row.get("Best_Run", best_row.get("run", 1)))
best_sil_val = float(best_row[sil_col])

print(f"K* by silhouette = {Kstar} (best run = {best_run}) -> silhouette = {best_sil_val:.6f}")

# load dataset and preprocess
df = pd.read_csv(DATA_CSV)

#  identify label column
label_candidates = [c for c in df.columns if c.lower() in ("class", "label", "labels", "target")]
if label_candidates:
    label_col = label_candidates[0]
else:
    # fallback: if there's a column named 'Sequence Name' and something else, try 'Class' as specified
    if "Class" in df.columns:
        label_col = "Class"
    else:
        raise ValueError("Não foi possível identificar a coluna de rótulos no dataset (procure 'Class' ou 'label').")
# features: remove known non-feature columns
drop_cols = [c for c in ("Class", "Sequence Name", "Sequence_Name", "SeqName") if c in df.columns]
X = df.drop(columns=drop_cols).copy()
y = LabelEncoder().fit_transform(df[label_col].astype(str))

print("Data shapes -> X:", X.shape, "y:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  refit 
try:
    from kfcm import KFCM_K_W_1
except Exception as e:
    print("ERRO: não foi possível importar KFCM_K_W_1 from kfcm.py:", e)
    sys.exit(1)

seed = (1000 * Kstar) + (best_run - 1) + 1
model_kstar = KFCM_K_W_1(n_clusters=Kstar, m=1.1, max_iter=200, random_state=seed)
print(f"[Init] fitting model for K={Kstar} with seed={seed} ...")
model_kstar.fit(X_scaled)

# crisp labels
labels_kstar = model_kstar.predict(X_scaled)

# outputs
print("\nPrototypes (G):")
print(getattr(model_kstar, "G", "NO G attribute"))

print("\nWidths vector s:")
print(getattr(model_kstar, "s", "NO s attribute"))

# confusion and ARI
cm = confusion_matrix(y, labels_kstar)
ari = adjusted_rand_score(y, labels_kstar)

print("\nConfusion matrix (raw true vs predicted):")
print(cm)
print(f"\nAdjusted Rand Index (K*={Kstar}): {ari:.6f}")

# --- remap predicted labels -> true labels (Hungarian) 
unique_true = np.unique(y)
n_true = len(unique_true)
# build contingency matrix true_label x predicted_label
pred_labels = np.unique(labels_kstar)
K_pred = len(pred_labels)
# rows: true classes, cols: predicted clusters (use counts)
cont = np.zeros((n_true, Kstar), dtype=int)
for i, true_val in enumerate(unique_true):
    for j in range(Kstar):
        cont[i, j] = np.sum((y == true_val) & (labels_kstar == j))

# cost matrix for Hungarian (maximize overlap -> minimize -overlap)
cost = -cont
try:
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    mapped_preds = np.array([mapping.get(lbl, lbl) for lbl in labels_kstar])
    cm_mapped = confusion_matrix(y, mapped_preds)
    print("\nConfusion matrix after optimal mapping (true vs mapped-pred):")
    print(cm_mapped)
except Exception as e:
    print("Warning: mapping via Hungarian failed:", e)
    cm_mapped = cm
    mapped_preds = labels_kstar

# classification report (using mapped preds)
print("\nClassification report (true vs mapped cluster labels):")
print(classification_report(y, mapped_preds, zero_division=0))

# silhouette computed on crisp labels (optional)
sil_crisp = None
if silhouette_score is not None:
    try:
        sil_crisp = silhouette_score(X_scaled, labels_kstar)
        print(f"\nSilhouette (computed on crisp labels) = {sil_crisp:.6f}")
    except Exception as e:
        print("Could not compute silhouette_score on crisp labels:", e)

#  plot function objective vs iterations
J_hist = None
if hasattr(model_kstar, "get_cost_history"):
    try:
        J_hist = model_kstar.get_cost_history()
    except Exception:
        J_hist = None

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.figure(figsize=(10, 5))
if J_hist is not None:
    plt.plot(np.arange(len(J_hist)), J_hist, marker="o")
    plt.title(f"Cost function vs iterations (K={Kstar})")
    plt.xlabel("Iteration")
    plt.ylabel("Objective J")
    plt.grid(True)
else:
    plt.text(0.5, 0.5, "No cost history available", ha="center")
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, f"cost_history_K{Kstar}.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved plot: {plot_path}")

# save outputs to files

out_prototypes = os.path.join(RESULTS_DIR, f"prototypes_K{Kstar}.csv")
out_s = os.path.join(RESULTS_DIR, f"s_widths_K{Kstar}.csv")
out_cm = os.path.join(RESULTS_DIR, f"confusion_matrix_K{Kstar}.csv")
out_cost = os.path.join(RESULTS_DIR, f"cost_history_K{Kstar}.csv")
out_model = os.path.join(RESULTS_DIR, f"best_model_K{Kstar}.joblib")
summary_txt = os.path.join(RESULTS_DIR, f"summary_K{Kstar}.txt")

print("\nSaving files to:", RESULTS_DIR)
print("-", out_prototypes)
print("-", out_s)
print("-", out_cm)
print("-", out_cost)
print("-", out_model)
print("-", summary_txt)

# save prototypes
try:
    if hasattr(model_kstar, "G"):
        np.savetxt(out_prototypes, model_kstar.G, delimiter=",", header="features", comments="")
    else:
        print("WARNING: model has no attribute G; skipping prototypes save.")
except Exception as e:
    print("ERROR saving prototypes:", e)

# save s widths
try:
    if hasattr(model_kstar, "s"):
        np.savetxt(out_s, model_kstar.s, delimiter=",", header="s_values", comments="")
    else:
        print("WARNING: model has no attribute s; skipping s widths save.")
except Exception as e:
    print("ERROR saving s widths:", e)

# save confusion (mapped preferred)
try:
    pd.DataFrame(cm_mapped).to_csv(out_cm, index=True)
except Exception as e:
    print("ERROR saving confusion matrix:", e)

# save cost history
try:
    if J_hist is not None:
        pd.DataFrame({"J": J_hist}).to_csv(out_cost, index=False)
    else:
        print("WARNING: no J_hist to save.")
except Exception as e:
    print("ERROR saving cost history:", e)

# save model
try:
    joblib.dump(model_kstar, out_model)
except Exception as e:
    print("ERROR saving model:", e)

# save summary txt
try:
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"K* = {Kstar}\n")
        f.write(f"best_run = {best_run}\n")
        f.write(f"silhouette_in_results = {best_sil_val}\n")
        f.write(f"silhouette_crisp_computed = {sil_crisp}\n")
        f.write(f"ARI = {ari}\n\n")
        f.write("Prototypes (G):\n")
        if hasattr(model_kstar, "G"):
            np.savetxt(f, model_kstar.G, delimiter=",", header="", comments="")
        f.write("\nWidths s:\n")
        if hasattr(model_kstar, "s"):
            np.savetxt(f, model_kstar.s, delimiter=",")
        f.write("\n\nConfusion matrix (mapped):\n")
        f.write(np.array2string(cm_mapped))
        f.write("\n\nCost history (last 10):\n")
        if J_hist is not None:
            f.write(np.array2string(J_hist[-10:] if len(J_hist) > 10 else J_hist))
except Exception as e:
    print("ERROR writing summary txt:", e)

# final message 
print("\nSave attempts finished. Check the files listed above in the script folder.")
