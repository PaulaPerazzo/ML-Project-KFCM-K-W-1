import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from kfcm import KFCM_K_W_1

# load train set
df = pd.read_csv("data/yeast.csv")

X = df.drop(columns=["Class", "Sequence Name"])

# normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

# Executar o algoritmo KFCM-K-W-1 50 vezes para cada K, sendo K ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}. 
K_clusters = range(2, 11)
n_runs = 50

# store rsults
results = []

def evaluate_run(k, run, X_train_scaled):
  model = KFCM_K_W_1(n_clusters=k, m=1.1, max_iter=100, random_state=((100 * k) + run + 1))
  model.fit(X_train_scaled)

  cost = model.get_cost_history()[-1]
  return {"run": run, "cost": cost, "model": model}

for k in K_clusters:
  print(f"for k={k}: ")

  results_k = Parallel(n_jobs=-3)(
    delayed(evaluate_run)(k, run, X_train_scaled)
    for run in range(n_runs)
  )

  best = min(results_k, key=lambda r: r["cost"])

  labels = best["model"].predict(X_train_scaled)
  sil = silhouette_score(X_train_scaled, labels)

  print(f"best sil for k={k}: {sil:.4f} (run {best['run']+1})")

  results.append({
    "k": k,
    "labels": labels,
    "silhouette": sil,
    "cost": best["cost"],
    "model": best["model"],
    "run": best["run"] + 1,
  })

# save csv
results_df = pd.DataFrame([{k: v for k, v in d.items() if k != "model"} for d in results])
results_df.to_csv("results/best_cost_for_every_k.csv", index=False)

best = max(results, key=lambda r: r["silhouette"])
print(f'The best model was for K={best["k"]} in the run {best["run"]}. The values of cost are: {best["model"].get_cost_history()}')

np.savetxt("results/cost_history.txt", best["model"].get_cost_history())
np.savetxt("results/crisp_partition.txt", best["labels"])

# Fazer o plot Sil × K para cada K
plt.figure(figsize=(8,5))
plt.plot(list([res["k"] for res in results]), list([res["silhouette"] for res in results]), marker="o")
plt.title("Silhouette × Número de Clusters (K)")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("results/silhouette_vs_K.png")
plt.show()

