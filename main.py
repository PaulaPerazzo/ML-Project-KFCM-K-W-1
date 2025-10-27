import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

from kfcm import KFCM_K_W_1

# load train set
train_data = pd.read_csv("data/yeast_train.csv")

X_train = train_data.drop(columns=["Class", "Sequence Name"])
y_train = LabelEncoder().fit_transform(train_data["Class"])

# normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Executar o algoritmo KFCM-K-W-1 50 vezes para cada K, sendo K ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}. 
K_clusters = range(2, 11)
n_runs = 50

# store rsults
results = []

best_sil_scores = {}
best_models = {}

for k in K_clusters:
    print(f"for k={k}: ")

    best_cost = 999999
    best_model = None
    best_labels = None
    best_sil = -1
    best_run = -1

    for run in range(n_runs):
        model = KFCM_K_W_1(n_clusters=k, m=1.1, max_iter=100)
        model.fit(X_train_scaled)

        labels = model.predict(X_train_scaled)

        # Calcular a silhueta (Sil) para cada K e partição crisp. 
        try:
            sil = silhouette_score(X_train_scaled, labels)
        except:
            continue

        current_cost = model.get_cost_history()[-1]

        if current_cost < best_cost:  
            best_cost = current_cost
            best_model = model
            best_labels = labels
            best_sil = sil
            best_run = run + 1

    best_models[k] = best_model
    best_sil_scores[k] = best_sil
    
    print(f"best sil for k={k}: {best_sil:.4f} (run {best_run})")

    # Selecionar (e salvar) o melhor resultado para cada K. 
    results.append({
        "K": k,
        "Best_Silhouette": best_sil,
        "Best_Cost": best_cost,
        "Best_Run": best_run
    })

# save csv
results_df = pd.DataFrame(results)
results_df.to_csv("results_kfcm_yeast.csv", index=False)

# Fazer o plot Sil × K para cada K
plt.figure(figsize=(8,5))
plt.plot(list(best_sil_scores.keys()), list(best_sil_scores.values()), marker="o")
plt.title("Silhouette × Número de Clusters (K)")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("plot_silhouette_vs_K.png")
plt.show()

