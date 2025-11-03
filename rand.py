import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score
from kfcm import KFCM_K_W_1

best_k, best_run = 2, 22

df = pd.read_csv("data/yeast.csv")
X = df.drop(columns=["Class", "Sequence Name"])
y = LabelEncoder().fit_transform(df["Class"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_kstar = KFCM_K_W_1(n_clusters=best_k, m=1.1, max_iter=100, random_state=((100 * best_k) + best_run + 1))
model_kstar.fit(X_scaled)

labels_kstar = model_kstar.predict(X_scaled)
ari_kstar_train = adjusted_rand_score(y, labels_kstar)

print(model_kstar.get_cost_history())

print(f"rand to K*={best_k}: {ari_kstar_train:.4f}")

# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import adjusted_rand_score
# from kfcm import KFCM_K_W_1

# best_k, best_run = 9, 26

# df = pd.read_csv("data/yeast.csv")
# X = df.drop(columns=["Class", "Sequence Name"])
# y = LabelEncoder().fit_transform(df["Class"])

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# results = []

# for i in range(2,11):
#   model_kstar = KFCM_K_W_1(n_clusters=i, m=1.1, max_iter=100)
#   model_kstar.fit(X_scaled)

#   labels_kstar = model_kstar.predict(X_scaled)
#   ari_kstar_train = adjusted_rand_score(y, labels_kstar)

#   print(y[:5])
#   print(labels_kstar[:5])

#   print(model_kstar.get_cost_history())

#   print(f"rand to K*={i}: {ari_kstar_train:.4f}")
#   results.append(ari_kstar_train)

# print(results)
