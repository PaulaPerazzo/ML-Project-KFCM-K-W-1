import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score
from kfcm import KFCM_K_W_1

train_data = pd.read_csv("data/yeast_train.csv")
X_train = train_data.drop(columns=["Class", "Sequence Name"])
y_train = LabelEncoder().fit_transform(train_data["Class"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model_kstar = KFCM_K_W_1(n_clusters=2, m=1.1, max_iter=100)
model_kstar.fit(X_train_scaled)

labels_kstar = model_kstar.predict(X_train_scaled)
ari_kstar_train = adjusted_rand_score(y_train, labels_kstar)

print(f"rand to K*=2: {ari_kstar_train:.4f}")
