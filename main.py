import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from kfcm import KFCM_K_W_1

# load dataset
train_data = pd.read_csv("data/yeast_train.csv")
test_data = pd.read_csv("data/yeast_test.csv")
val_data = pd.read_csv("data/yeast_val.csv")

# remove unusefull columns
X_train = train_data.drop(columns=["Class", "Sequence Name"])
X_test = test_data.drop(columns=["Class", "Sequence Name"])
X_val = val_data.drop(columns=["Class", "Sequence Name"])

# encode categorical data
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["Class"])
y_test = label_encoder.transform(test_data["Class"])
y_val = label_encoder.transform(val_data["Class"])

# normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# data training
model = KFCM_K_W_1(n_clusters=9, m=2, max_iter=50)
model.fit(X_train_scaled)

# train evaluation
train_clusters = model.predict(X_train_scaled)

ari_train = adjusted_rand_score(y_train, train_clusters)
nmi_train = normalized_mutual_info_score(y_train, train_clusters)
sil_train = silhouette_score(X_train_scaled, train_clusters)

print("evaluation on training set")
print(f"ARI (Adjusted Rand Index): {ari_train:.4f}")
print(f"NMI (Normalized Mutual Info): {nmi_train:.4f}")
print(f"Silhouette Score: {sil_train:.4f}")

# test evaluation
test_clusters = model.predict(X_test_scaled)

ari_test = adjusted_rand_score(y_test, test_clusters)
nmi_test = normalized_mutual_info_score(y_test, test_clusters)
sil_test = silhouette_score(X_test_scaled, test_clusters)

print("evaluation on test set")
print(f"ARI (Adjusted Rand Index): {ari_test:.4f}")
print(f"NMI (Normalized Mutual Info): {nmi_test:.4f}")
print(f"Silhouette Score: {sil_test:.4f}")
