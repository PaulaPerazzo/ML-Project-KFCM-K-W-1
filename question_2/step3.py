"""
================================================================================
ETAPA 3: IMPLEMENTAÇÃO DOS 5 CLASSIFICADORES
================================================================================

Os 5 classificadores pedidos no enunciado:
1. Bayesiano Gaussiano (Normal Multivariada)
2. Bayesiano baseado em K-Vizinhos (KNN Bayesiano - NÃO é o KNN tradicional!)
3. Bayesiano Janela de Parzen (KDE)
4. Regressão Logística
5. Voto Majoritário (dos 3 primeiros)

IMPORTANTE: O classificador 2 é o KNN BAYESIANO, que é diferente do KNN
tradicional por votação. O KNN Bayesiano estima p(x|ωi) e aplica Bayes.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ETAPA 3: IMPLEMENTAÇÃO DOS 5 CLASSIFICADORES")
print("="*70)

# =============================================================================
# CARREGAR DADOS
# =============================================================================
print("\n[3.0] Carregando dados...")

# Carregar dataset Yeast do CSV
df = pd.read_csv('yeast.csv')
X = df.drop(['Sequence Name', 'Class'], axis=1).values
y_original = df['Class'].values

# Codificar labels das 10 classes originais
le_10 = LabelEncoder()
y_10classes = le_10.fit_transform(y_original)

# Carregar labels do clustering KFCM-K-W-1 (Questão 1)
y_2classes = np.loadtxt('crisp_partition.txt').astype(int)

print(f"      Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"      Versão 1: 10 classes originais")
print(f"      Versão 2: 2 clusters (KFCM-K-W-1)")


# =============================================================================
# CLASSIFICADOR 1: Bayesiano Gaussiano
# =============================================================================
print("\n" + "═"*70)
print("CLASSIFICADOR 1: BAYESIANO GAUSSIANO")
print("═"*70)

print("""
Teoria:
-------
Baseado no Teorema de Bayes com distribuição Normal Multivariada.

Regra de decisão: Classificar x na classe ωl se P(ωl|x) = max P(ωi|x)

Onde:
    P(ωi|x) = p(x|ωi) * P(ωi) / p(x)
    
    p(x|ωi) = N(x; μi, Σi)  (Normal Multivariada)
    P(ωi) = ni/n            (estimativa MV - proporção)
    μi = média amostral da classe i
    Σi = covariância amostral da classe i

Hiperparâmetros: Nenhum
""")

class BayesianoGaussiano(BaseEstimator, ClassifierMixin):
    """
    Classificador Bayesiano com distribuição Normal Multivariada.
    Implementação manual seguindo as equações do enunciado.
    """
    def __init__(self):
        self.classes_ = None
        self.priors_ = {}      # P(ωi)
        self.means_ = {}       # μi
        self.covs_ = {}        # Σi
        self.covs_inv_ = {}    # Σi^(-1)
        self.covs_det_ = {}    # |Σi|
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes_:
            X_c = X[y == c]
            n_c = len(X_c)
            
            # P(ωi) = ni/n (MV)
            self.priors_[c] = n_c / n_samples
            
            # μi = média amostral (MV)
            self.means_[c] = np.mean(X_c, axis=0)
            
            # Σi = covariância amostral (MV)
            # Adiciona regularização para evitar matriz singular
            cov = np.cov(X_c.T) + 1e-6 * np.eye(n_features)
            self.covs_[c] = cov
            self.covs_inv_[c] = np.linalg.inv(cov)
            self.covs_det_[c] = np.linalg.det(cov)
        
        return self
    
    def _log_likelihood(self, X, c):
        """
        Calcula log p(x|ωi) para a classe c.
        log N(x; μ, Σ) = -d/2 * log(2π) - 1/2 * log|Σ| - 1/2 * (x-μ)ᵀΣ⁻¹(x-μ)
        """
        d = X.shape[1]
        diff = X - self.means_[c]
        
        # Termo quadrático: (x-μ)ᵀ Σ⁻¹ (x-μ)
        mahalanobis = np.sum(diff @ self.covs_inv_[c] * diff, axis=1)
        
        log_likelihood = (
            -0.5 * d * np.log(2 * np.pi)
            - 0.5 * np.log(self.covs_det_[c])
            - 0.5 * mahalanobis
        )
        
        return log_likelihood
    
    def predict_proba(self, X):
        """Calcula P(ωi|x) para todas as classes."""
        log_posteriors = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            # log P(ωi|x) ∝ log p(x|ωi) + log P(ωi)
            log_posteriors[:, i] = (
                self._log_likelihood(X, c) + np.log(self.priors_[c])
            )
        
        # Converter para probabilidades (normalizar)
        log_posteriors_max = np.max(log_posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors - log_posteriors_max)
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        
        return posteriors
    
    def predict(self, X):
        """Classifica x na classe com maior P(ωi|x)."""
        posteriors = self.predict_proba(X)
        return self.classes_[np.argmax(posteriors, axis=1)]


print("✓ Classe BayesianoGaussiano implementada")


# =============================================================================
# CLASSIFICADOR 2: BAYESIANO K-VIZINHOS (KNN BAYESIANO)
# =============================================================================
print("\n" + "═"*70)
print("CLASSIFICADOR 2: BAYESIANO K-VIZINHOS (KNN BAYESIANO)")
print("═"*70)

print("""
⚠️  IMPORTANTE: Este NÃO é o KNN tradicional por votação!

Teoria:
-------
O KNN Bayesiano estima a densidade p(x|ωi) usando os k-vizinhos e aplica Bayes.

Regra de decisão: Classificar x na classe ωl se P(ωl|x) = max P(ωi|x)

Onde:
    P(ωi|x) = p(x|ωi) * P(ωi) / p(x)
    
    p(x|ωi) ≈ ki / (ni * V)
    
    - ki = número de vizinhos da classe ωi entre os k vizinhos de x
    - ni = número total de amostras da classe ωi no treino
    - V = volume da hiperesfera contendo os k vizinhos
    - P(ωi) = ni/n (prior)

Hiperparâmetros (ajustar via CV 5-folds):
- k: número de vizinhos [1, 3, 5, 7, 9, 11, 15, 21]
- metric: ['euclidean', 'manhattan', 'chebyshev']
""")


class KNNBayesiano(BaseEstimator, ClassifierMixin):
    """
    Classificador Bayesiano baseado em K-Vizinhos.
    
    Diferente do KNN tradicional:
    - KNN tradicional: vota pela classe mais frequente entre k vizinhos
    - KNN Bayesiano: estima p(x|ωi) usando k-vizinhos e aplica Bayes
    
    P(ωi|x) ∝ p(x|ωi) * P(ωi)
    
    p(x|ωi) = ki / (ni * V)
    onde:
        ki = vizinhos da classe i entre os k vizinhos
        ni = total de amostras da classe i
        V = volume da hiperesfera (depende da distância ao k-ésimo vizinho)
    """
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.classes_ = None
        self.n_samples_per_class_ = {}
        self.priors_ = {}
        self.nn_model_ = None
        self.X_train_ = None
        self.y_train_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        n_samples = len(y)
        
        # Calcular priors e contagens por classe
        for c in self.classes_:
            n_c = np.sum(y == c)
            self.n_samples_per_class_[c] = n_c
            self.priors_[c] = n_c / n_samples
        
        # Modelo de vizinhos mais próximos
        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        self.nn_model_.fit(X)
        
        return self
    
    def predict_proba(self, X):
        """
        Calcula P(ωi|x) usando estimativa bayesiana com k-vizinhos.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Encontrar k vizinhos mais próximos
        distances, indices = self.nn_model_.kneighbors(X)
        
        # Calcular probabilidades posteriores
        posteriors = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            # Classes dos k vizinhos
            neighbor_classes = self.y_train_[indices[i]]
            
            # Distância ao k-ésimo vizinho (para calcular volume)
            r_k = distances[i, -1]
            if r_k == 0:
                r_k = 1e-10  # Evitar divisão por zero
            
            # Volume da hiperesfera d-dimensional: V = C_d * r^d
            # C_d é constante, então podemos ignorar para comparação
            # V ∝ r_k^d
            d = X.shape[1]
            V = r_k ** d
            
            for j, c in enumerate(self.classes_):
                # ki = número de vizinhos da classe c
                k_i = np.sum(neighbor_classes == c)
                
                # ni = número total de amostras da classe c
                n_i = self.n_samples_per_class_[c]
                
                # p(x|ωi) = ki / (ni * V)
                if n_i > 0 and V > 0:
                    likelihood = k_i / (n_i * V)
                else:
                    likelihood = 0
                
                # P(ωi|x) ∝ p(x|ωi) * P(ωi)
                posteriors[i, j] = likelihood * self.priors_[c]
        
        # Normalizar para somar 1
        row_sums = posteriors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Evitar divisão por zero
        posteriors = posteriors / row_sums
        
        return posteriors
    
    def predict(self, X):
        """Classifica x na classe com maior P(ωi|x)."""
        posteriors = self.predict_proba(X)
        return self.classes_[np.argmax(posteriors, axis=1)]


def ajustar_knn_bayesiano(X_train, y_train, n_folds=5):
    """
    Ajusta hiperparâmetros do KNN Bayesiano usando validação cruzada 5-folds.
    
    Hiperparâmetros:
    - n_neighbors (k): [1, 3, 5, 7, 9, 11, 15, 21]
    - metric: ['euclidean', 'manhattan', 'chebyshev']
    """
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    
    best_score = -1
    best_params = {'n_neighbors': 5, 'metric': 'euclidean'}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for k in k_values:
        for metric in metrics:
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                clf = KNNBayesiano(n_neighbors=k, metric=metric)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
            
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'n_neighbors': k, 'metric': metric}
    
    return best_params, best_score


print("✓ Classe KNNBayesiano implementada")
print("✓ Função ajustar_knn_bayesiano implementada")


# =============================================================================
# CLASSIFICADOR 3: BAYESIANO JANELA DE PARZEN
# =============================================================================
print("\n" + "═"*70)
print("CLASSIFICADOR 3: BAYESIANO JANELA DE PARZEN")
print("═"*70)

print("""
Teoria:
-------
Estima p(x|ωi) usando Kernel Density Estimation (KDE).

p(x|ωi) = (1/ni) * Σ K((x - xj)/h)
          j∈classe i

Onde:
- K = kernel Gaussiano multivariado produto
- h = bandwidth (largura da janela)
- xj = amostras da classe i

Regra de decisão: P(ωi|x) ∝ p(x|ωi) * P(ωi)

Hiperparâmetros (ajustar via CV 5-folds):
- bandwidth (h): [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
""")


class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador Bayesiano baseado em Janela de Parzen (KDE).
    Usa kernel Gaussiano multivariado produto.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.classes_ = None
        self.kde_models_ = {}
        self.priors_ = {}
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes_:
            X_c = X[y == c]
            
            # P(ωi) = ni/n (MV)
            self.priors_[c] = len(X_c) / n_samples
            
            # KDE para estimar p(x|ωi)
            self.kde_models_[c] = KernelDensity(
                kernel='gaussian',
                bandwidth=self.bandwidth
            )
            self.kde_models_[c].fit(X_c)
        
        return self
    
    def predict_proba(self, X):
        """Calcula P(ωi|x) para todas as classes."""
        log_posteriors = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            # log p(x|ωi)
            log_likelihood = self.kde_models_[c].score_samples(X)
            # log P(ωi)
            log_prior = np.log(self.priors_[c])
            # log P(ωi|x) ∝ log p(x|ωi) + log P(ωi)
            log_posteriors[:, i] = log_likelihood + log_prior
        
        # Converter para probabilidades
        log_posteriors_max = np.max(log_posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors - log_posteriors_max)
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        
        return posteriors
    
    def predict(self, X):
        """Classifica x na classe com maior P(ωi|x)."""
        posteriors = self.predict_proba(X)
        return self.classes_[np.argmax(posteriors, axis=1)]


def ajustar_parzen(X_train, y_train, n_folds=5):
    """
    Ajusta o bandwidth do Parzen usando validação cruzada 5-folds.
    """
    bandwidth_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    best_score = -1
    best_params = {'bandwidth': 1.0}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for bw in bandwidth_values:
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            try:
                clf = ParzenWindowClassifier(bandwidth=bw)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
            except:
                continue
        
        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'bandwidth': bw}
    
    return best_params, best_score


print("✓ Classe ParzenWindowClassifier implementada")
print("✓ Função ajustar_parzen implementada")


# =============================================================================
# CLASSIFICADOR 4: REGRESSÃO LOGÍSTICA
# =============================================================================
print("\n" + "═"*70)
print("CLASSIFICADOR 4: REGRESSÃO LOGÍSTICA")
print("═"*70)

print("""
Teoria:
-------
Modelo discriminativo que modela diretamente P(ωi|x).

Para multiclasse (softmax):
    P(ωi|x) = exp(wi·x + bi) / Σ exp(wj·x + bj)

Treinado minimizando cross-entropy loss com regularização L2.

Hiperparâmetros (ajustar via CV 5-folds):
- C: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] (inverso da regularização)
""")


def ajustar_logistica(X_train, y_train, n_folds=5):
    """
    Ajusta hiperparâmetros da Regressão Logística usando CV 5-folds.
    """
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    best_score = -1
    best_params = {'C': 1.0}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for C in C_values:
        try:
            lr = LogisticRegression(
                C=C, 
                penalty='l2', 
                max_iter=1000,
                solver='lbfgs', 
                multi_class='multinomial'
            )
            scores = cross_val_score(lr, X_train, y_train, cv=skf, scoring='accuracy')
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'C': C}
        except:
            continue
    
    return best_params, best_score


print("✓ Função ajustar_logistica implementada")


# =============================================================================
# CLASSIFICADOR 5: VOTO MAJORITÁRIO
# =============================================================================
print("\n" + "═"*70)
print("CLASSIFICADOR 5: VOTO MAJORITÁRIO")
print("═"*70)

print("""
Teoria:
-------
Combina as predições dos 3 classificadores bayesianos:
1. Bayesiano Gaussiano
2. KNN Bayesiano  
3. Parzen Window

Regra: A classe final é a mais votada pelos 3 classificadores.
Em caso de empate: usa ordem de prioridade.

Hiperparâmetros: Herda dos classificadores base (knn_params, parzen_params)
""")


class VotoMajoritario(BaseEstimator, ClassifierMixin):
    """
    Classificador por Voto Majoritário.
    Combina: Gaussiano + KNN Bayesiano + Parzen
    """
    def __init__(self, knn_params=None, parzen_params=None):
        self.knn_params = knn_params or {'n_neighbors': 5, 'metric': 'euclidean'}
        self.parzen_params = parzen_params or {'bandwidth': 1.0}
        self.classifiers = []
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        # Criar e treinar os 3 classificadores bayesianos
        clf_gaussian = BayesianoGaussiano()
        clf_knn = KNNBayesiano(**self.knn_params)
        clf_parzen = ParzenWindowClassifier(**self.parzen_params)
        
        clf_gaussian.fit(X, y)
        clf_knn.fit(X, y)
        clf_parzen.fit(X, y)
        
        self.classifiers = [clf_gaussian, clf_knn, clf_parzen]
        self.classifier_names = ['Gaussiano', 'KNN Bayesiano', 'Parzen']
        
        return self
    
    def predict(self, X):
        """Predição por voto majoritário."""
        # Coletar predições dos 3 classificadores
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        
        # Voto majoritário para cada amostra
        majority_votes = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            counter = Counter(votes)
            # Classe mais votada (empate: primeiro na ordem)
            majority_votes.append(counter.most_common(1)[0][0])
        
        return np.array(majority_votes)


print("✓ Classe VotoMajoritario implementada")


# =============================================================================
# DEMONSTRAÇÃO
# =============================================================================
print("\n" + "═"*70)
print("DEMONSTRAÇÃO DOS CLASSIFICADORES")
print("═"*70)

# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Testar com as 10 classes
print("\n" + "-"*50)
print("TESTE COM VERSÃO 1 (10 classes)")
print("-"*50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_10classes, test_size=0.3, random_state=42, stratify=y_10classes
)

print(f"\nDados: {len(X_train)} treino, {len(X_test)} teste")
print("\n--- Classificadores (parâmetros default) ---\n")

# 1. Bayesiano Gaussiano
clf1 = BayesianoGaussiano()
clf1.fit(X_train, y_train)
acc1 = accuracy_score(y_test, clf1.predict(X_test))
print(f"1. Bayesiano Gaussiano:    Acurácia = {acc1:.4f}")

# 2. KNN Bayesiano
clf2 = KNNBayesiano(n_neighbors=5, metric='euclidean')
clf2.fit(X_train, y_train)
acc2 = accuracy_score(y_test, clf2.predict(X_test))
print(f"2. KNN Bayesiano (k=5):    Acurácia = {acc2:.4f}")

# 3. Parzen Window
clf3 = ParzenWindowClassifier(bandwidth=1.0)
clf3.fit(X_train, y_train)
acc3 = accuracy_score(y_test, clf3.predict(X_test))
print(f"3. Parzen (h=1.0):         Acurácia = {acc3:.4f}")

# 4. Regressão Logística
clf4 = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial')
clf4.fit(X_train, y_train)
acc4 = accuracy_score(y_test, clf4.predict(X_test))
print(f"4. Regressão Logística:    Acurácia = {acc4:.4f}")

# 5. Voto Majoritário
clf5 = VotoMajoritario()
clf5.fit(X_train, y_train)
acc5 = accuracy_score(y_test, clf5.predict(X_test))
print(f"5. Voto Majoritário:       Acurácia = {acc5:.4f}")


# Testar com 2 classes (KFCM-K-W-1)
print("\n" + "-"*50)
print("TESTE COM VERSÃO 2 (2 clusters KFCM-K-W-1)")
print("-"*50)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled, y_2classes, test_size=0.3, random_state=42, stratify=y_2classes
)

print(f"\nDados: {len(X_train2)} treino, {len(X_test2)} teste")
print(f"Distribuição: Cluster 0 = {np.sum(y_2classes==0)} ({100*np.mean(y_2classes==0):.1f}%), "
      f"Cluster 1 = {np.sum(y_2classes==1)} ({100*np.mean(y_2classes==1):.1f}%)")
print("\n--- Classificadores (parâmetros default) ---\n")

# 1. Bayesiano Gaussiano
clf1.fit(X_train2, y_train2)
acc1_2 = accuracy_score(y_test2, clf1.predict(X_test2))
print(f"1. Bayesiano Gaussiano:    Acurácia = {acc1_2:.4f}")

# 2. KNN Bayesiano
clf2.fit(X_train2, y_train2)
acc2_2 = accuracy_score(y_test2, clf2.predict(X_test2))
print(f"2. KNN Bayesiano (k=5):    Acurácia = {acc2_2:.4f}")

# 3. Parzen Window
clf3.fit(X_train2, y_train2)
acc3_2 = accuracy_score(y_test2, clf3.predict(X_test2))
print(f"3. Parzen (h=1.0):         Acurácia = {acc3_2:.4f}")

# 4. Regressão Logística
clf4.fit(X_train2, y_train2)
acc4_2 = accuracy_score(y_test2, clf4.predict(X_test2))
print(f"4. Regressão Logística:    Acurácia = {acc4_2:.4f}")

# 5. Voto Majoritário
clf5.fit(X_train2, y_train2)
acc5_2 = accuracy_score(y_test2, clf5.predict(X_test2))
print(f"5. Voto Majoritário:       Acurácia = {acc5_2:.4f}")


# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO DA ETAPA 3")
print("="*70)
print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLASSIFICADORES IMPLEMENTADOS                         │
├────────────────────────┬────────────────────────────────────────────────────┤
│ Classificador          │ Hiperparâmetros                                    │
├────────────────────────┼────────────────────────────────────────────────────┤
│ 1. Bayesiano Gaussiano │ Nenhum (usa MV para μ, Σ, P(ω))                   │
│ 2. KNN Bayesiano       │ k (vizinhos), métrica de distância                │
│ 3. Parzen Window       │ h (bandwidth)                                      │
│ 4. Regressão Logística │ C (inverso da regularização)                       │
│ 5. Voto Majoritário    │ Herda de KNN e Parzen                             │
└────────────────────────┴────────────────────────────────────────────────────┘

⚠️  NOTA IMPORTANTE:
    O classificador 2 (KNN Bayesiano) é DIFERENTE do KNN tradicional!
    - KNN tradicional: vota pela classe mais frequente entre k vizinhos
    - KNN Bayesiano: estima p(x|ωi) = ki/(ni*V) e aplica regra de Bayes

Funções de ajuste de hiperparâmetros:
    - ajustar_knn_bayesiano(X_train, y_train, n_folds=5)
    - ajustar_parzen(X_train, y_train, n_folds=5)
    - ajustar_logistica(X_train, y_train, n_folds=5)
""")