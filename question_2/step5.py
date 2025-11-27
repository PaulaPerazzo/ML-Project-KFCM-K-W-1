"""
================================================================================
ETAPA 5: FUNÇÕES DE AJUSTE DE HIPERPARÂMETROS
================================================================================

O enunciado especifica:
"Quando necessário, faça validação cruzada 5-folds nos 9 folds restantes 
para fazer ajuste de hiper-parâmetros e depois treine o modelo novamente 
com o conjunto aprendizagem de 9-folds usando os valores selecionados 
para os hiper-parâmetros."

Classificadores que precisam de ajuste:
- KNN Bayesiano: k (vizinhos) e métrica de distância
- Parzen Window: h (bandwidth)
- Regressão Logística: C (regularização)

Classificadores sem hiperparâmetros:
- Bayesiano Gaussiano: usa MV diretamente
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ETAPA 5: FUNÇÕES DE AJUSTE DE HIPERPARÂMETROS")
print("="*70)


# =============================================================================
# Importar classificadores do step3
# =============================================================================
# Nota: Em uso real, importar de step3.py
# from step3 import KNNBayesiano, ParzenWindowClassifier

# Para este arquivo ser standalone, redefinimos as classes necessárias
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors, KernelDensity


class KNNBayesiano(BaseEstimator, ClassifierMixin):
    """
    Classificador Bayesiano baseado em K-Vizinhos.
    P(ωi|x) ∝ p(x|ωi) * P(ωi), onde p(x|ωi) = ki / (ni * V)
    """
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        n_samples = len(y)
        
        self.n_samples_per_class_ = {}
        self.priors_ = {}
        for c in self.classes_:
            n_c = np.sum(y == c)
            self.n_samples_per_class_[c] = n_c
            self.priors_[c] = n_c / n_samples
        
        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        self.nn_model_.fit(X)
        return self
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        distances, indices = self.nn_model_.kneighbors(X)
        posteriors = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            neighbor_classes = self.y_train_[indices[i]]
            r_k = max(distances[i, -1], 1e-10)
            V = r_k ** X.shape[1]
            
            for j, c in enumerate(self.classes_):
                k_i = np.sum(neighbor_classes == c)
                n_i = self.n_samples_per_class_[c]
                likelihood = k_i / (n_i * V) if n_i > 0 else 0
                posteriors[i, j] = likelihood * self.priors_[c]
        
        row_sums = posteriors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return posteriors / row_sums
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador Bayesiano baseado em Janela de Parzen (KDE).
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = len(y)
        self.kde_models_ = {}
        self.priors_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / n_samples
            self.kde_models_[c] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            self.kde_models_[c].fit(X_c)
        return self
    
    def predict_proba(self, X):
        log_posteriors = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            log_likelihood = self.kde_models_[c].score_samples(X)
            log_prior = np.log(self.priors_[c])
            log_posteriors[:, i] = log_likelihood + log_prior
        
        log_posteriors_max = np.max(log_posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors - log_posteriors_max)
        return posteriors / np.sum(posteriors, axis=1, keepdims=True)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =============================================================================
# FUNÇÕES DE AJUSTE DE HIPERPARÂMETROS
# =============================================================================

def ajustar_knn_bayesiano(X_train, y_train, n_folds=5, verbose=False):
    """
    Ajusta hiperparâmetros do KNN Bayesiano usando CV 5-folds.
    
    Hiperparâmetros testados:
    - n_neighbors (k): [1, 3, 5, 7, 9, 11, 15, 21]
    - metric: ['euclidean', 'manhattan', 'chebyshev']
    
    Parâmetros:
    -----------
    X_train : array-like
        Features de treino (9 folds)
    y_train : array-like
        Labels de treino
    n_folds : int
        Número de folds para CV interno (default: 5)
    verbose : bool
        Se True, imprime progresso
    
    Retorna:
    --------
    dict : Melhores hiperparâmetros {'n_neighbors': k, 'metric': m}
    float : Melhor score de validação
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
                
                try:
                    clf = KNNBayesiano(n_neighbors=k, metric=metric)
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_val)
                    scores.append(accuracy_score(y_val, y_pred))
                except Exception as e:
                    if verbose:
                        print(f"Erro com k={k}, metric={metric}: {e}")
                    continue
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'n_neighbors': k, 'metric': metric}
                    if verbose:
                        print(f"  Novo melhor: k={k}, metric={metric}, score={mean_score:.4f}")
    
    return best_params, best_score


def ajustar_parzen(X_train, y_train, n_folds=5, verbose=False):
    """
    Ajusta bandwidth do Parzen Window usando CV 5-folds.
    
    Hiperparâmetros testados:
    - bandwidth (h): [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    Parâmetros:
    -----------
    X_train : array-like
        Features de treino (9 folds)
    y_train : array-like
        Labels de treino
    n_folds : int
        Número de folds para CV interno (default: 5)
    verbose : bool
        Se True, imprime progresso
    
    Retorna:
    --------
    dict : Melhores hiperparâmetros {'bandwidth': h}
    float : Melhor score de validação
    """
    bandwidth_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    
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
            except Exception as e:
                if verbose:
                    print(f"Erro com bandwidth={bw}: {e}")
                continue
        
        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'bandwidth': bw}
                if verbose:
                    print(f"  Novo melhor: bandwidth={bw}, score={mean_score:.4f}")
    
    return best_params, best_score


def ajustar_logistica(X_train, y_train, n_folds=5, verbose=False):
    """
    Ajusta hiperparâmetros da Regressão Logística usando CV 5-folds.
    
    Hiperparâmetros testados:
    - C: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    Parâmetros:
    -----------
    X_train : array-like
        Features de treino (9 folds)
    y_train : array-like
        Labels de treino
    n_folds : int
        Número de folds para CV interno (default: 5)
    verbose : bool
        Se True, imprime progresso
    
    Retorna:
    --------
    dict : Melhores hiperparâmetros {'C': c}
    float : Melhor score de validação
    """
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    best_score = -1
    best_params = {'C': 1.0}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for C in C_values:
        scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            try:
                clf = LogisticRegression(
                    C=C,
                    penalty='l2',
                    max_iter=1000,
                    solver='lbfgs',
                    multi_class='multinomial'
                )
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
            except Exception as e:
                if verbose:
                    print(f"Erro com C={C}: {e}")
                continue
        
        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'C': C}
                if verbose:
                    print(f"  Novo melhor: C={C}, score={mean_score:.4f}")
    
    return best_params, best_score


# =============================================================================
# DEMONSTRAÇÃO
# =============================================================================
if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    print("\n" + "═"*70)
    print("DEMONSTRAÇÃO DO AJUSTE DE HIPERPARÂMETROS")
    print("═"*70)
    
    # Carregar dados
    print("\n[1] Carregando dados...")
    df = pd.read_csv('yeast.csv')
    X = df.drop(['Sequence Name', 'Class'], axis=1).values
    y = LabelEncoder().fit_transform(df['Class'].values)
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"    Amostras: {len(y)}")
    print(f"    Features: {X.shape[1]}")
    
    # Simular um fold de treino (90% dos dados)
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )
    
    print(f"    Treino (9 folds): {len(y_train)} amostras")
    
    # Ajustar KNN Bayesiano
    print("\n[2] Ajustando KNN Bayesiano (CV 5-folds)...")
    knn_params, knn_score = ajustar_knn_bayesiano(X_train, y_train, verbose=True)
    print(f"    Melhores parâmetros: {knn_params}")
    print(f"    Score de validação: {knn_score:.4f}")
    
    # Ajustar Parzen
    print("\n[3] Ajustando Parzen Window (CV 5-folds)...")
    parzen_params, parzen_score = ajustar_parzen(X_train, y_train, verbose=True)
    print(f"    Melhores parâmetros: {parzen_params}")
    print(f"    Score de validação: {parzen_score:.4f}")
    
    # Ajustar Logística
    print("\n[4] Ajustando Regressão Logística (CV 5-folds)...")
    lr_params, lr_score = ajustar_logistica(X_train, y_train, verbose=True)
    print(f"    Melhores parâmetros: {lr_params}")
    print(f"    Score de validação: {lr_score:.4f}")
    
    print("\n" + "="*70)
    print("RESUMO DOS HIPERPARÂMETROS SELECIONADOS")
    print("="*70)
    print(f"""
┌────────────────────────┬─────────────────────────────────────┐
│ Classificador          │ Hiperparâmetros Selecionados        │
├────────────────────────┼─────────────────────────────────────┤
│ KNN Bayesiano          │ k={knn_params['n_neighbors']}, metric={knn_params['metric']:<12} │
│ Parzen Window          │ bandwidth={parzen_params['bandwidth']:<24} │
│ Regressão Logística    │ C={lr_params['C']:<32} │
└────────────────────────┴─────────────────────────────────────┘
    """)
    print("✓ Funções de ajuste de hiperparâmetros implementadas")