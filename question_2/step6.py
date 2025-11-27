"""
================================================================================
ETAPA 6: VALIDAÇÃO CRUZADA 30×10-FOLDS COM NESTED CV
================================================================================

Estrutura da validação cruzada (conforme enunciado):

30 repetições × 10-fold CV (loop externo)
│
└── Para cada fold externo (9 folds treino, 1 fold teste):
    │
    ├── Classificadores COM hiperparâmetros (KNN, Parzen, Logística):
    │   └── CV 5-folds nos 9 folds de treino → selecionar hiperparâmetros
    │   └── Treinar com hiperparâmetros selecionados nos 9 folds completos
    │   └── Avaliar no fold de teste
    │
    ├── Classificadores SEM hiperparâmetros (Gaussiano):
    │   └── Treinar diretamente nos 9 folds
    │   └── Avaliar no fold de teste
    │
    └── Voto Majoritário:
        └── Usa hiperparâmetros ajustados de KNN e Parzen
        └── Combina Gaussiano + KNN + Parzen (3 primeiros)

Total de avaliações: 30 × 10 = 300 por classificador
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors, KernelDensity
from collections import Counter
import warnings
import time
warnings.filterwarnings('ignore')

print("="*70)
print("ETAPA 6: VALIDAÇÃO CRUZADA 30×10-FOLDS")
print("="*70)


# =============================================================================
# CLASSIFICADORES (copiados do step3 para ser standalone)
# =============================================================================

class BayesianoGaussiano(BaseEstimator, ClassifierMixin):
    """Classificador Bayesiano com Normal Multivariada."""
    def __init__(self):
        self.classes_ = None
        self.priors_ = {}
        self.means_ = {}
        self.covs_inv_ = {}
        self.covs_det_ = {}
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / n_samples
            self.means_[c] = np.mean(X_c, axis=0)
            cov = np.cov(X_c.T) + 1e-6 * np.eye(n_features)
            self.covs_inv_[c] = np.linalg.inv(cov)
            self.covs_det_[c] = np.linalg.det(cov)
        return self
    
    def _log_likelihood(self, X, c):
        d = X.shape[1]
        diff = X - self.means_[c]
        mahalanobis = np.sum(diff @ self.covs_inv_[c] * diff, axis=1)
        return -0.5 * d * np.log(2*np.pi) - 0.5 * np.log(self.covs_det_[c]) - 0.5 * mahalanobis
    
    def predict_proba(self, X):
        log_post = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            log_post[:, i] = self._log_likelihood(X, c) + np.log(self.priors_[c])
        log_post_max = np.max(log_post, axis=1, keepdims=True)
        post = np.exp(log_post - log_post_max)
        return post / np.sum(post, axis=1, keepdims=True)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNNBayesiano(BaseEstimator, ClassifierMixin):
    """Classificador Bayesiano baseado em K-Vizinhos."""
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        n_samples = len(y)
        self.n_samples_per_class_ = {c: np.sum(y == c) for c in self.classes_}
        self.priors_ = {c: np.sum(y == c) / n_samples for c in self.classes_}
        self.nn_model_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.nn_model_.fit(X)
        return self
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        distances, indices = self.nn_model_.kneighbors(X)
        posteriors = np.zeros((n_samples, len(self.classes_)))
        
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
    """Classificador Bayesiano baseado em Janela de Parzen."""
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
        log_post = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            log_post[:, i] = self.kde_models_[c].score_samples(X) + np.log(self.priors_[c])
        log_post_max = np.max(log_post, axis=1, keepdims=True)
        post = np.exp(log_post - log_post_max)
        return post / np.sum(post, axis=1, keepdims=True)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class VotoMajoritario(BaseEstimator, ClassifierMixin):
    """Classificador por Voto Majoritário (Gaussiano + KNN + Parzen)."""
    def __init__(self, knn_params=None, parzen_params=None):
        self.knn_params = knn_params or {'n_neighbors': 5, 'metric': 'euclidean'}
        self.parzen_params = parzen_params or {'bandwidth': 1.0}
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clf_gaussian = BayesianoGaussiano()
        self.clf_knn = KNNBayesiano(**self.knn_params)
        self.clf_parzen = ParzenWindowClassifier(**self.parzen_params)
        
        self.clf_gaussian.fit(X, y)
        self.clf_knn.fit(X, y)
        self.clf_parzen.fit(X, y)
        return self
    
    def predict(self, X):
        pred_gaussian = self.clf_gaussian.predict(X)
        pred_knn = self.clf_knn.predict(X)
        pred_parzen = self.clf_parzen.predict(X)
        
        predictions = np.array([pred_gaussian, pred_knn, pred_parzen])
        majority = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            majority.append(Counter(votes).most_common(1)[0][0])
        return np.array(majority)


# =============================================================================
# FUNÇÕES DE AJUSTE DE HIPERPARÂMETROS (CV 5-folds interno)
# =============================================================================

def ajustar_knn_bayesiano(X_train, y_train, n_folds=5):
    """Ajusta KNN Bayesiano com CV 5-folds."""
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    
    best_score = -1
    best_params = {'n_neighbors': 5, 'metric': 'euclidean'}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for k in k_values:
        for metric in metrics:
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                try:
                    clf = KNNBayesiano(n_neighbors=k, metric=metric)
                    clf.fit(X_train[train_idx], y_train[train_idx])
                    scores.append(accuracy_score(y_train[val_idx], clf.predict(X_train[val_idx])))
                except:
                    continue
            if scores and np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_params = {'n_neighbors': k, 'metric': metric}
    
    return best_params


def ajustar_parzen(X_train, y_train, n_folds=5):
    """Ajusta Parzen Window com CV 5-folds."""
    bandwidth_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    best_score = -1
    best_params = {'bandwidth': 1.0}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for bw in bandwidth_values:
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            try:
                clf = ParzenWindowClassifier(bandwidth=bw)
                clf.fit(X_train[train_idx], y_train[train_idx])
                scores.append(accuracy_score(y_train[val_idx], clf.predict(X_train[val_idx])))
            except:
                continue
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_params = {'bandwidth': bw}
    
    return best_params


def ajustar_logistica(X_train, y_train, n_folds=5):
    """Ajusta Regressão Logística com CV 5-folds."""
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    best_score = -1
    best_params = {'C': 1.0}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for C in C_values:
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            try:
                clf = LogisticRegression(C=C, penalty='l2', max_iter=1000, solver='lbfgs', multi_class='multinomial')
                clf.fit(X_train[train_idx], y_train[train_idx])
                scores.append(accuracy_score(y_train[val_idx], clf.predict(X_train[val_idx])))
            except:
                continue
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_params = {'C': C}
    
    return best_params


# =============================================================================
# FUNÇÕES DE MÉTRICAS
# =============================================================================

def calculate_metrics(y_true, y_pred, average='weighted'):
    """Calcula as 4 métricas pedidas."""
    return {
        'error_rate': 1 - accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


# =============================================================================
# VALIDAÇÃO CRUZADA 30×10-FOLDS
# =============================================================================

def run_30x10_fold_cv(X, y, n_repeats=30, n_folds=10, verbose=True):
    """
    Executa validação cruzada estratificada 30×10-folds.
    
    Para cada fold externo:
    1. Ajusta hiperparâmetros com CV 5-folds interno (KNN, Parzen, Logística)
    2. Treina classificadores com hiperparâmetros selecionados
    3. Avalia no fold de teste
    
    Parâmetros:
    -----------
    X : array-like
        Features (já normalizadas)
    y : array-like
        Labels
    n_repeats : int
        Número de repetições (default: 30)
    n_folds : int
        Número de folds (default: 10)
    verbose : bool
        Se True, imprime progresso
    
    Retorna:
    --------
    dict : Resultados por classificador e métrica
           results[classificador][metrica] = lista de 300 valores
    """
    # Inicializar estrutura de resultados
    classifiers = ['Gaussiano', 'KNN Bayesiano', 'Parzen', 'Logística', 'Voto Majoritário']
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    
    results = {clf: {m: [] for m in metrics} for clf in classifiers}
    
    # Armazenar hiperparâmetros selecionados
    hyperparams_history = {
        'KNN Bayesiano': [],
        'Parzen': [],
        'Logística': []
    }
    
    total_iterations = n_repeats * n_folds
    iteration = 0
    start_time = time.time()
    
    for repeat in range(n_repeats):
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Repetição {repeat + 1}/{n_repeats}")
            print(f"{'─'*50}")
        
        # Criar folds com seed diferente para cada repetição
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            iteration += 1
            
            if verbose and fold % 2 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / iteration) * (total_iterations - iteration)
                print(f"  Fold {fold + 1}/{n_folds} | "
                      f"Progresso: {100*iteration/total_iterations:.1f}% | "
                      f"ETA: {eta/60:.1f} min")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # ─────────────────────────────────────────────────────────────
            # 1. BAYESIANO GAUSSIANO (sem hiperparâmetros)
            # ─────────────────────────────────────────────────────────────
            clf_gaussian = BayesianoGaussiano()
            clf_gaussian.fit(X_train, y_train)
            y_pred_gaussian = clf_gaussian.predict(X_test)
            metrics_gaussian = calculate_metrics(y_test, y_pred_gaussian)
            
            # ─────────────────────────────────────────────────────────────
            # 2. KNN BAYESIANO (ajustar hiperparâmetros com CV 5-folds)
            # ─────────────────────────────────────────────────────────────
            knn_params = ajustar_knn_bayesiano(X_train, y_train, n_folds=5)
            hyperparams_history['KNN Bayesiano'].append(knn_params)
            
            clf_knn = KNNBayesiano(**knn_params)
            clf_knn.fit(X_train, y_train)
            y_pred_knn = clf_knn.predict(X_test)
            metrics_knn = calculate_metrics(y_test, y_pred_knn)
            
            # ─────────────────────────────────────────────────────────────
            # 3. PARZEN WINDOW (ajustar hiperparâmetros com CV 5-folds)
            # ─────────────────────────────────────────────────────────────
            parzen_params = ajustar_parzen(X_train, y_train, n_folds=5)
            hyperparams_history['Parzen'].append(parzen_params)
            
            clf_parzen = ParzenWindowClassifier(**parzen_params)
            clf_parzen.fit(X_train, y_train)
            y_pred_parzen = clf_parzen.predict(X_test)
            metrics_parzen = calculate_metrics(y_test, y_pred_parzen)
            
            # ─────────────────────────────────────────────────────────────
            # 4. REGRESSÃO LOGÍSTICA (ajustar hiperparâmetros com CV 5-folds)
            # ─────────────────────────────────────────────────────────────
            lr_params = ajustar_logistica(X_train, y_train, n_folds=5)
            hyperparams_history['Logística'].append(lr_params)
            
            clf_lr = LogisticRegression(**lr_params, penalty='l2', max_iter=1000, 
                                        solver='lbfgs', multi_class='multinomial')
            clf_lr.fit(X_train, y_train)
            y_pred_lr = clf_lr.predict(X_test)
            metrics_lr = calculate_metrics(y_test, y_pred_lr)
            
            # ─────────────────────────────────────────────────────────────
            # 5. VOTO MAJORITÁRIO (usa hiperparâmetros de KNN e Parzen)
            # ─────────────────────────────────────────────────────────────
            clf_majority = VotoMajoritario(knn_params=knn_params, parzen_params=parzen_params)
            clf_majority.fit(X_train, y_train)
            y_pred_majority = clf_majority.predict(X_test)
            metrics_majority = calculate_metrics(y_test, y_pred_majority)
            
            # ─────────────────────────────────────────────────────────────
            # Armazenar resultados
            # ─────────────────────────────────────────────────────────────
            for metric in metrics:
                results['Gaussiano'][metric].append(metrics_gaussian[metric])
                results['KNN Bayesiano'][metric].append(metrics_knn[metric])
                results['Parzen'][metric].append(metrics_parzen[metric])
                results['Logística'][metric].append(metrics_lr[metric])
                results['Voto Majoritário'][metric].append(metrics_majority[metric])
    
    # Tempo total
    total_time = time.time() - start_time
    if verbose:
        print(f"\n{'='*50}")
        print(f"Tempo total: {total_time/60:.2f} minutos")
        print(f"Total de avaliações: {total_iterations}")
        print(f"{'='*50}")
    
    return results, hyperparams_history


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def print_results_summary(results, version_name):
    """Imprime resumo dos resultados."""
    from scipy import stats
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS - {version_name}")
    print(f"{'='*80}")
    
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    metric_names = ['Taxa de Erro', 'Precisão', 'Cobertura', 'F-measure']
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{'-'*70}")
        print(f"{metric_name}:")
        print(f"{'-'*70}")
        print(f"{'Classificador':<20} {'Média':<10} {'Desvio':<10} {'IC 95%':<25}")
        print(f"{'-'*70}")
        
        for clf_name in results.keys():
            data = results[clf_name][metric]
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n = len(data)
            se = std / np.sqrt(n)
            t_val = stats.t.ppf(0.975, n-1)
            margin = t_val * se
            
            print(f"{clf_name:<20} {mean:.4f}     {std:.4f}     [{mean-margin:.4f}, {mean+margin:.4f}]")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "═"*70)
    print("EXECUÇÃO DA VALIDAÇÃO CRUZADA 30×10-FOLDS")
    print("═"*70)
    
    # ─────────────────────────────────────────────────────────────────────
    # Carregar dados
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1] Carregando dados...")
    
    df = pd.read_csv('yeast.csv')
    X = df.drop(['Sequence Name', 'Class'], axis=1).values
    y_original = df['Class'].values
    
    # Labels do KFCM-K-W-1
    y_2classes = np.loadtxt('crisp_partition.txt').astype(int)
    
    # Codificar 10 classes
    le = LabelEncoder()
    y_10classes = le.fit_transform(y_original)
    
    print(f"    Amostras: {len(y_original)}")
    print(f"    Features: {X.shape[1]}")
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ─────────────────────────────────────────────────────────────────────
    # VERSÃO 1: 10 Classes
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VERSÃO 1: 10 CLASSES ORIGINAIS")
    print("="*70)
    
    results_10, hyperparams_10 = run_30x10_fold_cv(
        X_scaled, y_10classes, 
        n_repeats=30, n_folds=10, verbose=True
    )
    
    print_results_summary(results_10, "10 Classes")
    
    # Salvar resultados
    np.save('results_10classes.npy', results_10)
    
    # ─────────────────────────────────────────────────────────────────────
    # VERSÃO 2: 2 Classes (KFCM-K-W-1)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VERSÃO 2: 2 CLUSTERS (KFCM-K-W-1)")
    print("="*70)
    
    results_2, hyperparams_2 = run_30x10_fold_cv(
        X_scaled, y_2classes,
        n_repeats=30, n_folds=10, verbose=True
    )
    
    print_results_summary(results_2, "2 Clusters")
    
    # Salvar resultados
    np.save('results_2classes.npy', results_2)
    
    print("\n" + "="*70)
    print("✓ Validação cruzada concluída!")
    print("  Arquivos salvos: results_10classes.npy, results_2classes.npy")
    print("="*70)