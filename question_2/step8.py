"""
================================================================================
ETAPA 8: CURVAS DE APRENDIZAGEM
================================================================================

O enunciado pede:
"d) Para cada métrica de avaliação, plot a curva de aprendizagem para o
classificador bayesiano Gaussiano. Mais precisamente, considere conjuntos
de treinamento e teste de (5%, 95%) a (95%, 5%) do conjunto original de
treinamento, com passo de 5% (usando amostragem estratificada). Para cada
par de conjuntos de treinamento e teste, compute as métricas de avaliação
tanto no conjunto de treinamento como no conjunto de teste. Comente."

Objetivo: Verificar overfitting/underfitting do Bayesiano Gaussiano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ETAPA 8: CURVAS DE APRENDIZAGEM")
print("="*70)


# =============================================================================
# CLASSIFICADOR BAYESIANO GAUSSIANO
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


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calculate_metrics(y_true, y_pred, average='weighted'):
    """Calcula as 4 métricas."""
    return {
        'error_rate': 1 - accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def stratified_sample(X, y, train_size, random_state=42):
    """
    Amostragem estratificada para dividir dados em treino/teste.
    
    Parâmetros:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    train_size : float
        Proporção para treino (0.05 a 0.95)
    random_state : int
        Seed para reprodutibilidade
    
    Retorna:
    --------
    X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for c in classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        
        n_train = max(1, int(len(class_indices) * train_size))
        
        train_indices.extend(class_indices[:n_train])
        test_indices.extend(class_indices[n_train:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def compute_learning_curves(X, y, train_sizes=None, n_runs=10):
    """
    Computa curvas de aprendizagem para o Bayesiano Gaussiano.
    
    Parâmetros:
    -----------
    X : array-like
        Features (já normalizadas)
    y : array-like
        Labels
    train_sizes : array-like
        Proporções de treino a testar (default: 5% a 95%, passo 5%)
    n_runs : int
        Número de execuções para cada tamanho (para média)
    
    Retorna:
    --------
    dict : Métricas de treino e teste para cada tamanho
    """
    if train_sizes is None:
        train_sizes = np.arange(0.05, 1.0, 0.05)  # 5% a 95%, passo 5%
    
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    
    results = {
        'train_sizes': train_sizes,
        'train': {m: [] for m in metrics},
        'test': {m: [] for m in metrics},
        'train_std': {m: [] for m in metrics},
        'test_std': {m: [] for m in metrics}
    }
    
    print(f"\nComputando curvas de aprendizagem...")
    print(f"Tamanhos de treino: {len(train_sizes)} ({train_sizes[0]*100:.0f}% a {train_sizes[-1]*100:.0f}%)")
    print(f"Execuções por tamanho: {n_runs}")
    
    for i, train_size in enumerate(train_sizes):
        train_metrics_runs = {m: [] for m in metrics}
        test_metrics_runs = {m: [] for m in metrics}
        
        for run in range(n_runs):
            # Amostragem estratificada
            X_train, X_test, y_train, y_test = stratified_sample(
                X, y, train_size, random_state=run
            )
            
            # Verificar se há amostras suficientes
            if len(X_train) < 2 or len(X_test) < 2:
                continue
            
            # Treinar classificador
            try:
                clf = BayesianoGaussiano()
                clf.fit(X_train, y_train)
                
                # Predições
                y_pred_train = clf.predict(X_train)
                y_pred_test = clf.predict(X_test)
                
                # Métricas
                m_train = calculate_metrics(y_train, y_pred_train)
                m_test = calculate_metrics(y_test, y_pred_test)
                
                for m in metrics:
                    train_metrics_runs[m].append(m_train[m])
                    test_metrics_runs[m].append(m_test[m])
            except Exception as e:
                continue
        
        # Média e desvio padrão
        for m in metrics:
            if train_metrics_runs[m]:
                results['train'][m].append(np.mean(train_metrics_runs[m]))
                results['test'][m].append(np.mean(test_metrics_runs[m]))
                results['train_std'][m].append(np.std(train_metrics_runs[m]))
                results['test_std'][m].append(np.std(test_metrics_runs[m]))
            else:
                results['train'][m].append(np.nan)
                results['test'][m].append(np.nan)
                results['train_std'][m].append(np.nan)
                results['test_std'][m].append(np.nan)
        
        if (i + 1) % 5 == 0:
            print(f"  Progresso: {i+1}/{len(train_sizes)} tamanhos")
    
    return results


def plot_learning_curves(results, version_name, output_file=None):
    """
    Plota as curvas de aprendizagem.
    
    Parâmetros:
    -----------
    results : dict
        Resultados de compute_learning_curves
    version_name : str
        Nome da versão do dataset
    output_file : str
        Caminho para salvar a figura (opcional)
    """
    train_sizes = results['train_sizes'] * 100  # Converter para percentual
    
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    metric_names = ['Taxa de Erro', 'Precisão', 'Cobertura (Recall)', 'F-measure']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Curvas de Aprendizagem - Bayesiano Gaussiano\n{version_name}', 
                 fontsize=14, fontweight='bold')
    
    for ax, metric, title in zip(axes.flatten(), metrics, metric_names):
        train_mean = np.array(results['train'][metric])
        test_mean = np.array(results['test'][metric])
        train_std = np.array(results['train_std'][metric])
        test_std = np.array(results['test_std'][metric])
        
        # Linha de treino
        ax.plot(train_sizes, train_mean, 'b-o', label='Treino', markersize=4, linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='blue')
        
        # Linha de teste
        ax.plot(train_sizes, test_mean, 'r-s', label='Teste', markersize=4, linewidth=2)
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                       alpha=0.2, color='red')
        
        ax.set_xlabel('% dos dados no conjunto de treino', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'Curva de Aprendizagem - {title}', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        # Definir limites do eixo Y baseado na métrica
        if metric == 'error_rate':
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nFigura salva em: {output_file}")
    
    plt.show()


def analyze_learning_curves(results, version_name):
    """
    Analisa as curvas de aprendizagem e gera comentários.
    
    Parâmetros:
    -----------
    results : dict
        Resultados das curvas
    version_name : str
        Nome da versão
    """
    print(f"\n{'='*70}")
    print(f"ANÁLISE DAS CURVAS DE APRENDIZAGEM - {version_name}")
    print(f"{'='*70}")
    
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    metric_names = ['Taxa de Erro', 'Precisão', 'Cobertura', 'F-measure']
    
    for metric, name in zip(metrics, metric_names):
        train_final = results['train'][metric][-1]
        test_final = results['test'][metric][-1]
        train_initial = results['train'][metric][0]
        test_initial = results['test'][metric][0]
        
        gap = abs(train_final - test_final)
        
        print(f"\n{name}:")
        print(f"  Inicial (5% treino): Treino={train_initial:.4f}, Teste={test_initial:.4f}")
        print(f"  Final (95% treino):  Treino={train_final:.4f}, Teste={test_final:.4f}")
        print(f"  Gap final (treino-teste): {gap:.4f}")
        
        # Diagnóstico
        if metric == 'error_rate':
            if gap > 0.1:
                print(f"  → Possível OVERFITTING (gap alto)")
            elif train_final > 0.4:
                print(f"  → Possível UNDERFITTING (erro alto no treino)")
            else:
                print(f"  → Modelo bem ajustado")
        else:
            if gap > 0.1:
                print(f"  → Possível OVERFITTING")
            elif train_final < 0.6:
                print(f"  → Possível UNDERFITTING")
            else:
                print(f"  → Modelo bem ajustado")
    
    print(f"\n{'─'*70}")
    print("COMENTÁRIOS GERAIS:")
    print("─"*70)
    
    # Análise geral
    error_gap = abs(results['train']['error_rate'][-1] - results['test']['error_rate'][-1])
    error_train = results['train']['error_rate'][-1]
    
    if error_gap < 0.05 and error_train < 0.3:
        print("""
✓ O classificador Bayesiano Gaussiano apresenta BOM comportamento:
  - As curvas de treino e teste convergem (baixo gap)
  - O modelo generaliza bem para dados não vistos
  - Não há sinais claros de overfitting ou underfitting
        """)
    elif error_gap > 0.1:
        print("""
⚠️ O classificador mostra sinais de OVERFITTING:
  - Grande diferença entre desempenho no treino e teste
  - O modelo pode estar memorizando os dados de treino
  - Possível causa: classes raras com poucas amostras para estimar parâmetros
        """)
    elif error_train > 0.4:
        print("""
⚠️ O classificador mostra sinais de UNDERFITTING:
  - Alto erro mesmo no conjunto de treino
  - O modelo é muito simples para capturar os padrões
  - A suposição de normalidade multivariada pode não ser adequada
        """)
    else:
        print("""
O classificador apresenta comportamento intermediário.
Mais dados podem ajudar a melhorar o desempenho.
        """)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "═"*70)
    print("GERAÇÃO DAS CURVAS DE APRENDIZAGEM")
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
    
    results_10 = compute_learning_curves(X_scaled, y_10classes, n_runs=10)
    plot_learning_curves(results_10, "10 Classes", output_file='curvas_aprendizagem_10classes.png')
    analyze_learning_curves(results_10, "10 Classes")
    
    # ─────────────────────────────────────────────────────────────────────
    # VERSÃO 2: 2 Classes (KFCM-K-W-1)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VERSÃO 2: 2 CLUSTERS (KFCM-K-W-1)")
    print("="*70)
    
    results_2 = compute_learning_curves(X_scaled, y_2classes, n_runs=10)
    plot_learning_curves(results_2, "2 Clusters (KFCM-K-W-1)", output_file='curvas_aprendizagem_2classes.png')
    analyze_learning_curves(results_2, "2 Clusters")
    
    print("\n" + "="*70)
    print("✓ Curvas de aprendizagem concluídas!")
    print("  Arquivos salvos:")
    print("    - curvas_aprendizagem_10classes.png")
    print("    - curvas_aprendizagem_2classes.png")
    print("="*70)