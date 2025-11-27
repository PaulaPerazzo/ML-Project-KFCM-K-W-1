"""
================================================================================
ETAPA 4: FUNÇÕES DE MÉTRICAS E INTERVALOS DE CONFIANÇA
================================================================================

Métricas pedidas no enunciado:
- Taxa de erro
- Precisão
- Cobertura (Recall)
- F-measure

Com estimativa pontual e intervalo de confiança 95%.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

print("="*70)
print("ETAPA 4: FUNÇÕES DE MÉTRICAS E INTERVALOS DE CONFIANÇA")
print("="*70)


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calcula as 4 métricas de avaliação pedidas no enunciado.
    
    Parâmetros:
    -----------
    y_true : array-like
        Labels verdadeiros
    y_pred : array-like
        Labels preditos
    average : str
        Tipo de média para métricas multiclasse ('weighted', 'macro', 'micro')
        - 'weighted': pondera pela quantidade de amostras em cada classe
        - 'macro': média simples entre classes
        - 'micro': calcula globalmente
    
    Retorna:
    --------
    dict : Dicionário com as 4 métricas
    """
    metrics = {
        'error_rate': 1 - accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics


def confidence_interval(data, confidence=0.95):
    """
    Calcula intervalo de confiança usando distribuição t de Student.
    
    Parâmetros:
    -----------
    data : array-like
        Dados para calcular o intervalo
    confidence : float
        Nível de confiança (default: 0.95 para 95%)
    
    Retorna:
    --------
    tuple : (média, limite_inferior, limite_superior)
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 para desvio padrão amostral
    se = std / np.sqrt(n)  # erro padrão
    
    # t-value para o intervalo de confiança
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin = t_value * se
    
    return mean, mean - margin, mean + margin


def print_metrics_summary(results, version_name):
    """
    Imprime resumo das métricas com estimativas pontuais e ICs.
    
    Parâmetros:
    -----------
    results : dict
        Dicionário com resultados por classificador e métrica
        Formato: results[classificador][metrica] = lista de valores
    version_name : str
        Nome da versão do dataset (ex: "10 Classes")
    """
    print(f"\n{'='*80}")
    print(f"RESULTADOS - {version_name}")
    print(f"{'='*80}")
    
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    metric_names = ['Taxa de Erro', 'Precisão', 'Cobertura (Recall)', 'F-measure']
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{'-'*70}")
        print(f"{metric_name}:")
        print(f"{'-'*70}")
        print(f"{'Classificador':<25} {'Média':<12} {'IC 95%':<30}")
        print(f"{'-'*70}")
        
        for clf_name in results.keys():
            if metric in results[clf_name]:
                mean, lower, upper = confidence_interval(results[clf_name][metric])
                print(f"{clf_name:<25} {mean:.4f}       [{lower:.4f}, {upper:.4f}]")


def create_results_dataframe(results, version_name):
    """
    Cria DataFrame com resultados para exportação.
    
    Parâmetros:
    -----------
    results : dict
        Dicionário com resultados
    version_name : str
        Nome da versão
    
    Retorna:
    --------
    pd.DataFrame : DataFrame com resultados formatados
    """
    import pandas as pd
    
    rows = []
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    metric_names_pt = {
        'error_rate': 'Taxa de Erro',
        'precision': 'Precisão',
        'recall': 'Cobertura',
        'f1': 'F-measure'
    }
    
    for clf_name in results.keys():
        for metric in metrics:
            if metric in results[clf_name]:
                mean, lower, upper = confidence_interval(results[clf_name][metric])
                rows.append({
                    'Versão': version_name,
                    'Classificador': clf_name,
                    'Métrica': metric_names_pt[metric],
                    'Média': mean,
                    'IC_Inferior': lower,
                    'IC_Superior': upper,
                    'Desvio_Padrão': np.std(results[clf_name][metric], ddof=1)
                })
    
    return pd.DataFrame(rows)


# =============================================================================
# DEMONSTRAÇÃO
# =============================================================================
if __name__ == "__main__":
    print("\n" + "═"*70)
    print("DEMONSTRAÇÃO DAS FUNÇÕES DE MÉTRICAS")
    print("═"*70)
    
    # Exemplo com dados simulados
    np.random.seed(42)
    
    # Simular 300 avaliações (30 repetições × 10 folds)
    n_evaluations = 300
    
    # Resultados simulados
    results_example = {
        'Gaussiano': {
            'error_rate': np.random.normal(0.45, 0.05, n_evaluations),
            'precision': np.random.normal(0.55, 0.05, n_evaluations),
            'recall': np.random.normal(0.55, 0.05, n_evaluations),
            'f1': np.random.normal(0.55, 0.05, n_evaluations)
        },
        'KNN Bayesiano': {
            'error_rate': np.random.normal(0.40, 0.04, n_evaluations),
            'precision': np.random.normal(0.60, 0.04, n_evaluations),
            'recall': np.random.normal(0.60, 0.04, n_evaluations),
            'f1': np.random.normal(0.60, 0.04, n_evaluations)
        }
    }
    
    # Demonstrar funções
    print("\n1. Cálculo de métricas para uma predição:")
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 0])
    metrics = calculate_metrics(y_true, y_pred)
    for m, v in metrics.items():
        print(f"   {m}: {v:.4f}")
    
    print("\n2. Intervalo de confiança:")
    data = np.random.normal(0.6, 0.05, 300)
    mean, lower, upper = confidence_interval(data)
    print(f"   Média: {mean:.4f}")
    print(f"   IC 95%: [{lower:.4f}, {upper:.4f}]")
    
    print("\n3. Resumo de resultados:")
    print_metrics_summary(results_example, "Exemplo")
    
    print("\n" + "="*70)
    print("✓ Funções de métricas implementadas")
    print("="*70)