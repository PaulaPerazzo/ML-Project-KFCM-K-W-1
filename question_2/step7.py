"""
================================================================================
ETAPA 7: TESTES ESTATÍSTICOS (FRIEDMAN E NEMENYI)
================================================================================

O enunciado pede:
"c) Usar o Friedman test (teste não paramétrico) para comparar os classificadores,
e o pós teste (Nemenyi test), usando cada uma das métricas"

Teste de Friedman:
- H0: Não há diferença significativa entre os classificadores
- H1: Há pelo menos uma diferença significativa

Se p-valor < 0.05: Rejeita H0, aplica pós-teste de Nemenyi

Teste de Nemenyi:
- Compara pares de classificadores
- Identifica quais são significativamente diferentes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Tentar importar scikit-posthocs para Nemenyi
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("⚠️ scikit-posthocs não instalado. Instale com: pip install scikit-posthocs")

print("="*70)
print("ETAPA 7: TESTES ESTATÍSTICOS (FRIEDMAN E NEMENYI)")
print("="*70)


def friedman_test(results, metric_name):
    """
    Realiza o teste de Friedman para comparar classificadores.
    
    Parâmetros:
    -----------
    results : dict
        Dicionário com resultados por classificador
        results[classificador][metrica] = lista de valores
    metric_name : str
        Nome da métrica a testar
    
    Retorna:
    --------
    tuple : (estatística, p-valor)
    """
    classifiers = list(results.keys())
    data = [results[clf][metric_name] for clf in classifiers]
    
    # Teste de Friedman
    stat, p_value = friedmanchisquare(*data)
    
    return stat, p_value


def nemenyi_test(results, metric_name):
    """
    Realiza o pós-teste de Nemenyi.
    
    Parâmetros:
    -----------
    results : dict
        Dicionário com resultados
    metric_name : str
        Nome da métrica
    
    Retorna:
    --------
    pd.DataFrame : Matriz de p-valores do teste de Nemenyi
    """
    if not HAS_POSTHOCS:
        print("scikit-posthocs não disponível para teste de Nemenyi")
        return None
    
    classifiers = list(results.keys())
    data = [results[clf][metric_name] for clf in classifiers]
    
    # Converter para DataFrame
    df = pd.DataFrame(data).T
    df.columns = classifiers
    
    # Teste de Nemenyi
    nemenyi_results = sp.posthoc_nemenyi_friedman(df)
    
    return nemenyi_results


def perform_statistical_tests(results, metric_name, alpha=0.05, verbose=True):
    """
    Realiza teste de Friedman e, se necessário, pós-teste de Nemenyi.
    
    Parâmetros:
    -----------
    results : dict
        Resultados dos classificadores
    metric_name : str
        Métrica a testar
    alpha : float
        Nível de significância (default: 0.05)
    verbose : bool
        Se True, imprime resultados detalhados
    
    Retorna:
    --------
    dict : Dicionário com resultados dos testes
    """
    metric_names_pt = {
        'error_rate': 'Taxa de Erro',
        'precision': 'Precisão',
        'recall': 'Cobertura (Recall)',
        'f1': 'F-measure'
    }
    
    metric_display = metric_names_pt.get(metric_name, metric_name)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TESTE DE FRIEDMAN - {metric_display.upper()}")
        print(f"{'='*70}")
    
    # Calcular ranks médios
    classifiers = list(results.keys())
    data = np.array([results[clf][metric_name] for clf in classifiers]).T
    
    # Para taxa de erro, menor é melhor (rank invertido)
    if metric_name == 'error_rate':
        ranks = np.apply_along_axis(lambda x: rankdata(x), 1, data)
    else:
        # Para outras métricas, maior é melhor (rank invertido)
        ranks = np.apply_along_axis(lambda x: rankdata(-x), 1, data)
    
    mean_ranks = np.mean(ranks, axis=0)
    
    if verbose:
        print(f"\nRanks médios:")
        for clf, rank in zip(classifiers, mean_ranks):
            print(f"  {clf:<20}: {rank:.4f}")
    
    # Teste de Friedman
    stat, p_value = friedman_test(results, metric_name)
    
    if verbose:
        print(f"\nEstatística de Friedman: {stat:.4f}")
        print(f"p-valor: {p_value:.6f}")
    
    test_results = {
        'metric': metric_name,
        'friedman_stat': stat,
        'friedman_p': p_value,
        'mean_ranks': dict(zip(classifiers, mean_ranks)),
        'reject_h0': p_value < alpha,
        'nemenyi': None
    }
    
    if p_value < alpha:
        if verbose:
            print(f"\n✓ RESULTADO: Rejeita H0 (p < {alpha})")
            print(f"  Há diferença significativa entre os classificadores")
            print(f"\n{'─'*70}")
            print(f"PÓS-TESTE DE NEMENYI")
            print(f"{'─'*70}")
        
        # Pós-teste de Nemenyi
        nemenyi_results = nemenyi_test(results, metric_name)
        test_results['nemenyi'] = nemenyi_results
        
        if verbose and nemenyi_results is not None:
            print(f"\nMatriz de p-valores (Nemenyi):")
            print(nemenyi_results.round(4).to_string())
            
            # Identificar pares significativamente diferentes
            print(f"\nPares significativamente diferentes (p < {alpha}):")
            significant_pairs = []
            for i, clf1 in enumerate(classifiers):
                for j, clf2 in enumerate(classifiers):
                    if i < j:
                        p_val = nemenyi_results.iloc[i, j]
                        if p_val < alpha:
                            significant_pairs.append((clf1, clf2, p_val))
                            print(f"  {clf1} vs {clf2}: p = {p_val:.4f}")
            
            if not significant_pairs:
                print("  Nenhum par significativamente diferente")
    else:
        if verbose:
            print(f"\n✗ RESULTADO: Não rejeita H0 (p >= {alpha})")
            print(f"  Não há diferença significativa entre os classificadores")
    
    return test_results


def plot_critical_difference_diagram(results, metric_name, alpha=0.05, output_file=None):
    """
    Plota diagrama de diferença crítica (CD diagram).
    
    Parâmetros:
    -----------
    results : dict
        Resultados dos classificadores
    metric_name : str
        Métrica a visualizar
    alpha : float
        Nível de significância
    output_file : str
        Caminho para salvar a figura (opcional)
    """
    classifiers = list(results.keys())
    n_classifiers = len(classifiers)
    n_samples = len(results[classifiers[0]][metric_name])
    
    # Calcular ranks
    data = np.array([results[clf][metric_name] for clf in classifiers]).T
    
    if metric_name == 'error_rate':
        ranks = np.apply_along_axis(lambda x: rankdata(x), 1, data)
    else:
        ranks = np.apply_along_axis(lambda x: rankdata(-x), 1, data)
    
    mean_ranks = np.mean(ranks, axis=0)
    
    # Calcular diferença crítica (CD) de Nemenyi
    # CD = q_alpha * sqrt(k(k+1)/(6N))
    # q_alpha para alpha=0.05
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q_alpha = q_alpha_table.get(n_classifiers, 2.728)
    cd = q_alpha * np.sqrt(n_classifiers * (n_classifiers + 1) / (6 * n_samples))
    
    # Ordenar por rank médio
    sorted_indices = np.argsort(mean_ranks)
    sorted_classifiers = [classifiers[i] for i in sorted_indices]
    sorted_ranks = mean_ranks[sorted_indices]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Linha horizontal
    ax.hlines(1, 1, n_classifiers, colors='black', linewidth=1)
    
    # Marcar posições dos classificadores
    for i, (clf, rank) in enumerate(zip(sorted_classifiers, sorted_ranks)):
        ax.plot(rank, 1, 'o', markersize=10, color='blue')
        ax.annotate(f"{clf}\n({rank:.2f})", 
                   xy=(rank, 1), xytext=(rank, 1.15 if i % 2 == 0 else 0.85),
                   ha='center', fontsize=9,
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    # Barra de CD
    ax.hlines(0.7, 1, 1 + cd, colors='red', linewidth=3, label=f'CD = {cd:.3f}')
    ax.annotate(f'CD = {cd:.3f}', xy=(1 + cd/2, 0.65), ha='center', fontsize=10, color='red')
    
    ax.set_xlim(0.5, n_classifiers + 0.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel('Rank Médio', fontsize=12)
    ax.set_title(f'Diagrama de Diferença Crítica - {metric_name}', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Diagrama salvo em: {output_file}")
    
    plt.show()
    
    return cd


def run_all_statistical_tests(results, version_name, alpha=0.05):
    """
    Executa todos os testes estatísticos para todas as métricas.
    
    Parâmetros:
    -----------
    results : dict
        Resultados dos classificadores
    version_name : str
        Nome da versão do dataset
    alpha : float
        Nível de significância
    
    Retorna:
    --------
    dict : Resultados de todos os testes
    """
    print(f"\n{'#'*70}")
    print(f"# TESTES ESTATÍSTICOS - {version_name.upper()}")
    print(f"{'#'*70}")
    
    metrics = ['error_rate', 'precision', 'recall', 'f1']
    all_results = {}
    
    for metric in metrics:
        all_results[metric] = perform_statistical_tests(results, metric, alpha=alpha)
    
    # Resumo
    print(f"\n{'='*70}")
    print(f"RESUMO DOS TESTES - {version_name}")
    print(f"{'='*70}")
    print(f"\n{'Métrica':<20} {'Friedman χ²':<15} {'p-valor':<15} {'Resultado':<20}")
    print(f"{'-'*70}")
    
    for metric in metrics:
        res = all_results[metric]
        resultado = "Diferença significativa" if res['reject_h0'] else "Sem diferença"
        print(f"{metric:<20} {res['friedman_stat']:<15.4f} {res['friedman_p']:<15.6f} {resultado:<20}")
    
    return all_results


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "═"*70)
    print("EXECUÇÃO DOS TESTES ESTATÍSTICOS")
    print("═"*70)
    
    # Tentar carregar resultados salvos
    try:
        results_10 = np.load('results_10classes.npy', allow_pickle=True).item()
        print("\n✓ Resultados de 10 classes carregados")
        
        # Executar testes para 10 classes
        test_results_10 = run_all_statistical_tests(results_10, "10 Classes")
        
    except FileNotFoundError:
        print("\n⚠️ Arquivo results_10classes.npy não encontrado")
        print("   Execute step6.py primeiro para gerar os resultados")
        
        # Criar dados de exemplo para demonstração
        print("\n   Gerando dados de exemplo para demonstração...")
        np.random.seed(42)
        n = 300
        
        results_10 = {
            'Gaussiano': {
                'error_rate': np.random.normal(0.45, 0.05, n),
                'precision': np.random.normal(0.55, 0.05, n),
                'recall': np.random.normal(0.55, 0.05, n),
                'f1': np.random.normal(0.55, 0.05, n)
            },
            'KNN Bayesiano': {
                'error_rate': np.random.normal(0.40, 0.04, n),
                'precision': np.random.normal(0.60, 0.04, n),
                'recall': np.random.normal(0.60, 0.04, n),
                'f1': np.random.normal(0.60, 0.04, n)
            },
            'Parzen': {
                'error_rate': np.random.normal(0.38, 0.04, n),
                'precision': np.random.normal(0.62, 0.04, n),
                'recall': np.random.normal(0.62, 0.04, n),
                'f1': np.random.normal(0.62, 0.04, n)
            },
            'Logística': {
                'error_rate': np.random.normal(0.39, 0.04, n),
                'precision': np.random.normal(0.61, 0.04, n),
                'recall': np.random.normal(0.61, 0.04, n),
                'f1': np.random.normal(0.61, 0.04, n)
            },
            'Voto Majoritário': {
                'error_rate': np.random.normal(0.37, 0.04, n),
                'precision': np.random.normal(0.63, 0.04, n),
                'recall': np.random.normal(0.63, 0.04, n),
                'f1': np.random.normal(0.63, 0.04, n)
            }
        }
        
        test_results_10 = run_all_statistical_tests(results_10, "10 Classes (Exemplo)")
    
    # Tentar carregar resultados de 2 classes
    try:
        results_2 = np.load('results_2classes.npy', allow_pickle=True).item()
        print("\n✓ Resultados de 2 classes carregados")
        test_results_2 = run_all_statistical_tests(results_2, "2 Clusters")
    except FileNotFoundError:
        print("\n⚠️ Arquivo results_2classes.npy não encontrado")
    
    print("\n" + "="*70)
    print("✓ Testes estatísticos concluídos!")
    print("="*70)