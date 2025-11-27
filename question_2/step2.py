"""
================================================================================
ETAPA 2: CRIAR AS DUAS VERSÕES DO DATASET
================================================================================

O enunciado pede para trabalhar com 2 versões do Yeast:
- Versão 1: 10 classes originais
- Versão 2: K* classes (K* = 2, obtido na Questão 1 via KFCM-K-W-1) (crisp_partition.txt),

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import json

print("="*70)
print("ETAPA 2: CRIAR AS DUAS VERSÕES DO DATASET")
print("="*70)

# -----------------------------------------------------------------------------
# 2.1 Carregar os dados
# -----------------------------------------------------------------------------
print("\n[2.1] Carregando os dados...")

df = pd.read_csv('yeast.csv')
X = df.drop(['Sequence Name', 'Class'], axis=1).values
y_original = df['Class'].values

print(f"Amostras: {len(y_original)}")
print(f"Features: {X.shape[1]}")
print(f"Classes originais: {list(np.unique(y_original))}")

# -----------------------------------------------------------------------------
# 2.2 VERSÃO 1: Dataset com 10 classes
# -----------------------------------------------------------------------------
print("\n"+ "-"*70)
print("[2.2] VERSÃO 1: Dataset com 10 classes originais")
print("-"*70)

# Codificar labels para números (necessário para sklearn)
le_10 = LabelEncoder()
y_10classes = le_10.fit_transform(y_original)

print(f"Mapeamento das classes:")
for i, cls in enumerate(le_10.classes_):
    count = np.sum(y_original == cls)
    pct = 100 * count / len(y_original)
    print(f" {cls} → {i} ({count:4d} amostras, {pct:5.1f}%)")

print(f"Shape X: {X.shape}")
print(f"Shape y: {y_10classes.shape}")
print(f"Classes únicas: {np.unique(y_10classes)}")

# -----------------------------------------------------------------------------
# 2.3 VERSÃO 2: Dataset com K*=2 classes (LABELS DO KFCM-K-W-1!)
# -----------------------------------------------------------------------------
print("\n"+ "-"*70)
print("[2.3] VERSÃO 2: Dataset com K*=2 classes (KFCM-K-W-1)")
print("-"*70)

# Carregar labels do arquivo crisp_partition.txt
y_2classes = np.loadtxt('crisp_partition.txt').astype(int)

print(f"Labels carregados de crisp_partition.txt")
print(f"Total de labels: {len(y_2classes)}")

# Verificar consistência
assert len(y_2classes) == len(y_original), "ERRO: Número de labels diferente do número de amostras!"

# Contagem
count_0 = np.sum(y_2classes == 0)
count_1 = np.sum(y_2classes == 1)

print(f"Distribuição dos clusters:")
print(f" Cluster 0: {count_0:4d} amostras ({100*count_0/len(y_2classes):5.1f}%)")
print(f" Cluster 1: {count_1:4d} amostras ({100*count_1/len(y_2classes):5.1f}%)")

# Analisar composição dos clusters
print("Composição dos clusters (classes originais em cada cluster):")
for cluster in [0, 1]:
    print(f"📦 CLUSTER {cluster}:")
    mask = y_2classes == cluster
    classes_in_cluster = y_original[mask]
    class_counts = Counter(classes_in_cluster)
    
    total = sum(class_counts.values())
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█"* int(pct/5)
        print(f"  {cls:4s}: {count:4d} ({pct:5.1f}%) {bar}")

# -----------------------------------------------------------------------------
# 2.4 Visualizar as duas versões
# -----------------------------------------------------------------------------
print("\n[2.4] Gerando visualização comparativa...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Versão 1: 10 classes
ax1 = axes[0]
class_counts_10 = pd.Series(y_original).value_counts().sort_values(ascending=False)
colors_10 = plt.cm.tab10(np.linspace(0, 1, 10))
bars1 = ax1.bar(range(len(class_counts_10)), class_counts_10.values, color=colors_10)
ax1.set_xticks(range(len(class_counts_10)))
ax1.set_xticklabels(class_counts_10.index, rotation=45, ha='right')
ax1.set_xlabel('Classe')
ax1.set_ylabel('Quantidade')
ax1.set_title('VERSÃO 1: 10 Classes Originais', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, count in zip(bars1, class_counts_10.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom', fontsize=9)

# Versão 2: 2 clusters (KFCM-K-W-1)
ax2 = axes[1]
class_counts_2 = [count_0, count_1]
colors_2 = ['#e74c3c', '#3498db']
bars2 = ax2.bar(['Cluster 0', 'Cluster 1'], class_counts_2, color=colors_2, width=0.5)
ax2.set_ylabel('Quantidade')
ax2.set_title('VERSÃO 2: K*=2 Clusters (KFCM-K-W-1)\nSilhouette = 0.927', 
              fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, count in zip(bars2, class_counts_2):
    pct = 100 * count / len(y_2classes)
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('step2_duas_versoes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Salvo: results/step2_duas_versoes.png")

# -----------------------------------------------------------------------------
# 2.5 Salvar dados para próximas etapas
# -----------------------------------------------------------------------------
print("\n[2.5] Salvando dados processados...")

np.save('results/X_data.npy', X)
np.save('results/y_10classes.npy', y_10classes)
np.save('results/y_2classes.npy', y_2classes)

# Salvar mapeamento das 10 classes
with open('results/label_mapping_10classes.json', 'w') as f:
    json.dump({int(i): cls for i, cls in enumerate(le_10.classes_)}, f, indent=2)

print("results/X_data.npy (features)")
print("results/y_10classes.npy (10 classes originais)")
print("results/y_2classes.npy (2 clusters do KFCM-K-W-1)")
print("results/label_mapping_10classes.json")

# -----------------------------------------------------------------------------
# RESUMO DA ETAPA 2
# -----------------------------------------------------------------------------
print("\n"+ "="*70)
print("RESUMO DA ETAPA 2")
print("="*70)
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ VERSÃO 1: 10 Classes Originais                                      │
├─────────────────────────────────────────────────────────────────────┤
│ Classes: CYT, NUC, MIT, ME3, ME2, ME1, EXC, VAC, POX, ERL          │
│ Codificadas como: 0-9 (ordem alfabética)                           │
│ Problema: Muito desbalanceado (CYT: 31.2% vs ERL: 0.3%)            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ VERSÃO 2: K*=2 Clusters (KFCM-K-W-1 da Questão 1)                  │
├─────────────────────────────────────────────────────────────────────┤
│ Cluster 0: {count_0:4d} amostras ({100*count_0/len(y_2classes):4.1f}%)                                   │
│ Cluster 1: {count_1:4d} amostras ({100*count_1/len(y_2classes):4.1f}%)                                   │
│ Silhouette Score: 0.927                                             │
│                                                                     │
│ ⚠️  Problema EXTREMAMENTE desbalanceado ({100*count_0/len(y_2classes):.0f}% vs {100*count_1/len(y_2classes):.0f}%)              │
│ O KFCM-K-W-1 identificou um cluster de "outliers"           │
└─────────────────────────────────────────────────────────────────────┘

Arquivos gerados em results/:
  - step2_duas_versoes.png (visualização)
  - X_data.npy, y_10classes.npy, y_2classes.npy (dados)
  - label_mapping_10classes.json (mapeamento)
""")