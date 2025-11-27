"""
================================================================================
ETAPA 1: CARREGAR E EXPLORAR OS DADOS
================================================================================

O dataset Yeast contém informações sobre localização de proteínas em células
de levedura. É um problema de classificação multiclasse.

Features (8 atributos numéricos entre 0 e 1):
- mcg: McGeoch's method for signal sequence recognition
- gvh: von Heijne's method for signal sequence recognition  
- alm: Score of the ALOM membrane spanning region prediction program
- mit: Score of discriminant analysis of the amino acid content
- erl: Presence of "HDEL" substring (binary)
- pox: Peroxisomal targeting signal
- vac: Score of discriminant analysis of amino acid content of vacuolar proteins
- nuc: Score of discriminant analysis of nuclear localization signals

Classes (10 localizações celulares):
- CYT (cytosolic or cytoskeletal)
- NUC (nuclear)
- MIT (mitochondrial)
- ME3 (membrane protein, no N-terminal signal)
- ME2 (membrane protein, uncleaved signal)
- ME1 (membrane protein, cleaved signal)
- EXC (extracellular)
- VAC (vacuolar)
- POX (peroxisomal)
- ERL (endoplasmic reticulum lumen)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*70)
print("ETAPA 1: CARREGAR E EXPLORAR OS DADOS")
print("="*70)

# -----------------------------------------------------------------------------
# 1.1 Carregar o dataset
# -----------------------------------------------------------------------------
print("\n[1.1] Carregando o dataset...")

df = pd.read_csv('/mnt/user-data/uploads/yeast__3_.csv')

print(f"      Dataset carregado com sucesso!")
print(f"      Shape: {df.shape}")
print(f"      Colunas: {list(df.columns)}")

# -----------------------------------------------------------------------------
# 1.2 Visualizar primeiras linhas
# -----------------------------------------------------------------------------
print("\n[1.2] Primeiras 5 linhas do dataset:")
print(df.head().to_string())

# -----------------------------------------------------------------------------
# 1.3 Informações estatísticas
# -----------------------------------------------------------------------------
print("\n[1.3] Estatísticas das features:")
print(df.describe().round(3).to_string())

# -----------------------------------------------------------------------------
# 1.4 Verificar valores faltantes
# -----------------------------------------------------------------------------
print("\n[1.4] Valores faltantes por coluna:")
missing = df.isnull().sum()
print(f"      Total de valores faltantes: {missing.sum()}")

# -----------------------------------------------------------------------------
# 1.5 Distribuição das classes
# -----------------------------------------------------------------------------
print("\n[1.5] Distribuição das 10 classes originais:")
print("-"*50)

class_counts = df['Class'].value_counts().sort_values(ascending=False)
total = len(df)

for cls, count in class_counts.items():
    pct = 100 * count / total
    bar = "█" * int(pct/2)
    print(f"      {cls:4s}: {count:4d} ({pct:5.1f}%) {bar}")

print(f"\n      Total de amostras: {total}")
print(f"      Número de classes: {len(class_counts)}")

# -----------------------------------------------------------------------------
# 1.6 Separar features (X) e target (y)
# -----------------------------------------------------------------------------
print("\n[1.6] Separando features e target...")

# Remover coluna de nome da sequência (não é feature)
X = df.drop(['Sequence Name', 'Class'], axis=1).values
y = df['Class'].values

print(f"      X shape: {X.shape} (amostras × features)")
print(f"      y shape: {y.shape} (amostras)")
print(f"      Features: {list(df.drop(['Sequence Name', 'Class'], axis=1).columns)}")

# -----------------------------------------------------------------------------
# 1.7 Visualizar correlação entre features
# -----------------------------------------------------------------------------
print("\n[1.7] Gerando matriz de correlação...")

fig, ax = plt.subplots(figsize=(10, 8))
feature_cols = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
corr_matrix = df[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, ax=ax)
ax.set_title('Matriz de Correlação das Features')
plt.tight_layout()
plt.savefig('etapa1_correlacao.png', dpi=150)
plt.close()
print("Salvo: etapa1_correlacao.png")

# -----------------------------------------------------------------------------
# 1.8 Visualizar distribuição das classes
# -----------------------------------------------------------------------------
print("\n[1.8] Gerando gráfico de distribuição das classes...")

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
bars = ax.bar(class_counts.index, class_counts.values, color=colors)
ax.set_xlabel('Classe')
ax.set_ylabel('Quantidade')
ax.set_title('Distribuição das 10 Classes Originais do Dataset Yeast')

# Adicionar valores nas barras
for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            str(count), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('etapa1_distribuicao_classes.png', dpi=150)
plt.close()
print("Salvo: etapa1_distribuicao_classes.png")

# -----------------------------------------------------------------------------
# RESUMO DA ETAPA 1
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("RESUMO DA ETAPA 1")
print("="*70)
print(f"""
✓ Dataset: Yeast (UCI Machine Learning Repository)
✓ Amostras: {len(df)}
✓ Features: 8 (todas numéricas, escala 0-1)
✓ Classes: 10 (problema multiclasse desbalanceado)
✓ Valores faltantes: {missing.sum()}

Observações importantes:
- O dataset é DESBALANCEADO (CYT tem 463 amostras, ERL tem apenas 5)
- Isso afeta a escolha das métricas (usar weighted average)
- A classe ERL é muito rara (0.3%)
""")

# Salvar dados processados
np.save('X_data.npy', X)
np.save('y_data.npy', y)