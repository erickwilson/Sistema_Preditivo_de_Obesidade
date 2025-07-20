# treinamento_modelo.py

# --- 1. Importação das Bibliotecas ---
# Importamos as ferramentas necessárias para nosso trabalho.
# pandas: para carregar e manipular nossos dados (o arquivo .csv).
# scikit-learn: a principal biblioteca de Machine Learning em Python.
#   - train_test_split: para dividir os dados em conjuntos de treino e teste.
#   - RandomForestClassifier: o algoritmo de Machine Learning que escolhemos.
#   - LabelEncoder, OrdinalEncoder: para transformar texto em números que o modelo entenda.
#   - accuracy_score: para medir o quão bom nosso modelo é.
# joblib: para salvar nosso modelo treinado e usá-lo mais tarde na aplicação.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- 2. Carregamento e Preparação dos Dados ---
print("Iniciando o processo de treinamento...")

try:
    # Usamos o pandas para ler o arquivo CSV e carregá-lo em um DataFrame.
    # Um DataFrame é como uma tabela ou planilha do Excel, mas dentro do Python.
    dataframe_original = pd.read_csv('obesity.csv')
    print("Arquivo 'obesity.csv' carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: O arquivo 'obesity.csv' não foi encontrado.")
    print("Por favor, certifique-se de que o arquivo está na mesma pasta que este script.")
    exit() # Encerra o script se o arquivo não for encontrado.

# Para garantir consistência, renomeamos colunas que possam ter nomes diferentes.
# 'errors='ignore'' evita que o programa quebre se a coluna já tiver o nome correto.
dataframe_original.rename(columns={'family_history': 'family_history_with_overweight'}, inplace=True, errors='ignore')
dataframe_original.rename(columns={'Obesity_level': 'Obesity'}, inplace=True, errors='ignore')

# --- 3. Separação das Features e da Variável Alvo ---
# O modelo precisa saber o que são as "perguntas" (features) e o que é a "resposta" (variável alvo).

# 'X' conterá todas as colunas, exceto a que queremos prever ('Obesity').
features = dataframe_original.drop('Obesity', axis=1)
# 'y' conterá apenas a coluna 'Obesity', que é o nosso objetivo de previsão.
alvo = dataframe_original['Obesity']

print("\nDados separados em features (características) e alvo (o que queremos prever).")

# --- 4. Pré-processamento: Transformando Texto em Números ---
# Modelos de Machine Learning não entendem texto como "Masculino" ou "Sim".
# Precisamos converter todas as colunas de texto em números.

# Identificamos quais colunas são de texto (categóricas).
colunas_categoricas = features.select_dtypes(include=['object', 'category']).columns

# Criamos um dicionário para guardar os "tradutores" (encoders) que usarmos.
# Isso é crucial para que a aplicação web possa usar a mesma tradução depois.
mapeamento_encoders = {}

print(f"Iniciando a conversão das colunas de texto: {list(colunas_categoricas)}")

for coluna in colunas_categoricas:
    # Se a coluna tem apenas duas opções (ex: Sim/Não), usamos o LabelEncoder.
    if features[coluna].nunique() == 2:
        encoder = LabelEncoder()
        features[coluna] = encoder.fit_transform(features[coluna])
    # Se tiver mais de duas opções, usamos o OrdinalEncoder.
    else:
        # Capturamos a ordem original das categorias para manter a consistência.
        ordem_categorias = list(features[coluna].unique())
        encoder = OrdinalEncoder(categories=[ordem_categorias])
        features[coluna] = encoder.fit_transform(features[[coluna]])
    
    # Guardamos o encoder usado para esta coluna.
    mapeamento_encoders[coluna] = encoder

print("Conversão de texto para número concluída.")

# --- 5. Divisão dos Dados em Treino e Teste ---
# Vamos dividir nosso conjunto de dados em duas partes:
# - 80% para treinar o modelo (para ele aprender).
# - 20% para testar o modelo (para ver se ele aprendeu direito, com dados que nunca viu).
# 'random_state=42' garante que a divisão seja sempre a mesma, para resultados reproduzíveis.
# 'stratify=alvo' garante que a proporção das classes de obesidade seja a mesma nos dois conjuntos.
features_treino, features_teste, alvo_treino, alvo_teste = train_test_split(
    features, alvo, test_size=0.2, random_state=42, stratify=alvo
)

print(f"\nDados divididos: {len(features_treino)} amostras para treino e {len(features_teste)} para teste.")

# --- 6. Treinamento do Modelo ---
# Agora, a parte principal: criar e treinar o modelo.
# Usamos o RandomForestClassifier, que é como um comitê de "árvores de decisão".
# Ele é poderoso e bom para evitar overfitting (quando o modelo decora em vez de aprender).
print("\nIniciando o treinamento do modelo... Isso pode levar um momento.")
modelo = RandomForestClassifier(
    n_estimators=200,      # Número de "árvores" no comitê.
    random_state=42,       # Para resultados consistentes.
    max_depth=20,          # Profundidade máxima de cada árvore.
    min_samples_leaf=1,    # Mínimo de amostras em uma folha final.
    min_samples_split=2    # Mínimo de amostras para dividir um nó.
)

# O comando .fit() é o que efetivamente treina o modelo com os dados de treino.
modelo.fit(features_treino, alvo_treino)
print("Modelo treinado com sucesso!")

# --- 7. Avaliação da Performance ---
# Vamos usar o conjunto de teste (que o modelo nunca viu) para ver o quão bom ele é.
print("\nAvaliando a performance do modelo com os dados de teste...")
previsoes_teste = modelo.predict(features_teste)

# Comparamos as previsões do modelo com as respostas reais do conjunto de teste.
acuracia = accuracy_score(alvo_teste, previsoes_teste)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

# Verificamos se atingimos a meta do desafio.
if acuracia > 0.75:
    print("Meta de acurácia (> 75%) atingida com sucesso!")
else:
    print("ATENÇÃO: A meta de acurácia não foi atingida. Considere ajustar o modelo.")

    
print("\n--- Avaliação Detalhada do Modelo ---")
print("\nMatriz de Confusão:")
# A matriz mostra nas linhas o que era real e nas colunas o que o modelo previu.
print(confusion_matrix(alvo_teste, previsoes_teste))

print("\nRelatório de Classificação:")
# Este relatório é a melhor forma de ver a performance para cada classe.
print(classification_report(alvo_teste, previsoes_teste))

# --- 8. Salvando o Modelo e os Encoders ---
# Salvamos o modelo treinado e o dicionário de encoders em arquivos.
# Assim, nossa aplicação web pode carregá-los sem precisar treinar tudo de novo.
joblib.dump(modelo, 'modelo_obesidade.pkl')
joblib.dump(mapeamento_encoders, 'encoders.pkl')

print("\nModelo salvo como 'modelo_obesidade.pkl'.")
print("Mapeamento de encoders salvo como 'encoders.pkl'.")
print("\nProcesso de treinamento finalizado. Você já pode executar a aplicação web.")
