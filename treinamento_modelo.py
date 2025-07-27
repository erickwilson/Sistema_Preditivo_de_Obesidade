# --- 1. Importação das Bibliotecas ---
# Importamos as bibliotecas essenciais para o processo de Machine Learning.
# pandas: para manipulação de dados em formato de tabela.
# joblib: para salvar o modelo e os encoders treinados.
# scikit-learn:
#   - train_test_split: divide os dados em treino e teste.
#   - GridSearchCV: busca os melhores hiperparâmetros para o modelo.
#   - LabelEncoder, OrdinalEncoder: convertem dados categóricos em números.
#   - accuracy_score, classification_report: avaliam a performance do modelo.
# xgboost: um algoritmo robusto e eficiente para classificação multiclasse.
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb  # Classificador XGBoost para tarefas de classificação multiclasse

# --- 2. Carregamento e Preparação dos Dados ---
# Iniciamos lendo os dados de um arquivo CSV, que precisa estar na mesma pasta que o script.
print("Iniciando o processo de treinamento otimizado...")

try:
    dataframe_original = pd.read_csv('obesity.csv')
    print("Arquivo 'obesity.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo 'obesity.csv' não encontrado.")
    exit()  # Encerra o script caso o arquivo não seja encontrado.

# Renomeamos colunas para manter consistência com outros scripts e facilitar a leitura.
dataframe_original.rename(columns={
    'family_history_with_overweight': 'family_history',
    'Obesity_level': 'Obesity'
}, inplace=True, errors='ignore')

# --- 3. Engenharia de Features - Cálculo do IMC ---
# Calculamos uma nova feature chamada IMC (Índice de Massa Corporal),
# que pode ajudar o modelo a entender melhor o estado físico da pessoa.
dataframe_original['IMC'] = dataframe_original['Weight'] / (dataframe_original['Height'] ** 2)
print("Feature 'IMC' calculada e adicionada ao conjunto de dados.")

# --- 4. Separação das Features e do Alvo ---
# Separamos os dados em:
# - 'features': todas as colunas que o modelo pode usar para aprender.
# - 'alvo': a coluna 'Obesity', que queremos prever.
features = dataframe_original.drop('Obesity', axis=1)
alvo = dataframe_original['Obesity']

# Como a coluna alvo contém texto, usamos o LabelEncoder para convertê-la em números.
encoder_alvo = LabelEncoder()
alvo_encoded = encoder_alvo.fit_transform(alvo)
print("Coluna alvo codificada para formato numérico.")

# --- 5. Pré-processamento das Features ---
# Precisamos transformar colunas de texto em números, pois o modelo não entende texto diretamente.
colunas_categoricas = features.select_dtypes(include=['object', 'category']).columns
mapeamento_encoders = {}

print(f"Codificando colunas categóricas: {list(colunas_categoricas)}")

# Para cada coluna categórica:
for coluna in colunas_categoricas:
    # Se houver apenas duas categorias (ex: Sim/Não), usamos LabelEncoder.
    if features[coluna].nunique() == 2:
        encoder = LabelEncoder()
        features[coluna] = encoder.fit_transform(features[coluna])
    # Caso contrário, usamos OrdinalEncoder com a ordem original.
    else:
        ordem_categorias = list(features[coluna].unique())
        encoder = OrdinalEncoder(categories=[ordem_categorias])
        features[coluna] = encoder.fit_transform(features[[coluna]])
    
    # Guardamos o encoder no dicionário para reutilizar depois.
    mapeamento_encoders[coluna] = encoder

print("Codificação das variáveis categóricas concluída.")

# --- 6. Divisão dos Dados em Treino e Teste ---
# Separamos os dados para que o modelo aprenda com parte deles e seja testado com outra parte.
# - 80% para treino, 20% para teste.
# - stratify=alvo_encoded garante que todas as classes estejam bem distribuídas nos dois grupos.
features_treino, features_teste, alvo_treino, alvo_teste = train_test_split(
    features, alvo_encoded, test_size=0.2, random_state=42, stratify=alvo_encoded
)
print(f"Dados divididos: {len(features_treino)} para treino e {len(features_teste)} para teste.")

# --- 7. Otimização de Hiperparâmetros com GridSearchCV ---
# Usamos o GridSearchCV para testar várias combinações de parâmetros do XGBoost
# e encontrar a que traz melhor performance.
print("Iniciando busca por hiperparâmetros ideais com GridSearchCV...")

parametros_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.05, 0.1, 0.2],
    'colsample_bytree': [0.7, 1.0]
}

# Configuramos o modelo com parâmetros básicos e deixamos o GridSearch ajustar os melhores.
modelo_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

# Executamos o GridSearch com validação cruzada.
grid_search = GridSearchCV(
    estimator=modelo_xgb,
    param_grid=parametros_grid,
    scoring='accuracy',
    cv=3,           # 3 divisões para validação cruzada.
    n_jobs=-1,      # Usa todos os núcleos disponíveis para acelerar.
    verbose=1       # Mostra o progresso no terminal.
)

# Treinamos o modelo com os melhores parâmetros encontrados.
grid_search.fit(features_treino, alvo_treino)

melhor_modelo = grid_search.best_estimator_
print("Otimização concluída.")
print(f"Melhores parâmetros: {grid_search.best_params_}")

# --- 8. Avaliação do Modelo ---
# Agora, vamos ver o quão bom o modelo ficou, usando os dados de teste.
print("Iniciando avaliação do modelo otimizado...")

previsoes_teste = melhor_modelo.predict(features_teste)
acuracia = accuracy_score(alvo_teste, previsoes_teste)
print(f"Acurácia final do modelo: {acuracia * 100:.2f}%")

# Verificamos se atingimos a meta de desempenho.
if acuracia > 0.75:
    print("Meta alcançada: acurácia superior a 75%.")
else:
    print("Meta não atingida: considere ajustar parâmetros ou testar outros modelos.")

# Geramos o relatório detalhado de performance para cada classe de obesidade.
alvo_teste_texto = encoder_alvo.inverse_transform(alvo_teste)
previsoes_teste_texto = encoder_alvo.inverse_transform(previsoes_teste)
print("Relatório de classificação:")
print(classification_report(alvo_teste_texto, previsoes_teste_texto))

# --- 9. Salvamento do Modelo e dos Encoders ---
# Salvamos o modelo treinado e os encoders em arquivos.
# Esses arquivos serão usados pela aplicação web para fazer previsões sem treinar novamente.
joblib.dump(melhor_modelo, 'modelo_obesidade.pkl')
mapeamento_encoders['encoder_alvo'] = encoder_alvo
joblib.dump(mapeamento_encoders, 'encoders.pkl')

print("Modelo e encoders salvos com sucesso.")
