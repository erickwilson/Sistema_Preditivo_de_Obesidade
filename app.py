# app.py — Interface Web para Previsão de Obesidade com Streamlit e XGBoost

# --- 1. Importação das Bibliotecas ---
# streamlit: biblioteca para a interface web.
# pandas: manipulação dos dados de entrada e saída.
# joblib: carregamento do modelo e encoders treinados.
import streamlit as st
import pandas as pd
import joblib

# --- 2. Configuração Inicial da Página ---
# Define título da aba, ícone e layout da interface Streamlit.
st.set_page_config(
    page_title="Previsão de Obesidade",
    page_icon="🩺",
    layout="centered"
)

# --- 3. Carregamento do Modelo e Encoders ---
# Usa cache para evitar recarregamento desnecessário dos arquivos.
@st.cache_resource
def load_model_and_encoders():
    """Carrega o modelo treinado e os encoders salvos."""
    try:
        modelo_carregado = joblib.load('modelo_obesidade.pkl')
        encoders_carregados = joblib.load('encoders.pkl')
        return modelo_carregado, encoders_carregados
    except FileNotFoundError:
        st.error("Arquivos 'modelo_obesidade.pkl' ou 'encoders.pkl' não encontrados.")
        st.error("Execute o script de treinamento antes de iniciar esta aplicação.")
        return None, None

# Carrega modelo e encoders em memória
modelo, todos_os_encoders = load_model_and_encoders()

# --- 4. Dicionários para Tradução da Interface ---
# Dicionários para tradução dos dados exibidos ao usuário (português) e enviados ao modelo (inglês).
dicionario_traducao_genero = {'Male': 'Masculino', 'Female': 'Feminino'}
dicionario_traducao_sim_nao = {'yes': 'Sim', 'no': 'Não'}
dicionario_traducao_refeicoes = {'no': 'Não', 'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre'}
dicionario_traducao_alcool = dicionario_traducao_refeicoes
dicionario_traducao_transporte = {
    'Automobile': 'Automóvel', 'Motorbike': 'Moto', 'Bike': 'Bicicleta',
    'Public_Transportation': 'Transporte Público', 'Walking': 'A pé'
}

# Tradução das classes de saída do modelo
dicionario_traducao_previsao = {
    'Insufficient_Weight': 'Peso Insuficiente',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III'
}

# --- 5. Construção da Interface Gráfica (UI) ---
st.title("🩺 Sistema Preditivo de Obesidade")
st.markdown("Utilize os campos da barra lateral para inserir os dados do paciente e obter uma previsão do nível de obesidade.")

# Cria barra lateral com campos de entrada
st.sidebar.header("Parâmetros do Paciente")

def obter_dados_do_usuario():
    """
    Cria os campos de entrada na barra lateral e retorna os dados em dois formatos:
    - Um DataFrame para o modelo (em inglês)
    - Um dicionário para exibição (em português)
    """
    # --- Informações Pessoais ---
    st.sidebar.markdown("### 🧍 Informações Pessoais")
    genero_pt = st.sidebar.selectbox('Gênero', options=list(dicionario_traducao_genero.values()))
    idade = st.sidebar.slider('Idade', 14, 100, 25)
    altura_m = st.sidebar.slider('Altura (m)', 1.20, 2.20, 1.70, 0.01)
    peso_kg = st.sidebar.slider('Peso (kg)', 30.0, 200.0, 70.0, 0.5)
    historia_familiar_pt = st.sidebar.radio('Histórico familiar de sobrepeso?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)

    # --- Hábitos Alimentares ---
    st.sidebar.markdown("### 🍎 Hábitos Alimentares")
    comida_calorica_pt = st.sidebar.radio('Consumo frequente de alimentos calóricos?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)
    consumo_vegetais = st.sidebar.slider('Frequência de consumo de vegetais', 1, 3, 2, help="1 = Nunca, 2 = Às vezes, 3 = Sempre")
    num_refeicoes = st.sidebar.slider('Nº de refeições diárias', 1, 4, 3)
    come_entre_refeicoes_pt = st.sidebar.selectbox('Come entre as refeições?', options=list(dicionario_traducao_refeicoes.values()))
    consumo_agua = st.sidebar.slider('Consumo de água por dia', 1, 3, 2, help="1 = <1L, 2 = 1-2L, 3 = >2L")
    monitora_calorias_pt = st.sidebar.radio('Monitora calorias?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)
    consumo_alcool_pt = st.sidebar.selectbox('Frequência de consumo de álcool', options=list(dicionario_traducao_alcool.values()))

    # --- Hábitos de Vida ---
    st.sidebar.markdown("### 🏃 Hábitos de Vida")
    fumante_pt = st.sidebar.radio('Fumante?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)
    freq_atividade_fisica = st.sidebar.slider('Frequência de atividade física (semana)', 0, 3, 1, help="0 = Nenhuma, 1 = 1-2x, 2 = 2-4x, 3 = 4-5x")
    tempo_telas = st.sidebar.slider('Tempo em telas por dia', 0, 2, 1, help="0 = 0-2h, 1 = 3-5h, 2 = >5h")
    transporte_pt = st.sidebar.selectbox('Principal meio de transporte', options=list(dicionario_traducao_transporte.values()))

    # Conversão para inglês (valores esperados pelo modelo)
    def traduzir(valor_pt, dicionario): return [k for k, v in dicionario.items() if v == valor_pt][0]
    
    # Cálculo do IMC (adicionado nesta nova versão)
    imc = peso_kg / (altura_m ** 2)

    dados_modelo = {
        'Gender': traduzir(genero_pt, dicionario_traducao_genero),
        'Age': idade,
        'Height': altura_m,
        'Weight': peso_kg,
        'family_history': traduzir(historia_familiar_pt, dicionario_traducao_sim_nao),
        'FAVC': traduzir(comida_calorica_pt, dicionario_traducao_sim_nao),
        'FCVC': consumo_vegetais,
        'NCP': num_refeicoes,
        'CAEC': traduzir(come_entre_refeicoes_pt, dicionario_traducao_refeicoes),
        'SMOKE': traduzir(fumante_pt, dicionario_traducao_sim_nao),
        'CH2O': consumo_agua,
        'SCC': traduzir(monitora_calorias_pt, dicionario_traducao_sim_nao),
        'FAF': freq_atividade_fisica,
        'TUE': tempo_telas,
        'CALC': traduzir(consumo_alcool_pt, dicionario_traducao_alcool),
        'MTRANS': traduzir(transporte_pt, dicionario_traducao_transporte),
        'IMC': imc
    }

    dados_exibicao = {
        'Gênero': genero_pt, 'Idade': idade, 'Altura': altura_m, 'Peso': peso_kg,
        'IMC Calculado': f'{imc:.2f}', 'Histórico Familiar': historia_familiar_pt,
        'Alimentos Calóricos': comida_calorica_pt, 'Vegetais': consumo_vegetais,
        'Refeições': num_refeicoes, 'Lanches': come_entre_refeicoes_pt, 'Fumante': fumante_pt,
        'Água (L/dia)': consumo_agua, 'Monitora Calorias': monitora_calorias_pt,
        'Atividade Física': freq_atividade_fisica, 'Tempo em Telas': tempo_telas,
        'Álcool': consumo_alcool_pt, 'Transporte': transporte_pt
    }

    return pd.DataFrame(dados_modelo, index=[0]), dados_exibicao

# --- 6. Lógica Principal da Aplicação ---
if modelo and todos_os_encoders:
    df_modelo, exibicao = obter_dados_do_usuario()

    # Verifica se os valores excedem os limites de treino do modelo
    limites_treino = {'Age': 62, 'Weight': 173, 'Height': 1.98}
    if (exibicao['Idade'] > limites_treino['Age'] or
        exibicao['Peso'] > limites_treino['Weight'] or
        exibicao['Altura'] > limites_treino['Height']):
        st.warning("Os valores inseridos excedem os limites observados durante o treinamento. A precisão da previsão pode ser afetada.")

    # Exibe dados inseridos pelo usuário em duas colunas
    with st.expander("Resumo dos Dados Informados"):
        col1, col2 = st.columns(2)
        lista = list(exibicao.items())
        for i, (chave, valor) in enumerate(lista):
            (col1 if i < len(lista)//2 else col2).markdown(f"**{chave}:** {valor}")

    st.write("")
    _, botao, _ = st.columns([1, 2, 1])
    with botao:
        if st.button('**Realizar Predição**', use_container_width=True):

            # --- Preparação para Predição ---
            encoders = todos_os_encoders.copy()
            encoder_alvo = encoders.pop('encoder_alvo', None)

            if not encoder_alvo:
                st.error("Encoder do alvo não encontrado.")
            else:
                # Codificação dos dados categóricos
                df_encoded = df_modelo.copy()
                for coluna, encoder in encoders.items():
                    if coluna in df_encoded.columns:
                        if hasattr(encoder, 'categories_'):
                            df_encoded[coluna] = encoder.transform(df_encoded[[coluna]])
                        else:
                            df_encoded[coluna] = encoder.transform(df_encoded[coluna])

                # --- Realiza a predição ---
                pred_numerica = modelo.predict(df_encoded)[0]
                pred_label = encoder_alvo.inverse_transform([pred_numerica])[0]
                pred_proba = modelo.predict_proba(df_encoded)

                # Exibição dos resultados
                pred_pt = dicionario_traducao_previsao.get(pred_label, pred_label)
                confianca = pred_proba.max() * 100

                st.subheader("Resultado da Predição")
                st.success(f"Nível previsto: **{pred_pt}**")
                st.info(f"Confiança do modelo: **{confianca:.2f}%**")

                # --- Exibição do gráfico de probabilidades ---
                st.subheader("Distribuição das Probabilidades")
                df_proba = pd.DataFrame({
                    'Classe': encoder_alvo.classes_,
                    'Probabilidade': pred_proba[0] * 100
                })
                df_proba['Classe Traduzida'] = df_proba['Classe'].map(dicionario_traducao_previsao)
                ordem = list(dicionario_traducao_previsao.values())
                df_proba['Classe Traduzida'] = pd.Categorical(df_proba['Classe Traduzida'], categories=ordem, ordered=True)
                df_proba = df_proba.sort_values('Classe Traduzida')

                st.bar_chart(df_proba, x='Categoria de Peso ', y='Probabilidade')

# Caso o modelo ou encoders não tenham sido carregados corretamente
else:
    st.error("A aplicação não foi iniciada. Verifique o carregamento do modelo.")
