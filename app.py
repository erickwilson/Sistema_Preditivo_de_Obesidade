# app.py

# --- 1. Importação das Bibliotecas ---
# streamlit: a biblioteca principal para criar a interface web.
# pandas: para criar o DataFrame com os dados do usuário.
# joblib: para carregar nosso modelo e encoders que foram salvos anteriormente.
import streamlit as st
import pandas as pd
import joblib

# --- 2. Configuração Inicial da Página ---
# Este comando configura o título que aparece na aba do navegador e o ícone.
# Deve ser o primeiro comando do Streamlit a ser executado no script.
st.set_page_config(
    page_title="Previsão de Obesidade",
    page_icon="🩺",  # Ícone de estetoscópio
    layout="centered" # Centraliza o conteúdo na página
)

# --- 3. Carregamento do Modelo e Encoders ---
# Usamos uma função com o decorador @st.cache_resource para que o modelo
# seja carregado apenas uma vez, otimizando a performance da aplicação.
@st.cache_resource
def carregar_modelo_e_encoders():
    """
    Carrega o modelo de Machine Learning e os encoders salvos pelo script de treinamento.
    Retorna None se os arquivos não forem encontrados.
    """
    try:
        modelo_carregado = joblib.load('modelo_obesidade.pkl')
        encoders_carregados = joblib.load('encoders.pkl')
        return modelo_carregado, encoders_carregados
    except FileNotFoundError:
        st.error("ERRO: Arquivos 'modelo_obesidade.pkl' ou 'encoders.pkl' não encontrados.")
        st.error("Por favor, execute o script 'treinamento_modelo.py' primeiro para gerar os arquivos.")
        return None, None

# Carrega os artefatos na memória.
modelo, mapeamento_encoders = carregar_modelo_e_encoders()

# --- 4. Dicionários para Tradução da Interface ---
# Estes dicionários ajudam a traduzir a interface para o português,
# enquanto mantêm os valores em inglês que o modelo espera.
dicionario_traducao_genero = {'Male': 'Masculino', 'Female': 'Feminino'}
dicionario_traducao_sim_nao = {'yes': 'Sim', 'no': 'Não'}
dicionario_traducao_refeicoes = {'no': 'Não', 'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre'}
dicionario_traducao_alcool = {'no': 'Não', 'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre'}
dicionario_traducao_transporte = {'Automobile': 'Automóvel', 'Motorbike': 'Moto', 'Bike': 'Bicicleta', 'Public_Transportation': 'Transporte Público', 'Walking': 'A pé'}

# Dicionário para traduzir o resultado final do modelo.
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
st.markdown("Insira os dados do paciente na barra lateral à esquerda para obter uma previsão sobre o nível de obesidade.")

# A barra lateral (sidebar) é usada para agrupar os controles de entrada.
st.sidebar.header("Parâmetros do Paciente")

def obter_dados_do_usuario():
    """
    Cria todos os campos de entrada na barra lateral e coleta os dados do usuário.
    Retorna dois dicionários: um para o modelo (em inglês) e um para exibição (em português).
    """
    # --- Seção de Informações Pessoais ---
    st.sidebar.markdown("### 🧍 Informações Pessoais")
    genero_pt = st.sidebar.selectbox('Gênero', options=list(dicionario_traducao_genero.values()))
    idade = st.sidebar.slider('Idade', 14, 100, 25)
    altura_m = st.sidebar.slider('Altura (em metros)', 1.20, 2.20, 1.70, 0.01)
    peso_kg = st.sidebar.slider('Peso (em kgs)', 30.0, 200.0, 70.0, 0.5)
    historia_familiar_pt = st.sidebar.radio('Histórico familiar de sobrepeso?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)

    # --- Seção de Hábitos Alimentares ---
    st.sidebar.markdown("### 🍎 Hábitos Alimentares")
    comida_calorica_pt = st.sidebar.radio('Come alimentos de alta caloria com frequência?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)
    consumo_vegetais = st.sidebar.slider('Frequência de consumo de vegetais', 1, 3, 2,
        help="1 = Nunca, 2 = Às vezes, 3 = Sempre")
    num_refeicoes = st.sidebar.slider('Nº de refeições principais diárias', 1, 4, 3)
    come_entre_refeicoes_pt = st.sidebar.selectbox('Come entre as refeições?', options=list(dicionario_traducao_refeicoes.values()))
    consumo_agua = st.sidebar.slider('Consumo diário de água', 1, 3, 2,
        help="1 = Menos de 1 Litro, 2 = Entre 1 e 2 Litros, 3 = Mais de 2 Litros")
    monitora_calorias_pt = st.sidebar.radio('Monitora as calorias que ingere?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)
    consumo_alcool_pt = st.sidebar.selectbox('Frequência de consumo de álcool?', options=list(dicionario_traducao_alcool.values()))

    # --- Seção de Hábitos de Vida ---
    st.sidebar.markdown("### 🏃 Hábitos de Vida")
    fumante_pt = st.sidebar.radio('Você fuma?', options=list(dicionario_traducao_sim_nao.values()), horizontal=True)
    freq_atividade_fisica = st.sidebar.slider('Frequência de atividade física', 0, 3, 1,
        help="0 = Nenhuma, 1 = 1 a 2 dias/semana, 2 = 2 a 4 dias/semana, 3 = 4 a 5 dias/semana")
    tempo_telas = st.sidebar.slider('Tempo diário em telas (celular, TV, etc)', 0, 2, 1,
        help="0 = 0 a 2 horas, 1 = 3 a 5 horas, 2 = Mais de 5 horas")
    transporte_pt = st.sidebar.selectbox('Transporte principal?', options=list(dicionario_traducao_transporte.values()))

    # Converte as seleções em português de volta para inglês para o modelo.
    genero_en = [k for k, v in dicionario_traducao_genero.items() if v == genero_pt][0]
    historia_familiar_en = [k for k, v in dicionario_traducao_sim_nao.items() if v == historia_familiar_pt][0]
    comida_calorica_en = [k for k, v in dicionario_traducao_sim_nao.items() if v == comida_calorica_pt][0]
    come_entre_refeicoes_en = [k for k, v in dicionario_traducao_refeicoes.items() if v == come_entre_refeicoes_pt][0]
    monitora_calorias_en = [k for k, v in dicionario_traducao_sim_nao.items() if v == monitora_calorias_pt][0]
    consumo_alcool_en = [k for k, v in dicionario_traducao_alcool.items() if v == consumo_alcool_pt][0]
    fumante_en = [k for k, v in dicionario_traducao_sim_nao.items() if v == fumante_pt][0]
    transporte_en = [k for k, v in dicionario_traducao_transporte.items() if v == transporte_pt][0]

    # Dicionário com os dados no formato que o modelo espera (em inglês).
    dados_para_modelo = {'Gender': genero_en, 'Age': idade, 'Height': altura_m, 'Weight': peso_kg, 'family_history_with_overweight': historia_familiar_en, 'FAVC': comida_calorica_en, 'FCVC': consumo_vegetais, 'NCP': num_refeicoes, 'CAEC': come_entre_refeicoes_en, 'SMOKE': fumante_en, 'CH2O': consumo_agua, 'SCC': monitora_calorias_en, 'FAF': freq_atividade_fisica, 'TUE': tempo_telas, 'CALC': consumo_alcool_en, 'MTRANS': transporte_en}
    
    # Dicionário com os dados formatados para exibição na tela (em português).
    dados_para_exibicao = {'Gênero': genero_pt, 'Idade': idade, 'Altura': altura_m, 'Peso': peso_kg, 'Histórico Familiar de Sobrepeso': historia_familiar_pt, 'Consumo Frequente de Alim. Calóricos': comida_calorica_pt, 'Consumo de Vegetais': consumo_vegetais, 'Nº de Refeições Principais': num_refeicoes, 'Come entre Refeições': come_entre_refeicoes_pt, 'Fumante': fumante_pt, 'Consumo de Água': consumo_agua, 'Monitora Calorias': monitora_calorias_pt, 'Atividade Física': freq_atividade_fisica, 'Tempo em Dispositivos': tempo_telas, 'Consumo de Álcool': consumo_alcool_pt, 'Transporte Principal': transporte_pt}
    
    return pd.DataFrame(dados_para_modelo, index=[0]), dados_para_exibicao

# --- 6. Lógica Principal da Aplicação ---
# A aplicação só continua se o modelo e os encoders foram carregados com sucesso.
if modelo and mapeamento_encoders:
    # Coleta os dados do usuário a partir da barra lateral.
    dataframe_modelo, dicionario_exibicao = obter_dados_do_usuario()

    # --- Seção de Aviso de Confiabilidade ---
    limites_originais_treino = {'Age': 62, 'Weight': 173, 'Height': 1.98}
    if (dicionario_exibicao['Idade'] > limites_originais_treino['Age'] or
        dicionario_exibicao['Peso'] > limites_originais_treino['Weight'] or
        dicionario_exibicao['Altura'] > limites_originais_treino['Height']):
        st.warning(
            "**Atenção:** Os valores de Idade, Peso ou Altura inseridos estão fora da faixa de dados "
            "utilizada para treinar o modelo. A previsão pode ter uma confiabilidade menor."
        )

    # Cria uma seção expansível para mostrar o resumo dos dados inseridos.
    with st.expander("Ver Resumo dos Dados Fornecidos", expanded=False):
        col1, col2 = st.columns(2)
        itens_dicionario = list(dicionario_exibicao.items())
        ponto_medio = len(itens_dicionario) // 2
        with col1:
            for chave, valor in itens_dicionario[:ponto_medio]:
                st.markdown(f"**{chave}:** {valor}")
        with col2:
            for chave, valor in itens_dicionario[ponto_medio:]:
                st.markdown(f"**{chave}:** {valor}")

    # Adiciona um espaço e centraliza o botão de predição.
    st.write("")
    _ , coluna_botao, _ = st.columns([1, 2, 1])
    with coluna_botao:
        # Se o botão for clicado, o código dentro do 'if' é executado.
        if st.button('**Realizar Predição**', use_container_width=True):
            
            # Prepara os dados para o modelo, aplicando a mesma transformação numérica do treino.
            dataframe_encoded = dataframe_modelo.copy()
            for coluna, encoder in mapeamento_encoders.items():
                if coluna in dataframe_encoded.columns:
                    # Transforma os dados usando o encoder correto.
                    if hasattr(encoder, 'categories_'): # OrdinalEncoder
                        dataframe_encoded[coluna] = encoder.transform(dataframe_encoded[[coluna]])
                    else: # LabelEncoder
                        dataframe_encoded[coluna] = encoder.transform(dataframe_encoded[coluna])

            # --- Execução da Predição ---
            previsao_bruta = modelo.predict(dataframe_encoded)[0]
            previsao_probabilidades = modelo.predict_proba(dataframe_encoded)
            
            # Traduz o resultado da previsão para português.
            resultado_final_traduzido = dicionario_traducao_previsao.get(previsao_bruta, previsao_bruta)
            # Calcula a confiança do modelo na previsão.
            confianca_previsao = previsao_probabilidades.max() * 100

            # --- Exibição dos Resultados ---
            st.subheader('Resultado da Predição')
            st.success(f'O nível de obesidade previsto é: **{resultado_final_traduzido}**')
            st.info(f'Confiança do modelo nesta previsão: **{confianca_previsao:.2f}%**')

            st.subheader('Confiança do Modelo por Categoria')
            
            # Prepara os dados para o gráfico de barras ordenado.
            df_probabilidades = pd.DataFrame({
                'Classe': modelo.classes_, 
                'Probabilidade': previsao_probabilidades[0] * 100  # Multiplica por 100 para exibir em porcentagem
            })
            df_probabilidades['Classe Traduzida'] = df_probabilidades['Classe'].map(dicionario_traducao_previsao)
            
            ordem_grafico = list(dicionario_traducao_previsao.values())
            df_probabilidades['Classe Traduzida'] = pd.Categorical(df_probabilidades['Classe Traduzida'], categories=ordem_grafico, ordered=True)
            df_probabilidades = df_probabilidades.sort_values('Classe Traduzida')
            
            # Exibe o gráfico de barras.
            st.bar_chart(df_probabilidades, x='Classe Traduzida', y='Probabilidade')

# Se o modelo não puder ser carregado, exibe um aviso final.
else:
    st.error("Aplicação não pode ser iniciada. Verifique o carregamento do modelo.")
