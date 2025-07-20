# painel_analitico.py

import streamlit as st
import pandas as pd
import plotly.express as px

# Função para carregar e preparar os dados (cache para performance)
@st.cache_data
def carregar_dados():
    """Carrega, traduz e prepara o dataset para análise."""
    try:
        df = pd.read_csv('obesity.csv')
    except FileNotFoundError:
        st.error("Arquivo 'obesity.csv' não encontrado. Certifique-se de que ele está na pasta.")
        return None

    # Renomear colunas para consistência
    df.rename(columns={'family_history_with_overweight': 'family_history', 'Obesity': 'Obesity'}, inplace=True, errors='ignore')

    # Dicionários para tradução dos valores
    traducao_sim_nao = {'yes': 'Sim', 'no': 'Não'}
    traducao_genero = {'Male': 'Masculino', 'Female': 'Feminino'}
    traducao_nivel_obesidade = {
        'Insufficient_Weight': 'Peso Insuficiente',
        'Normal_Weight': 'Peso Normal',
        'Overweight_Level_I': 'Sobrepeso Nível I',
        'Overweight_Level_II': 'Sobrepeso Nível II',
        'Obesity_Type_I': 'Obesidade Tipo I',
        'Obesity_Type_II': 'Obesidade Tipo II',
        'Obesity_Type_III': 'Obesidade Tipo III'
    }

    # Aplicando as traduções para os gráficos
    df['Tem_Historico_Familiar'] = df['family_history'].map(traducao_sim_nao)
    df['Genero'] = df['Gender'].map(traducao_genero)
    df['Nivel_Obesidade'] = df['Obesity'].map(traducao_nivel_obesidade)
    
    # Definindo a ordem correta das categorias para os gráficos
    ordem_niveis = list(traducao_nivel_obesidade.values())
    df['Nivel_Obesidade'] = pd.Categorical(df['Nivel_Obesidade'], categories=ordem_niveis, ordered=True)

    return df

# --- Início da Construção da Página ---
st.set_page_config(page_title="Painel Analítico de Obesidade", page_icon="📊", layout="wide")

st.title("📊 Painel Analítico sobre Obesidade")
st.markdown("Esta página apresenta insights e padrões obtidos a partir dos dados de pacientes.")

# Carrega os dados
dataframe = carregar_dados()

if dataframe is not None:
    # --- Layout do Painel ---
    st.markdown("### 1. Distribuição Geral dos Níveis de Obesidade")
    st.markdown("Este gráfico mostra a quantidade de pacientes em cada categoria de peso, permitindo identificar quais são as mais prevalentes no conjunto de dados.")
    
    # Gráfico de barras da distribuição geral
    fig_distribuicao = px.histogram(dataframe.sort_values('Nivel_Obesidade'), x='Nivel_Obesidade', title='Contagem de Pacientes por Nível de Peso', text_auto=True)
    fig_distribuicao.update_layout(xaxis_title="Nível de Peso", yaxis_title="Quantidade de Pacientes")
    st.plotly_chart(fig_distribuicao, use_container_width=True)

    st.markdown("---")

    # Layout em duas colunas para os próximos gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 2. Relação entre Histórico Familiar e Obesidade")
        st.markdown("Pacientes com histórico familiar de sobrepeso têm uma tendência maior a desenvolver obesidade? Este gráfico compara as duas populações.")
        
        # Gráfico de barras agrupado
        fig_familia = px.histogram(
            dataframe.sort_values('Nivel_Obesidade'), 
            x="Nivel_Obesidade", 
            color="Tem_Historico_Familiar", 
            barmode='group',
            title='Nível de Obesidade vs. Histórico Familiar',
            text_auto=True
        )
        fig_familia.update_layout(xaxis_title="Nível de Peso", yaxis_title="Quantidade de Pacientes", legend_title="Tem Histórico Familiar?")
        st.plotly_chart(fig_familia, use_container_width=True)

    with col2:
        st.markdown("### 3. Consumo de Álcool e Níveis de Peso")
        st.markdown("Como o hábito de consumir bebidas alcoólicas se correlaciona com os diferentes níveis de peso dos pacientes?")
        
        # Gráfico de barras empilhado 100%
        fig_alcool = px.histogram(
            dataframe, 
            x="Nivel_Obesidade", 
            color="CALC", 
            title='Influência do Consumo de Álcool no Nível de Peso',
            barnorm='percent', # Mostra em percentual
            text_auto='.2f'
        )
        fig_alcool.update_layout(xaxis_title="Nível de Peso", yaxis_title="Percentual de Pacientes (%)", legend_title="Frequência de Consumo de Álcool")
        st.plotly_chart(fig_alcool, use_container_width=True)

    st.markdown("---")

    st.markdown("### 4. Análise Bivariada: Idade, Peso e Gênero")
    st.markdown("Este gráfico de dispersão (scatter plot) nos ajuda a visualizar a relação entre idade e peso, separando os pontos por gênero e nível de obesidade. Podemos identificar tendências, como o aumento de peso com a idade.")

    # Gráfico de dispersão
    fig_idade_peso = px.scatter(
        dataframe,
        x="Age",
        y="Weight",
        color="Nivel_Obesidade",
        hover_data=['Genero'], # Mostra o gênero ao passar o mouse
        title='Relação entre Idade, Peso e Nível de Obesidade'
    )
    fig_idade_peso.update_layout(xaxis_title="Idade", yaxis_title="Peso (kg)")
    st.plotly_chart(fig_idade_peso, use_container_width=True)

    # --- Principais Insights para a Equipe Médica ---
    st.markdown("---")
    st.header("💡 Principais Insights para a Equipe Médica")
    st.markdown("""
    - **Prevalência de Sobrepeso e Obesidade:** Os dados indicam que uma parcela significativa dos pacientes se encontra nas categorias de **Sobrepeso** e **Obesidade Tipo I, II e III**, superando os casos de Peso Normal. Isso reforça a importância de programas de prevenção e tratamento.

    - **Forte Influência do Histórico Familiar:** O Gráfico 2 demonstra claramente que pacientes com histórico familiar de sobrepeso têm uma probabilidade muito maior de estarem em categorias de sobrepeso e obesidade em comparação com aqueles sem histórico. Aconselhamento genético e acompanhamento precoce para essas famílias podem ser estratégias eficazes.

    - **Álcool e Categorias de Risco:** O Gráfico 3 sugere que o consumo frequente de álcool está mais presente em pacientes com **Sobrepeso** e **Obesidade Tipo I**. Embora não seja uma causa única, pode ser um fator de risco importante a ser abordado nas consultas.

    - **Relação Idade-Peso:** O Gráfico 4 mostra uma tendência de aumento de peso com a idade. Intervenções preventivas focadas em faixas etárias mais jovens podem ser cruciais para evitar a progressão para níveis mais graves de obesidade na vida adulta.
    """)

