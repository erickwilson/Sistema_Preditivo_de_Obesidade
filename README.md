# 🩺 Sistema Preditivo de Obesidade - Tech Challenge

https://sistema-preditivo-obesidade.streamlit.app/

Este repositório contém a solução completa para o Tech Challenge, que consiste no desenvolvimento de um sistema de Machine Learning para prever o nível de obesidade de um paciente com base em suas características físicas, hábitos alimentares e estilo de vida.

O projeto abrange todo o pipeline de ciência de dados, desde a análise exploratória e treinamento do modelo até o deploy de uma aplicação web interativa e um painel analítico para extração de insights.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/en/stable/)


## 🚀 Funcionalidades

O projeto é dividido em duas principais funcionalidades entregues como uma aplicação multi-página:

1.  **Sistema de Previsão Individual:**
    *   Uma interface interativa onde a equipe médica pode inserir os dados de um paciente através de um formulário na barra lateral.
    *   O sistema utiliza um modelo de `XGBoostClassifier` treinado para prever em qual das 7 categorias de peso o paciente se enquadra.
    *   Exibe o resultado da previsão de forma clara, junto com a probabilidade de confiança do modelo.
    *   Apresenta um gráfico de barras com a distribuição de probabilidade para todas as classes possíveis, ordenado de forma clínica (de Peso Insuficiente a Obesidade Tipo III).

2.  **Painel Analítico para Insights:**
    *   Uma página de dashboard com visualizações de dados para apoiar a tomada de decisão da equipe médica.
    *   Gráficos interativos que exploram a relação entre o nível de obesidade e fatores como:
        *   Distribuição geral dos níveis de peso na base de dados.
        *   Influência do histórico familiar de sobrepeso.
        *   Correlação com o consumo de álcool.
        *   Relação entre idade, peso e gênero.
    *   Uma seção com os principais insights extraídos da análise, traduzindo os dados em informações acionáveis.

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python 3
*   **Análise e Modelagem:**
    *   **Pandas:** Para manipulação e limpeza dos dados.
    *   **Scikit-learn:** Para o pré-processamento, treinamento e avaliação do modelo de Machine Learning.
    *   **Joblib:** Para serialização (salvar e carregar) do modelo treinado.
*   **Visualização e Interface:**
    *   **Streamlit:** Para a construção da aplicação web interativa e do dashboard.
    *   **Plotly Express:** Para a criação dos gráficos dinâmicos no painel analítico.

## ⚙️ Como Executar o Projeto Localmente

Siga os passos abaixo para executar a aplicação na sua máquina.

### 1. Pré-requisitos

*   Ter o [Python](https://www.python.org/downloads/) (versão 3.8 ou superior) instalado.
*   Ter o `pip` (gerenciador de pacotes do Python) disponível.

### 2. Clone o Repositório

Abra o seu terminal e clone este repositório:
```bash
git clone https://github.com/erickwilson/Sistema_Preditivo_de_Obesidade.git
cd Sistema_Preditivo_de_Obesidade
```

### 3. Crie um Ambiente Virtual (Recomendado)

É uma boa prática isolar as dependências do projeto.
```bash
# Criar o ambiente
python -m venv venv

# Ativar o ambiente
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

### 4. Instale as Dependências

Instale todas as bibliotecas necessárias com um único comando:```bash
pip install pandas scikit-learn joblib streamlit plotly.express
```

### 5. Treine o Modelo

Antes de iniciar a aplicação, você precisa treinar o modelo. Este passo irá ler o `obesity.csv` e criar os arquivos `modelo_obesidade.pkl` e `encoders.pkl`.
```bash
python treinamento_modelo.py
```

### 6. Inicie a Aplicação Streamlit

Com o modelo treinado, inicie a aplicação web:
```bash
streamlit run app.py
```
O Streamlit abrirá a aplicação automaticamente no seu navegador. Você poderá navegar entre o "Sistema Preditivo" e o "Painel Analítico" pela barra lateral.

## 📁 Estrutura do Repositório

```
.
├── app.py                  # Script principal da aplicação de previsão
├── treinamento_modelo.py   # Script para o pipeline de ML e treinamento
├── obesity.csv             # Dataset utilizado no projeto
├── modelo_obesidade.pkl    # Modelo treinado e salvo
├── encoders.pkl            # Encoders salvos para pré-processamento
├── pages/
│   └── painel_analitico.py # Script da página do dashboard de insights
└── README.md               # Este arquivo
```

