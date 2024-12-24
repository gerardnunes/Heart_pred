import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Importando os dados
df = pd.read_csv('heart.csv')

# Renomeando colunas
df = df.rename(columns={
    'chol': 'coleste',
    'fbs': 'acucars',
    'cp': 'dorpeito'
})

# Separando os dados
X = df.drop(columns='target')
y = df['target']

# Escalando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Treinando os modelos
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

logistic_model = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model.fit(X_train, y_train)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Função para prever o novo dado
def prever_dado(dado, modelo, tipo='classificacao'):
    dado = scaler.transform([dado])  # Escalando o dado de entrada
    if tipo == 'classificacao':
        predicao = modelo.predict(dado)
        return 'Doente' if predicao[0] == 1 else 'Não Doente'
    elif tipo == 'regressao':
        predicao = modelo.predict(dado)
        return f'Probabilidade: {predicao[0]:.2f}'

# Configurando o Streamlit
st.title('Análise e Predição de Doenças Cardíacas')
st.sidebar.title('Menu')
menu = st.sidebar.radio("Selecione uma opção", ['Análises', 'Predição'])

if menu == 'Análises':
    st.header('Análises de Dados')
    st.write("### Distribuição dos Dados")
    st.bar_chart(df['target'].value_counts())

    st.write("### Correlação entre as Variáveis")
    st.write(df.corr())

else:
    st.header('Predição de Doenças Cardíacas')
    st.write("### Insira os Dados do Paciente")

    # Entrada de dados do usuário
    idade = st.slider('Idade', 20, 80, 50)
    sexo = st.selectbox('Sexo (0 = Feminino, 1 = Masculino)', [0, 1])
    dorpeito = st.slider('Dor no Peito (0-3)', 0, 3, 1)
    pressão = st.slider('Pressão Arterial em Repouso', 80, 200, 120)
    coleste = st.slider('Colesterol', 100, 400, 200)
    acucars = st.selectbox('Açúcar no Sangue > 120 mg/dl (1 = Sim, 0 = Não)', [0, 1])
    ecg = st.slider('Eletrocardiograma (0-2)', 0, 2, 1)
    freq = st.slider('Frequência Cardíaca Máxima', 60, 220, 150)
    angina = st.selectbox('Angina Induzida por Exercício (1 = Sim, 0 = Não)', [0, 1])
    depressão = st.slider('Depressão Induzida por Exercício', 0.0, 6.0, 1.0, step=0.1)
    inclinação = st.slider('Inclinação do Segmento ST (0-2)', 0, 2, 1)
    vasos = st.slider('Número de Vasos Principais (0-4)', 0, 4, 0)
    thal = st.slider('Thalassemia (1-3)', 1, 3, 2)

    novo_dado = [
        idade, sexo, dorpeito, pressão, coleste, acucars, ecg, freq, angina, depressão, inclinação, vasos, thal
    ]

    # Botões para previsão
    if st.button('Prever com Random Forest'):
        resultado = prever_dado(novo_dado, rf_model)
        st.success(f'Resultado da Previsão: {resultado}')

    if st.button('Prever com Regressão Logística'):
        resultado = prever_dado(novo_dado, logistic_model)
        st.success(f'Resultado da Previsão: {resultado}')

    if st.button('Prever com Regressão Linear'):
        resultado = prever_dado(novo_dado, linear_model, tipo='regressao')
        st.success(f'Resultado da Previsão: {resultado}')
