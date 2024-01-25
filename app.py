import streamlit as st
import pandas as pd
import plotly.express as px
from google.cloud import bigquery
from prophet import Prophet
import numpy as np
from google.oauth2 import service_account
import datetime
from prophet.serialize import model_from_json


__LAST_DAY = "2024-01-16"

def predict(model, data):

    date1 = datetime.strptime(__LAST_DAY, '%Y-%m-%d')

    date2 = datetime.strptime(data, '%Y-%m-%d')

    futuro = (date2 - date1).days

    print(futuro)
    
    fut = model.make_future_dataframe(periods=futuro, include_history=False, freq='D')
    
    forecast = model.predict(fut)

    return forecast

def load_model(path):

    with open('serialized_model.json', 'r') as fin:

        m = model_from_json(fin.read())  # Load model

    return m

def get_prices():
    
    modelo_carregado = load_model("serialized_model.json")

    predito = predict(model=modelo_carregado, data="2024-01-25")

    result = predict.query('ds == "2024-01-28"')

    result[['ds','yhat']]


# Título ----------------------------------------------------------
st.title('Preço por barril do petróleo bruto Brent (FOB) :chart:')  


# Dataframe ----------------------------------------------
# Autenticação para o BigQuery usando arquivo de credenciais
projeto_id = 'pos-tech-403001'
dataset_id = 'tech_challenge'
tabela_id = 'raw_petr_brent'


credentials = service_account.Credentials.from_service_account_file('./chave.json')
client = bigquery.Client(credentials=credentials, project=projeto_id)


# Consulta SQL para selecionar todos os dados da tabela
consulta_sql = f'SELECT * FROM `{projeto_id}.{dataset_id}.{tabela_id}`'
resultado = client.query(consulta_sql)
# Converte o resultado em um DataFrame do Pandas
df = resultado.to_dataframe()
df.columns = ['Data','Preço']
df['Data'] = pd.to_datetime(df['Data'],format='%d/%m/%Y')
df = df.sort_values(by='Data', ascending=True)
df.reset_index(inplace=True, drop=True)

df = df.query('Data >= "2000-01-01"')

# Visual --------------------------------------------------------
aba1, aba2, aba3 = st.tabs(['Análise dos Preços', 'Previsão de Preços', 'Highlights'])

with aba1:
    # KPI -----------
    coluna1,coluna2,coluna3 = st.columns(3)
    with coluna1:
        st.metric('Preço Mínimo', df['Preço'].min())
    with coluna2:
        st.metric('Preço Médio', df['Preço'].mean().round(2))
    with coluna3:
        st.metric('Preço Máximo', df['Preço'].max())

    # Gráficos ----------
    fig = px.line(df, x = 'Data', y='Preço',
                title='Série Histórica Preço Petróleo bruto')
    st.plotly_chart(fig)


    df_mes = df.groupby(df['Data'].dt.year)['Preço'].mean().round(2).reset_index()
    df_mes.columns= ['Ano','Preço']
    df_mes['Ano'] = df_mes['Ano'].astype('str')
    fig_bar = px.bar(df_mes, x = 'Ano', y='Preço',
                    text_auto=True,
                    title='Preço Médio por Ano')
    st.plotly_chart(fig_bar)

    # Dataframe------
    st.dataframe(df)

with aba2:
    # KPI -----------
    coluna1,coluna2,coluna3 = st.columns(3)
    with coluna1:
        st.metric('Preço Mínimo', df['Preço'].min())
    with coluna2:
        st.metric('Preço Médio', df['Preço'].mean().round(2))
    with coluna3:
        st.metric('Preço Máximo', df['Preço'].max())

    # Previsão
    st.write("### Escolha uma data para ver o preço previsto:")
    d = st.date_input("Data", value=None,format='DD/MM/YYYY')
    st.write('Data:', d)
    #print(type(d))

    if st.button('Enviar'):
        model = joblib.load('model_prophet.joblib')
        final_pred = model.predict(pd.to_datetime(d,format='%d/%m/%Y'))
        st.write('Preço:', final_pred) 



with aba3:
    st.write("### Queda 2014")
    st.write('Os principais apontados como "culpados" pela queda dos preços são o aumento de produção, em especial nas áreas de xisto dos EUA, e uma demanda menor que a esperada na Europa e na Ásia.')