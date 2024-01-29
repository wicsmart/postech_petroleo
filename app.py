import streamlit as st
import pandas as pd
import plotly.express as px
from google.cloud import bigquery
from prophet import Prophet
import numpy as np
from google.oauth2 import service_account
import datetime
import joblib
import time
from prophet.serialize import model_from_json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def read_last_day(filepath):

    with open(filepath, 'r') as file:
        # Read the first line
        first_line = file.readline()
    
    return first_line


def predict(model, data):

    last_day = read_last_day("/shared/lastday.txt")
    try:
      

        date_time_object = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
    
        timestamp = date_time_object.timestamp()

        # date1 = datetime.strptime(last_day, '%Y-%m-%d')

        # date2 = datetime.strptime(data, '%Y-%m-%d')

        futuro = (data - timestamp).days

    except:
        futuro = 1
    
    print(futuro)
    
    fut = model.make_future_dataframe(periods=futuro, include_history=False, freq='D')
    
    forecast = model.predict(fut)

    result = forecast.query('ds == "2024-01-28"')

    return result[['ds','yhat']]


def load_model(path):

    with open(path, 'r') as fin:

        m = model_from_json(fin.read())  # Load model

    return m

def get_price():
    
    modelo_carregado = load_model("serialized_model.json")

    predito = predict(model=modelo_carregado, data="2024-01-25")

    return predito

# Funções -----
@st.cache_data
def converte_csv(df):
    return df.to_csv(index=False).encode('latin1')

def mensagem_sucesso():
    sucesso = st.success("Download concluído", icon="✅")
    time.sleep(5)
    sucesso.empty()

# Título ----------------------------------------------------------
st.title('Preço por barril do petróleo bruto Brent (FOB) :chart:')  

# # Consulta SQL para selecionar todos os dados da tabela
# consulta_sql = f'SELECT * FROM `{projeto_id}.{dataset_id}.{tabela_id}`'
# resultado = client.query(consulta_sql)

# Dataframe ----------------------------------------------
# Autenticação para o BigQuery usando arquivo de credenciais
# projeto_id = 'pos-tech-403001'
# dataset_id = 'tech_challenge'
# tabela_id = 'raw_petr_brent'
# credentials = service_account.Credentials.from_service_account_file('./pos-tech-403001-25c18098d334.json')
# client = bigquery.Client(credentials=credentials, project=projeto_id)
# # Consulta SQL para selecionar todos os dados da tabela
# consulta_sql = f'SELECT * FROM `{projeto_id}.{dataset_id}.{tabela_id}`'
# resultado = client.query(consulta_sql)


# Converte o resultado em um DataFrame do Pandas


df = pd.read_parquet('/shared/refined_data.parquet')

print(df.head())

df.rename(columns={'ds': 'Data', 'y': 'Preço'}, inplace=True)

# df['Data'] = pd.to_datetime(df['Data'],format='%d/%m/%Y')
# df = df.sort_values(by='Data', ascending=True)
# df.reset_index(inplace=True, drop=True)

# df = df.query('Data >= "2000-01-01"')

# Filtro -------------------------------------------------------
with st.sidebar:
    MIN_MAX_RANGE = (datetime.datetime(1987,5,20), datetime.datetime(2024,7,1))
    selected_min, selected_max = st.slider(
        "Período",
        value=(MIN_MAX_RANGE[0],MIN_MAX_RANGE[1]),
        min_value=MIN_MAX_RANGE[0],
        max_value=MIN_MAX_RANGE[1])
df_filter = df.query("Data >= @selected_min \
                       and Data < @selected_max")
# Visual --------------------------------------------------------


aba1, aba2, aba3 = st.tabs(['Análise dos Preços', 'Previsão de Preços', 'Highlights'])

with aba1:
    # KPI -----------
    coluna1,coluna2,coluna3 = st.columns(3)
    with coluna1:
        st.metric('Preço Mínimo', df_filter['Preço'].min())
    with coluna2:
        st.metric('Preço Médio', df_filter['Preço'].mean().round(2))
    with coluna3:
        st.metric('Preço Máximo', df_filter['Preço'].max())

    # Gráficos ----------
    fig = px.line(df_filter, x = 'Data', y='Preço',
                title='Série Histórica Preço Petróleo bruto')
    st.plotly_chart(fig)


    df_mes = df_filter.groupby(df['Data'].dt.year)['Preço'].mean().round(2).reset_index()
    df_mes.columns= ['Ano','Preço']
    df_mes['Ano'] = df_mes['Ano'].astype('str')
    fig_bar = px.bar(df_mes, x = 'Ano', y='Preço',
                    text_auto=True,
                    title='Preço Médio por Ano')
    st.plotly_chart(fig_bar)

    # Dataframe------
    st.dataframe(df_filter,hide_index=True)
    st.download_button('Exportar csv', data=converte_csv(df_filter),file_name='file.csv', mime='text/csv', on_click=mensagem_sucesso)
    

with aba2:
    # KPI -----------
    coluna1,coluna2,coluna3 = st.columns(3)
    with coluna1:
        st.metric('Preço Mínimo', df_filter['Preço'].min())
    with coluna2:
        st.metric('Preço Médio', df_filter['Preço'].mean().round(2))
    with coluna3:
        st.metric('Preço Máximo', df_filter['Preço'].max())

    # Previsão
    st.write("### Escolha uma data para ver o preço previsto:")
    d = st.date_input("Data", value=None,format='DD/MM/YYYY')
    out = pd.DataFrame([d], columns = ['ds'])
    out['ds'] = pd.to_datetime(out['ds'])

    
    if st.button('Enviar'):

        model = joblib.load('model_prophet.joblib')
        final_pred = model.predict(out)
        print(final_pred)
        st.write('O preço previsto para a data selecionada é:', final_pred['yhat'].values[0].round(2))


with aba3:
    st.write("### Queda 2014")
    st.write('Os principais apontados como "culpados" pela queda dos preços são o aumento de produção, em especial nas áreas de xisto dos EUA, e uma demanda menor que a esperada na Europa e na Ásia.')