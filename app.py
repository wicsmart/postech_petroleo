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



def predict(model):

    futuro = 30

    fut = model.make_future_dataframe(periods=futuro, include_history=False, freq='D')
    
    forecast = model.predict(fut)

    return forecast


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

# Carrega o dado tratado

df = pd.read_parquet('shared/refined_data.parquet')

df.rename(columns={'ds': 'Data', 'y': 'Preço'}, inplace=True)


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
        model = load_model('shared/serialized_model.json')
        # model = joblib.load('shared/model_prophet.joblib')
        final_pred = model.predict(out)
        print(final_pred)
        st.write('O preço previsto para a data selecionada é:', final_pred['yhat'].values[0].round(2))


with aba3:
    st.write("### Queda 2014")
    st.write('Os principais apontados como "culpados" pela queda dos preços são o aumento de produção, em especial nas áreas de xisto dos EUA, e uma demanda menor que a esperada na Europa e na Ásia.')