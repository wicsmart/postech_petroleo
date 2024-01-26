import streamlit as st
import pandas as pd
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account
import datetime
import joblib


# Título ----------------------------------------------------------
st.title('Preço por barril do petróleo bruto Brent (FOB) :chart:')  


# Dataframe ----------------------------------------------
# Autenticação para o BigQuery usando arquivo de credenciais
projeto_id = 'pos-tech-403001'
dataset_id = 'tech_challenge'
tabela_id = 'raw_petr_brent'
credentials = service_account.Credentials.from_service_account_file('./pos-tech-403001-25c18098d334.json')
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