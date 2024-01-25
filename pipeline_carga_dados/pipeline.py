#!/usr/bin/env python0
# coding: utf-8

from datetime import datetime
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from prophet import Prophet
import numpy as np
from prophet.serialize import model_to_json

import warnings
warnings.filterwarnings('ignore')


__LAST_DAY = None

def extrac(url):

    df = pd.read_html(url, decimal=',', thousands='.')[2]

    df.columns = df.iloc[0]
    
    df = df[1:]

    df.rename(columns={'PreÃ§o - petrÃ³leo bruto - Brent (FOB)': 'preco_petroleo_bruto'}, inplace=True)
    
    df['preco_petroleo_bruto'] = df['preco_petroleo_bruto'].astype(float)

    return df

def save_data_raw(df):
    # Salvar o DataFrame como arquivo Parquet
    try:
        arquivo_parquet = 'petr_price.parquet'
    
        df.to_parquet(arquivo_parquet, index=False)
    
    except Exception as err:
    
        print(str(err))
        return False
    
    return True

def load_to_bigquery(arquivo_parquet):
    # Autenticação para o BigQuery usando arquivo de credenciais
    pk_json_input = './chave.json'

    projeto_bigquery = 'pos-tech-403001'
    dataset_bigquery = 'tech_challenge'
    tabela_bigquery = 'raw_petr_brent'

    # Autenticação para o BigQuery usando arquivo de credenciais
    credentials = service_account.Credentials.from_service_account_file(pk_json_input)
    client = bigquery.Client(credentials=credentials, project=projeto_bigquery)

    table_id = f'{projeto_bigquery}.{dataset_bigquery}.{tabela_bigquery}'

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition='WRITE_TRUNCATE'
    )

    with open(arquivo_parquet, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_id, job_config=job_config)

    job.result()

    print(f"Dados carregados para {table_id} no BigQuery.")


def transform(df):
    # transformando a coluna com as datas para Datetime, e ordernando essa coluna
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
    
    df = df.sort_values(by='Data', ascending=True)
    
    df.reset_index(inplace=True, drop=True)

    df_1 = df[['Data', 'preco_petroleo_bruto']]

    df_1.columns = ['ds','y']

    df_1.tail()

    df_1 = df_1.set_index('ds').asfreq('d')

    df_1 = df_1.fillna(method='ffill')

    df_1.reset_index(inplace=True)

    df_1['unique_id'] = 'petro'

    return df_1


def train_split_data(dff, start_train):

    train =  dff.loc[(dff['ds'] >= start_train)]
    
    last_date = dff['ds'].iloc[-1]
    
    return train, last_date

def wmape(y_true,y_pred):
    y_true = y_true.values
    y_pred = y_pred.values
    return np.mean(np.abs((y_true - y_pred) / y_true))

def save_model(model):

    with open('serialized_model.json', 'w') as fout:

        fout.write(model_to_json(model))  # Save model


def predict(model, data):

    date2 = datetime.strptime(data, '%Y-%m-%d')

    futuro = (date2 - __LAST_DAY).days

    print(futuro)
    
    fut = model.make_future_dataframe(periods=futuro, include_history=False, freq='D')
    
    forecast = model.predict(fut)

    return forecast




if __name__ == "__main__":
    
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

    df = extrac(url=url)

    df_refined = transform(df)

    df_train, __LAST_DAY = train_split_data(dff=df_refined, start_train= "2018-01-01")

    model = Prophet(interval_width=0.95)

    model.fit(df_train)

    try:
    
        save_model(model=model)

        print("Modelo salvo com sucesso")

    except Exception as err:
         
        print(str(err))