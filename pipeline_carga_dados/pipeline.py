#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from io import BytesIO
from google.cloud import bigquery
from google.oauth2 import service_account
import warnings
warnings.filterwarnings('ignore')


# In[2]:


url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"


# In[3]:


df = pd.read_html(url, decimal=',', thousands='.')[2]


# In[4]:


df


# In[5]:


df.columns = df.iloc[0]
df = df[1:]


# In[6]:


df.rename(columns={'PreÃ§o - petrÃ³leo bruto - Brent (FOB)': 'preco_petroleo_bruto'}, inplace=True)
df['preco_petroleo_bruto'] = df['preco_petroleo_bruto'].astype(float)


# In[7]:


# Salvar o DataFrame como arquivo Parquet
arquivo_parquet = 'petr_price.parquet'
df.to_parquet(arquivo_parquet, index=False)


# In[8]:


# Autenticação para o BigQuery usando arquivo de credenciais
pk_json_input = './pos-tech-403001-813583759ec1.json'


# In[9]:


projeto_bigquery = 'pos-tech-403001'
dataset_bigquery = 'tech_challenge'
tabela_bigquery = 'raw_petr_brent'


# In[10]:


# Autenticação para o BigQuery usando arquivo de credenciais
credentials = service_account.Credentials.from_service_account_file(pk_json_input)
client = bigquery.Client(credentials=credentials, project=projeto_bigquery)


# In[11]:


table_id = f'{projeto_bigquery}.{dataset_bigquery}.{tabela_bigquery}'
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.PARQUET,
    write_disposition='WRITE_TRUNCATE'
)


# In[14]:


with open(arquivo_parquet, "rb") as source_file:
    job = client.load_table_from_file(source_file, table_id, job_config=job_config)


# In[15]:


job.result()


# In[17]:


print(f"Dados carregados para {table_id} no BigQuery.")

