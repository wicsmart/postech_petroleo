# Analise de Dados da base IPEA - Pós Tech Fase 4

## Índice

- [Introdução](#Introdução)
- [Ambiente](#Ambiente)
- [Divisão do projeto no Github](#Divisão-do-projeto-no-Github)
- [Como acessar a aplicação?](#Como-acessar-a-aplicação?)

## Introdução

Nessa fase 4 da Pós Tech fomos desafiados a criar um modelo de machine learning que faça a previsão dos preços do pretroleo, obtendo os dados do site do IPEA. Além de prever os preços foi necessário criar uma aplicação aonde os usuários pudessem selecionar para ver a previsão de preços com atualização diária.

## Ambiente

Para implementar a solução com base no que foi proposto no projeto, foi pensando em usar um pipeline de dados que faz o webscrapping no site do IPEA, obtendo os preços, e em cima dessa base a criação de uma modelo de machine learning com uma aplicação no Streamlit para que o usuário possa selecionar uma data para ter a previsão do modelo, e uma analise histórico de como ocorreu a variação dos preços no decorrer dos anos.

O Pilar do projeto é executado por baixo do panos é o Docker Compose, que ao executar o comando "docker-compose up" para iniciar os conteiners.

# ETL
Para esse projeto temos como base um conteiner que através de uma imagem prepara todo ambiente para executar o ETL. Esse ETL é o responsável por fazer o webscrapping do site, realizar a limpeza dos dados, e após isso preparar os dados limpos para realizar o treino do modelo, para que os dados possam ser consumidos na aplicação do Streamlit.

# APP
O segundo container é o do APP que é responsável por ler os dados do arquivo **refined_data** e carregar as analises para alimentar a aplicação do Streamlit.

Exemplo de como a arquitetura ficou estruturada nesse projeto:

![arquitetura](https://github.com/wicsmart/postech_petroleo/assets/82483612/992e6a7b-a38e-4ea0-bbc3-535a8760341a)

Através da imagem conseguimos identificar como o Pipeline foi pensando para ese projeto e como ele é alimentado e atualizado.


## Divisão do projeto no Github

Na pasta **pipeline_carga_dados** é onde contém o arquivo Python que será o responsável por fazer o ETL e gerar o modelo com os dados já tratados, utilizamos a biblioteca Prophet para treinar e refinar o modelo. Dentro dessa pasta também contém o a imagem dockerfile com as bibliotecas usadas no conteiner ETL mencionado anteriormente e um arquivo cron com o disparo do script python as 12h diariamente.

A pasta **shared** contém dois arquivos no formato parquet e o arquivo que contém o modelo que será lido e exceutado posteriormente no Streamlit.

Na pasta raíz do Github, temos a aplicação, o arquivo com a configuração do Docker Compose, mais uma imagem dockefile que contém as bilbiotecas usadas para o APP, que serão usadas pelo Streamlit é dois arquivos Jupyter Notebook com algumas analises exploratórias de exemplo de como foi válidado alguns pontos referente ao modelo Prophet.

## Como acessar a aplicação?

A aplicação pode ser acessada através do link https://postechpetroleo-wgp3rymxhr53gckghd2nu5.streamlit.app 

Na aplicação do Streamlit temos 3 abas sendo dividas:

Aba 1 é a **Análise dos preços** com o histórico atual de 1987 até os dias atuais.

Aba 2 é a **Previsão de Preços** aonde podemos pesquisar uma data futura para o modelo exibir uma previsão de qual será o preço naquela data.

Aba 3 é a **Highlights** onde podemos encontrar uma análise histórica com diversos fatores geográficos, e economicos de como o preço do Petroléo variou no decorrer dos anos.