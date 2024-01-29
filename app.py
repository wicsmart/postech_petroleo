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
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def predict(model):

    futuro = 30

    fut = model.make_future_dataframe(
        periods=futuro, include_history=False, freq='D')

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
    MIN_MAX_RANGE = (datetime.datetime(1987, 5, 20),
                     datetime.datetime(2024, 7, 1))
    selected_min, selected_max = st.slider(
        "Período",
        value=(MIN_MAX_RANGE[0], MIN_MAX_RANGE[1]),
        min_value=MIN_MAX_RANGE[0],
        max_value=MIN_MAX_RANGE[1])
df_filter = df.query("Data >= @selected_min \
                       and Data < @selected_max")
# Visual --------------------------------------------------------


aba1, aba2, aba3 = st.tabs(
    ['Análise dos Preços', 'Previsão de Preços', 'Highlights'])

with aba1:
    # KPI -----------
    coluna1, coluna2, coluna3 = st.columns(3)
    with coluna1:
        st.metric('Preço Mínimo', df_filter['Preço'].min())
    with coluna2:
        st.metric('Preço Médio', df_filter['Preço'].mean().round(2))
    with coluna3:
        st.metric('Preço Máximo', df_filter['Preço'].max())

    # Gráficos ----------
    fig = px.line(df_filter, x='Data', y='Preço',
                  title='Série Histórica Preço Petróleo bruto')
    st.plotly_chart(fig)

    df_mes = df_filter.groupby(df['Data'].dt.year)[
        'Preço'].mean().round(2).reset_index()
    df_mes.columns = ['Ano', 'Preço']
    df_mes['Ano'] = df_mes['Ano'].astype('str')
    fig_bar = px.bar(df_mes, x='Ano', y='Preço',
                     text_auto=True,
                     title='Preço Médio por Ano')
    st.plotly_chart(fig_bar)

    # Dataframe------
    st.dataframe(df_filter, hide_index=True)
    st.download_button('Exportar csv', data=converte_csv(
        df_filter), file_name='file.csv', mime='text/csv', on_click=mensagem_sucesso)


with aba2:
    # KPI -----------
    coluna1, coluna2, coluna3 = st.columns(3)
    with coluna1:
        st.metric('Preço Mínimo', df_filter['Preço'].min())
    with coluna2:
        st.metric('Preço Médio', df_filter['Preço'].mean().round(2))
    with coluna3:
        st.metric('Preço Máximo', df_filter['Preço'].max())

    # Previsão
    st.write("### Escolha uma data para ver o preço previsto:")
    d = st.date_input("Data", value=None, format='DD/MM/YYYY')
    out = pd.DataFrame([d], columns=['ds'])
    out['ds'] = pd.to_datetime(out['ds'])

    if st.button('Enviar'):
        model = load_model('shared/serialized_model.json')
        # model = joblib.load('shared/model_prophet.joblib')
        final_pred = model.predict(out)
        print(final_pred)
        st.write('O preço previsto para a data selecionada é:',
                 final_pred['yhat'].values[0].round(2))


with aba3:
    # Foi agrupado os dados por ano, calculado a média dos preços e a variação percentual entre os anos.
    # A variação percentual é atribuída à coluna 'percentual', com o primeiro valor definido como 0 para evitar problemas de cálculo
    df_agrupado = df.groupby(df['Data'].dt.year)[
        'Preço'].mean().round(2).reset_index()
    df_agrupado['percentual'] = (
        df_agrupado['Preço'].pct_change() * 100).round(2)
    df_agrupado.loc[df_agrupado.index[0], 'percentual'] = 0

    # Título do aplicativo
    st.subheader("Introdução")

    st.caption("Tornando-se uma das fontes de energia mais cruciais globalmente, o petróleo ainda desempenha um papel vital na base de diversas economias. "
               "Atualmente, flutuações nos preços do barril no mercado global têm a capacidade de desencadear crises econômicas significativas. Assim como, fatores externos têm o potencial de impactar os valores do barril."
               "Desde os grandes investidores até os consumidores comuns na cadeia econômica, todos se tornam suscetíveis às flutuações do \"diamante negro\".")

    #  gráfico de linha
    fig = px.line(df_agrupado, x='Data', y='Preço',
                  title='Preço Petróleo bruto X Variação percentual')
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço')

    # Adicionar a série temporal de percentuais como gráfico de barras

    def get_marker_color(val):
        return 'red' if val < 0 else 'green'

    fig.add_trace(go.Bar(x=df_agrupado['Data'], y=df_agrupado['percentual'],
                         marker=dict(color=[get_marker_color(val)
                                            for val in df_agrupado['percentual']]),
                         name='Percentuais Positivos'))

    # Adicionar grades verticais
    for date in df_agrupado['Data']:
        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='gray', width=1)
        ))

    # Ajustar para exibir todos os anos no eixo x
    fig.update_xaxes(tickmode='linear',
                     tick0=df_agrupado['Data'].min(), dtick='M1')
    fig.update_xaxes(tickangle=100, tickmode='array',
                     tickvals=df_agrupado['Data'])

    #  altura
    fig.update_layout(height=600)
    # largura
    fig.update_layout(width=1000)
    # legendas
    fig.update_traces(name='Preço', showlegend=True, selector=dict(
        type='scatter'))  # Legenda para o preço

    # Adicionar legenda específica para os percentuais negativo, o positivo estou ousando o default
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(
        color='red'), name='Percentuais Negativos'))  # Legenda para percentuais negativos

    # Exibir o gráfico usando st.plotly_chart se colocar print simples o streamlit plota o gráfico em uma outra página
    st.plotly_chart(fig)

    st.subheader("História")

    st.caption("A nossa análise se inicia em em 1987 e a primeira flutuação mais drástica dos preços que conseguimos identificar foi em 1991")

    # Filtrar os dados para o intervalo
    df_intervalo = df_agrupado.query('Data >= 1990 and Data <= 1995')

    # Criar o gráfico de linha
    fig = px.line(df_intervalo, x='Data', y='Preço',
                  title='Preço Petróleo bruto X Variação percentual')
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço')

    # Adicionar a série temporal de percentuais como gráfico de barras
    fig.add_trace(go.Bar(x=df_intervalo['Data'], y=df_intervalo['percentual'],
                         marker=dict(color=[get_marker_color(val)
                                            for val in df_agrupado['percentual']]),
                         name='Percentuais Positivos'))

    # Adicionar grades verticais
    for date in df_intervalo['Data']:
        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='gray', width=1)
        ))

    # Ajustar para exibir todos os anos no eixo x
    fig.update_xaxes(tickmode='linear',
                     tick0=df_intervalo['Data'].min(), dtick=1)
    fig.update_xaxes(tickangle=45, tickmode='array',
                     tickvals=df_intervalo['Data'])

    # Adicionar legendas
    fig.update_traces(name='Preço', showlegend=True, selector=dict(
        type='scatter'))  # Legenda para o preço

    # Adicionar legenda específica para os percentuais
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(
        color='red'), name='Percentuais Negativos'))  # Legenda para percentuais negativos

    # Exibir o gráfico usando st.plotly_chart
    st.plotly_chart(fig)

    st.subheader("Guerra do Golfo")

    st.caption("""
    A Guerra do Golfo, que ocorreu entre agosto de 1990 e fevereiro de 1991, teve um impacto significativo nos preços do petróleo devido a vários fatores:

    1. **Invasão do Kuwait:** O conflito começou quando o Iraque, liderado por Saddam Hussein, invadiu o Kuwait em agosto de 1990. Isso levou a uma resposta internacional liderada pelos Estados Unidos.

    2. **Interrupção da produção:** A invasão resultou em uma interrupção significativa na produção de petróleo no Kuwait, um importante produtor na região. Isso reduziu a oferta global de petróleo, levando a preocupações sobre uma possível escassez.

    3. **Temor de expansão do conflito:** A comunidade internacional temia que o conflito se espalhasse para outros países produtores de petróleo na região do Golfo Pérsico, como a Arábia Saudita e os Emirados Árabes Unidos. Esses temores aumentaram a incerteza no mercado de petróleo.

    4. **Restrições à produção iraquiana:** Durante a guerra, a coalizão liderada pelos Estados Unidos impôs sanções ao Iraque, incluindo restrições severas à sua produção e exportação de petróleo. Isso contribuiu para uma redução adicional na oferta global.

    5. **Aumento da demanda por segurança energética:** A instabilidade na região do Golfo Pérsico aumentou a percepção de risco para o fornecimento global de petróleo. Isso levou muitos países a buscar medidas para garantir sua segurança energética, como estoques estratégicos, o que aumentou a demanda por petróleo no curto prazo.

    Esses fatores combinados resultaram em uma diminuição da oferta e em preocupações sobre a estabilidade do fornecimento global de petróleo, levando a um aumento nos preços do petróleo durante e após a Guerra do Golfo. O impacto do conflito na região continuou a influenciar os mercados de petróleo nos anos subsequentes.
    """)

    st.subheader("Crise Imobiliária")

    st.caption("A segunda flutuação mais drástica dos preços foi em 2008:")

    # Filtrar os dados para o intervalo
    df_intervalo = df_agrupado.query('Data >= 2007 and Data <= 2011')

    # Criar o gráfico de linha
    fig = px.line(df_intervalo, x='Data', y='Preço',
                  title='Preço Petróleo bruto X Variação percentual')
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço')

    # Adicionar a série temporal de percentuais como gráfico de barras
    fig.add_trace(go.Bar(x=df_intervalo['Data'], y=df_intervalo['percentual'],
                         marker=dict(color=[get_marker_color(val)
                                            for val in df_agrupado['percentual']]),
                         name='Percentuais Positivos'))

    # Adicionar grades verticais
    for date in df_intervalo['Data']:
        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='gray', width=1)
        ))

    # Ajustar para exibir todos os anos no eixo x
    fig.update_xaxes(tickmode='linear',
                     tick0=df_intervalo['Data'].min(), dtick=1)
    fig.update_xaxes(tickangle=45, tickmode='array',
                     tickvals=df_intervalo['Data'])

    # Adicionar legendas
    fig.update_traces(name='Preço', showlegend=True, selector=dict(
        type='scatter'))  # Legenda para o preço

    # Adicionar legenda específica para os percentuais
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(
        color='red'), name='Percentuais Negativos'))  # Legenda para percentuais negativos

    # Exibir o gráfico usando st.plotly_chart
    st.plotly_chart(fig)

    st.caption("""

    1. **Colapso do Mercado Imobiliário:** A crise foi desencadeada por um colapso no mercado imobiliário dos EUA, marcado por hipotecas de alto risco (subprime) e uma bolha imobiliária que estourou.

    2. **Crise Financeira:** O colapso teve repercussões financeiras globais, levando a falências bancárias e uma crise sistêmica.

    **Impactos nos Preços do Petróleo:**

    1. **Redução da Demanda:** A crise resultou em uma recessão global, reduzindo a atividade econômica e a demanda por petróleo.

    2. **Queda nos Preços:** A diminuição da demanda contribuiu para a queda nos preços do petróleo, já que a oferta superava a demanda.

    **Preço do Petróleo e Impacto na Crise Imobiliária:**

    1. **Custo dos Insumos:** Os preços mais altos do petróleo antes da crise aumentaram os custos de construção e transporte, contribuindo para a pressão nos custos imobiliários.

    2. **Impacto na Economia Geral:** A alta nos preços do petróleo aumentou os custos de vida e contribuiu para a pressão inflacionária, afetando a capacidade de pagamento das hipotecas.

    **Variação de Preços:**

    1. **Círculo Vicioso:** A crise imobiliária e a recessão reduziram a demanda por petróleo, levando a uma queda nos preços. Por sua vez, a queda nos preços do petróleo afetou negativamente as economias dependentes do setor, contribuindo para a persistência da crise.

    A crise imobiliária e os preços do petróleo estavam interligados, com a recessão global afetando a demanda por petróleo e vice-versa. Esses eventos desencadearam um ciclo de retroalimentação negativa, contribuindo para a magnitude da crise econômica de 2008.
    """)

    st.subheader("Impactos nos Preços do Petróleo (2011-2017)")

    # Filtrar os dados para o intervalo
    df_intervalo = df_agrupado.query('Data >= 2007 and Data <= 2017')

    # Criar o gráfico de linha
    fig = px.line(df_intervalo, x='Data', y='Preço',
                  title='Preço Petróleo bruto X Variação percentual')
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço')

    # Adicionar a série temporal de percentuais como gráfico de barras
    fig.add_trace(go.Bar(x=df_intervalo['Data'], y=df_intervalo['percentual'],
                         marker=dict(color=[get_marker_color(val)
                                            for val in df_agrupado['percentual']]),
                         name='Percentuais Positivos'))

    # Adicionar grades verticais
    for date in df_intervalo['Data']:
        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='gray', width=1)
        ))

    # Ajustar para exibir todos os anos no eixo x
    fig.update_xaxes(tickmode='linear',
                     tick0=df_intervalo['Data'].min(), dtick=1)
    fig.update_xaxes(tickangle=45, tickmode='array',
                     tickvals=df_intervalo['Data'])

    # Adicionar legendas
    fig.update_traces(name='Preço', showlegend=True, selector=dict(
        type='scatter'))  # Legenda para o preço

    # Adicionar legenda específica para os percentuais
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(
        color='red'), name='Percentuais Negativos'))  # Legenda para percentuais negativos

    # Exibir o gráfico usando st.plotly_chart
    st.plotly_chart(fig)

    st.caption(
        """
            O período de 2011 a 2017 foi marcado por diversas mudanças e eventos que impactaram os preços do petróleo. 
            Alguns dos fatores mais significativos incluem:

            **Desaceleração econômica global:**
            Como mencionamos no tópico superior, crise financeira global de 2008-2009 teve repercussões que se estenderam até 2011. A desaceleração econômica global 
            afetou a demanda por petróleo, resultando em uma pressão de baixa nos preços.

            **Produção de petróleo de xisto nos EUA:**
            Durante esse período, houve um aumento significativo na produção de petróleo de xisto nos Estados Unidos. 
            Isso foi impulsionado por avanços tecnológicos, como a perfuração horizontal e a fratura hidráulica (fracking). 
            O aumento na produção dos EUA contribuiu para um aumento na oferta global de petróleo.

            **Instabilidade geopolítica:**
            Vários eventos geopolíticos contribuíram para a volatilidade nos preços do petróleo. Isso incluiu conflitos no Oriente 
            Médio, como a Primavera Árabe, tensões no Golfo Pérsico e eventos relacionados à Rússia, que afetaram a oferta e a percepção 
            de risco no mercado de petróleo.

            **Decisões da OPEP (Organização dos Países Exportadores de Petróleo):**
            A OPEP desempenhou um papel importante na gestão da produção global de petróleo. Decisões tomadas pelos países membros da 
            OPEP, como cortes ou aumentos na produção, tiveram impacto direto nos preços do petróleo.

            **Dólar americano e política monetária:**
            A relação inversa entre o dólar americano e os preços das commodities, incluindo o petróleo, também influenciou os preços. 
            Mudanças nas políticas monetárias e econômicas nos EUA afetaram a taxa de câmbio e, consequentemente, os preços do petróleo 
            denominados em dólares.

            **Inovações tecnológicas e eficiência energética:**
            Avanços em eficiência energética e a crescente ênfase em fontes de energia alternativas contribuíram para uma mudança nas 
            perspectivas de demanda futura por petróleo, impactando os preços.

            **Acordo nuclear com o Irã:**
            O acordo nuclear com o Irã em 2015 levou à suspensão de sanções econômicas, permitindo que o país aumentasse sua produção 
            de petróleo. Isso também influenciou a oferta global e os preços do petróleo.

            Esses são apenas alguns dos fatores que contribuíram para a variabilidade nos preços do petróleo durante o período de 2011 a 2017. 
            É importante notar que os mercados de commodities, incluindo o petróleo, são altamente complexos e estão sujeitos a uma série de 
            influências econômicas, geopolíticas e tecnológicas.
            """
    )
    st.subheader("Impactos nos Preços do Petróleo (2019-2021)")

    # Filtrar os dados para o intervalo
    df_intervalo = df_agrupado.query('Data >= 2018 and Data <= 2021')

    # Criar o gráfico de linha
    fig = px.line(df_intervalo, x='Data', y='Preço',
                  title='Preço Petróleo bruto X Variação percentual')
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço')

    # Adicionar a série temporal de percentuais como gráfico de barras
    fig.add_trace(go.Bar(x=df_intervalo['Data'], y=df_intervalo['percentual'],
                         marker=dict(color=[get_marker_color(val)
                                            for val in df_agrupado['percentual']]),
                         name='Percentuais Positivos'))

    # Adicionar grades verticais
    for date in df_intervalo['Data']:
        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='gray', width=1)
        ))

    # Ajustar para exibir todos os anos no eixo x
    fig.update_xaxes(tickmode='linear',
                     tick0=df_intervalo['Data'].min(), dtick=1)
    fig.update_xaxes(tickangle=45, tickmode='array',
                     tickvals=df_intervalo['Data'])

    # Adicionar legendas
    fig.update_traces(name='Preço', showlegend=True, selector=dict(
        type='scatter'))  # Legenda para o preço

    # Adicionar legenda específica para os percentuais
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(
        color='red'), name='Percentuais Negativos'))  # Legenda para percentuais negativos

    # Exibir o gráfico usando st.plotly_chart
    st.plotly_chart(fig)

    st.caption("""
            
            **Pandemia de COVID-19:**
            A pandemia teve um impacto significativo na demanda global por petróleo, uma vez que as restrições de viagem e lockdowns 
            em muitas partes do mundo reduziram drasticamente o consumo de combustíveis. A queda na demanda contribuiu para o excesso 
            de oferta e pressionou os preços para baixo.
            
            **Guerra de Preços entre Arábia Saudita e Rússia:**
            Em março de 2020, a Arábia Saudita e a Rússia não conseguiram chegar a um acordo sobre os cortes na produção 
            de petróleo para sustentar os preços em meio à queda da demanda global devido à pandemia de COVID-19. Como resposta, 
            a Arábia Saudita aumentou sua produção e iniciou uma guerra de preços, inundando o mercado com petróleo, o que levou a 
            uma queda acentuada nos preços.

            **Acordo OPEP+ e Cortes na Produção:**
            Em abril de 2020, a OPEP+ (Organização dos Países Exportadores de Petróleo e aliados) chegou a um acordo para reduzir a 
            produção global de petróleo em resposta à crise induzida pela pandemia. Os cortes na produção ajudaram a estabilizar os 
            preços após a queda inicial.

            **Recuperação Econômica:**
            Com o avanço das campanhas de vacinação contra a COVID-19 e a perspectiva de uma recuperação econômica global, a demanda 
            por petróleo começou a se recuperar em 2021. A expectativa de uma maior demanda influenciou positivamente os preços do petróleo.
        
        

            """
               )
    st.subheader("Impactos nos Preços do Petróleo (2022-2023)")
    # Filtrar os dados para o intervalo
    df_intervalo = df_agrupado.query('Data >= 2022 and Data <= 2023')

    # Criar o gráfico de linha
    fig = px.line(df_intervalo, x='Data', y='Preço',
                  title='Preço Petróleo bruto X Variação percentual')
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço')

    # Adicionar a série temporal de percentuais como gráfico de barras
    fig.add_trace(go.Bar(x=df_intervalo['Data'], y=df_intervalo['percentual'],
                         marker=dict(
        color=['red' if val < 0 else 'green' for val in df_intervalo['percentual']]),
        name='Percentuais Positivos'))

    # Adicionar grades verticais
    for date in df_intervalo['Data']:
        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='gray', width=1)
        ))

    # Ajustar para exibir todos os anos no eixo x
    fig.update_xaxes(tickmode='linear',
                     tick0=df_intervalo['Data'].min(), dtick=1)
    fig.update_xaxes(tickangle=45, tickmode='array',
                     tickvals=df_intervalo['Data'])

    # Adicionar legendas
    fig.update_traces(name='Preço', showlegend=True, selector=dict(
        type='scatter'))  # Legenda para o preço

    # Adicionar legenda específica para os percentuais
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(
        color='red'), name='Percentuais Negativos'))  # Legenda para percentuais negativos

    # Exibir o gráfico usando st.plotly_chart
    st.plotly_chart(fig)

    st.caption("""
            
    **Sanções à Rússia:**
    Desde a invasão da Ucrânia pela Rússia, o país tem enfrentado sanções internacionais com o objetivo de reduzir o comércio com seus parceiros. Isso tem contribuído para a escalada dos preços, uma vez que há preocupações de que as sanções à Rússia possam prejudicar o fornecimento global de energia. Além disso, mesmo antes da invasão, a oferta global já não conseguia acompanhar a demanda, devido à flexibilização das medidas contra a COVID-19, intensificando as preocupações em relação aos preços das commodities..

    **Alta nos Preços devido a Restrições na Oferta:**
    Os preços do petróleo atingiram o maior nível de 2023 devido às expectativas de oferta mais restrita, superando preocupações 
    com o crescimento econômico mais fraco e o aumento dos estoques dos EUA. 

    """)

    st.header("O que está acontecendo hoje: 2024")

    st.caption("A volatilidade nos preços do petróleo, que observamos diariamente, muitas vezes é influenciada pelos acontecimentos no Oriente Médio. Isso se deve à presença de cinco dos dez maiores produtores mundiais na região: Arábia Saudita, Iraque, Emirados Árabes Unidos, Irã e Kuwait. Além disso, duas das principais rotas comerciais globais passam por lá: o Estreito de Ormuz, responsável pelo transporte de mais de 3/10 da produção mundial de petróleo, e o Canal de Suez, que conecta o Mediterrâneo ao Mar Vermelho. A instabilidade na região, causada por conflitos como a guerra entre Israel e Hamas e confrontos no Iêmen, afeta diretamente essas rotas cruciais. Bloqueios temporários nesses pontos estratégicos podem ocorrer durante momentos de tensão, impactando significativamente o mercado global de petróleo.")

    st.header("Conclusão")

    st.caption(" Os preços do petróleo são influenciados por uma interação complexa de fatores econômicos, geopolíticos, tecnológicos e ambientais, resultando na volatilidade nos mercados de commodities. Tensões geopolíticas, condições econômicas globais, decisões da OPEP+, desastres naturais, dinâmicas de oferta e demanda, e desenvolvimentos na transição energética são todos elementos cruciais que moldam o cenário dos preços do petróleo. Essa complexidade destaca a necessidade de uma abordagem abrangente ao analisar e compreender as flutuações nos mercados energéticos.")

    st.header("Referência")

    # Substituindo URLs
    url_bbc1 = "https://www.bbc.com/portuguese/articles/cld19n1dzy7o"
    url_veja = "https://veja.abril.com.br/economia/a-nova-era-do-petroleo-comecou"
    url_brasil_escola = "https://brasilescola.uol.com.br/geografia/crise-financeira-global.htm"
    url_ibp = "https://www.ibp.org.br/observatorio-do-setor/analises/covid-19-e-os-impactos-sobre-o-mercado-de-petroleo/#:~:text=A%20disseminação%20do%20COVID-19,efeitos%20da%20pandemia%20na%20economia."
    url_cnn = "https://www.cnnbrasil.com.br/economia/entenda-por-que-o-preco-do-petroleo-disparou-com-a-guerra-entre-ucrania-e-russia/"
    url_pantheon = "https://pantheon.ufrj.br/bitstream/11422/830/3/DCSDuarte.pdf"
    url_bbc2 = "https://www.bbc.com/portuguese/internacional-51799906"

    # Configurando a data de acesso
    data_acesso = "22/01/2024"

    # Texto formatado
    st.write("BBC (Artigo 1). Disponível em:",
             f"[<URL>]({url_bbc1}). Acesso em: {data_acesso}.")
    st.write("VEJA. Disponível em:",
             f"[<URL>]({url_veja}). Acesso em: {data_acesso}.")
    st.write("BRASIL ESCOLA. Disponível em:",
             f"[<URL>]({url_brasil_escola}). Acesso em: {data_acesso}.")
    st.write("INSTITUTO BRASILEIRO DE PETRÓLEO (IBP). Disponível em:",
             f"[<URL>]({url_ibp}). Acesso em: {data_acesso}.")
    st.write("CNN Brasil. Disponível em:",
             f"[<URL>]({url_cnn}). Acesso em: {data_acesso}.")
    st.write("Pantheon UFRJ (PDF). Disponível em:",
             f"[<URL>]({url_pantheon}). Acesso em: {data_acesso}.")
    st.write("BBC (Artigo 2). Disponível em:",
             f"[<URL>]({url_bbc2}). Acesso em: {data_acesso}.")
