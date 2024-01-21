# Use uma imagem base do Python
FROM python:3.10.13-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /home/project

# Copie os arquivos do projeto para o contêiner
COPY . /home/project

# Instale as bibliotecas necessárias
RUN pip install --upgrade pip
RUN pip install pandas \
               lxml \
               pandas-gbq -U

CMD ["python", "pipeline.py"]
