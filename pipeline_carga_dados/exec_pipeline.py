import docker

def run_container_script():
    # Cria um cliente Docker
    client = docker.from_env()

    # Nome do container
    container_name = 'pipeline-project'

    # Nome da imagem Docker
    image_name = 'tech-challenge'

    # Nome do script a ser executado dentro do container
    script_name = 'pipeline.py'

    try:
        # Inicia o container
        container = client.containers.run(
            image=image_name,
            detach=True,
            name=container_name
        )

        # Executa o script dentro do container
        response = container.exec_run(['python', script_name])

        # Captura a saída do script
        output = response.output.decode('utf-8')

        # Imprime a saída do script
        print("Saída do script dentro do container:")
        print(output)

    finally:
        # Remove o container após a execução
        container.remove(force=True)

if __name__ == "__main__":
    run_container_script()

