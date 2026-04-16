# Baixando os banco de dados
  Este aplicativo serve para baixar os dados do SINAN (meningite), SIM e CIHA (anos 2007 a 2025) referentes ao estado do Rio de Janeiro e convertê-los para os formatos parquet e duckdb, para fins de análise epidemiológica.
  
  Também é possível baixar tudo (formatos parquet, duckdb e dbc) diretamente por meio da pasta "Bases_Datasus_Municipio_Rio_de_Janeiro" no seguinte link: https://drive.google.com/drive/u/0/folders/1JrFZ1PN3kU11ab2xZWmoO06K1HRPUmM4.

- **SINAN**: notificações/casos
- **SIM**: óbitos
- **CIHA**: internações/atendimentos

# Painel Streamlit para Parquets do SINAN, SIM e CIHA

Este app em Python foi feito para análise epidemiológica a partir de arquivos `.parquet` do DATASUS, com foco em três bases:

- **SINAN**: notificações/casos
- **SIM**: óbitos
- **CIHA**: internações/atendimentos

## O que o app faz

- lê um ou mais Parquets por base
- aceita **upload** ou **caminho local/glob**
- detecta automaticamente colunas prováveis de:
  - data
  - sexo
  - idade
  - município
  - desfecho/classificação
  - diagnóstico/CID
- gera gráficos epidemiológicos interativos
- permite download em CSV das tabelas agregadas de cada gráfico
- compara séries temporais entre bases

## Gráficos incluídos

### Para SINAN
- série temporal por ano/mês/semana
- heatmap ano × mês
- distribuição etária em faixas de 5 anos
- pirâmide etária por sexo
- top diagnósticos/agravos
- top desfechos/classificações
- top municípios
- completude de campos-chave

### Para SIM
- série temporal de óbitos
- heatmap ano × mês
- distribuição etária
- pirâmide etária por sexo
- top causas básicas / diagnóstico
- top variáveis de desfecho/local do óbito
- completude

### Para CIHA
- série temporal de internações/atendimentos
- heatmap ano × mês
- distribuição etária
- pirâmide etária por sexo
- top diagnósticos/procedimentos
- top desfechos de saída
- completude

## Instalação

Crie e ative um ambiente virtual, se desejar, e depois instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

No diretório do projeto, rode:

```bash
streamlit run app_streamlit_epidemiologia.py
```

## Como usar

### Opção 1: upload
Envie um ou mais arquivos `.parquet` em cada aba da base desejada.

### Opção 2: pasta/glob local
Informe um padrão local, por exemplo:

```text
Bases_Datasus_Municipio_Rio_de_Janeiro/SINAN/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/SIM/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/CIHA/data/parquet/*.parquet
```

## Observações importantes

- O app foi desenhado para funcionar com layouts **variáveis** do DATASUS, mas pode ser necessário ajustar manualmente as colunas detectadas.
- Para idade codificada do DATASUS, há a opção **"DATASUS codificada"**. A conversão para anos é aproximada para registros em horas, dias e meses.
- Se os parquets já estiverem filtrados para um município específico, os gráficos respeitarão esse recorte.
- A comparação entre bases é **exploratória** e faz mais sentido quando o agravo, o território e a janela temporal são os mesmos.

## Sugestões de uso epidemiológico

- Use a **série temporal** como gráfico principal para monitorar tendência.
- Use o **heatmap ano × mês** para sazonalidade.
- Use a **pirâmide etária por sexo** para perfil demográfico.
- Use **top diagnósticos/desfechos** para caracterização clínica e gravidade.
- Use a **completude** para avaliar qualidade da informação antes de interpretar resultados.
