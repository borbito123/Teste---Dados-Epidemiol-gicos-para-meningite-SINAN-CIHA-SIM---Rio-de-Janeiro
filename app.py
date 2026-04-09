import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="Dashboard Epidemiológico", layout="wide")

st.title("🔬 Painel de Análise Epidemiológica (SINAN, SIM, CIHA)")
st.markdown("Carregue seu arquivo `.parquet` e selecione as variáveis para gerar os gráficos interativos.")

# Sidebar para upload de dados
st.sidebar.header("1. Importação de Dados")
uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo Parquet", type=['parquet'])

if uploaded_file is not None:
    # Carregando os dados
    @st.cache_data # Faz cache para não recarregar o arquivo toda vez que interagir
    def load_data(file):
        return pd.read_parquet(file)
    
    df = load_data(uploaded_file)
    
    st.success(f"Arquivo carregado com sucesso! {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
    with st.expander("Visualizar amostra dos dados (5 primeiras linhas)"):
        st.dataframe(df.head())

    st.sidebar.header("2. Mapeamento de Colunas")
    st.sidebar.info("Selecione as colunas correspondentes no seu banco de dados.")
    
    colunas_disp = ['Nenhuma'] + list(df.columns)
    
    col_data = st.sidebar.selectbox("Coluna de Data (Ex: DT_NOTIFIC, DTOBITO)", colunas_disp)
    col_sexo = st.sidebar.selectbox("Coluna de Sexo (Ex: CS_SEXO, SEXO)", colunas_disp)
    col_categoria = st.sidebar.selectbox("Agravo / CID (Ex: ID_AGRAVO, CAUSABAS)", colunas_disp)

    # Filtros
    st.sidebar.header("3. Filtros")
    if col_sexo != 'Nenhuma':
        sexo_selecionado = st.sidebar.multiselect("Filtrar por Sexo", df[col_sexo].dropna().unique())
        if sexo_selecionado:
            df = df[df[col_sexo].isin(sexo_selecionado)]

    # Área de Gráficos
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    # 1. SÉRIE TEMPORAL (Curva Epidêmica)
    with col1:
        st.subheader("📈 Série Temporal de Casos/Registros")
        if col_data != 'Nenhuma':
            try:
                # Converte para datetime caso não esteja
                df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
                # Conta registros por Ano/Mês
                df_tempo = df.groupby(df[col_data].dt.to_period('M')).size().reset_index(name='Casos')
                df_tempo[col_data] = df_tempo[col_data].dt.to_timestamp()
                
                fig_tempo = px.line(df_tempo, x=col_data, y='Casos', markers=True, 
                                    title="Evolução Temporal (Mensal)",
                                    labels={col_data: 'Data', 'Casos': 'Número de Registros'})
                st.plotly_chart(fig_tempo, use_container_width=True)
            except Exception as e:
                st.error("Erro ao processar a data. Verifique se a coluna está em formato válido.")
        else:
            st.info("Selecione a coluna de Data no menu lateral para ver este gráfico.")

    # 2. TOP CAUSAS / AGRAVOS
    with col2:
        st.subheader("📊 Principais Ocorrências (Top 10)")
        if col_categoria != 'Nenhuma':
            df_cat = df[col_categoria].value_counts().head(10).reset_index()
            df_cat.columns = [col_categoria, 'Frequência']
            
            fig_bar = px.bar(df_cat, x='Frequência', y=col_categoria, orientation='h',
                             title=f"Top 10: {col_categoria}",
                             color='Frequência', color_continuous_scale='Blues')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Selecione a coluna de Agravo/CID no menu lateral.")

    # 3. DISTRIBUIÇÃO POR SEXO
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("👥 Distribuição por Sexo")
        if col_sexo != 'Nenhuma':
            df_sexo = df[col_sexo].value_counts().reset_index()
            df_sexo.columns = [col_sexo, 'Contagem']
            
            fig_pie = px.pie(df_sexo, names=col_sexo, values='Contagem', hole=0.4,
                             title="Proporção por Sexo",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Selecione a coluna de Sexo no menu lateral.")

else:
    st.info("Aguardando o upload de um arquivo Parquet válido. Extraia seus dados do TabNet/DataSUS ou de seus scripts R/Python e faça o upload acima.")
