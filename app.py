import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="TCC - Predição de Engajamento", page_icon="📊", layout="wide")

st.title(" TCC: Pipeline de Predição de Engajamento no YouTube")
st.markdown("Dashboard interativo desenvolvido para apresentar os resultados da Prova de Conceito (PoC) com **BERTweet-pt** e **Random Forest**.")

# Seção de Métricas
st.header(" Desempenho dos Modelos")
metrics_path = "outputs/metrics/metrics.csv"
comparison_path = "outputs/metrics/comparacao_modelos.csv"

if os.path.exists(metrics_path):
    df_metrics = pd.read_csv(metrics_path)
    st.subheader("Métricas Atuais de Regressão e Classificação")
    st.dataframe(df_metrics, use_container_width=True)
else:
    st.warning(" Arquivo `metrics.csv` não encontrado. Execute `python main.py` primeiro.")

if os.path.exists(comparison_path):
    df_comp = pd.read_csv(comparison_path)
    st.subheader(" Comparativo de Modelos")
    st.dataframe(df_comp, use_container_width=True)

# Seção de Visualizações (Gráficos)
st.header(" Gráficos Gerados pelo Pipeline")
figures_dir = "outputs/figures"

if os.path.exists(figures_dir):
    figures = [f for f in os.listdir(figures_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if figures:
        selected_fig = st.selectbox("Selecione um gráfico para visualizar:", figures)
        st.image(os.path.join(figures_dir, selected_fig), caption=selected_fig, use_container_width=True)
    else:
        st.info("Nenhuma imagem encontrada na pasta de figuras.")
else:
    st.info("A pasta `outputs/figures` ainda não foi gerada.")

st.sidebar.info(" **Dica:** Você pode tirar prints das tabelas e gráficos desta tela para montar a sua apresentação para a professora!")
