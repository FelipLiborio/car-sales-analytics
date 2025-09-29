import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================================
# --- Carregar e tratar dados ---
# ==========================================================
df = pd.read_csv("../data/car_prices.csv")  # ajuste o caminho

# Converter data
saledate_clean = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
df["saledate"] = saledate_clean.dt.tz_convert(None)

# Remover nulos relevantes
df = df.dropna(subset=["sellingprice", "saledate", "odometer", "condition"])

# Tratar variáveis categóricas
categorical_cols = ["make", "model", "trim", "body",
                    "transmission", "color", "interior"]
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

for col in ["make", "model", "body"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Padronizar estados
df["state"] = df["state"].astype(str).str.upper().str.strip()

# ==========================================================
# --- Sidebar: Navegação ---
# ==========================================================
st.sidebar.header("⚙️ Configurações do Dashboard")

aba = st.sidebar.radio(
    "Selecione a seção:",
    ["Análises Descritivas", "Regressão Linear"]
)

# ==========================================================
# --- Aba 1: Análises Descritivas ---
# ==========================================================
if aba == "Análises Descritivas":
    st.header("📊 Análises Descritivas")
    st.markdown("---")

    colunas_categoricas = ["make", "year", "body",
                           "transmission", "state", "color", "interior"]

    coluna = st.sidebar.selectbox("Selecione a variável categórica:", colunas_categoricas)

    tipo_rank = st.sidebar.radio("Selecione:", ["10 Maiores", "10 Menores"])

    make_sel = st.sidebar.selectbox("Selecione a Marca (Make):", sorted(df["make"].unique()))

    metrica = st.sidebar.radio(
        "Selecione a métrica para os modelos:",
        ["Mais vendidos", "Maior preço médio", "Maior MMR médio", "Relação Preço/MMR"]
    )

    # ----------------------------------------------------------
    # Gráfico 1 → Número de Vendas por Categoria
    # ----------------------------------------------------------
    st.subheader("📊 Número de Vendas por Categoria")

    df_plot = df[coluna].value_counts().reset_index()
    df_plot.columns = [coluna, "Contagem"]

    if tipo_rank == "10 Maiores":
        df_plot = df_plot.nlargest(10, "Contagem").sort_values(by="Contagem", ascending=False)
    else:
        df_plot = df_plot.nsmallest(10, "Contagem").sort_values(by="Contagem", ascending=True)

    fig1 = px.bar(
        df_plot,
        x=coluna,
        y="Contagem",
        text="Contagem",
        title=f"{tipo_rank} categorias de {coluna} por vendas",
        color=coluna,
        color_discrete_sequence=px.colors.qualitative.Pastel1
    )
    fig1.update_layout(xaxis_tickangle=-45, height=500, title_x=0.5, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # ----------------------------------------------------------
    # Gráfico 2 → Modelos dentro de um Make
    # ----------------------------------------------------------
    st.subheader(f"🚗 Análise de Modelos ({make_sel})")

    df_make = df[df["make"] == make_sel]

    if metrica == "Mais vendidos":
        df_plot2 = df_make["model"].value_counts().reset_index().head(10)
        df_plot2.columns = ["model", "Valor"]
        title = f"Modelos mais vendidos da {make_sel}"

    elif metrica == "Maior preço médio":
        df_plot2 = df_make.groupby("model")["sellingprice"].mean().reset_index()
        df_plot2.columns = ["model", "Valor"]
        df_plot2 = df_plot2.sort_values(by="Valor", ascending=False).head(10)
        title = f"Top 10 modelos da {make_sel} por preço médio"

    elif metrica == "Maior MMR médio":
        df_plot2 = df_make.groupby("model")["mmr"].mean().reset_index()
        df_plot2.columns = ["model", "Valor"]
        df_plot2 = df_plot2.sort_values(by="Valor", ascending=False).head(10)
        title = f"Top 10 modelos da {make_sel} por MMR médio"

    else:  # Relação Preço/MMR
        df_make = df_make[df_make["mmr"] > 0]
        df_make["relacao"] = df_make["sellingprice"] / df_make["mmr"]
        df_plot2 = df_make.groupby("model")["relacao"].mean().reset_index()
        df_plot2.columns = ["model", "Valor"]
        df_plot2 = df_plot2.sort_values(by="Valor", ascending=False).head(10)
        title = f"Top 10 modelos da {make_sel} por relação Preço/MMR"

    fig2 = px.bar(
        df_plot2,
        x="model",
        y="Valor",
        text="Valor",
        title=title,
        color="model",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig2.update_layout(xaxis_tickangle=-45, height=500, title_x=0.5, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------------------------------
    # Gráfico → Vendas por Estado (Mapa)
    # ----------------------------------------------------------
    st.subheader("🗺️ Vendas por Estado")

    # Criar seção separada na sidebar
    st.sidebar.markdown("### 🌎 Filtros - Mapa por Estado")

    make_options = ["Todas"] + sorted(df["make"].unique())
    make_filter = st.sidebar.selectbox("Filtrar por Marca (Make):", make_options)

    model_filter = None
    if make_filter != "Todas":
        model_options = ["Todos"] + sorted(df[df["make"] == make_filter]["model"].unique())
        model_filter = st.sidebar.selectbox(f"Filtrar por Modelo da {make_filter}:", model_options)

    # Filtrar o dataframe
    df_map = df.copy()
    if make_filter != "Todas":
        df_map = df_map[df_map["make"] == make_filter]
        if model_filter and model_filter != "Todos":
            df_map = df_map[df_map["model"] == model_filter]

    # Contagem por estado
    df_state = df_map["state"].value_counts().reset_index()
    df_state.columns = ["state", "Contagem"]

    # Criar mapa
    fig_map = px.choropleth(
        df_state,
        locations="state",
        locationmode="USA-states",
        color="Contagem",
        scope="usa",
        color_continuous_scale="Blues",
        title="Distribuição de Vendas por Estado"
    )
    fig_map.update_layout(height=500, title_x=0.5)
    st.plotly_chart(fig_map, use_container_width=True)

    # ----------------------------------------------------------
    # Gráfico → Comparação Preço vs Odômetro/Condição
    # ----------------------------------------------------------
    st.subheader("📈 Comparação: Preço de Venda vs Odômetro/Condição")

    eixo_x = st.sidebar.radio("Comparar preço com:", ["odometer", "condition"],
                            format_func=lambda x: "Odômetro" if x == "odometer" else "Condição")

    escala_cor = "Blues" if eixo_x == "odometer" else "Viridis"

    fig_comp = px.scatter(
        df, x=eixo_x, y="sellingprice",
        opacity=0.6, color=df[eixo_x],
        color_continuous_scale=escala_cor,
        title=f"Preço de venda em função de {eixo_x}"
    )
    fig_comp.update_layout(height=500, title_x=0.5)
    st.plotly_chart(fig_comp, use_container_width=True)


    # ----------------------------------------------------------
    # Gráfico → Ano de Fabricação
    # ----------------------------------------------------------
    st.subheader("📅 Análise por Ano de Fabricação")
    st.markdown("---")

    # Criar seção separada na sidebar
    st.sidebar.markdown("### 📅 Filtros - Ano de Fabricação")
    metrica_ano = st.sidebar.radio(
        "Selecione a métrica:",
        ["Preço médio de venda", "Quantidade de vendas"]
    )

    # Agrupar dados
    if metrica_ano == "Preço médio de venda":
        df_year = df.groupby("year")["sellingprice"].mean().reset_index()
        df_year.columns = ["year", "Valor"]
        titulo = "Preço médio de venda por ano de fabricação"
        eixo_y = "Preço Médio"
    else:
        df_year = df["year"].value_counts().reset_index()
        df_year.columns = ["year", "Valor"]
        df_year = df_year.sort_values("year")
        titulo = "Quantidade de vendas por ano de fabricação"
        eixo_y = "Quantidade de Vendas"

    # Criar gráfico
    fig_year = px.line(
        df_year, x="year", y="Valor", markers=True,
        title=titulo, labels={"year": "Ano de Fabricação", "Valor": eixo_y}
    )
    fig_year.update_layout(height=500, title_x=0.5)
    st.plotly_chart(fig_year, use_container_width=True)

# ==========================================================
# --- Aba 2: Regressão Linear + Análise de Correlação ---
# ==========================================================
else:
    st.header("📉 Regressão Linear e Correlação de Variáveis")

    st.markdown("""
    Nesta etapa, trabalhamos com **regressão linear** e **análise de correlação entre variáveis**.

    O objetivo geral é responder: **quais fatores realmente influenciam variáveis importantes e em que intensidade?**
    """)

    # ==========================================================
    # Cenário 1
    # ==========================================================
    st.subheader("📊 Cenário 1: Ano, Quilometragem e Condição → Preço de Venda")

    st.latex(r"""
    \hat{SellingPrice} = -1.514 \times 10^{6} \;+\; 760.66 \cdot \text{year} 
            \;-\; 0.056 \cdot \text{odometer} 
            \;+\; 88.30 \cdot \text{condition}
    """)

    st.markdown("""
    - **Coeficientes estimados (β):**
      - Ano (year): **+760,66**
      - Quilometragem (odometer): **-0,056**
      - Condição (condition): **+88,30**

    - **Erros padrão:** baixos, indicando boa precisão.  
    - **p-valores:** todos iguais a **0.0**, confirmando significância estatística.  
    - **Intervalos de confiança (95%):**
      - Ano: [750,84 ; 770,47]  
      - Quilometragem: [-0,0566 ; -0,0552]  
      - Condição: [86,38 ; 90,23]  

    ✅ **Interpretação:** carros mais novos e em melhor condição custam mais; carros com maior quilometragem custam menos.
    """)

    # ==========================================================
    # Cenário 2
    # ==========================================================
    st.subheader("📊 Cenário 2: Ano e Quilometragem → Condição do Carro")

    st.latex(r"""
    \widehat{Condition} = -1583.87 \;+\; 0.804 \cdot \text{year} 
            \;-\; 0.000035 \cdot \text{odometer}
    """)

    st.markdown("""
    - **Coeficientes estimados (β):**
      - Ano (year): **+0,804**
      - Quilometragem (odometer): **-0,000035**

    - **Erros padrão:** muito baixos, indicando alta precisão.  
    - **p-valores:** todos iguais a **0.0**, confirmando significância estatística.  
    - **Intervalos de confiança (95%):**
      - Ano: [0,788 ; 0,820]  
      - Quilometragem: [-0,000036 ; -0,000033]  

    ✅ **Interpretação:** cada ano a mais aumenta a condição em **~0,8 pontos**, enquanto a cada **10.000 km rodados** a condição cai em **~0,35 pontos**. Carros mais novos e menos rodados tendem a estar em melhores condições.
    """)

