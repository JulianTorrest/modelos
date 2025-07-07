import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis de Baloto",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Análisis Exploratorio de Datos del Baloto 🇨🇴")

# URL del archivo CSV en GitHub
url = "https://raw.githubusercontent.com/JulianTorrest/modelos/refs/heads/main/Baloto.csv"

# Usar caché para cargar los datos solo una vez
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        # Convertir la columna 'Fecha' a formato de fecha
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
        # Extraer el año, mes y día para posibles análisis
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month
        df['Dia'] = df['Fecha'].dt.day
        # Calcular la suma de balotas
        df['SumaBalotas'] = df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5']].sum(axis=1)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error

df = load_data(url)

if not df.empty:
    st.success("¡Datos de Baloto cargados exitosamente!")

    # --- Mostrar Primeras Filas ---
    st.header("🔍 Primeras Filas del Conjunto de Datos")
    st.dataframe(df.head())

    # --- Información General ---
    st.header("📊 Información General y Estadísticas Descriptivas")
    st.subheader("Tipos de Datos y Valores No Nulos")
    st.write(df.info(buf=st.io.StringIO())) # Redirige la salida de info() a Streamlit

    st.subheader("Estadísticas Descriptivas de las Balotas")
    st.dataframe(df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].describe())

    # --- Distribución de Cada Balota ---
    st.header("📈 Distribución de Frecuencia de las Balotas")
    st.write("Observa qué números han salido con mayor frecuencia en cada posición.")

    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
    axes1 = axes1.flatten() # Aplanar para facilitar la iteración

    for i in range(1, 6):
        sns.histplot(df[f'Balota {i}'], bins=range(1, 44), kde=True, ax=axes1[i-1])
        axes1[i-1].set_title(f'Distribución Balota {i}')
        axes1[i-1].set_xticks(range(1, 44, 4)) # Ajuste de ticks para mejor visualización

    sns.histplot(df['SuperBalota'], bins=range(1, 17), kde=True, ax=axes1[5])
    axes1[5].set_title('Distribución SuperBalota')
    axes1[5].set_xticks(range(1, 17))

    plt.tight_layout()
    st.pyplot(fig1) # Muestra la figura en Streamlit

    # --- Top 10 Balotas Más Frecuentes ---
    st.header("⭐ Balotas Más Frecuentes")
    st.write("Descubre los números que más han aparecido en el histórico del Baloto.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Balotas Regulares")
        all_balotas = pd.concat([df['Balota 1'], df['Balota 2'], df['Balota 3'], df['Balota 4'], df['Balota 5']])
        top_balotas = all_balotas.value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_balotas.index, y=top_balotas.values, palette='viridis', ax=ax2)
        ax2.set_title('Top 10 Balotas Más Frecuentes (excluyendo SuperBalota)')
        ax2.set_xlabel('Número de Balota')
        ax2.set_ylabel('Frecuencia')
        st.pyplot(fig2)

    with col2:
        st.subheader("Top 10 SuperBalotas")
        top_superbalotas = df['SuperBalota'].value_counts().head(10)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_superbalotas.index, y=top_superbalotas.values, palette='magma', ax=ax3)
        ax3.set_title('Top 10 SuperBalotas Más Frecuentes')
        ax3.set_xlabel('Número de SuperBalota')
        ax3.set_ylabel('Frecuencia')
        st.pyplot(fig3)

    # --- Análisis de Correlación ---
    st.header("🔗 Matriz de Correlación entre Balotas")
    st.write("Aunque las balotas son teóricamente independientes, esta matriz muestra cualquier correlación observada.")
    numeric_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
    correlation_matrix = df[numeric_cols].corr()

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
    ax4.set_title('Matriz de Correlación entre Balotas')
    st.pyplot(fig4)

    # --- Tendencia de la Suma de Balotas a lo largo del Tiempo ---
    st.header("⏳ Tendencia de la Suma de Balotas")
    st.write("Visualiza cómo la suma total de las balotas ha variado a lo largo de los años.")

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Fecha', y='SumaBalotas', data=df, ax=ax5)
    ax5.set_title('Suma de Balotas a lo largo del Tiempo')
    ax5.set_xlabel('Fecha')
    ax5.set_ylabel('Suma de las Balotas')
    st.pyplot(fig5)

    # --- Suma de Balotas por Año ---
    st.header("📅 Suma de Balotas por Año")
    st.write("Un vistazo a la distribución de la suma de las balotas para cada año.")

    fig6, ax6 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Año', y='SumaBalotas', data=df, ax=ax6)
    ax6.set_title('Suma de Balotas por Año')
    ax6.set_xlabel('Año')
    ax6.set_ylabel('Suma de las Balotas')
    st.pyplot(fig6)

else:
    st.warning("No se pudieron cargar los datos. Por favor, verifica la URL del archivo CSV.")
