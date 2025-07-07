import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io # Importa el módulo io para manejar la salida de df.info()

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Análisis de Baloto",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Análisis Exploratorio de Datos del Baloto 🇨🇴")
st.write("Bienvenido al panel interactivo de análisis de los resultados históricos del Baloto colombiano.")

# --- URL del archivo CSV en GitHub ---
url = "https://raw.githubusercontent.com/JulianTorrest/modelos/refs/heads/main/Baloto.csv"

# --- Función para Cargar y Preprocesar los Datos (con caché) ---
@st.cache_data
def load_data(data_url):
    """
    Carga el archivo CSV desde la URL, convierte la columna de fecha
    y calcula la suma de las balotas.
    Utiliza caché para evitar recargas innecesarias.
    """
    try:
        df = pd.read_csv(data_url)
        # Convertir la columna 'Fecha' a formato de fecha (día/mes/año)
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
        # Extraer el año, mes y día para posibles análisis
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month
        df['Dia'] = df['Fecha'].dt.day
        # Calcular la suma de balotas principales
        df['SumaBalotas'] = df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5']].sum(axis=1)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo desde {data_url}: {e}")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error

# --- Cargar los datos ---
df = load_data(url)

# --- Verificar si los datos se cargaron correctamente ---
if not df.empty:
    st.success("¡Datos de Baloto cargados exitosamente! Fecha de la última actualización: " + df['Fecha'].max().strftime('%d/%m/%Y'))

    # --- Sección 1: Primeras Filas y Estructura del DataFrame ---
    st.header("🔍 Primeras Filas del Conjunto de Datos")
    st.dataframe(df.head())

    st.header("📊 Información General y Estadísticas Descriptivas")
    st.subheader("Tipos de Datos y Valores No Nulos")
    # Captura la salida de df.info() en un buffer de texto y la muestra
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Estadísticas Descriptivas de las Balotas")
    st.write("Resumen estadístico de las balotas y la SuperBalota.")
    st.dataframe(df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].describe())

    # --- Sección 2: Distribución de Frecuencia de las Balotas ---
    st.header("📈 Distribución de Frecuencia de las Balotas")
    st.write("Estos histogramas muestran la frecuencia con la que ha aparecido cada número en cada posición de las balotas y en la SuperBalota.")

    # Crear una figura con múltiples subplots
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
    axes1 = axes1.flatten() # Aplanar el array de ejes para una fácil iteración

    # Distribución de las 5 Balotas Regulares
    for i in range(1, 6):
        sns.histplot(df[f'Balota {i}'], bins=range(1, 44), kde=True, ax=axes1[i-1], color='skyblue')
        axes1[i-1].set_title(f'Distribución Balota {i}')
        axes1[i-1].set_xlabel('Número')
        axes1[i-1].set_ylabel('Frecuencia')
        axes1[i-1].set_xticks(range(1, 44, 4)) # Mostrar ticks cada 4 números para claridad

    # Distribución de la SuperBalota
    sns.histplot(df['SuperBalota'], bins=range(1, 17), kde=True, ax=axes1[5], color='lightcoral')
    axes1[5].set_title('Distribución SuperBalota')
    axes1[5].set_xlabel('Número')
    axes1[5].set_ylabel('Frecuencia')
    axes1[5].set_xticks(range(1, 17))

    plt.tight_layout() # Ajusta automáticamente los parámetros de la subtrama para un diseño ajustado
    st.pyplot(fig1) # Muestra la figura completa en Streamlit

    # --- Sección 3: Balotas Más Frecuentes ---
    st.header("⭐ Balotas Más Frecuentes")
    st.write("Identifica los números que han sido los más 'afortunados' en la historia del Baloto.")

    col1, col2 = st.columns(2) # Divide la interfaz en dos columnas para mejor visualización

    with col1:
        st.subheader("Top 10 Balotas Regulares")
        # Concatena todas las balotas regulares para contar la frecuencia global
        all_balotas = pd.concat([df['Balota 1'], df['Balota 2'], df['Balota 3'], df['Balota 4'], df['Balota 5']])
        top_balotas = all_balotas.value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Se corrige el FutureWarning asignando `x` a `hue` y `legend=False`
        sns.barplot(x=top_balotas.index, y=top_balotas.values, palette='viridis', ax=ax2, hue=top_balotas.index, legend=False)
        ax2.set_title('Top 10 Balotas Más Frecuentes (excluyendo SuperBalota)')
        ax2.set_xlabel('Número de Balota')
        ax2.set_ylabel('Frecuencia')
        st.pyplot(fig2)

    with col2:
        st.subheader("Top 10 SuperBalotas")
        top_superbalotas = df['SuperBalota'].value_counts().head(10)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        # Se corrige el FutureWarning asignando `x` a `hue` y `legend=False`
        sns.barplot(x=top_superbalotas.index, y=top_superbalotas.values, palette='magma', ax=ax3, hue=top_superbalotas.index, legend=False)
        ax3.set_title('Top 10 SuperBalotas Más Frecuentes')
        ax3.set_xlabel('Número de SuperBalota')
        ax3.set_ylabel('Frecuencia')
        st.pyplot(fig3)

    # --- Sección 4: Análisis de Correlación ---
    st.header("🔗 Matriz de Correlación entre Balotas")
    st.write("Aunque las balotas en un sorteo son teóricamente independientes, esta matriz muestra si existe alguna correlación numérica observada entre ellas a lo largo del tiempo.")
    numeric_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
    correlation_matrix = df[numeric_cols].corr()

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
    ax4.set_title('Matriz de Correlación entre Balotas')
    st.pyplot(fig4)

    # --- Sección 5: Análisis Temporal de la Suma de Balotas ---
    st.header("⏳ Tendencia de la Suma de Balotas a lo largo del Tiempo")
    st.write("Esta gráfica muestra cómo ha evolucionado la suma de las cinco balotas principales en cada sorteo a lo largo de los años.")

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Fecha', y='SumaBalotas', data=df, ax=ax5, color='darkgreen')
    ax5.set_title('Suma de Balotas a lo largo del Tiempo')
    ax5.set_xlabel('Fecha del Sorteo')
    ax5.set_ylabel('Suma de las Balotas')
    st.pyplot(fig5)

    # --- Sección 6: Suma de Balotas por Año ---
    st.header("📅 Distribución de la Suma de Balotas por Año")
    st.write("Los diagramas de caja muestran la distribución de la suma de las balotas para cada año, incluyendo medianas, cuartiles y valores atípicos.")

    fig6, ax6 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Año', y='SumaBalotas', data=df, ax=ax6, palette='Pastel1')
    ax6.set_title('Suma de Balotas por Año')
    ax6.set_xlabel('Año del Sorteo')
    ax6.set_ylabel('Suma de las Balotas')
    st.pyplot(fig6)

    st.markdown("---")
    st.write("Análisis completado. ¡Esperamos que esta información te sea útil!")

else:
    st.error("No se pudieron cargar los datos del Baloto. Por favor, asegúrate de que la URL sea correcta y el archivo esté accesible.")
    st.info("Intenta revisar la URL del archivo en tu repositorio de GitHub o la conexión a internet.")
