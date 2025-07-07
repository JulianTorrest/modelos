import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import google.generativeai as genai # Importa la librería de Gemini

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Análisis y Predicción de Baloto con Gemini AI",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Análisis Exploratorio de Datos y Predicción de Baloto con Gemini AI 🇨🇴")
st.write("Bienvenido al panel interactivo de análisis de los resultados históricos del Baloto colombiano. Explora tendencias pasadas y experimenta con la IA de Gemini para posibles predicciones o insights.")

# --- Configuración de la API Key de Gemini ---
# IMPORTANTE: No hardcodear la API key directamente en el código de producción.
# Para Streamlit Cloud, usa los secretos de la aplicación.
# Por ahora, la cargaremos del secrets.toml
# Crea un archivo .streamlit/secrets.toml con:
#GOOGLE_API_KEY = "AIzaSyAo1mZnBvslWoUKot7svYIo2K3fZIrLgRk"
try:
    gemini_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    st.success("API de Gemini configurada exitosamente.")
except KeyError:
    st.error("Error: La API Key de Gemini no se encontró en los secretos de Streamlit. "
             "Asegúrate de que 'GOOGLE_API_KEY' esté definida en tu archivo .streamlit/secrets.toml")
    model = None # Asegurarse de que el modelo sea None si la clave no está configurada

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
    st.success("¡Datos de Baloto cargados exitosamente! Fecha del último sorteo registrado: " + df['Fecha'].max().strftime('%d/%m/%Y'))

    # --- Pestañas para organizar el contenido ---
    tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🤖 Predicción/Insights con IA", "ℹ️ Acerca de"])

    with tab1:
        st.header("Análisis Exploratorio de Datos Históricos")

        # --- Sección 1: Primeras Filas y Estructura del DataFrame ---
        st.subheader("🔍 Primeras Filas del Conjunto de Datos")
        st.dataframe(df.head())

        st.subheader("📊 Información General y Estadísticas Descriptivas")
        st.write("Resumen de los tipos de datos, valores no nulos y uso de memoria.")
        # Captura la salida de df.info() en un buffer de texto y la muestra
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write("Estadísticas descriptivas básicas para las balotas:")
        st.dataframe(df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].describe())

        # --- Sección 2: Distribución de Frecuencia de las Balotas ---
        st.subheader("📈 Distribución de Frecuencia de las Balotas")
        st.write("Estos histogramas muestran la frecuencia con la que ha aparecido cada número **en su respectiva posición de balota** y en la SuperBalota. Recuerda que las balotas 1 a 5 están ordenadas numéricamente.")

        # Crear una figura con múltiples subplots
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
        axes1 = axes1.flatten() # Aplanar el array de ejes para una fácil iteración

        # Distribución de las 5 Balotas Regulares
        for i in range(1, 6):
            sns.histplot(df[f'Balota {i}'], bins=range(1, 44), kde=True, ax=axes1[i-1], color='skyblue')
            axes1[i-1].set_title(f'Distribución Balota {i} (1-43)')
            axes1[i-1].set_xlabel('Número')
            axes1[i-1].set_ylabel('Frecuencia')
            axes1[i-1].set_xticks(range(1, 44, 4)) # Mostrar ticks cada 4 números para claridad

        # Distribución de la SuperBalota
        sns.histplot(df['SuperBalota'], bins=range(1, 17), kde=True, ax=axes1[5], color='lightcoral')
        axes1[5].set_title('Distribución SuperBalota (1-16)')
        axes1[5].set_xlabel('Número')
        axes1[5].set_ylabel('Frecuencia')
        axes1[5].set_xticks(range(1, 17))

        plt.tight_layout()
        st.pyplot(fig1)

        # --- Sección 3: Balotas Más Frecuentes (Global y SuperBalota) ---
        st.subheader("⭐ Balotas Más Frecuentes")
        st.write("Identifica los números que han sido los más 'afortunados' en la historia del Baloto, considerando todas las posiciones para las balotas regulares.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Top 10 Balotas Regulares (Global)")
            # Concatena todas las balotas regulares para contar la frecuencia global
            all_balotas = pd.concat([df['Balota 1'], df['Balota 2'], df['Balota 3'], df['Balota 4'], df['Balota 5']])
            top_balotas = all_balotas.value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_balotas.index, y=top_balotas.values, palette='viridis', ax=ax2, hue=top_balotas.index, legend=False)
            ax2.set_title('Top 10 Balotas Más Frecuentes (1-43)')
            ax2.set_xlabel('Número de Balota')
            ax2.set_ylabel('Frecuencia')
            st.pyplot(fig2)

        with col2:
            st.markdown("##### Top 10 SuperBalotas")
            top_superbalotas = df['SuperBalota'].value_counts().head(10)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_superbalotas.index, y=top_superbalotas.values, palette='magma', ax=ax3, hue=top_superbalotas.index, legend=False)
            ax3.set_title('Top 10 SuperBalotas Más Frecuentes (1-16)')
            ax3.set_xlabel('Número de SuperBalota')
            ax3.set_ylabel('Frecuencia')
            st.pyplot(fig3)

        # --- Sección 4: Análisis de Correlación ---
        st.subheader("🔗 Matriz de Correlación entre Balotas")
        st.write("Aunque las balotas de un sorteo individual son independientes, esta matriz muestra si existe alguna correlación numérica **observada** entre las *posiciones* de las balotas a lo largo del tiempo, teniendo en cuenta su orden ascendente.")
        numeric_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
        correlation_matrix = df[numeric_cols].corr()

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
        ax4.set_title('Matriz de Correlación entre Balotas')
        st.pyplot(fig4)

        # --- Sección 5: Análisis Temporal de la Suma de Balotas ---
        st.subheader("⏳ Tendencia de la Suma de Balotas a lo largo del Tiempo")
        st.write("Esta gráfica muestra cómo ha evolucionado la suma de las cinco balotas principales en cada sorteo a lo largo de los años.")

        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='Fecha', y='SumaBalotas', data=df, ax=ax5, color='darkgreen')
        ax5.set_title('Suma de Balotas a lo largo del Tiempo')
        ax5.set_xlabel('Fecha del Sorteo')
        ax5.set_ylabel('Suma de las Balotas')
        st.pyplot(fig5)

        # --- Sección 6: Suma de Balotas por Año ---
        st.subheader("📅 Distribución de la Suma de Balotas por Año")
        st.write("Los diagramas de caja muestran la distribución de la suma de las balotas para cada año, incluyendo medianas, cuartiles y valores atípicos.")

        fig6, ax6 = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Año', y='SumaBalotas', data=df, ax=ax6, palette='Pastel1')
        ax6.set_title('Suma de Balotas por Año')
        ax6.set_xlabel('Año del Sorteo')
        ax6.set_ylabel('Suma de las Balotas')
        st.pyplot(fig6)

    with tab2:
        st.header("🤖 Interacción con Gemini AI")
        st.write("Aquí puedes usar la inteligencia artificial de Google Gemini para obtener insights, predicciones o análisis adicionales basados en los datos del Baloto.")

        if model:
            st.subheader("Generar una 'predicción' o insight")
            st.markdown("""
            **Descargo de responsabilidad:** La IA de Gemini puede generar texto predictivo basado en patrones, pero **no puede predecir números de lotería reales**. Los sorteos de lotería son eventos de probabilidad puramente aleatorios.
            """)

            # Obtener los últimos resultados para el prompt
            latest_results = df.sort_values(by='Fecha', ascending=False).head(5)
            latest_results_str = latest_results.to_string(index=False)

            prompt = st.text_area(
                "Ingresa tu pregunta o solicitud para Gemini sobre los datos del Baloto:",
                f"Basado en los siguientes últimos resultados del Baloto:\n\n{latest_results_str}\n\n"
                "¿Podrías identificar alguna tendencia interesante o sugerir un posible conjunto de números (puramente por curiosidad y sin garantía de ser ganadores) y justificar tu razonamiento? Ten en cuenta que las balotas 1 a 5 están ordenadas numéricamente y la SuperBalota es independiente. Considera los rangos de balotas (1-43) y SuperBalota (1-16)."
            )

            if st.button("Generar con Gemini"):
                with st.spinner("Generando respuesta..."):
                    try:
                        response = model.generate_content(prompt)
                        st.markdown("**Respuesta de Gemini:**")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error al comunicarse con la API de Gemini: {e}")
                        st.info("Esto puede deberse a un límite de cuota, un problema de red, o un problema con el prompt.")
        else:
            st.warning("La funcionalidad de Gemini AI no está disponible. Por favor, configura tu API Key.")

    with tab3:
        st.header("ℹ️ Acerca de esta Aplicación")
        st.write("""
        Esta aplicación de Streamlit fue creada para realizar un Análisis Exploratorio de Datos (EDA) sobre los resultados históricos del Baloto colombiano.
        Los datos se cargan directamente desde un archivo CSV alojado en GitHub.

        **Características Clave:**
        * Visualización de distribuciones de frecuencia de balotas individuales y SuperBalota.
        * Identificación de las balotas más frecuentes.
        * Análisis de correlación entre las posiciones de las balotas.
        * Tendencias de la suma de balotas a lo largo del tiempo.
        * **Integración con Google Gemini AI** para explorar insights adicionales y "predicciones" (puramente con fines ilustrativos y de entretenimiento, ya que las loterías son aleatorias).

        **Desarrollado por:** Julian Torres (con asistencia de un modelo de lenguaje de Google).
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Google_Gemini_logo.svg/1200px-Google_Gemini_logo.svg.png", width=150)
        st.write("Los resultados de Baloto son aleatorios. Por favor, juega responsablemente.")

else:
    st.error("No se pudieron cargar los datos del Baloto. Por favor, asegúrate de que la URL sea correcta y el archivo esté accesible.")
    st.info("Intenta revisar la URL del archivo en tu repositorio de GitHub o la conexión a internet. Si el problema persiste, verifica el formato del CSV.")

st.markdown("---")
st.write("¡Gracias por usar la aplicación!")
