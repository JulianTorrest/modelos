import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import google.generativeai as genai

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Análisis y Predicción de Baloto con Gemini AI",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Análisis Exploratorio de Datos y Predicción de Baloto con Gemini AI 🇨🇴")
st.write("Bienvenido al panel interactivo de análisis de los resultados históricos del Baloto colombiano. Explora tendencias pasadas y experimenta con la IA de Gemini para posibles predicciones o insights.")

# --- Configuración de la API Key de Gemini ---
# ADVERTENCIA: La API Key está hardcodeada directamente en el código.
# Esto NO es una práctica recomendada para producción debido a riesgos de seguridad.
# Sin embargo, se hace a petición explícita.
gemini_api_key = "AIzaSyAo1mZnBvslWoUKot7svYIo2K3fZIrLgRk" # ¡TU API KEY AQUÍ!

try:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    st.success("API de Gemini configurada exitosamente (API Key incrustada en el código).")
except Exception as e:
    st.error(f"Error al configurar la API de Gemini: {e}")
    st.warning("La funcionalidad de Gemini AI podría no estar disponible.")
    model = None

# --- URL del archivo CSV en GitHub ---
url = "https://raw.githubusercontent.com/JulianTorrest/modelos/refs/heads/main/Baloto.csv"

# --- Función para Cargar y Preprocesar los Datos (con caché) ---
@st.cache_data
def load_data(data_url):
    """
    Carga el archivo CSV desde la URL, convierte la columna de fecha
    y extrae el año, mes y día.
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
        # No se calcula SumaBalotas ya que el análisis ha cambiado
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

        # --- Sección 5: Tendencia del Promedio de Cada Balota por Año ---
        st.subheader("⏳ Tendencia Anual del Promedio de Cada Balota")
        st.write("Esta gráfica muestra cómo ha variado el **promedio de los números** para cada balota (Balota 1 a Balota 5) y la SuperBalota a lo largo de los años. Esto puede indicar si los números tendieron a ser más altos o bajos en ciertos años para cada posición.")

        # Calcular el promedio de cada balota por año
        df_avg_by_year = df.groupby('Año')[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].mean().reset_index()

        fig5, ax5 = plt.subplots(figsize=(14, 7))
        # Melt el dataframe para usar Seaborn.lineplot con múltiples líneas
        df_avg_by_year_melted = df_avg_by_year.melt('Año', var_name='Balota', value_name='Promedio')
        sns.lineplot(x='Año', y='Promedio', hue='Balota', data=df_avg_by_year_melted, marker='o', ax=ax5)
        ax5.set_title('Promedio Anual de los Números para Cada Balota')
        ax5.set_xlabel('Año')
        ax5.set_ylabel('Promedio del Número')
        ax5.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig5)

        # --- Sección 6: Distribución de Números por Balota y Año (Boxplots) ---
        st.subheader("📅 Distribución Anual de Números por Balota")
        st.write("Estos diagramas de caja muestran la distribución de los números para cada balota (Balota 1 a Balota 5) y la SuperBalota, agrupados por año. Puedes observar la mediana, los cuartiles y los valores atípicos.")

        # Crear subplots para cada balota + SuperBalota
        fig6, axes6 = plt.subplots(2, 3, figsize=(18, 12))
        axes6 = axes6.flatten()

        balota_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
        for i, col in enumerate(balota_cols):
            sns.boxplot(x='Año', y=col, data=df, ax=axes6[i], palette='Pastel1')
            axes6[i].set_title(f'Distribución Anual de {col}')
            axes6[i].set_xlabel('Año')
            axes6[i].set_ylabel('Número')
            axes6[i].tick_params(axis='x', rotation=45) # Rotar etiquetas para años si son muchos

        plt.tight_layout()
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
            st.warning("La funcionalidad de Gemini AI no está disponible debido a un error de configuración de la API Key.")

    with tab3:
        st.header("ℹ️ Acerca de esta Aplicación")
        st.write("""
        Esta aplicación de Streamlit fue creada para realizar un Análisis Exploratorio de Datos (EDA) sobre los resultados históricos del Baloto colombiano.
        Los datos se cargan directamente desde un archivo CSV alojado en GitHub.

        **Características Clave:**
        * Visualización de distribuciones de frecuencia de balotas individuales y SuperBalota.
        * Identificación de las balotas más frecuentes.
        * Análisis de correlación entre las posiciones de las balotas.
        * **Nuevas tendencias:** Análisis del promedio y la distribución de números para *cada balota individualmente* a lo largo del tiempo y por año.
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
