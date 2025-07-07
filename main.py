import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import google.generativeai as genai
import numpy as np # Importar numpy para opciones de muestreo

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Análisis y Predicción de Baloto con Gemini AI",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Análisis Exploratorio de Datos y Predicción de Baloto con Gemini AI 🇨🇴")
st.write("Bienvenido al panel interactivo de análisis de los resultados históricos del Baloto colombiano. Explora tendencias pasadas y experimenta con la IA de Gemini para posibles predicciones o insights.")

# --- Configuración de la API Key de Gemini ---
gemini_api_key = "AIzaSyAo1mZnBvslWoUKot7svYIo2K3fZIrLgRk" # ¡TU API KEY AQUÍ!

try:
    genai.configure(api_key=gemini_api_key) 
    model = genai.GenerativeModel('gemini-1.5-flash') # Usando gemini-1.5-flash para mejor disponibilidad
    st.success("API de Gemini configurada exitosamente con 'gemini-1.5-flash'.")
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
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month
        df['Dia'] = df['Fecha'].dt.day
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo desde {data_url}: {e}")
        return pd.DataFrame()

# --- Cargar los datos ---
df = load_data(url)

# --- Funciones de Pronóstico y Simulación (sin cambios aquí) ---
# ... (Mantén las funciones generate_montecarlo_draw y get_hot_numbers_recommendation tal cual) ...
def generate_montecarlo_draw(df, num_simulations=10000):
    """
    Genera combinaciones de baloto usando Montecarlo,
    respetando el orden y las distribuciones históricas de cada balota.
    """
    balota_cols = [f'Balota {i}' for i in range(1, 6)]
    simulated_draws = []

    for _ in range(num_simulations):
        current_draw = []
        prev_num = 0 # Para asegurar el orden ascendente

        for i in range(1, 6):
            col_name = f'Balota {i}'
            
            # Filtrar candidatos: deben ser mayores que el número anterior
            # y dejar espacio para las balotas restantes (max_balota - (5 - i + 1))
            min_allowed = prev_num + 1
            max_allowed = 43 - (5 - i) # Asegurarse de que queden números suficientes para las siguientes balotas
            
            candidates = df[col_name][(df[col_name] >= min_allowed) & (df[col_name] <= max_allowed)].unique()
            
            if len(candidates) == 0:
                candidates = np.arange(min_allowed, max_allowed + 1)
                if len(candidates) == 0:
                    current_draw = []
                    break

            frequencies = df[col_name].value_counts(normalize=True).sort_index()
            
            weights = [frequencies.get(n, 0.0001) for n in candidates]
            weights_sum = sum(weights)
            if weights_sum == 0:
                weights = [1/len(candidates)] * len(candidates)
            else:
                weights = [w / weights_sum for w in weights]

            chosen_num = np.random.choice(candidates, p=weights)
            current_draw.append(chosen_num)
            prev_num = chosen_num
        
        if len(current_draw) == 5:
            superbalota_frequencies = df['SuperBalota'].value_counts(normalize=True).sort_index()
            sb_candidates = np.arange(1, 17)
            sb_weights = [superbalota_frequencies.get(n, 0.0001) for n in sb_candidates]
            sb_weights_sum = sum(sb_weights)
            if sb_weights_sum == 0:
                sb_chosen = np.random.choice(sb_candidates)
            else:
                sb_weights = [w / sb_weights_sum for w in sb_weights]
                sb_chosen = np.random.choice(sb_candidates, p=sb_weights)
            
            simulated_draws.append((tuple(current_draw), sb_chosen))

    if not simulated_draws:
        return None, "No se pudieron generar sorteos simulados que cumplan las reglas. Intente aumentar el número de simulaciones."

    draw_counts = pd.Series(simulated_draws).value_counts().head(5)
    return draw_counts, None


def get_hot_numbers_recommendation(df):
    """
    Genera una recomendación de balotas basadas en los números más frecuentes
    para cada posición, respetando el orden.
    """
    recommended_balotas = []
    
    prev_num = 0
    for i in range(1, 6):
        col_name = f'Balota {i}'
        
        min_allowed = prev_num + 1
        max_allowed = 43 - (5 - i) 
        
        possible_numbers = df[col_name][(df[col_name] >= min_allowed) & (df[col_name] <= max_allowed)]
        
        if possible_numbers.empty:
            chosen_num = min_allowed
        else:
            chosen_num = possible_numbers.value_counts().sort_index(ascending=True).idxmax()

        recommended_balotas.append(chosen_num)
        prev_num = chosen_num

    hot_superbalota = df['SuperBalota'].value_counts().idxmax()
    
    return recommended_balotas, hot_superbalota

# --- Verificar si los datos se cargaron correctamente ---
if not df.empty:
    st.success("¡Datos de Baloto cargados exitosamente! Fecha del último sorteo registrado: " + df['Fecha'].max().strftime('%d/%m/%Y'))

    # --- Pestañas para organizar el contenido ---
    tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🤖 Pronósticos y Simulación", "ℹ️ Acerca de"])

    with tab1:
        st.header("Análisis Exploratorio de Datos Históricos")

        st.subheader("🔍 Primeras Filas del Conjunto de Datos")
        st.dataframe(df.head())

        st.subheader("📊 Información General y Estadísticas Descriptivas")
        st.write("Resumen de los tipos de datos, valores no nulos y uso de memoria.")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write("Estadísticas descriptivas básicas para las balotas:")
        st.dataframe(df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].describe())

        st.header("🔥 Mapa de Calor Consolidado por Balota")
        st.write("Explora la distribución consolidada de números para cada posición de balota, eligiendo entre el conteo de apariciones, el promedio o la mediana.")

        metric_selection = st.radio(
            "Selecciona la métrica a visualizar:",
            ('Conteo', 'Promedio', 'Mediana'),
            horizontal=True
        )

        balotas_reg_cols = [f'Balota {i}' for i in range(1, 6)]
        
        if metric_selection == 'Conteo':
            df_melted_balotas = df[balotas_reg_cols].melt(var_name='Balota', value_name='Numero')
            heatmap_data_regular = pd.crosstab(df_melted_balotas['Numero'], df_melted_balotas['Balota'])
            heatmap_data_regular = heatmap_data_regular.reindex(columns=balotas_reg_cols)
            heatmap_data_regular = heatmap_data_regular.fillna(0)
            
            superbalota_counts = df['SuperBalota'].value_counts().sort_index()
            superbalota_full_range = pd.Series(0, index=range(1, 17))
            superbalota_full_range.update(superbalota_counts)
            
            max_num_regular = 43
            all_numbers = pd.RangeIndex(start=1, stop=max_num_regular + 1)
            
            heatmap_data_regular = heatmap_data_regular.reindex(all_numbers, fill_value=0)

            heatmap_final = heatmap_data_regular.copy()
            heatmap_final['SuperBalota'] = superbalota_full_range.reindex(all_numbers, fill_value=pd.NA)

            fig7, ax7 = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                heatmap_final,
                annot=True,
                fmt=".0f",
                cmap='viridis',
                linewidths=.5,
                linecolor='black',
                ax=ax7
            )
            ax7.set_title(f'Mapa de Calor Consolidado: {metric_selection} por Número y Balota')
            ax7.set_xlabel('Tipo de Balota')
            ax7.set_ylabel('Número de Balota')
            st.pyplot(fig7)
        else:
            st.info("Para un 'Mapa de Calor Consolidado por Balota' (Número vs Posición), la métrica de 'Conteo' es la más significativa. El 'Promedio' o 'Mediana' de los números en sí mismos no tienen una variación útil en esta vista. Los gráficos de tendencias por año ya muestran promedios y medianas a lo largo del tiempo. Por favor, selecciona 'Conteo' para ver el mapa de calor.")
            

        st.subheader("📈 Distribución de Frecuencia de las Balotas")
        st.write("Estos histogramas muestran la frecuencia con la que ha aparecido cada número **en su respectiva posición de balota** y en la SuperBalota. Recuerda que las balotas 1 a 5 están ordenadas numéricamente.")

        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
        axes1 = axes1.flatten()

        for i in range(1, 6):
            sns.histplot(df[f'Balota {i}'], bins=range(1, 44), kde=True, ax=axes1[i-1], color='skyblue')
            axes1[i-1].set_title(f'Distribución Balota {i} (1-43)')
            axes1[i-1].set_xlabel('Número')
            axes1[i-1].set_ylabel('Frecuencia')
            axes1[i-1].set_xticks(range(1, 44, 4))

        sns.histplot(df['SuperBalota'], bins=range(1, 17), kde=True, ax=axes1[5], color='lightcoral')
        axes1[5].set_title('Distribución SuperBalota (1-16)')
        axes1[5].set_xlabel('Número')
        axes1[5].set_ylabel('Frecuencia')
        axes1[5].set_xticks(range(1, 17))

        plt.tight_layout()
        st.pyplot(fig1)

        st.subheader("⭐ Balotas Más Frecuentes")
        st.write("Identifica los números que han sido los más 'afortunados' en la historia del Baloto, considerando todas las posiciones para las balotas regulares.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Top 10 Balotas Regulares (Global)")
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

        st.subheader("🔗 Matriz de Correlación entre Balotas")
        st.write("Aunque las balotas de un sorteo individual son independientes, esta matriz muestra si existe alguna correlación numérica **observada** entre las *posiciones* de las balotas a lo largo del tiempo, teniendo en cuenta su orden ascendente.")
        numeric_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
        correlation_matrix = df[numeric_cols].corr()

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
        ax4.set_title('Matriz de Correlación entre Balotas')
        st.pyplot(fig4)

        st.subheader("⏳ Tendencia Anual del Promedio de Cada Balota")
        st.write("Esta gráfica muestra cómo ha variado el **promedio de los números** para cada balota (Balota 1 a Balota 5) y la SuperBalota a lo largo de los años. Esto puede indicar si los números tendieron a ser más altos o bajos en ciertos años para cada posición.")

        df_avg_by_year = df.groupby('Año')[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].mean().reset_index()

        fig5, ax5 = plt.subplots(figsize=(14, 7))
        df_avg_by_year_melted = df_avg_by_year.melt('Año', var_name='Balota', value_name='Promedio')
        sns.lineplot(x='Año', y='Promedio', hue='Balota', data=df_avg_by_year_melted, marker='o', ax=ax5)
        ax5.set_title('Promedio Anual de los Números para Cada Balota')
        ax5.set_xlabel('Año')
        ax5.set_ylabel('Promedio del Número')
        ax5.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig5)

        st.subheader("📅 Distribución Anual de Números por Balota")
        st.write("Estos diagramas de caja muestran la distribución de los números para cada balota (Balota 1 a Balota 5) y la SuperBalota, agrupados por año. Puedes observar la mediana, los cuartiles y los valores atípicos.")

        fig6, axes6 = plt.subplots(2, 3, figsize=(18, 12))
        axes6 = axes6.flatten()

        balota_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
        for i, col in enumerate(balota_cols):
            sns.boxplot(x='Año', y=col, data=df, ax=axes6[i], palette='Pastel1')
            axes6[i].set_title(f'Distribución Anual de {col}')
            axes6[i].set_xlabel('Año')
            axes6[i].set_ylabel('Número')
            axes6[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig6)

    with tab2:
        st.header("🤖 Herramientas de Pronóstico y Simulación")
        st.write("Aquí puedes explorar diferentes enfoques para generar posibles combinaciones de Baloto, incluyendo simulaciones y recomendaciones basadas en datos históricos. **Recuerda:** Los sorteos de lotería son aleatorios y estas herramientas son para fines de entretenimiento y análisis, no garantizan resultados.")

        if model:
            st.subheader("1. Pregunta a Gemini AI")
            st.markdown("""
            Usa la inteligencia artificial de Google Gemini para obtener insights o sugerencias de números. Gemini intentará seguir tus instrucciones de orden y rango, **basándose en el resumen histórico proporcionado**.
            """)

            latest_results = df.sort_values(by='Fecha', ascending=False).head(5)
            latest_results_str = latest_results.to_string(index=False)

            # --- PREPARACIÓN DE DATOS HISTÓRICOS PARA EL PROMPT ---
            all_balotas = pd.concat([df['Balota 1'], df['Balota 2'], df['Balota 3'], df['Balota 4'], df['Balota 5']])
            top_10_balotas_global = all_balotas.value_counts().head(10)
            top_balotas_str = ", ".join([f"{num} ({freq} veces)" for num, freq in top_10_balotas_global.items()])

            top_10_superbalotas = df['SuperBalota'].value_counts().head(10)
            top_superbalotas_str = ", ".join([f"{num} ({freq} veces)" for num, freq in top_10_superbalotas.items()])
            # --- FIN DE PREPARACIÓN DE DATOS ---

            # --- EL PROMPT CLAVE CON LA INFORMACIÓN HISTÓRICA ADICIONAL ---
            prompt = st.text_area(
                "Ingresa tu pregunta o solicitud para Gemini sobre los datos del Baloto:",
                f"Basado en los siguientes últimos 5 resultados del Baloto:\n\n{latest_results_str}\n\n"
                f"**Información Histórica Adicional:**\n"
                f"- Los números de Balota regular más frecuentes históricamente (en cualquier posición, del 1 al 43) son: {top_balotas_str}.\n"
                f"- Los números de SuperBalota más frecuentes históricamente (del 1 al 16) son: {top_superbalotas_str}.\n\n"
                "Estoy buscando un posible conjunto de 5 números de balota y 1 SuperBalota. "
                "Las 5 balotas deben estar en el rango de 1 a 43 y **estrictamente ordenadas de forma ascendente (Balota 1 < Balota 2 < Balota 3 < Balota 4 < Balota 5)**. "
                "La SuperBalota debe estar en el rango de 1 a 16 y es independiente de las otras 5. "
                "Por favor, sugiere un conjunto de números y justifica brevemente tu razonamiento, basándote en los datos proporcionados (últimos resultados y números frecuentes históricos). "
                "Formato de salida deseado: Balotas: [N1, N2, N3, N4, N5], SuperBalota: [SB]."
            )
            # --- FIN DEL PROMPT CLAVE ---

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
        
        st.markdown("---")

        st.subheader("2. Simulación de Montecarlo")
        st.write("Genera sorteos hipotéticos basados en las probabilidades históricas de aparición de cada número, respetando el orden. Esto muestra las combinaciones que *podrían* ser más probables si el pasado influyera en el futuro.")
        
        num_sims = st.slider("Número de simulaciones (Montecarlo):", min_value=1000, max_value=100000, value=20000, step=1000)

        if st.button("Ejecutar Simulación Montecarlo"):
            if not df.empty:
                with st.spinner(f"Ejecutando {num_sims} simulaciones de Montecarlo..."):
                    most_frequent_sims, error_msg = generate_montecarlo_draw(df, num_sims)
                    if most_frequent_sims is not None:
                        st.markdown("##### Top 5 Combinaciones Más Frecuentes en la Simulación:")
                        for (balotas_tuple, superbalota_num), count in most_frequent_sims.items():
                            st.write(f"- Balotas: {list(balotas_tuple)}, SuperBalota: {superbalota_num} (Apareció {count} veces)")
                        st.info("Estas combinaciones son las que se generaron más a menudo en el conjunto de simulaciones, respetando las distribuciones históricas y el orden.")
                    else:
                        st.error(error_msg)
            else:
                st.error("No se pueden ejecutar simulaciones sin datos cargados.")
        
        st.markdown("---")

        st.subheader("3. Recomendación de Números 'Calientes'")
        st.write("Obtén una combinación de balotas seleccionando los números más frecuentes para cada posición de balota, asegurando que se cumpla el orden ascendente.")

        if st.button("Generar Números 'Calientes'"):
            if not df.empty:
                recommended_balotas, recommended_superbalota = get_hot_numbers_recommendation(df)
                st.markdown("##### Tu Combinación 'Caliente' Sugerida:")
                st.write(f"**Balotas:** {recommended_balotas}")
                st.write(f"**SuperBalota:** {recommended_superbalota}")
                st.info("Esta combinación se construye eligiendo el número más frecuente para cada posición que cumple la condición de ser mayor que el número anterior.")
            else:
                st.error("No se pueden generar recomendaciones sin datos cargados.")


    with tab3:
        st.header("ℹ️ Acerca de esta Aplicación")
        st.write("""
        Esta aplicación de Streamlit fue creada para realizar un Análisis Exploratorio de Datos (EDA) sobre los resultados históricos del Baloto colombiano.
        Los datos se cargan directamente desde un archivo CSV alojado en GitHub.

        **Características Clave:**
        * Visualización de distribuciones de frecuencia de balotas individuales y SuperBalota.
        * Identificación de las balotas más frecuentes.
        * Análisis de correlación entre las posiciones de las balotas.
        * Tendencias del promedio y la distribución de números para *cada balota individualmente* a lo largo del tiempo y por año.
        * Mapa de Calor Consolidado para visualizar el conteo de números por balota.
        * **Nuevas herramientas de Pronóstico y Simulación:**
            * **Integración con Google Gemini AI** para explorar insights adicionales y sugerencias de números.
            * **Simulación de Montecarlo** para generar combinaciones hipotéticas basadas en probabilidades históricas.
            * **Recomendación de Números 'Calientes'** basada en la frecuencia de aparición por posición.

        **Desarrollado por:** Julian Torres (con asistencia de un modelo de lenguaje de Google).
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Google_Gemini_logo.svg/1200px-Google_Gemini_logo.svg.png", width=150)
        st.write("Los resultados de Baloto son aleatorios. Por favor, juega responsablemente.")

else:
    st.error("No se pudieron cargar los datos del Baloto. Por favor, asegúrate de que la URL sea correcta y el archivo esté accesible.")
    st.info("Intenta revisar la URL del archivo en tu repositorio de GitHub o la conexión a internet. Si el problema persiste, verifica el formato del CSV.")

st.markdown("---")
st.write("¡Gracias por usar la aplicación!")
