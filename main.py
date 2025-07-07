import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import google.generativeai as genai
import numpy as np
import re
from itertools import combinations

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
    model = genai.GenerativeModel('gemini-1.5-flash')
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

# --- Inicializar session_state para almacenar resultados ---
if 'all_forecasts' not in st.session_state:
    st.session_state.all_forecasts = []
if 'gemini_raw_response' not in st.session_state:
    st.session_state.gemini_raw_response = ""

# --- Funciones de Utilidad ---
@st.cache_data
def calculate_historical_frequency_score(df, balotas_list, superbalota_num):
    """
    Calcula un puntaje basado en la suma de las frecuencias históricas
    de cada número en su respectiva posición de balota.
    """
    score = 0
    if not isinstance(balotas_list, list) or len(balotas_list) != 5:
        return 0

    for i in range(5):
        col_name = f'Balota {i+1}'
        score += df[col_name].value_counts().get(balotas_list[i], 0)

    score += df['SuperBalota'].value_counts().get(superbalota_num, 0)
    return score

@st.cache_data
def get_max_possible_score(df):
    """
    Calcula el score máximo posible buscando la combinación de 5 balotas ordenadas
    que maximiza la suma de sus frecuencias posicionales, más la SuperBalota más frecuente.
    """
    max_score = 0
    
    # Calcular las frecuencias de cada número por posición
    freq_by_pos = {}
    for i in range(1, 6):
        freq_by_pos[f'Balota {i}'] = df[f'Balota {i}'].value_counts()
    superbalota_freq = df['SuperBalota'].value_counts()
    
    # Obtener el número más frecuente de la SuperBalota
    max_sb_freq = superbalota_freq.max() if not superbalota_freq.empty else 0

    # Iterar sobre todas las combinaciones posibles de 5 números (1-43) ordenados
    # Esto es computacionalmente INTENSO. Para loterías grandes, es mejor aproximar o limitar.
    # Baloto (43,5) son 962,598 combinaciones posibles. Sumar frecuencias para cada una.
    # Podríamos simplificar tomando los N más frecuentes para cada posición.
    
    # Aproximación: encontrar la combinación de 5 balotas ordenadas
    # que maximice el score, sin recorrer *todas* las combinaciones,
    # sino construyéndolas a partir de los números más frecuentes.
    
    # Una forma más eficiente es usar la función get_hot_numbers_recommendations
    # para obtener la combinación "pura" más caliente y usar su score como base,
    # aunque no es el máximo teórico global, es el máximo bajo la lógica de "hot numbers".
    # Sin embargo, para un *máximo posible*, debemos considerar todas las balotas.

    # Esto es una simplificación razonable para evitar un cálculo excesivo
    # y sigue la lógica de los números más frecuentes en cada posición.
    # No es el *máximo global* de todas las 962k combinaciones,
    # sino el score de la combinación construida con los números más frecuentes
    # por posición, manteniendo el orden.
    
    current_balotas = []
    prev_num = 0
    for i in range(1, 6):
        col_name = f'Balota {i}'
        max_freq_num_for_pos = 0
        best_num = 0
        
        # Iterar sobre números válidos para la posición, mayores al anterior
        for num in range(prev_num + 1, 44 - (5 - i)): # 44-(5-i) es el límite superior para dejar espacio para los siguientes
            freq = freq_by_pos[col_name].get(num, 0)
            if freq > max_freq_num_for_pos:
                max_freq_num_for_pos = freq
                best_num = num
        
        if best_num == 0: # Fallback if no specific hot number found or range issue
            best_num = prev_num + 1 # Take the next valid number
            max_freq_num_for_pos = freq_by_pos[col_name].get(best_num, 0) # Use its freq

        current_balotas.append(best_num)
        prev_num = best_num
        max_score += max_freq_num_for_pos
    
    max_score += max_sb_freq # Sumar la frecuencia del SB más frecuente
    
    return max_score


# Pre-calcular el score máximo posible una vez
# Asegúrate de que df no esté vacío antes de llamar a esto
MAX_POSSIBLE_SCORE = 0
if not df.empty:
    MAX_POSSIBLE_SCORE = get_max_possible_score(df)
    if MAX_POSSIBLE_SCORE == 0: # Fallback si no se pudo calcular un score máximo (e.g., df muy pequeño)
        st.warning("No se pudo calcular un Score Máximo Posible válido. El 'Nivel de Calidez' no se mostrará.")

# --- Funciones de Pronóstico y Simulación Actualizadas ---

def generate_montecarlo_draws(df, num_simulations=10000):
    """
    Genera 5 combinaciones de baloto usando Montecarlo,
    respetando el orden y las distribuciones históricas de cada balota,
    y calcula su puntaje de frecuencia.
    """
    balota_cols = [f'Balota {i}' for i in range(1, 6)]
    simulated_draws_with_scores = []

    temp_simulated_draws = []
    for _ in range(num_simulations):
        current_draw = []
        prev_num = 0

        for i in range(1, 6):
            col_name = f'Balota {i}'
            min_allowed = prev_num + 1
            max_allowed = 43 - (5 - i)
            
            candidates = df[col_name][(df[col_name] >= min_allowed) & (df[col_name] <= max_allowed)].unique()
            
            if len(candidates) == 0:
                # Fallback: if no historical data for specific range, use general range
                candidates = np.arange(min_allowed, max_allowed + 1)
                if len(candidates) == 0:
                    current_draw = []
                    break # Break inner loop, invalid combination

            frequencies = df[col_name].value_counts(normalize=True).sort_index()
            # Ensure all candidates have a weight; if not in frequencies, assign a small non-zero
            weights = [frequencies.get(n, 0.0001) for n in candidates]
            weights_sum = sum(weights)
            if weights_sum == 0: # Should not happen with 0.0001 fallback, but safety
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
            
            temp_simulated_draws.append((tuple(current_draw), sb_chosen))

    if not temp_simulated_draws:
        st.warning("No se pudieron generar sorteos simulados que cumplan las reglas. Intente aumentar el número de simulaciones.")
        return []

    draw_counts = pd.Series(temp_simulated_draws).value_counts()
    
    unique_draws = []
    seen_combinations = set() # To store (balotas_tuple, superbalota) for strict uniqueness
    
    for (balotas_tuple, superbalota_num) in draw_counts.index:
        if len(unique_draws) >= 5:
            break
        
        combo_key = (balotas_tuple, superbalota_num)
        if combo_key not in seen_combinations:
            score = calculate_historical_frequency_score(df, list(balotas_tuple), superbalota_num)
            calidez = (score / MAX_POSSIBLE_SCORE * 100) if MAX_POSSIBLE_SCORE > 0 else 0
            unique_draws.append({
                'balotas': list(balotas_tuple),
                'superbalota': superbalota_num,
                'score': score,
                'calidez_pct': calidez,
                'method': 'Monte Carlo'
            })
            seen_combinations.add(combo_key)
    
    return unique_draws


def get_hot_numbers_recommendations(df):
    """
    Genera 5 recomendaciones de balotas basadas en los números más frecuentes
    para cada posición, respetando el orden, y calcula su puntaje.
    """
    recommendations = []
    seen_combinations = set()

    # Pre-calcular las frecuencias por posición para un acceso rápido
    freq_by_pos = {}
    for i in range(1, 6):
        freq_by_pos[f'Balota {i}'] = df[f'Balota {i}'].value_counts()
    superbalota_freq = df['SuperBalota'].value_counts()
    
    hot_superbalota_candidates = superbalota_freq.index.tolist()
    if not hot_superbalota_candidates: hot_superbalota_candidates = list(range(1, 17))

    # --- Generar la combinación más caliente "pura" (la primera) ---
    current_balotas = []
    prev_num = 0
    for i in range(1, 6):
        col_name = f'Balota {i}'
        possible_numbers = freq_by_pos[col_name].index.tolist()
        
        # Filter for order and range
        valid_candidates = [n for n in possible_numbers if n > prev_num and n <= (43 - (5 - i))]
        
        if not valid_candidates:
            # Fallback to next possible number if no hot number fits
            chosen_num = prev_num + 1
        else:
            # Pick the hottest among valid candidates
            chosen_num = max(valid_candidates, key=lambda x: freq_by_pos[col_name].get(x, 0))
        
        current_balotas.append(chosen_num)
        prev_num = chosen_num
    
    hot_superbalota = hot_superbalota_candidates[0] if hot_superbalota_candidates else 1 # Fallback
    
    score = calculate_historical_frequency_score(df, current_balotas, hot_superbalota)
    calidez = (score / MAX_POSSIBLE_SCORE * 100) if MAX_POSSIBLE_SCORE > 0 else 0
    
    recommendation = {
        'balotas': current_balotas,
        'superbalota': hot_superbalota,
        'score': score,
        'calidez_pct': calidez,
        'method': 'Números Calientes'
    }
    recommendations.append(recommendation)
    seen_combinations.add((tuple(current_balotas), hot_superbalota))

    # --- Generar 4 variaciones "calientes" ---
    generated_count = 0
    max_attempts = 200 # Para evitar bucles infinitos

    while generated_count < 4 and max_attempts > 0:
        temp_balotas = []
        prev_num = 0
        valid_combination = True

        for i in range(1, 6):
            col_name = f'Balota {i}'
            # Take top N candidates for this position, then filter by order/range
            top_n_candidates = freq_by_pos[col_name].head(5).index.tolist() # Top 5 hot numbers for this position
            
            candidates_for_pos = [n for n in top_n_candidates if n > prev_num and n <= (43 - (5 - i))]
            
            if not candidates_for_pos:
                # Fallback to any number in range if no hot candidate fits
                candidates_for_pos = list(range(prev_num + 1, 44 - (5 - i)))
                if not candidates_for_pos: # If still no candidates, this path is dead
                    valid_combination = False
                    break
            
            chosen_num = np.random.choice(candidates_for_pos)
            temp_balotas.append(chosen_num)
            prev_num = chosen_num
        
        if valid_combination and len(temp_balotas) == 5:
            temp_superbalota = np.random.choice(hot_superbalota_candidates)
            
            combo_key = (tuple(temp_balotas), temp_superbalota)
            if combo_key not in seen_combinations:
                score = calculate_historical_frequency_score(df, temp_balotas, temp_superbalota)
                calidez = (score / MAX_POSSIBLE_SCORE * 100) if MAX_POSSIBLE_SCORE > 0 else 0
                
                recommendations.append({
                    'balotas': temp_balotas,
                    'superbalota': temp_superbalota,
                    'score': score,
                    'calidez_pct': calidez,
                    'method': 'Números Calientes'
                })
                seen_combinations.add(combo_key)
                generated_count += 1
        max_attempts -= 1
    
    return recommendations


def get_gemini_recommendations(df, model, latest_results_str, top_balotas_str, top_superbalotas_str):
    """
    Obtiene 5 recomendaciones de Gemini AI y calcula su puntaje.
    """
    if not model:
        st.warning("Gemini AI no está disponible para generar recomendaciones.")
        return []

    gemini_forecasts = []
    
    # --- EL PROMPT CLAVE CON LA INFORMACIÓN HISTÓRICA ADICIONAL ---
    prompt = (
        f"Basado en los siguientes últimos 5 resultados del Baloto:\n\n{latest_results_str}\n\n"
        f"**Información Histórica Adicional:**\n"
        f"- Los números de Balota regular más frecuentes históricamente (en cualquier posición, del 1 al 43) son: {top_balotas_str}.\n"
        f"- Los números de SuperBalota más frecuentes históricamente (del 1 al 16) son: {top_superbalotas_str}.\n\n"
        "Por favor, sugiere **5 conjuntos distintos** de 5 números de balota y 1 SuperBalota. "
        "Para cada conjunto, las 5 balotas deben estar en el rango de 1 a 43 y **estrictamente ordenadas de forma ascendente (Balota 1 < Balota 2 < Balota 3 < Balota 4 < Balota 5)**. "
        "La SuperBalota debe estar en el rango de 1 a 16 y es independiente de las otras 5. "
        "Justifica brevemente tu razonamiento para cada conjunto, basándote en los datos proporcionados (últimos resultados y números frecuentes históricos). "
        "**Formato de salida deseado para cada conjunto (importante para el parsing):**\n"
        "**Conjunto N:** Balotas: [N1, N2, N3, N4, N5], SuperBalota: [SB]. Razón: [Tu justificación]\n"
        "Asegúrate de que cada conjunto sea único y siga el formato exacto."
    )
    # --- FIN DEL PROMPT CLAVE ---

    try:
        response = model.generate_content(prompt)
        st.session_state.gemini_raw_response = response.text # Guardar la respuesta cruda para depuración
        
        pattern = r"Balotas: \[(\d{1,2}), (\d{1,2}), (\d{1,2}), (\d{1,2}), (\d{1,2})\], SuperBalota: \[(\d{1,2})\]"
        matches = re.findall(pattern, response.text)
        
        parsed_count = 0
        seen_combinations = set()
        for match in matches:
            if parsed_count >= 5:
                break
            try:
                balotas = [int(n) for n in match[:5]]
                superbalota = int(match[5])

                # Validate ranges and order
                if (1 <= balotas[0] < balotas[1] < balotas[2] < balotas[3] < balotas[4] <= 43 and
                    1 <= superbalota <= 16):
                    
                    combo_key = (tuple(balotas), superbalota)
                    if combo_key not in seen_combinations:
                        score = calculate_historical_frequency_score(df, balotas, superbalota)
                        calidez = (score / MAX_POSSIBLE_SCORE * 100) if MAX_POSSIBLE_SCORE > 0 else 0
                        gemini_forecasts.append({
                            'balotas': balotas,
                            'superbalota': superbalota,
                            'score': score,
                            'calidez_pct': calidez,
                            'method': 'Gemini AI'
                        })
                        seen_combinations.add(combo_key)
                        parsed_count += 1
                # else: st.warning(f"Gemini sugirió una combinación inválida (rango/orden): Balotas: {balotas}, SuperBalota: {superbalota}")

            except ValueError:
                st.warning(f"No se pudo parsear una combinación de Gemini de '{match}'.")
        
        if parsed_count == 0:
            st.warning("Gemini no pudo generar combinaciones válidas en el formato esperado.")
            st.info("Respuesta cruda de Gemini (para depuración): " + response.text)

    except Exception as e:
        st.error(f"Error al comunicarse con la API de Gemini: {e}")
        st.info("Esto puede deberse a un límite de cuota, un problema de red, o un problema con el prompt.")
    
    while len(gemini_forecasts) < 5:
        gemini_forecasts.append({
            'balotas': [],
            'superbalota': None,
            'score': 0,
            'calidez_pct': 0,
            'method': 'Gemini AI (Inválido/No Generado)'
        })

    return gemini_forecasts


# --- Función para Simular Ganancias ---
def simulate_winnings(selected_balotas, selected_superbalota, historical_df, num_simulations):
    """
    Simula jugar una combinación N veces contra los sorteos históricos disponibles
    y calcula las ganancias basadas en acertar las 5 balotas y la SuperBalota
    en una fecha histórica.
    """
    if not selected_balotas or selected_superbalota is None:
        return 0, 0, "Combinaicón inválida para simulación."

    # Convertir la combinación seleccionada a un formato comparable (tuple)
    selected_combo = (tuple(selected_balotas), selected_superbalota)
    
    # Crear un DataFrame con los sorteos históricos en el mismo formato
    historical_combos = []
    for index, row in historical_df.iterrows():
        balotas_hist = tuple(sorted([row[f'Balota {i}'] for i in range(1, 6)])) # Ensure sorted
        superbalota_hist = row['SuperBalota']
        historical_combos.append((balotas_hist, superbalota_hist))
    
    historical_combos_series = pd.Series(historical_combos)
    
    winnings_count = 0
    # Simulate playing 'num_simulations' times.
    # For a historical check, this is like playing against 'num_simulations' random past draws.
    # Or, more directly, seeing how many times this specific combo *would have* hit in the history.

    # If num_simulations is very large, this is equivalent to checking against all historical combos multiple times.
    # For realism in "how many times this would win historically", we should limit to unique historical draws.
    
    # Let's clarify: "how many times that combination (or part of it) would have coincided with historical draws"
    # and "en la misma fecha todos los valores de las 5 balotas y la superbalota"
    # This means exact match of 5 balotas (ordered) AND SuperBalota.

    # Count exact matches in the historical data
    exact_matches_in_history = historical_combos_series.value_counts().get(selected_combo, 0)
    
    # Scale this by the number of simulations.
    # If a combo appeared X times in history, and we simulate N games:
    # If N <= len(historical_df), it's just X.
    # If N > len(historical_df), we assume historical patterns repeat.
    
    # Simplified approach: We assume playing N times means drawing N times from the historical pool (with replacement)
    # The probability of hitting a specific historical combo in one draw from the past is X/TotalDraws.
    # So over N simulations, expected wins = N * (X / TotalDraws)
    
    total_historical_draws = len(historical_df)
    
    if total_historical_draws == 0:
        return 0, 0, "No hay datos históricos para simular."
        
    expected_winnings_count = (exact_matches_in_history / total_historical_draws) * num_simulations
    
    # Round to nearest integer for display, or use floor for conservative estimate
    winnings_count = int(np.floor(expected_winnings_count)) # Use floor to be conservative

    total_winnings = winnings_count * 1_000_000
    
    message = f"Simulando jugar esta combinación {num_simulations} veces contra el historial de {total_historical_draws} sorteos."
    message += f" Basado en esto, se estima que habrías ganado el premio mayor {winnings_count} veces."

    return winnings_count, total_winnings, message


# --- Verificar si los datos se cargaron correctamente ---
if not df.empty:
    st.success("¡Datos de Baloto cargados exitosamente! Fecha del último sorteo registrado: " + df['Fecha'].max().strftime('%d/%m/%Y'))

    tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🤖 Pronósticos y Simulación", "ℹ️ Acerca de"])

    with tab1:
        st.header("Análisis Exploratorio de Datos Históricos")
        st.subheader("🔍 Primeras Filas del Conjunto de Datos")
        st.dataframe(df.head())
        st.subheader("📊 Información General y Estadísticas Descriptivas")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.write("Estadísticas descriptivas básicas para las balotas:")
        st.dataframe(df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].describe())

        st.header("🔥 Mapa de Calor Consolidado por Balota")
        metric_selection = st.radio("Selecciona la métrica a visualizar:", ('Conteo', 'Promedio', 'Mediana'), horizontal=True)
        balotas_reg_cols = [f'Balota {i}' for i in range(1, 6)]
        
        if metric_selection == 'Conteo':
            df_melted_balotas = df[balotas_reg_cols].melt(var_name='Balota', value_name='Numero')
            heatmap_data_regular = pd.crosstab(df_melted_balotas['Numero'], df_melted_balotas['Balota'])
            heatmap_data_regular = heatmap_data_regular.reindex(columns=balotas_reg_cols).fillna(0)
            
            superbalota_counts = df['SuperBalota'].value_counts().sort_index()
            superbalota_full_range = pd.Series(0, index=range(1, 17))
            superbalota_full_range.update(superbalota_counts)
            
            max_num_regular = 43
            all_numbers = pd.RangeIndex(start=1, stop=max_num_regular + 1)
            heatmap_data_regular = heatmap_data_regular.reindex(all_numbers, fill_value=0)

            heatmap_final = heatmap_data_regular.copy()
            heatmap_final['SuperBalota'] = superbalota_full_range.reindex(all_numbers, fill_value=pd.NA)

            fig7, ax7 = plt.subplots(figsize=(12, 10))
            sns.heatmap(heatmap_final, annot=True, fmt=".0f", cmap='viridis', linewidths=.5, linecolor='black', ax=ax7)
            ax7.set_title(f'Mapa de Calor Consolidado: {metric_selection} por Número y Balota')
            ax7.set_xlabel('Tipo de Balota')
            ax7.set_ylabel('Número de Balota')
            st.pyplot(fig7)
        else:
            st.info("Para un 'Mapa de Calor Consolidado por Balota' (Número vs Posición), la métrica de 'Conteo' es la más significativa. Por favor, selecciona 'Conteo' para ver el mapa de calor.")
            
        st.subheader("📈 Distribución de Frecuencia de las Balotas")
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
        numeric_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
        correlation_matrix = df[numeric_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
        ax4.set_title('Matriz de Correlación entre Balotas')
        st.pyplot(fig4)

        st.subheader("⏳ Tendencia Anual del Promedio de Cada Balota")
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
            num_sims = st.slider("Número de simulaciones para Montecarlo (y para la Simulación de Ganancias):", min_value=1000, max_value=100000, value=20000, step=1000)

            if st.button("Generar Todos los Pronósticos"):
                st.session_state.all_forecasts = [] # Limpiar resultados anteriores

                with st.spinner("Generando pronósticos con Gemini AI..."):
                    latest_results = df.sort_values(by='Fecha', ascending=False).head(5)
                    latest_results_str = latest_results.to_string(index=False)
                    all_balotas_combined = pd.concat([df['Balota 1'], df['Balota 2'], df['Balota 3'], df['Balota 4'], df['Balota 5']])
                    top_10_balotas_global = all_balotas_combined.value_counts().head(10)
                    top_balotas_str = ", ".join([f"{num} ({freq} veces)" for num, freq in top_10_balotas_global.items()])
                    top_10_superbalotas = df['SuperBalota'].value_counts().head(10)
                    top_superbalotas_str = ", ".join([f"{num} ({freq} veces)" for num, freq in top_10_superbalotas.items()])

                    gemini_results = get_gemini_recommendations(df, model, latest_results_str, top_balotas_str, top_superbalotas_str)
                    st.session_state.all_forecasts.extend(gemini_results)
                    st.success("Pronósticos de Gemini AI generados.")

                with st.spinner(f"Ejecutando {num_sims} simulaciones de Montecarlo..."):
                    montecarlo_results = generate_montecarlo_draws(df, num_sims)
                    st.session_state.all_forecasts.extend(montecarlo_results)
                    st.success("Simulación de Montecarlo completada.")

                with st.spinner("Generando recomendaciones de Números 'Calientes'..."):
                    hot_numbers_results = get_hot_numbers_recommendations(df)
                    st.session_state.all_forecasts.extend(hot_numbers_results)
                    st.success("Recomendaciones de Números 'Calientes' generadas.")
                
                st.info("Todos los pronósticos han sido generados y almacenados.")
                
            if st.session_state.all_forecasts:
                st.subheader("Resultados de los Pronósticos:")

                # Mostrar resultados de Gemini AI
                st.markdown("##### 1. Pronósticos de Gemini AI")
                gemini_df = pd.DataFrame([f for f in st.session_state.all_forecasts if f['method'].startswith('Gemini AI')])
                if not gemini_df.empty:
                    st.dataframe(gemini_df[['balotas', 'superbalota', 'score', 'calidez_pct']].style.format({'score': '{:.0f}', 'calidez_pct': '{:.1f}%'}))
                    if st.session_state.gemini_raw_response:
                        with st.expander("Ver respuesta cruda de Gemini (para depuración)"):
                            st.text(st.session_state.gemini_raw_response)
                else:
                    st.write("No se generaron pronósticos de Gemini AI.")

                # Mostrar resultados de Montecarlo
                st.markdown("##### 2. Pronósticos de Simulación Montecarlo")
                montecarlo_df = pd.DataFrame([f for f in st.session_state.all_forecasts if f['method'] == 'Monte Carlo'])
                if not montecarlo_df.empty:
                    st.dataframe(montecarlo_df[['balotas', 'superbalota', 'score', 'calidez_pct']].style.format({'score': '{:.0f}', 'calidez_pct': '{:.1f}%'}))
                else:
                    st.write("No se generaron pronósticos de Montecarlo.")

                # Mostrar resultados de Números Calientes
                st.markdown("##### 3. Pronósticos de Números 'Calientes'")
                hot_numbers_df = pd.DataFrame([f for f in st.session_state.all_forecasts if f['method'] == 'Números Calientes'])
                if not hot_numbers_df.empty:
                    st.dataframe(hot_numbers_df[['balotas', 'superbalota', 'score', 'calidez_pct']].style.format({'score': '{:.0f}', 'calidez_pct': '{:.1f}%'}))
                else:
                    st.write("No se generaron pronósticos de Números 'Calientes'.")

                st.markdown("---")
                st.subheader("🏆 Top 3 Combinaciones con Mayor Puntaje de Frecuencia Histórica")
                
                valid_forecasts = [f for f in st.session_state.all_forecasts if f['balotas'] and f['superbalota'] is not None]
                
                if valid_forecasts:
                    sorted_forecasts = sorted(valid_forecasts, key=lambda x: x['score'], reverse=True)
                    top_3 = sorted_forecasts[:3]

                    top_3_df = pd.DataFrame(top_3)
                    st.dataframe(top_3_df[['balotas', 'superbalota', 'score', 'calidez_pct', 'method']].style.format({'score': '{:.0f}', 'calidez_pct': '{:.1f}%'}))
                    st.info(f"El **'Puntaje de Frecuencia Histórica'** indica qué tan a menudo han aparecido los números de la combinación en los sorteos pasados. Un puntaje más alto sugiere que la combinación está compuesta por números históricamente más frecuentes.")
                    st.info(f"El **'Nivel de Calidez (%)'** compara el puntaje de la combinación con el puntaje máximo teórico posible ({MAX_POSSIBLE_SCORE:.0f}). Un porcentaje más alto significa que la combinación es más 'caliente' en relación con el ideal histórico.")
                else:
                    st.write("No hay combinaciones válidas para mostrar el Top 3.")
            else:
                st.write("Presiona 'Generar Todos los Pronósticos' para ver las sugerencias.")
            
            st.markdown("---")
            st.subheader("🎲 Simulación de Ganancias Históricas")
            st.write("Selecciona una de las combinaciones generadas anteriormente para simular cuántas veces habrías ganado el premio mayor si la hubieras jugado *N* veces contra los sorteos históricos. **Premio por acierto: $1.000.000**")

            if st.session_state.all_forecasts:
                # Crear opciones para el selectbox
                forecast_options = []
                for idx, f in enumerate(st.session_state.all_forecasts):
                    if f['balotas'] and f['superbalota'] is not None:
                        option_label = f"{f['method']} #{idx+1}: Balotas {f['balotas']}, SB {f['superbalota']} (Score: {f['score']:.0f})"
                        forecast_options.append((option_label, f['balotas'], f['superbalota']))
                
                if forecast_options:
                    selected_option_label = st.selectbox(
                        "Selecciona una combinación para simular:",
                        [opt[0] for opt in forecast_options]
                    )
                    
                    # Find the selected combination's actual numbers
                    selected_balotas_sim = None
                    selected_superbalota_sim = None
                    for opt_label, b, sb in forecast_options:
                        if opt_label == selected_option_label:
                            selected_balotas_sim = b
                            selected_superbalota_sim = sb
                            break

                    if st.button(f"Simular Ganancias con {num_sims} Juegos"):
                        if selected_balotas_sim and selected_superbalota_sim is not None:
                            winnings_count, total_winnings, message = simulate_winnings(selected_balotas_sim, selected_superbalota_sim, df, num_sims)
                            st.markdown(f"**Combinación Simulada:** Balotas: {selected_balotas_sim}, SuperBalota: {selected_superbalota_sim}")
                            st.write(message)
                            st.success(f"**Ganancia Total Estimada:** ${total_winnings:,.0f}")
                            st.warning("Esta simulación se basa en la frecuencia de coincidencias exactas con sorteos históricos y no garantiza ganancias futuras. El Baloto es un juego de azar.")
                        else:
                            st.error("Por favor, selecciona una combinación válida para simular.")
                else:
                    st.info("Genera pronósticos primero para poder simular ganancias.")
            else:
                st.info("Genera pronósticos primero para poder simular ganancias.")

        else:
            st.warning("La funcionalidad de Gemini AI no está disponible. Por favor, verifica tu API Key para habilitar los pronósticos.")


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
        * **Herramientas de Pronóstico y Simulación:**
            * **Integración con Google Gemini AI:** Genera 5 sugerencias de números con justificación.
            * **Simulación de Montecarlo:** Genera 5 combinaciones hipotéticas basadas en probabilidades históricas.
            * **Recomendación de Números 'Calientes':** Genera 5 combinaciones basadas en la frecuencia de aparición por posición.
            * **Puntaje de Frecuencia Histórica y Nivel de Calidez (%):** Evalúa la "calidez" de las combinaciones en relación con los patrones históricos.
            * **Almacenamiento y Comparación:** Guarda y muestra el Top 3 de los pronósticos generados.
            * **Simulación de Ganancias Históricas:** Estima ganancias al jugar una combinación N veces contra los sorteos históricos.

        **Desarrollado por:** Julian Torres (con asistencia de un modelo de lenguaje de Google).
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Google_Gemini_logo.svg/1200px-Google_Gemini_logo.svg.png", width=150)
        st.write("Los resultados de Baloto son aleatorios. Por favor, juega responsablemente.")

else:
    st.error("No se pudieron cargar los datos del Baloto. Por favor, asegúrate de que la URL sea correcta y el archivo esté accesible.")
    st.info("Intenta revisar la URL del archivo en tu repositorio de GitHub o la conexión a internet. Si el problema persiste, verifica el formato del CSV.")

st.markdown("---")
st.write("¡Gracias por usar la aplicación!")
