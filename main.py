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
    page_title="Análisis y Predicción de Baloto/Revancha con Gemini AI",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Análisis Exploratorio de Datos y Predicción de Baloto/Revancha con Gemini AI 🇨🇴")
st.write("Bienvenido al panel interactivo de análisis de los resultados históricos del Baloto colombiano. Explora tendencias pasadas y experimenta con la IA de Gemini para posibles predicciones o insights.")

# --- Configuración de la API Key de Gemini ---
# gemini_api_key = st.secrets["GEMINI_API_KEY"] # Ideal para producción con Streamlit Secrets
gemini_api_key = "AIzaSyAo1mZnBvslWoUKot7svYIo2K3fZIrLgRk"

try:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    st.success("API de Gemini configurada exitosamente con 'gemini-1.5-flash'.")
except Exception as e:
    st.error(f"Error al configurar la API de Gemini: {e}")
    st.warning("La funcionalidad de Gemini AI podría no estar disponible. Asegúrate de que tu API Key sea válida.")
    model = None

# --- URLs de los archivos CSV en GitHub ---
BALOTO_URL = "https://raw.githubusercontent.com/JulianTorrest/modelos/refs/heads/main/Baloto.csv"
REVANCHA_URL = "https://raw.githubusercontent.com/JulianTorrest/modelos/refs/heads/main/Revancha.csv"

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

# --- Inicializar session_state para almacenar resultados ---
if 'all_forecasts' not in st.session_state:
    st.session_state.all_forecasts = []
if 'gemini_raw_response' not in st.session_state:
    st.session_state.gemini_raw_response = ""
if 'selected_sorteo_type' not in st.session_state:
    st.session_state.selected_sorteo_type = 'Baloto' # Default selection

# --- Sorteo Selector ---
st.sidebar.header("Seleccionar Sorteo")
selected_sorteo_type = st.sidebar.radio(
    "Elige el tipo de sorteo:",
    ('Baloto', 'Revancha'),
    key='sorteo_selector'
)

# Load data based on selection
if selected_sorteo_type == 'Baloto':
    df = load_data(BALOTO_URL)
    st.session_state.selected_sorteo_type = 'Baloto'
else:
    df = load_data(REVANCHA_URL)
    st.session_state.selected_sorteo_type = 'Revancha'

# --- Funciones de Utilidad ---

@st.cache_data
def get_positional_stats(df_current):
    """
    Calcula las estadísticas de min, max y frecuencias para cada posición de balota.
    """
    positional_stats = {}
    if df_current.empty:
        return positional_stats

    for i in range(1, 6): # Balota 1 to Balota 5
        col_name = f'Balota {i}'
        data_for_pos = df_current[col_name]
        positional_stats[col_name] = {
            'min_observed': data_for_pos.min(),
            'max_observed': data_for_pos.max(),
            'frequencies': data_for_pos.value_counts(normalize=True).sort_index()
        }
    
    superbalota_data = df_current['SuperBalota']
    positional_stats['SuperBalota'] = {
        'min_observed': superbalota_data.min(),
        'max_observed': superbalota_data.max(),
        'frequencies': superbalota_data.value_counts(normalize=True).sort_index()
    }
    return positional_stats

# Pre-calcular estas estadísticas una vez al cargar los datos
POSITIONAL_STATS = {}
if not df.empty:
    POSITIONAL_STATS = get_positional_stats(df)
else:
    st.warning("Datos no cargados, las estadísticas posicionales no estarán disponibles.")


@st.cache_data
def calculate_historical_frequency_score(df_current, balotas_list, superbalota_num):
    """
    Calcula un puntaje basado en la suma de las frecuencias históricas
    de cada número en su respectiva posición de balota.
    """
    score = 0
    if not isinstance(balotas_list, list) or len(balotas_list) != 5:
        return 0

    for i in range(5):
        col_name = f'Balota {i+1}'
        # Usar las frecuencias de la posición específica
        if POSITIONAL_STATS and col_name in POSITIONAL_STATS:
            # Multiplicar por el total para revertir normalización y obtener conteo real de frecuencia
            score += POSITIONAL_STATS[col_name]['frequencies'].get(balotas_list[i], 0) * len(df_current) 
        else: # Fallback a conteo global si no hay stats posicionales
             score += df_current[col_name].value_counts().get(balotas_list[i], 0)

    if POSITIONAL_STATS and 'SuperBalota' in POSITIONAL_STATS:
        score += POSITIONAL_STATS['SuperBalota']['frequencies'].get(superbalota_num, 0) * len(df_current)
    else: # Fallback a conteo global
        score += df_current['SuperBalota'].value_counts().get(superbalota_num, 0)
    return score

@st.cache_data
def get_max_possible_score(df_current):
    """
    Calcula el score máximo posible buscando la combinación de 5 balotas ordenadas
    que maximiza la suma de sus frecuencias posicionales, más la SuperBalota más frecuente.
    """
    if df_current.empty or not POSITIONAL_STATS:
        return 1 # Fallback

    max_score = 0
    
    # Calcular las frecuencias de cada número por posición
    freq_by_pos = {col: stats['frequencies'] for col, stats in POSITIONAL_STATS.items() if col.startswith('Balota')}
    superbalota_freq = POSITIONAL_STATS['SuperBalota']['frequencies']
    
    # Obtener la frecuencia máxima de la SuperBalota
    max_sb_freq = superbalota_freq.max() * len(df_current) if not superbalota_freq.empty else 0

    current_balotas = []
    prev_num = 0
    for i in range(1, 6):
        col_name = f'Balota {i}'
        
        # Consider ranges: min from previous ball + 1, and max from overall total - remaining balls
        min_allowed_by_order = prev_num + 1
        max_allowed_by_remaining_slots = 43 - (5 - i) 
        
        # Also consider historical observed min/max for this specific position
        historical_min_for_pos = POSITIONAL_STATS[col_name]['min_observed']
        historical_max_for_pos = POSITIONAL_STATS[col_name]['max_observed']

        # Combine all constraints for candidates
        candidates_in_range = [n for n in freq_by_pos[col_name].index.tolist() 
                               if n >= min_allowed_by_order and 
                                  n <= max_allowed_by_remaining_slots and
                                  n >= historical_min_for_pos and
                                  n <= historical_max_for_pos]
        
        if not candidates_in_range:
            # Fallback if no hot numbers fit the range, try to pick smallest possible within combined range
            chosen_num = max(min_allowed_by_order, historical_min_for_pos)
            if chosen_num > min(max_allowed_by_remaining_slots, historical_max_for_pos):
                 return 0 # Cannot form a valid sequence
            
            # Ensure we can still pick 5 distinct increasing numbers
            if chosen_num + (5 - i) > 43: # If next balls won't fit 
                return 0
            
            # Use frequency of the chosen_num, or 0 if not found
            max_freq_num_for_pos = freq_by_pos[col_name].get(chosen_num, 0) * len(df_current)
            best_num = chosen_num
        else:
            # Pick the hottest among valid candidates
            best_num = max(candidates_in_range, key=lambda x: freq_by_pos[col_name].get(x, 0))
            max_freq_num_for_pos = freq_by_pos[col_name].get(best_num, 0) * len(df_current)
        
        current_balotas.append(best_num)
        prev_num = best_num
        max_score += max_freq_num_for_pos
    
    max_score += max_sb_freq # Sumar la frecuencia del SB más frecuente
    
    return max_score


# Pre-calcular el score máximo posible una vez
MAX_POSSIBLE_SCORE = 1 # Initialize to 1 to avoid ZeroDivisionError
_temp_max_score_calculated = 0

if not df.empty and POSITIONAL_STATS:
    _temp_max_score_calculated = get_max_possible_score(df)
    if _temp_max_score_calculated > 0:
        MAX_POSSIBLE_SCORE = _temp_max_score_calculated
    else:
        st.warning("Advertencia: El 'Score Máximo Posible' calculado fue 0 o inválido. El 'Nivel de Calidez (%)' se mostrará como 0% para todas las combinaciones.")


# --- Funciones de Pronóstico y Simulación ---

def generate_montecarlo_draws(df_current, num_simulations=10000):
    """
    Genera 5 combinaciones de baloto usando Montecarlo,
    respetando el orden, las distribuciones históricas de cada balota POSICIONAL,
    y los rangos observados para cada posición.
    """
    simulated_draws_with_scores = []
    seen_combinations = set()

    if not POSITIONAL_STATS:
        # Fallback si POSITIONAL_STATS no está disponible
        st.error("Estadísticas posicionales no disponibles para Monte Carlo. Generando aleatoriamente.")
        # Generación completamente aleatoria como fallback
        for _ in range(5):
            balotas = sorted(np.random.choice(range(1, 44), 5, replace=False).tolist())
            superbalota = np.random.choice(range(1, 17))
            simulated_draws_with_scores.append({
                'balotas': balotas,
                'superbalota': superbalota,
                'score': 0, # Score no aplicable sin stats
                'calidez_pct': 0,
                'method': 'Monte Carlo (Fallback Aleatorio)'
            })
        return simulated_draws_with_scores


    attempts = 0
    while len(simulated_draws_with_scores) < 5 and attempts < num_simulations * 2: # Intentar más veces para asegurar 5 únicas
        current_draw = []
        prev_num = 0

        valid_combination_path = True
        for i in range(1, 6):
            col_name = f'Balota {i}'
            
            # Constraints from previous ball and remaining slots
            min_allowed_by_order = prev_num + 1
            max_allowed_by_remaining_slots = 43 - (5 - i)
            
            # Constraints from historical observed range for this position
            historical_min_for_pos = POSITIONAL_STATS[col_name]['min_observed']
            historical_max_for_pos = POSITIONAL_STATS[col_name]['max_observed']

            # Combine all constraints
            effective_min = max(min_allowed_by_order, historical_min_for_pos)
            effective_max = min(max_allowed_by_remaining_slots, historical_max_for_pos)

            if effective_min > effective_max: # No valid numbers possible for this position
                valid_combination_path = False
                break
            
            # Filter candidates based on effective range and positional frequencies
            pos_frequencies = POSITIONAL_STATS[col_name]['frequencies']
            
            candidates = [n for n in pos_frequencies.index.tolist() if effective_min <= n <= effective_max]

            if not candidates:
                # Fallback: if no historical hot numbers in range, use the whole effective range
                candidates = list(range(effective_min, effective_max + 1))
                if not candidates: # Still no candidates, this path is invalid
                    valid_combination_path = False
                    break 

            # Calculate weights for candidates based on positional frequencies
            weights = [pos_frequencies.get(n, 0.0001) for n in candidates] # Small non-zero for unseen numbers
            weights_sum = sum(weights)
            if weights_sum == 0: 
                weights = [1/len(candidates)] * len(candidates) # Uniform if all weights are zero
            else:
                weights = [w / weights_sum for w in weights]

            chosen_num = np.random.choice(candidates, p=weights)
            current_draw.append(chosen_num)
            prev_num = chosen_num
        
        if valid_combination_path and len(current_draw) == 5:
            # SuperBalota selection (using its specific historical distribution)
            sb_frequencies = POSITIONAL_STATS['SuperBalota']['frequencies']
            sb_candidates = list(range(1, 17)) # Full possible range for SB
            sb_weights = [sb_frequencies.get(n, 0.0001) for n in sb_candidates]
            sb_weights_sum = sum(sb_weights)
            if sb_weights_sum == 0:
                sb_chosen = np.random.choice(sb_candidates)
            else:
                sb_weights = [w / sb_weights_sum for w in sb_weights]
                sb_chosen = np.random.choice(sb_candidates, p=sb_weights)
            
            combo_key = (tuple(current_draw), sb_chosen)
            if combo_key not in seen_combinations:
                score = calculate_historical_frequency_score(df_current, list(current_draw), sb_chosen)
                calidez = (score / MAX_POSSIBLE_SCORE * 100) if MAX_POSSIBLE_SCORE > 0 else 0
                simulated_draws_with_scores.append({
                    'balotas': list(current_draw),
                    'superbalota': sb_chosen,
                    'score': score,
                    'calidez_pct': calidez,
                    'method': 'Monte Carlo'
                })
                seen_combinations.add(combo_key)
        attempts += 1
    
    # Asegurar que siempre se devuelvan 5 simulaciones, incluso si algunas no fueron válidas o no se pudieron generar
    while len(simulated_draws_with_scores) < 5:
        simulated_draws_with_scores.append({
            'balotas': [],
            'superbalota': None,
            'score': 0,
            'calidez_pct': 0,
            'method': 'Monte Carlo (Inválido/No Generado)'
        })

    return simulated_draws_with_scores


def get_hot_numbers_recommendations(df_current):
    """
    Genera 5 recomendaciones de balotas basadas en los números más frecuentes
    para cada POSICIÓN, respetando el orden y los rangos históricos.
    """
    recommendations = []
    seen_combinations = set()

    if not POSITIONAL_STATS:
        # Fallback si POSITIONAL_STATS no está disponible
        st.error("Estadísticas posicionales no disponibles para Números Calientes. Generando aleatoriamente.")
        # Generación completamente aleatoria como fallback
        for _ in range(5):
            balotas = sorted(np.random.choice(range(1, 44), 5, replace=False).tolist())
            superbalota = np.random.choice(range(1, 17))
            recommendations.append({
                'balotas': balotas,
                'superbalota': superbalota,
                'score': 0, # Score no aplicable sin stats
                'calidez_pct': 0,
                'method': 'Números Calientes (Fallback Aleatorio)'
            })
        return recommendations


    # Pre-calcular las frecuencias por posición para un acceso rápido
    freq_by_pos = {col: stats['frequencies'] for col, stats in POSITIONAL_STATS.items() if col.startswith('Balota')}
    superbalota_freq = POSITIONAL_STATS['SuperBalota']['frequencies']
    
    # Get the hottest SuperBalota (or default to 1 if none)
    hot_superbalota_candidates = superbalota_freq.index.tolist()
    if not hot_superbalota_candidates: hot_superbalota_candidates = list(range(1, 17)) # Fallback if no data


    # --- Generar la combinación más caliente "pura" (la primera) ---
    # This tries to pick the *absolute* hottest for each position, given constraints.
    current_balotas = []
    prev_num = 0
    for i in range(1, 6):
        col_name = f'Balota {i}'
        
        min_allowed_by_order = prev_num + 1
        max_allowed_by_remaining_slots = 43 - (5 - i)
        
        historical_min_for_pos = POSITIONAL_STATS[col_name]['min_observed']
        historical_max_for_pos = POSITIONAL_STATS[col_name]['max_observed']

        effective_min = max(min_allowed_by_order, historical_min_for_pos)
        effective_max = min(max_allowed_by_remaining_slots, historical_max_for_pos)

        if effective_min > effective_max: # No valid number for this position
            current_balotas = [] # Mark as invalid path
            break

        # Filter numbers that have appeared historically in this position AND are within effective range
        valid_candidates_for_pos = [n for n in freq_by_pos[col_name].index.tolist() 
                                    if effective_min <= n <= effective_max]
        
        if not valid_candidates_for_pos:
            # Fallback: if no historical hot numbers fit the combined range, choose the smallest possible
            chosen_num = effective_min
            # Double check if this fallback number allows forming a valid sequence
            if chosen_num + (5 - i) > 43 or chosen_num > effective_max:
                current_balotas = []
                break # Cannot form a valid sequence
        else:
            # Pick the hottest among valid candidates
            chosen_num = max(valid_candidates_for_pos, key=lambda x: freq_by_pos[col_name].get(x, 0))
        
        current_balotas.append(chosen_num)
        prev_num = chosen_num
    
    if len(current_balotas) == 5: # Only if a valid 5-ball combo was formed
        hot_superbalota = hot_superbalota_candidates[0] if hot_superbalota_candidates else 1 # Fallback to 1
        
        score = calculate_historical_frequency_score(df_current, current_balotas, hot_superbalota)
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
    generated_count = len(recommendations) # Start from 1 if the first combo was valid
    max_attempts = 200 # Para evitar bucles infinitos

    while generated_count < 5 and max_attempts > 0:
        temp_balotas = []
        prev_num = 0
        valid_combination = True

        for i in range(1, 6):
            col_name = f'Balota {i}'
            
            min_allowed_by_order = prev_num + 1
            max_allowed_by_remaining_slots = 43 - (5 - i)
            
            historical_min_for_pos = POSITIONAL_STATS[col_name]['min_observed']
            historical_max_for_pos = POSITIONAL_STATS[col_name]['max_observed']

            effective_min = max(min_allowed_by_order, historical_min_for_pos)
            effective_max = min(max_allowed_by_remaining_slots, historical_max_for_pos)

            if effective_min > effective_max:
                valid_combination = False
                break
            
            # Take top N candidates for this position, then filter by combined constraints
            top_n_candidates_for_pos = freq_by_pos[col_name].head(10).index.tolist() # Consider top 10 hot numbers
            
            candidates_in_range_and_hot = [n for n in top_n_candidates_for_pos 
                                           if effective_min <= n <= effective_max]
            
            if not candidates_in_range_and_hot:
                # Fallback to any number in effective range if no hot candidate fits
                candidates_in_range_and_hot = list(range(effective_min, effective_max + 1))
                if not candidates_in_range_and_hot: 
                    valid_combination = False
                    break
            
            chosen_num = np.random.choice(candidates_in_range_and_hot)
            temp_balotas.append(chosen_num)
            prev_num = chosen_num
        
        if valid_combination and len(temp_balotas) == 5:
            temp_superbalota = np.random.choice(hot_superbalota_candidates) # Still pick from overall hot SB
            
            combo_key = (tuple(temp_balotas), temp_superbalota)
            if combo_key not in seen_combinations:
                score = calculate_historical_frequency_score(df_current, temp_balotas, temp_superbalota)
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
    
    # Ensure we always return 5 recommendations, even if some were invalid/couldn't be generated
    while len(recommendations) < 5:
        recommendations.append({
            'balotas': [],
            'superbalota': None,
            'score': 0,
            'calidez_pct': 0,
            'method': 'Números Calientes (Inválido/No Generado)'
        })
    
    return recommendations


def get_gemini_recommendations(df_current, model_ai, latest_results_str, top_balotas_str, top_superbalotas_str):
    """
    Obtiene 5 recomendaciones de Gemini AI y calcula su puntaje.
    """
    if not model_ai:
        st.warning("Gemini AI no está disponible para generar recomendaciones.")
        return []

    gemini_forecasts = []
    
    # --- Modificar el prompt para incluir información de rango POSICIONAL ---
    # Aquí es donde realmente instruimos a Gemini sobre la tendencia de rangos.
    # Extraer y formatar la información de rangos posicionales observados
    positional_range_info = ""
    if POSITIONAL_STATS:
        for i in range(1, 6):
            col_name = f'Balota {i}'
            if col_name in POSITIONAL_STATS:
                min_obs = POSITIONAL_STATS[col_name]['min_observed']
                max_obs = POSITIONAL_STATS[col_name]['max_observed']
                positional_range_info += f"  - Balota {i} (rango histórico observado): {min_obs} a {max_obs}\n"
        if positional_range_info:
            positional_range_info = "**Rangos Históricos Observados por Posición de Balota:**\n" + positional_range_info
    
    prompt = (
        f"Basado en los siguientes últimos 5 resultados del Baloto/Revancha:\n\n{latest_results_str}\n\n"
        f"**Información Histórica Adicional:**\n"
        f"- Los números de Balota regular más frecuentes históricamente (en cualquier posición, del 1 al 43) son: {top_balotas_str}.\n"
        f"- Los números de SuperBalota más frecuentes históricamente (del 1 al 16) son: {top_superbalotas_str}.\n"
        f"{positional_range_info}\n" # Incluir la nueva información
        "Por favor, sugiere **5 conjuntos distintos** de 5 números de balota y 1 SuperBalota. "
        "Para cada conjunto, las 5 balotas deben estar en el rango de 1 a 43 y **estrictamente ordenadas de forma ascendente (Balota 1 < Balota 2 < Balota 3 < Balota 4 < Balota 5)**. "
        "**Es CRÍTICO que los números para cada balota (Balota 1, Balota 2, etc.) tiendan a respetar sus rangos históricos observados para esa posición específica.** "
        "La SuperBalota debe estar en el rango de 1 a 16 y es independiente de las otras 5. "
        "Justifica brevemente tu razonamiento para cada conjunto, basándote en los datos proporcionados (últimos resultados, números frecuentes históricos y **rangos posicionales**).\n"
        "**Formato de salida deseado para cada conjunto (importante para el parsing):**\n"
        "**Conjunto N:** Balotas: [N1, N2, N3, N4, N5], SuperBalota: [SB]. Razón: [Tu justificación]\n"
        "Asegúrate de que cada conjunto sea único y siga el formato exacto."
    )
    # --- FIN DEL PROMPT CLAVE ---

    try:
        response = model_ai.generate_content(prompt)
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
                is_valid = True
                if not (1 <= balotas[0] < balotas[1] < balotas[2] < balotas[3] < balotas[4] <= 43 and
                        1 <= superbalota <= 16):
                    is_valid = False
                
                # Further validate against positional historical ranges for Gemini's output
                if is_valid and POSITIONAL_STATS:
                    for i in range(5):
                        col_name = f'Balota {i+1}'
                        num = balotas[i]
                        if col_name in POSITIONAL_STATS:
                            min_obs = POSITIONAL_STATS[col_name]['min_observed']
                            max_obs = POSITIONAL_STATS[col_name]['max_observed']
                            if not (min_obs <= num <= max_obs):
                                # If Gemini suggests a number outside its historical positional range,
                                # we mark it as invalid for our purpose here.
                                is_valid = False
                                # st.warning(f"DEBUG: Gemini sugirió un número fuera de rango posicional: Balota {i+1}: {num} (Rango: {min_obs}-{max_obs})")
                                break

                if is_valid:                    
                    combo_key = (tuple(balotas), superbalota)
                    if combo_key not in seen_combinations:
                        score = calculate_historical_frequency_score(df_current, balotas, superbalota)
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
                # else: st.warning(f"Gemini sugirió una combinación inválida (rango/orden/posicional): Balotas: {balotas}, SuperBalota: {superbalota}")

            except ValueError:
                st.warning(f"No se pudo parsear una combinación de Gemini de '{match}'.")
        
        if parsed_count == 0:
            st.warning("Gemini no pudo generar combinaciones válidas en el formato esperado o que cumplan los criterios posicionales.")
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


# --- Función para Simular Ganancias (directamente sobre una combinación) ---
def simulate_winnings(selected_balotas, selected_superbalota, historical_df, num_simulations):
    """
    Simula jugar una combinación N veces contra los sorteos históricos disponibles
    y calcula las ganancias basadas en acertar las 5 balotas y la SuperBalota
    en una fecha histórica.
    """
    if not selected_balotas or selected_superbalota is None:
        return 0, 0, "Combinación inválida para simulación."

    # Convertir la combinación seleccionada a un formato comparable (tuple)
    selected_combo = (tuple(sorted(selected_balotas)), selected_superbalota) # Ensure sorted for comparison
    
    # Crear un DataFrame con los sorteos históricos en el mismo formato
    historical_combos = []
    for index, row in historical_df.iterrows():
        balotas_hist = tuple(sorted([row[f'Balota {i}'] for i in range(1, 6)])) # Ensure sorted for comparison
        superbalota_hist = row['SuperBalota']
        historical_combos.append((balotas_hist, superbalota_hist))
    
    historical_combos_series = pd.Series(historical_combos)
    
    exact_matches_in_history = historical_combos_series.value_counts().get(selected_combo, 0)
    
    total_historical_draws = len(historical_df)
    
    if total_historical_draws == 0:
        return 0, 0, "No hay datos históricos para simular."
        
    expected_winnings_count = (exact_matches_in_history / total_historical_draws) * num_simulations
    
    winnings_count = int(np.floor(expected_winnings_count)) # Use floor to be conservative

    total_winnings = winnings_count * 1_000_000
    
    message = f"Simulando jugar esta combinación {num_simulations} veces contra el historial de {total_historical_draws} sorteos de {st.session_state.selected_sorteo_type}."
    message += f" Basado en esto, se estima que habrías ganado el premio mayor {winnings_count} veces."

    return winnings_count, total_winnings, message


# --- Nueva Función: Simulación Histórica "Inteligente" ---
def smart_historical_simulation(df_current, num_combinations_per_method=100,
                                match_criteria="Exacto (5 Balotas + SuperBalota)",
                                include_proximity_analysis=False):
    """
    Simula jugar N combinaciones generadas por cada método (Montecarlo, Hot Numbers)
    contra *cada sorteo histórico* para ver cuántas veces habrían acertado,
    considerando diferentes criterios de acierto y análisis de proximidad.
    """
    st.info(f"Realizando Simulación Histórica 'Inteligente' para {st.session_state.selected_sorteo_type}. Esto podría tomar unos segundos...")

    if df_current.empty or not POSITIONAL_STATS:
        st.error("Datos o estadísticas posicionales no disponibles para la simulación.")
        # Retornar valores predeterminados para evitar errores en el flujo principal
        return 0, 0, 0, "Simulación no ejecutada debido a datos faltantes.", {}


    # --- Generación de Combinaciones ---
    all_simulated_combinations_with_method = [] # List of {'combo': (balotas_tuple, sb_num), 'method': 'Method Name'}

    # Generate a pool of Monte Carlo combinations
    mc_pool_unique = set()
    mc_attempts = 0
    while len(mc_pool_unique) < num_combinations_per_method and mc_attempts < num_combinations_per_method * 10: # Attempt more times to get unique combos
        sim_draws_list = generate_montecarlo_draws(df_current, 1) # Generate one at a time
        if sim_draws_list and sim_draws_list[0]['balotas'] and sim_draws_list[0]['superbalota'] is not None:
            combo_tuple = (tuple(sim_draws_list[0]['balotas']), sim_draws_list[0]['superbalota'])
            mc_pool_unique.add(combo_tuple)
        mc_attempts += 1
    all_simulated_combinations_with_method.extend([{'combo': c, 'method': 'Monte Carlo'} for c in mc_pool_unique])

    # Generate a pool of Hot Numbers combinations
    hot_pool_unique = set()
    hot_attempts = 0
    while len(hot_pool_unique) < num_combinations_per_method and hot_attempts < num_combinations_per_method * 10: # Attempt more times
        hot_draws_batch = get_hot_numbers_recommendations(df_current) 
        for d in hot_draws_batch:
            if d['balotas'] and d['superbalota'] is not None:
                combo_tuple = (tuple(d['balotas']), d['superbalota'])
                hot_pool_unique.add(combo_tuple)
            if len(hot_pool_unique) >= num_combinations_per_method: # Stop if we have enough
                break
        hot_attempts += 1
    all_simulated_combinations_with_method.extend([{'combo': c, 'method': 'Números Calientes'} for c in hot_pool_unique])

    # --- Preparación de Datos Históricos ---
    historical_winning_combos_full = [] # List of (sorted_balotas_tuple, superbalota_num)
    for index, row in df_current.iterrows():
        balotas_hist = tuple(sorted([row[f'Balota {i}'] for i in range(1, 6)]))
        superbalota_hist = row['SuperBalota']
        historical_winning_combos_full.append((balotas_hist, superbalota_hist))

    # Convert to set for fast exact lookups (only if exact match criteria)
    historical_winning_combos_set = set(historical_winning_combos_full)

    st.subheader("📊 Depuración de Simulación Histórica")
    with st.expander("Ver detalles de las combinaciones generadas y comparadas"):
        if all_simulated_combinations_with_method:
            st.write(f"**Muestra de las primeras 10 combinaciones generadas ({len(all_simulated_combinations_with_method)} en total):**")
            sample_generated_df = pd.DataFrame([{'Balotas': c['combo'][0], 'SuperBalota': c['combo'][1], 'Método': c['method']}
                                                 for c in all_simulated_combinations_with_method[:min(10, len(all_simulated_combinations_with_method))]])
            st.dataframe(sample_generated_df)

        if historical_winning_combos_set:
            st.write(f"**Muestra de las primeras 10 combinaciones históricas ganadoras (únicas y normalizadas, {len(historical_winning_combos_set)} en total):**")
            sample_historical_df = pd.DataFrame([{'Balotas': c[0], 'SuperBalota': c[1]} for c in list(historical_winning_combos_set)[:min(10, len(historical_winning_combos_set))]])
            st.dataframe(sample_historical_df)

        st.write(f"**Número de combinaciones Monte Carlo únicas en el pool:** {len(mc_pool_unique)}")
        st.write(f"**Número de combinaciones Números Calientes únicas en el pool:** {len(hot_pool_unique)}")
        st.write(f"**Número de sorteos históricos únicos a verificar:** {len(historical_winning_combos_set)}")
        st.write(f"**Criterio de acierto seleccionado:** `{match_criteria}`")


    # --- Contadores de Aciertos y Proximidad ---
    wins_by_method = {'Monte Carlo': set(), 'Números Calientes': set()} # Para aciertos según el criterio
    
    # Para el análisis de proximidad
    proximity_analysis_results = {
        'Monte Carlo': {'Total_Combs_Checked': 0},
        'Números Calientes': {'Total_Combs_Checked': 0}
    }
    # Inicializar las claves de proximidad dinámicamente
    for method_key in proximity_analysis_results.keys():
        for i in range(6): # 0 to 5 matching balotas
            proximity_analysis_results[method_key][f'{i}+SB'] = 0
            proximity_analysis_results[method_key][f'{i}-SB'] = 0


    # --- Bucle de Comparación ---
    for sim_combo_info in all_simulated_combinations_with_method:
        generated_balotas = set(sim_combo_info['combo'][0]) # Convert to set for easy comparison
        generated_sb = sim_combo_info['combo'][1]
        method = sim_combo_info['method']

        for historical_balotas_tuple, historical_sb in historical_winning_combos_full:
            historical_balotas_set = set(historical_balotas_tuple)

            # Calcular aciertos para proximidad
            matching_balotas = len(generated_balotas.intersection(historical_balotas_set))
            superbalota_match = (generated_sb == historical_sb)

            # Acumular para análisis de proximidad
            if include_proximity_analysis:
                proximity_key = f"{matching_balotas}+SB" if superbalota_match else f"{matching_balotas}-SB"
                proximity_analysis_results[method][proximity_key] += 1
                proximity_analysis_results[method]['Total_Combs_Checked'] += 1


            # Evaluar el criterio de acierto seleccionado
            is_match = False
            if match_criteria == "Exacto (5 Balotas + SuperBalota)":
                is_match = (matching_balotas == 5 and superbalota_match)
            elif match_criteria == "5 Balotas (sin SuperBalota)":
                is_match = (matching_balotas == 5)
            elif match_criteria == "4 Balotas + SuperBalota":
                is_match = (matching_balotas == 4 and superbalota_match)
            elif match_criteria == "4 Balotas (sin SuperBalota)":
                is_match = (matching_balotas == 4)
            elif match_criteria == "3 Balotas + SuperBalota":
                is_match = (matching_balotas == 3 and superbalota_match)
            elif match_criteria == "3 Balotas (sin SuperBalota)":
                is_match = (matching_balotas == 3)
            # Puedes añadir más criterios si lo deseas

            if is_match:
                wins_by_method[method].add((historical_balotas_tuple, historical_sb)) # Añadir el sorteo histórico que fue acertado
                # Opcional: st.success(f"DEBUG: ¡Coincidencia encontrada! Combo: {sim_combo_info['combo']}, Histórico: ({historical_balotas_tuple}, {historical_sb}), Método: {method}")

    # --- Resumen de Resultados ---
    total_unique_wins_mc = len(wins_by_method['Monte Carlo'])
    total_unique_wins_hot = len(wins_by_method['Números Calientes'])
    
    all_unique_wins_set = wins_by_method['Monte Carlo'].union(wins_by_method['Números Calientes'])
    total_unique_wins = len(all_unique_wins_set)

    message = f"**Resultados de la Simulación Histórica 'Inteligente' para {st.session_state.selected_sorteo_type}**:\n\n"
    message += f"Se intentaron generar hasta **{num_combinations_per_method}** combinaciones *únicas* por cada método y se compararon con **{len(df_current)}** sorteos históricos reales.\n"
    message += f"**Criterio de Acierto Aplicado:** `{match_criteria}`\n"
    message += f"- **Combinaciones de Monte Carlo habrían acertado:** **{total_unique_wins_mc}** sorteos históricos únicos.\n"
    message += f"- **Combinaciones de Números Calientes habrían acertado:** **{total_unique_wins_hot}** sorteos históricos únicos.\n"
    message += f"**Total de sorteos históricos únicos cubiertos por al menos un método:** **{total_unique_wins}**.\n\n"
    message += "Esto muestra la capacidad de los modelos para generar combinaciones que, bajo el criterio seleccionado, habrían sido ganadoras en el pasado."
    message += "\n*Nota: Un resultado de cero puede indicar que las combinaciones ganadoras son extremadamente raras o que los métodos de generación necesitan ajuste para capturar patrones más diversos. La aleatoriedad intrínseca de los sorteos de lotería significa que estas simulaciones no son predictivas de resultados futuros.*"

    return total_unique_wins_mc, total_unique_wins_hot, total_unique_wins, message, proximity_analysis_results


# --- Verificar si los datos se cargaron correctamente ---
if not df.empty:
    st.success(f"¡Datos de {st.session_state.selected_sorteo_type} cargados exitosamente! Fecha del último sorteo registrado: " + df['Fecha'].max().strftime('%d/%m/%Y'))

    tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🤖 Pronósticos y Simulación", "ℹ️ Acerca de"])

    with tab1:
        st.header(f"Análisis Exploratorio de Datos Históricos de {st.session_state.selected_sorteo_type}")
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
        metric_selection = st.radio("Selecciona la métrica a visualizar:", ('Conteo', 'Promedio', 'Mediana'), horizontal=True, key='heatmap_metric')
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
            # Asegurarse de que el índice de superbalota_full_range se alinee correctamente
            heatmap_final['SuperBalota'] = superbalota_full_range.reindex(heatmap_final.index, fill_value=pd.NA)

            fig7, ax7 = plt.subplots(figsize=(12, 10))
            sns.heatmap(heatmap_final, annot=True, fmt=".0f", cmap='viridis', linewidths=.5, linecolor='black', ax=ax7)
            ax7.set_title(f'Mapa de Calor Consolidado: {metric_selection} por Número y Balota ({st.session_state.selected_sorteo_type})')
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
            ax2.set_title(f'Top 10 Balotas Más Frecuentes ({st.session_state.selected_sorteo_type}, 1-43)')
            ax2.set_xlabel('Número de Balota')
            ax2.set_ylabel('Frecuencia')
            st.pyplot(fig2)
        with col2:
            st.markdown("##### Top 10 SuperBalotas")
            top_superbalotas = df['SuperBalota'].value_counts().head(10)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_superbalotas.index, y=top_superbalotas.values, palette='magma', ax=ax3, hue=top_superbalotas.index, legend=False)
            ax3.set_title(f'Top 10 SuperBalotas Más Frecuentes ({st.session_state.selected_sorteo_type}, 1-16)')
            ax3.set_xlabel('Número de SuperBalota')
            ax3.set_ylabel('Frecuencia')
            st.pyplot(fig3)

        st.subheader("🔗 Matriz de Correlación entre Balotas")
        numeric_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
        correlation_matrix = df[numeric_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
        ax4.set_title(f'Matriz de Correlación entre Balotas ({st.session_state.selected_sorteo_type})')
        st.pyplot(fig4)

        st.subheader("⏳ Tendencia Anual del Promedio de Cada Balota")
        df_avg_by_year = df.groupby('Año')[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].mean().reset_index()
        fig5, ax5 = plt.subplots(figsize=(14, 7))
        df_avg_by_year_melted = df_avg_by_year.melt('Año', var_name='Balota', value_name='Promedio')
        sns.lineplot(x='Año', y='Promedio', hue='Balota', data=df_avg_by_year_melted, marker='o', ax=ax5)
        ax5.set_title(f'Promedio Anual de los Números para Cada Balota ({st.session_state.selected_sorteo_type})')
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
        st.header(f"🤖 Herramientas de Pronóstico y Simulación para {st.session_state.selected_sorteo_type}")
        st.write("Aquí puedes explorar diferentes enfoques para generar posibles combinaciones de Baloto, incluyendo simulaciones y recomendaciones basadas en datos históricos. **Recuerda:** Los sorteos de lotería son aleatorios y estas herramientas son para fines de entretenimiento y análisis, no garantizan resultados.")

        if model:
            num_sims = st.slider("Número de simulaciones para Montecarlo (y para la Simulación de Ganancias):", min_value=10000, max_value=1000000, value=20000, step=1000, key='num_sim_slider')
            
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
            st.subheader("🎲 Simulación de Ganancias Históricas (con una combinación específica)")
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
                        [opt[0] for opt in forecast_options],
                        key='specific_sim_combo'
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

            st.markdown("---")
            st.subheader("🧠 Simulación Histórica 'Inteligente' (Rendimiento del Modelo)")
            st.write("Esta simulación evalúa cuántas veces las combinaciones generadas por nuestros métodos habrían acertado los sorteos históricos reales de Baloto/Revancha.")
            
            num_smart_combos = st.slider(
                "Número de combinaciones a generar por método para la Simulación Histórica 'Inteligente':",
                min_value=100, # Aumentado el mínimo
                max_value=1_000_000, # Aumentado drásticamente el máximo (1 millón)
                value=5000, # Valor predeterminado más alto para empezar a ver algo
                step=100,
                key='num_smart_combos_slider_2' # Cambiado el key para evitar conflictos
            )

            match_criteria_option = st.selectbox(
                "Criterio de Acierto para la Simulación Histórica:",
                [
                    "Exacto (5 Balotas + SuperBalota)",
                    "5 Balotas (sin SuperBalota)",
                    "4 Balotas + SuperBalota",
                    "4 Balotas (sin SuperBalota)",
                    "3 Balotas + SuperBalota",
                    "3 Balotas (sin SuperBalota)"
                ],
                key='match_criteria_selector_2' # Cambiado el key
            )

            include_proximity_analysis = st.checkbox(
                "Incluir análisis de proximidad (cuántas balotas y SuperBalota coinciden)?",
                value=True, # Por defecto activado para ver más insights
                key='proximity_analysis_checkbox_2' # Cambiado el key
            )

            if st.button("Ejecutar Simulación Histórica 'Inteligente'"):
                total_unique_wins_mc, total_unique_wins_hot, total_unique_wins, message, proximity_results = \
                    smart_historical_simulation(df, num_smart_combos, match_criteria_option, include_proximity_analysis)
                
                st.markdown(message)

                if include_proximity_analysis:
                    st.subheader("📊 Resultados del Análisis de Proximidad")
                    st.write("Muestra cuántas veces las combinaciones generadas se acercaron a los sorteos históricos ganadores:")

                    proximity_df_data = []
                    for method, data in proximity_results.items():
                        total_checked = data.get('Total_Combs_Checked', 0) # Use .get with default 0
                        
                        row = {'Método': method}
                        for i in range(6): # 0 to 5 matching balotas
                            # Match with SuperBalota (+SB)
                            key_plus_sb = f"{i}+SB"
                            count_plus_sb = data.get(key_plus_sb, 0)
                            percentage_plus_sb = (count_plus_sb / total_checked * 100) if total_checked > 0 else 0
                            row[f'{i} B + SB'] = f"{count_plus_sb} ({percentage_plus_sb:.2f}%)"

                            # Match without SuperBalota (-SB)
                            key_minus_sb = f"{i}-SB"
                            count_minus_sb = data.get(key_minus_sb, 0)
                            percentage_minus_sb = (count_minus_sb / total_checked * 100) if total_checked > 0 else 0
                            row[f'{i} B - SB'] = f"{count_minus_sb} ({percentage_minus_sb:.2f}%)"
                        proximity_df_data.append(row)
                    
                    if proximity_df_data:
                        proximity_df = pd.DataFrame(proximity_df_data)
                        st.dataframe(proximity_df)
                        st.info("La columna 'Total_Combs_Checked' indica el número total de comparaciones realizadas (combinación generada vs. sorteo histórico). Los porcentajes se calculan sobre este total. Ten en cuenta que un mismo sorteo histórico puede ser acertado por múltiples combinaciones generadas.")
                    else:
                        st.info("No hay datos de proximidad para mostrar. Asegúrate de que se generaron combinaciones y se realizó la simulación.")

            st.warning("Esta simulación es una métrica de rendimiento histórico del modelo, no una predicción de sorteos futuros. Los sorteos son eventos independientes.")


        else:
            st.warning("La funcionalidad de Gemini AI no está disponible. Por favor, verifica tu API Key para habilitar los pronósticos.")


    with tab3:
        st.header("ℹ️ Acerca de esta Aplicación")
        st.write(f"""
        Esta aplicación de Streamlit fue creada para realizar un Análisis Exploratorio de Datos (EDA) sobre los resultados históricos del Baloto/Revancha colombiano.
        Los datos se cargan directamente desde archivos CSV alojados en GitHub.

        **Características Clave:**
        * **Selector de Sorteo:** Elige entre los datos históricos de **{st.session_state.selected_sorteo_type}**.
        * Visualización de distribuciones de frecuencia de balotas individuales y SuperBalota.
        * Identificación de las balotas más frecuentes.
        * Análisis de correlación entre las posiciones de las balotas.
        * Tendencias del promedio y la distribución de números para *cada balota individualmente* a lo largo del tiempo y por año.
        * Mapa de Calor Consolidado para visualizar el conteo de números por balota.
        * **Herramientas de Pronóstico y Simulación:**
            * **Integración con Google Gemini AI:** Genera 5 sugerencias de números con justificación, considerando **rangos posicionales y frecuencias**.
            * **Simulación de Montecarlo:** Genera 5 combinaciones hipotéticas basadas en probabilidades históricas y **rangos observados por posición**.
            * **Recomendación de Números 'Calientes':** Genera 5 combinaciones basadas en la frecuencia de aparición por posición y **rangos observados por posición**.
            * **Puntaje de Frecuencia Histórica y Nivel de Calidez (%):** Evalúa la "calidez" de las combinaciones en relación con los patrones históricos.
            * **Almacenamiento y Comparación:** Guarda y muestra el Top 3 de los pronósticos generados.
            * **Simulación de Ganancias Históricas (Específica):** Estima ganancias al jugar una combinación N veces contra los sorteos históricos.
            * **Simulación Histórica 'Inteligente' (Rendimiento del Modelo):** Evalúa cuántas veces las combinaciones generadas por nuestros métodos habrían coincidido con sorteos históricos reales, permitiendo **criterios de acierto flexibles** y un **análisis detallado de proximidad**.

        **Desarrollado por:** Julian Torres (con asistencia de un modelo de lenguaje de Google).
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Google_Gemini_logo.svg/1200px-Google_Gemini_logo.svg.png", width=150)
        st.write("Los resultados de Baloto/Revancha son aleatorios. Por favor, juega responsablemente.")

else:
    st.error(f"No se pudieron cargar los datos del Baloto/Revancha. Por favor, asegúrate de que la URL sea correcta y el archivo esté accesible para {st.session_state.selected_sorteo_type}.")
    st.info("Intenta revisar la URL del archivo en tu repositorio de GitHub o la conexión a internet. Si el problema persiste, verifica el formato del CSV.")

st.markdown("---")
st.write("¡Gracias por usar la aplicación!")
