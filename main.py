import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# URL del archivo CSV en GitHub
url = "https://raw.githubusercontent.com/JulianTorrest/modelos/refs/heads/main/Baloto.csv"

# Leer el archivo CSV
try:
    df = pd.read_csv(url)
    print("¡Archivo leído exitosamente!")
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit()

# Mostrar las primeras filas del DataFrame
print("\n--- Primeras 5 filas del DataFrame ---")
print(df.head())

# Información general del DataFrame
print("\n--- Información general del DataFrame ---")
df.info()

# Estadísticas descriptivas
print("\n--- Estadísticas descriptivas de las balotas ---")
print(df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']].describe())

# Convertir la columna 'Fecha' a formato de fecha
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')

# Extraer el año, mes y día para posibles análisis
df['Año'] = df['Fecha'].dt.year
df['Mes'] = df['Fecha'].dt.month
df['Dia'] = df['Fecha'].dt.day

print("\n--- DataFrame con columnas de fecha desglosadas ---")
print(df.head())

# Distribución de cada balota
plt.figure(figsize=(15, 10))

for i in range(1, 6):
    plt.subplot(2, 3, i)
    sns.histplot(df[f'Balota {i}'], bins=range(1, 44), kde=True)
    plt.title(f'Distribución Balota {i}')
    plt.xticks(range(1, 44, 2)) # Mostrar ticks para números pares para mejor lectura

plt.subplot(2, 3, 6)
sns.histplot(df['SuperBalota'], bins=range(1, 17), kde=True)
plt.title('Distribución SuperBalota')
plt.xticks(range(1, 17))

plt.tight_layout()
plt.suptitle('Distribución de Frecuencia de las Balotas', y=1.02, fontsize=16)
plt.show()

# Balotas más frecuentes
all_balotas = pd.concat([df['Balota 1'], df['Balota 2'], df['Balota 3'], df['Balota 4'], df['Balota 5']])
top_balotas = all_balotas.value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_balotas.index, y=top_balotas.values, palette='viridis')
plt.title('Top 10 Balotas Más Frecuentes (excluyendo SuperBalota)')
plt.xlabel('Número de Balota')
plt.ylabel('Frecuencia')
plt.show()

# SuperBalotas más frecuentes
top_superbalotas = df['SuperBalota'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_superbalotas.index, y=top_superbalotas.values, palette='magma')
plt.title('Top 10 SuperBalotas Más Frecuentes')
plt.xlabel('Número de SuperBalota')
plt.ylabel('Frecuencia')
plt.show()

# Análisis de correlación (puede no ser muy significativo en este contexto pero es parte de un EDA completo)
# Primero, asegurémonos de que las columnas sean numéricas y no haya valores nulos
numeric_cols = ['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5', 'SuperBalota']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación entre Balotas')
plt.show()

# Tendencia de la suma de balotas a lo largo del tiempo
df['SumaBalotas'] = df[['Balota 1', 'Balota 2', 'Balota 3', 'Balota 4', 'Balota 5']].sum(axis=1)

plt.figure(figsize=(12, 6))
sns.lineplot(x='Fecha', y='SumaBalotas', data=df)
plt.title('Suma de Balotas a lo largo del Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Suma de las Balotas')
plt.show()

# Análisis por año (si la columna 'Año' fue creada)
if 'Año' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Año', y='SumaBalotas', data=df)
    plt.title('Suma de Balotas por Año')
    plt.xlabel('Año')
    plt.ylabel('Suma de las Balotas')
    plt.show()
