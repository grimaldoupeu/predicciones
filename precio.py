import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================
# 1. GENERACIÓN DE DATOS DE EJEMPLO
# ============================================
print("Generando datos de ejemplo...")

# Semilla para reproducibilidad
np.random.seed(42)

# Número de muestras
n_samples = 1000

# Características de las casas
tamaño = np.random.uniform(500, 5000, n_samples)  # Pies cuadrados
habitaciones = np.random.randint(1, 8, n_samples)  # Número de habitaciones
año_construccion = np.random.randint(1950, 2024, n_samples)  # Año de construcción
distancia_centro = np.random.uniform(0.5, 50, n_samples)  # Km al centro
baños = np.random.randint(1, 5, n_samples)  # Número de baños

# Crear matriz de características
X = np.column_stack([tamaño, habitaciones, año_construccion, distancia_centro, baños])

# Generar precios basados en las características (con algo de ruido)
# Fórmula: precio base + factores que influyen en el precio
precio = (
    50000 +  # Precio base
    tamaño * 100 +  # $100 por pie cuadrado
    habitaciones * 15000 +  # $15,000 por habitación
    (año_construccion - 1950) * 500 +  # Más caro si es más nuevo
    -distancia_centro * 2000 +  # Más barato si está lejos
    baños * 10000 +  # $10,000 por baño
    np.random.normal(0, 50000, n_samples)  # Ruido aleatorio
)

y = precio

# Crear DataFrame para visualización
df = pd.DataFrame({
    'Tamaño (ft²)': tamaño,
    'Habitaciones': habitaciones,
    'Año': año_construccion,
    'Distancia (km)': distancia_centro,
    'Baños': baños,
    'Precio ($)': precio
})

print("\nPrimeras 5 casas del dataset:")
print(df.head())
print(f"\nTotal de casas: {len(df)}")
print(f"\nEstadísticas del precio:")
print(f"  Mínimo: ${precio.min():,.2f}")
print(f"  Máximo: ${precio.max():,.2f}")
print(f"  Promedio: ${precio.mean():,.2f}")

# ============================================
# 2. PREPROCESAMIENTO DE DATOS
# ============================================
print("\n" + "="*50)
print("Preprocesando datos...")

# Dividir en conjunto de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizar los datos (importante para redes neuronales)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"Datos de entrenamiento: {X_train_scaled.shape[0]}")
print(f"Datos de prueba: {X_test_scaled.shape[0]}")

# ============================================
# 3. CONSTRUCCIÓN DEL MODELO
# ============================================
print("\n" + "="*50)
print("Construyendo la red neuronal...")

model = keras.Sequential([
    # Capa de entrada
    keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    keras.layers.Dropout(0.2),  # Para prevenir overfitting
    
    # Capas ocultas
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(16, activation='relu'),
    
    # Capa de salida
    keras.layers.Dense(1)  # 1 neurona para predecir el precio
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)

print("\nArquitectura del modelo:")
model.summary()

# ============================================
# 4. ENTRENAMIENTO DEL MODELO
# ============================================
print("\n" + "="*50)
print("Entrenando el modelo...")

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0  # Cambiar a 1 para ver el progreso detallado
)

print("¡Entrenamiento completado!")

# ============================================
# 5. EVALUACIÓN DEL MODELO
# ============================================
print("\n" + "="*50)
print("Evaluando el modelo...")

# Hacer predicciones
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calcular métricas
mse = np.mean((y_test - y_pred.flatten())**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred.flatten()))
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

print(f"\nMétricas de Evaluación:")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAE: ${mae:,.2f}")
print(f"  MAPE: {mape:.2f}%")

# ============================================
# 6. VISUALIZACIÓN DE RESULTADOS
# ============================================
print("\n" + "="*50)
print("Generando visualizaciones...")

# Gráfico 1: Pérdida durante el entrenamiento
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True)

# Gráfico 2: Predicciones vs Valores Reales
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Precio Real ($)')
plt.ylabel('Precio Predicho ($)')
plt.title('Predicciones vs Valores Reales')
plt.grid(True)

# Gráfico 3: Distribución de errores
plt.subplot(1, 3, 3)
errores = y_test - y_pred.flatten()
plt.hist(errores, bins=50, edgecolor='black')
plt.xlabel('Error de Predicción ($)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True)

plt.tight_layout()
plt.savefig('resultados_prediccion.png', dpi=300, bbox_inches='tight')
print("Gráficos guardados en 'resultados_prediccion.png'")
plt.show()

# ============================================
# 7. EJEMPLO DE PREDICCIÓN
# ============================================
print("\n" + "="*50)
print("Ejemplo de predicción con una casa nueva:")

# Casa de ejemplo
casa_ejemplo = np.array([[
    2500,  # Tamaño en ft²
    3,     # Habitaciones
    2015,  # Año de construcción
    5,     # Distancia al centro (km)
    2      # Baños
]])

# Normalizar y predecir
casa_ejemplo_scaled = scaler_X.transform(casa_ejemplo)
precio_predicho_scaled = model.predict(casa_ejemplo_scaled, verbose=0)
precio_predicho = scaler_y.inverse_transform(precio_predicho_scaled)

print(f"\nCaracterísticas de la casa:")
print(f"  Tamaño: 2,500 ft²")
print(f"  Habitaciones: 3")
print(f"  Año de construcción: 2015")
print(f"  Distancia al centro: 5 km")
print(f"  Baños: 2")
print(f"\n💰 Precio predicho: ${precio_predicho[0][0]:,.2f}")

# ============================================
# 8. COMPARACIÓN CON CASAS SIMILARES
# ============================================
print("\n" + "="*50)
print("Comparando con casas similares del dataset:")

# Encontrar casas similares
similares_idx = np.where(
    (np.abs(df['Tamaño (ft²)'] - 2500) < 300) &
    (df['Habitaciones'] == 3) &
    (df['Baños'] == 2)
)[0][:5]

if len(similares_idx) > 0:
    print("\nCasas similares encontradas:")
    print(df.iloc[similares_idx][['Tamaño (ft²)', 'Habitaciones', 'Año', 'Baños', 'Precio ($)']])
    print(f"\nPrecio promedio de casas similares: ${df.iloc[similares_idx]['Precio ($)'].mean():,.2f}")

print("\n" + "="*50)
print("¡Análisis completado!")
print("="*50)