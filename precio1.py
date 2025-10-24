import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================
# 1. CARGA DE DATOS DEL CSV
# ============================================
print("="*60)
print("   SISTEMA DE PREDICCIÓN DE PRECIOS DE CASAS")
print("="*60)
print("\nCargando datos del archivo CSV...")

# Cargar el dataset
df = pd.read_csv('predicionCasa.csv')

# Seleccionar características relevantes
selected_features = ['livingArea', 'bedrooms', 'age', 'landValue', 'bathrooms']
X = df[selected_features].values
y = df['price'].values

print(f"✓ Dataset cargado: {len(X)} casas")
print(f"✓ Precio promedio: ${y.mean():,.2f}")

# ============================================
# 2. PREPROCESAMIENTO DE DATOS
# ============================================
print("\nPreprocesando datos...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"✓ Datos de entrenamiento: {X_train_scaled.shape[0]}")
print(f"✓ Datos de prueba: {X_test_scaled.shape[0]}")

# ============================================
# 3. CONSTRUCCIÓN DEL MODELO
# ============================================
print("\nConstruyendo la red neuronal...")

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("✓ Modelo creado")

# ============================================
# 4. ENTRENAMIENTO DEL MODELO
# ============================================
print("\nEntrenando el modelo (esto puede tomar unos segundos)...")

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print("✓ Entrenamiento completado!")

# ============================================
# 5. EVALUACIÓN DEL MODELO
# ============================================
print("\nEvaluando el modelo...")

y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(np.mean((y_test - y_pred.flatten())**2))
mae = np.mean(np.abs(y_test - y_pred.flatten()))
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

print(f"\nMétricas de Evaluación:")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAE: ${mae:,.2f}")
print(f"  MAPE: {mape:.2f}%")

# ============================================
# 6. FUNCIÓN PARA VALIDAR ENTRADA
# ============================================
def obtener_numero(prompt, minimo, maximo, default=None, tipo=float):
    """Función para validar entrada numérica del usuario"""
    while True:
        try:
            entrada = input(prompt)
            if entrada.strip() == '' and default is not None:
                return default
            valor = tipo(entrada)
            if minimo <= valor <= maximo:
                return valor
            else:
                print(f"❌ Por favor ingresa un valor entre {minimo} y {maximo}")
        except ValueError:
            print("❌ Por favor ingresa un número válido")

# ============================================
# 7. PREDICCIÓN INTERACTIVA
# ============================================
print("\n" + "="*60)
print("   SISTEMA DE PREDICCIÓN DE PRECIOS DE CASAS")
print("="*60)

# Bucle principal para múltiples predicciones
while True:
    print("\n🔍 Selecciona una opción:")
    print("1. Ingresar datos manualmente")
    print("2. Seleccionar una casa existente del dataset")
    print("3. Salir")
    
    opcion = input("\nIngrese el número de la opción deseada (1-3): ")
    
    if opcion == "3":
        print("\n" + "="*60)
        print("   ¡Gracias por usar el sistema de predicción!")
        print("="*60)
        break
    
    elif opcion == "2":
        print("\n📊 Mostrando primeras 10 casas del dataset:")
        print("\nÍndice | Área  | Habitaciones | Edad | Valor Terreno | Baños")
        print("-"*65)
        
        for i in range(min(10, len(df))):
            print(f"{i:6d} | {df.iloc[i]['livingArea']:5.0f} | {df.iloc[i]['bedrooms']:11.0f} | {df.iloc[i]['age']:4.0f} | ${df.iloc[i]['landValue']:11.2f} | {df.iloc[i]['bathrooms']:5.1f}")
        
        while True:
            try:
                indice = int(input("\nIngrese el índice de la casa a predecir (0-9) o -1 para ver más casas: "))
                if indice == -1:
                    start_idx = 10
                    while start_idx < len(df):
                        print("\nÍndice | Área  | Habitaciones | Edad | Valor Terreno | Baños")
                        print("-"*65)
                        for i in range(start_idx, min(start_idx + 10, len(df))):
                            print(f"{i:6d} | {df.iloc[i]['livingArea']:5.0f} | {df.iloc[i]['bedrooms']:11.0f} | {df.iloc[i]['age']:4.0f} | ${df.iloc[i]['landValue']:11.2f} | {df.iloc[i]['bathrooms']:5.1f}")
                        start_idx += 10
                        if start_idx < len(df):
                            continuar = input("\n¿Desea ver más casas? (s/n): ").lower()
                            if continuar != 's':
                                break
                    continue
                
                if 0 <= indice < len(df):
                    casa_seleccionada = df.iloc[indice]
                    casa_nueva = np.array([[
                        casa_seleccionada['livingArea'],
                        casa_seleccionada['bedrooms'],
                        casa_seleccionada['age'],
                        casa_seleccionada['landValue'],
                        casa_seleccionada['bathrooms']
                    ]])
                    break
                else:
                    print("❌ Por favor ingrese un índice válido")
            except ValueError:
                print("❌ Por favor ingrese un número válido")
    
    elif opcion == "1":
        print("\n📝 Ingresa las características de la casa:\n")
    
    # Solicitar datos al usuario
    area_input = obtener_numero(
        "1. Área habitable (en pies cuadrados, 500-5000): ",
        500, 5000, float
    )
    
    habitaciones_input = obtener_numero(
        "2. Número de habitaciones (1-7): ", 
        1, 7, int
    )
    
    edad_input = obtener_numero(
        "3. Edad de la casa (0-100 años): ",
        0, 200, int
    )
    
    valor_terreno_input = obtener_numero(
        "4. Valor del terreno (en USD, 1000-300000): ",
        1000, 300000, float
    )
    
    baños_input = obtener_numero(
        "5. Número de baños (1-7): ",
        1, 5, float
    )
    
    # Crear array con los datos ingresados
    casa_nueva = np.array([[
        area_input,
        habitaciones_input,
        edad_input,
        valor_terreno_input,
        baños_input
    ]])
    
    # Normalizar y predecir
    casa_nueva_scaled = scaler_X.transform(casa_nueva)
    precio_predicho_scaled = model.predict(casa_nueva_scaled, verbose=0)
    precio_predicho = scaler_y.inverse_transform(precio_predicho_scaled)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("   RESULTADO DE LA PREDICCIÓN")
    print("="*60)
    print("\n📊 Características ingresadas:")
    print(f"   • Área habitable: {area_input:,.0f} ft²")
    print(f"   • Habitaciones: {habitaciones_input:,.0f}")
    print(f"   • Edad de la casa: {edad_input} años")
    print(f"   • Valor del terreno: ${valor_terreno_input:,.2f}")
    print(f"   • Baños: {baños_input:.1f}")
    print(f"\n💰 PRECIO ESTIMADO: ${precio_predicho[0][0]:,.2f}")
    
    # Comparar con casas similares del dataset
    df_temp = pd.DataFrame({
        'Area': X[:, 0],
        'Habitaciones': X[:, 1],
        'Edad': X[:, 2],
        'Valor_Terreno': X[:, 3],
        'Baños': X[:, 4],
        'Precio': y
    })
    
    # Encontrar casas similares (tolerancia del 20% en área)
    tolerancia_area = area_input * 0.2
    similares = df_temp[
        (np.abs(df_temp['Area'] - area_input) <= tolerancia_area) &
        (df_temp['Habitaciones'] == habitaciones_input) &
        (df_temp['Baños'] == baños_input)
    ]
    
    if len(similares) > 0:
        print(f"\n📈 Comparación con casas similares en el dataset:")
        print(f"   • Casas similares encontradas: {len(similares)}")
        print(f"   • Precio promedio: ${similares['Precio'].mean():,.2f}")
        print(f"   • Precio mínimo: ${similares['Precio'].min():,.2f}")
        print(f"   • Precio máximo: ${similares['Precio'].max():,.2f}")
    
    # Calcular rango de confianza (±10%)
    precio_min = precio_predicho[0][0] * 0.9
    precio_max = precio_predicho[0][0] * 1.1
    print(f"\n📊 Rango de confianza (±10%):")
    print(f"   ${precio_min:,.2f} - ${precio_max:,.2f}")
    
    print("\n" + "="*60)
    
    # Preguntar si desea hacer otra predicción
    print("\n" + "="*60)
    
    if opcion == "2":
        print(f"\n💡 Precio real de la casa seleccionada: ${df.iloc[indice]['price']:,.2f}")
        print(f"📊 Diferencia con la predicción: ${abs(df.iloc[indice]['price'] - precio_predicho[0][0]):,.2f}")
        print(f"   Error porcentual: {abs(df.iloc[indice]['price'] - precio_predicho[0][0]) / df.iloc[indice]['price'] * 100:.2f}%")
    
    continuar = input("\n¿Deseas predecir el precio de otra casa? (s/n): ").lower()
    if continuar != 's':
        break

# ============================================
# 8. GUARDAR HISTORIAL (OPCIONAL)
# ============================================
guardar = input("\n¿Deseas guardar las visualizaciones del modelo? (s/n): ").lower()

if guardar == 's':
    print("\nGenerando visualizaciones...")
    
    plt.figure(figsize=(15, 5))
    
    # Gráfico 1: Pérdida
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 2: Predicciones vs Reales
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Precio Real ($)')
    plt.ylabel('Precio Predicho ($)')
    plt.title('Predicciones vs Valores Reales')
    plt.grid(True)
    
    # Gráfico 3: Errores
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
    print("✓ Gráficos guardados en 'resultados_prediccion.png'")
    plt.show()

# Finalizar el programa
print("\n✓ Programa finalizado")
