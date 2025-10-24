import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. CARGA DE DATOS DEL CSV
# ============================================
print("="*70)
print("   SISTEMA DE DIAGNÓSTICO DE DIABETES - RED NEURONAL")
print("="*70)
print("\nCargando datos del archivo CSV...")

# Cargar el dataset
df = pd.read_csv('diagnosticoDiabetes.csv')

# Renombrar columnas para facilitar su uso
df_renamed = df.rename(columns={
    'Pregnancies': 'Embarazos',
    'Glucose': 'Glucosa',
    'BloodPressure': 'Presión Arterial',
    'SkinThickness': 'Grosor Piel (mm)',
    'Insulin': 'Insulina',
    'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Función Pedigrí',
    'Age': 'Edad',
    'Outcome': 'Diabetes'
})

# Actualizar el DataFrame
df = df_renamed

# Preparar características y variable objetivo
X = df[['Embarazos', 'Glucosa', 'Presión Arterial', 'Grosor Piel (mm)', 'Insulina', 'IMC', 'Edad']].values
y = df['Diabetes'].values

total_pacientes = len(df)
pacientes_diabetes = y.sum()
porcentaje_diabetes = (pacientes_diabetes / total_pacientes) * 100
porcentaje_sin_diabetes = ((total_pacientes - pacientes_diabetes) / total_pacientes) * 100

print(f"✓ Dataset cargado: {total_pacientes} pacientes")
print(f"✓ Pacientes con diabetes: {pacientes_diabetes} ({porcentaje_diabetes:.1f}%)")
print(f"✓ Pacientes sin diabetes: {total_pacientes - pacientes_diabetes} ({porcentaje_sin_diabetes:.1f}%)")

print("\n📊 Primeros 5 registros del dataset:")
print(df.head())

print("\n📈 Estadísticas descriptivas:")
print(df.describe())

# ============================================
# 2. PREPROCESAMIENTO DE DATOS
# ============================================
print("\n" + "="*70)
print("Preprocesando datos...")

# Dividir en conjunto de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Datos de entrenamiento: {X_train_scaled.shape[0]}")
print(f"✓ Datos de prueba: {X_test_scaled.shape[0]}")

# ============================================
# 3. CONSTRUCCIÓN DEL MODELO
# ============================================
print("\n" + "="*70)
print("Construyendo la red neuronal...")

model = keras.Sequential([
    # Capa de entrada
    keras.layers.Dense(32, activation='relu', input_shape=(7,)),
    keras.layers.Dropout(0.3),
    
    # Capas ocultas
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(8, activation='relu'),
    
    # Capa de salida (clasificación binaria)
    keras.layers.Dense(1, activation='sigmoid')  # Sigmoid para clasificación binaria
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Para clasificación binaria
    metrics=['accuracy', 'AUC']
)

print("✓ Modelo creado")
print("\nArquitectura del modelo:")
model.summary()

# ============================================
# 4. ENTRENAMIENTO DEL MODELO
# ============================================
print("\n" + "="*70)
print("Entrenando el modelo (esto puede tomar unos segundos)...")

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print("✓ Entrenamiento completado!")

# ============================================
# 5. EVALUACIÓN DEL MODELO
# ============================================
print("\n" + "="*70)
print("Evaluando el modelo...")

# Hacer predicciones
y_pred_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n📊 Métricas de Evaluación:")
print(f"  Exactitud (Accuracy): {accuracy*100:.2f}%")

print("\n📋 Matriz de Confusión:")
print(f"  Verdaderos Negativos: {conf_matrix[0,0]}")
print(f"  Falsos Positivos: {conf_matrix[0,1]}")
print(f"  Falsos Negativos: {conf_matrix[1,0]}")
print(f"  Verdaderos Positivos: {conf_matrix[1,1]}")

print("\n📈 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Sin Diabetes', 'Con Diabetes']))

# ============================================
# 6. FUNCIÓN PARA VALIDAR ENTRADA
# ============================================
def obtener_numero(prompt, minimo, maximo, tipo=float):
    """Función para validar entrada numérica del usuario"""
    while True:
        try:
            valor = tipo(input(prompt))
            if minimo <= valor <= maximo:
                return valor
            else:
                print(f"❌ Por favor ingresa un valor entre {minimo} y {maximo}")
        except ValueError:
            print("❌ Por favor ingresa un número válido")

def interpretar_resultado(probabilidad):
    """Interpreta la probabilidad de diabetes"""
    prob_porcentaje = probabilidad * 100
    
    if prob_porcentaje < 30:
        return "BAJO", "🟢", "Es poco probable que tengas diabetes"
    elif prob_porcentaje < 60:
        return "MODERADO", "🟡", "Existe un riesgo moderado de diabetes"
    else:
        return "ALTO", "🔴", "Es muy probable que tengas diabetes"

# ============================================
# 7. PREDICCIÓN INTERACTIVA
# ============================================
print("\n" + "="*70)
print("   INGRESA TUS DATOS PARA DIAGNÓSTICO DE DIABETES")
print("="*70)

# Bucle principal para múltiples predicciones
while True:
    print("\n📝 Ingresa los datos del paciente:\n")
    
    print("\n🔍 Selecciona una opción:")
    print("1. Ingresar datos manualmente")
    print("2. Seleccionar un paciente existente del dataset")
    print("3. Salir")
    
    opcion = input("\nIngrese el número de la opción deseada (1-3): ")
    
    if opcion == "3":
        print("\n" + "="*70)
        print("   ⚕️  ¡Gracias por usar el sistema de diagnóstico!")
        print("   ⚠️  Recuerda: Esta es solo una herramienta de apoyo.")
        print("       Consulta siempre con un profesional médico.")
        print("="*70)
        break
        
    elif opcion == "2":
        print("\n📊 Mostrando primeros 10 pacientes del dataset:")
        print("\nÍnd | Emb | Glu | Pre | Piel | Ins | IMC  | Función | Edad | Diag")
        print("-"*70)
        
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            print(f"{i:3d} | {row['Embarazos']:3.0f} | {row['Glucosa']:3.0f} | "
                  f"{row['Presión Arterial']:3.0f} | {row['Grosor Piel (mm)']:4.0f} | "
                  f"{row['Insulina']:3.0f} | {row['IMC']:5.1f} | "
                  f"{row['Función Pedigrí']:7.3f} | {row['Edad']:4.0f} | "
                  f"{row['Diabetes']}")
        
        while True:
            try:
                indice = int(input("\nIngrese el índice del paciente a predecir (0-9) o -1 para ver más: "))
                if indice == -1:
                    start_idx = 10
                    while start_idx < len(df):
                        print("\nÍnd | Emb | Glu | Pre | Piel | Ins | IMC  | Función | Edad | Diag")
                        print("-"*70)
                        for i in range(start_idx, min(start_idx + 10, len(df))):
                            row = df.iloc[i]
                            print(f"{i:3d} | {row['Embarazos']:3.0f} | {row['Glucosa']:3.0f} | "
                                  f"{row['Presión Arterial']:3.0f} | {row['Grosor Piel (mm)']:4.0f} | "
                                  f"{row['Insulina']:3.0f} | {row['IMC']:5.1f} | "
                                  f"{row['Función Pedigrí']:7.3f} | {row['Edad']:4.0f} | "
                                  f"{row['Diabetes']}")
                        start_idx += 10
                        if start_idx < len(df):
                            continuar = input("\n¿Desea ver más pacientes? (s/n): ").lower()
                            if continuar != 's':
                                break
                    continue
                
                if 0 <= indice < len(df):
                    paciente = df.iloc[indice]
                    paciente_nuevo = np.array([[
                        paciente['Embarazos'],
                        paciente['Glucosa'],
                        paciente['Presión Arterial'],
                        paciente['Grosor Piel (mm)'],
                        paciente['Insulina'],
                        paciente['IMC'],
                        paciente['Edad']
                    ]])
                    break
                else:
                    print("❌ Por favor ingrese un índice válido")
            except ValueError:
                print("❌ Por favor ingrese un número válido")
    
    elif opcion == "1":
        print("\n📝 Ingresa los datos del paciente:\n")
        
        # Solicitar datos al usuario
        embarazos_input = obtener_numero(
            "1. Número de embarazos (0-20): ",
            0, 20, int
        )
        
        glucosa_input = obtener_numero(
            "2. Nivel de glucosa en sangre (mg/dL, 0-200): ",
            0, 200, int
        )
        
        presion_input = obtener_numero(
            "3. Presión arterial (mm Hg, 0-122): ",
            0, 122, int
        )
        
        grosor_input = obtener_numero(
            "4. Grosor del pliegue cutáneo del tríceps (mm, 0-100): ",
            0, 100, int
        )
        
        insulina_input = obtener_numero(
            "5. Insulina sérica en 2 horas (mu U/ml, 0-846): ",
            0, 846, int
        )
        
        imc_input = obtener_numero(
            "6. Índice de Masa Corporal - IMC (0-70): ",
            0, 70, float
        )
        
        edad_input = obtener_numero(
            "7. Edad (21-90 años): ",
            21, 90, int
        )
        
        # Crear array con los datos ingresados
        paciente_nuevo = np.array([[
            embarazos_input,
            glucosa_input,
            presion_input,
            grosor_input,
            insulina_input,
            imc_input,
            edad_input
        ]])
    
    # Normalizar y predecir
    paciente_scaled = scaler.transform(paciente_nuevo)
    probabilidad = model.predict(paciente_scaled, verbose=0)[0][0]
    prediccion = 1 if probabilidad > 0.5 else 0
    
    # Interpretar resultado
    nivel_riesgo, emoji, mensaje = interpretar_resultado(probabilidad)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("   RESULTADO DEL DIAGNÓSTICO")
    print("="*70)
    
    print("\n📋 Datos del Paciente:")
    if opcion == "2":
        print(f"   • Embarazos: {paciente['Embarazos']}")
        print(f"   • Glucosa: {paciente['Glucosa']} mg/dL")
        print(f"   • Presión Arterial: {paciente['Presión Arterial']} mm Hg")
        print(f"   • Grosor Piel: {paciente['Grosor Piel (mm)']} mm")
        print(f"   • Insulina: {paciente['Insulina']} mu U/ml")
        print(f"   • IMC: {paciente['IMC']:.1f}")
        print(f"   • Función Pedigrí: {paciente['Función Pedigrí']:.3f}")
        print(f"   • Edad: {paciente['Edad']} años")
    else:
        print(f"   • Embarazos: {embarazos_input}")
        print(f"   • Glucosa: {glucosa_input} mg/dL")
        print(f"   • Presión Arterial: {presion_input} mm Hg")
        print(f"   • Grosor Piel: {grosor_input} mm")
        print(f"   • Insulina: {insulina_input} mu U/ml")
        print(f"   • IMC: {imc_input:.1f}")
        print(f"   • Edad: {edad_input} años")
    
    print(f"\n{emoji} DIAGNÓSTICO: {'DIABETES DETECTADA' if prediccion == 1 else 'SIN DIABETES'}")
    print(f"\n📊 Probabilidad de diabetes: {probabilidad*100:.2f}%")
    print(f"🎯 Nivel de riesgo: {nivel_riesgo}")
    print(f"💬 {mensaje}")
    
    # Análisis de factores de riesgo
    print("\n⚠️  Análisis de Factores de Riesgo:")
    factores_riesgo = []
    
    # Obtener valores según la opción seleccionada
    if opcion == "2":
        glucosa_val = paciente['Glucosa']
        imc_val = paciente['IMC']
        edad_val = paciente['Edad']
        presion_val = paciente['Presión Arterial']
        insulina_val = paciente['Insulina']
    else:
        glucosa_val = glucosa_input
        imc_val = imc_input
        edad_val = edad_input
        presion_val = presion_input
        insulina_val = insulina_input
    
    if glucosa_val > 140:
        factores_riesgo.append("   🔴 Glucosa alta (>140 mg/dL)")
    if imc_val > 30:
        factores_riesgo.append("   🔴 Sobrepeso/Obesidad (IMC >30)")
    if edad_val > 45:
        factores_riesgo.append("   🟡 Edad mayor a 45 años")
    if presion_val > 90:
        factores_riesgo.append("   🟡 Presión arterial elevada")
    if insulina_val > 200:
        factores_riesgo.append("   🟡 Nivel de insulina elevado")
    
    if factores_riesgo:
        for factor in factores_riesgo:
            print(factor)
    else:
        print("   🟢 No se detectaron factores de riesgo significativos")
    
    # Recomendaciones
    print("\n💡 Recomendaciones:")
    if prediccion == 1 or probabilidad > 0.6:
        print("   • Consulta con un médico endocrinólogo lo antes posible")
        print("   • Realiza análisis de sangre completos")
        print("   • Monitorea tu glucosa regularmente")
        print("   • Considera cambios en tu dieta y ejercicio")
    elif probabilidad > 0.3:
        print("   • Mantén un estilo de vida saludable")
        print("   • Realiza chequeos médicos periódicos")
        print("   • Controla tu peso y haz ejercicio regularmente")
    else:
        print("   • Continúa con tus hábitos saludables")
        print("   • Realiza chequeos preventivos anuales")
    
    # Comparar con casos similares
    casos_similares = df[
        (np.abs(df['Edad'] - edad_val) <= 10) &
        (np.abs(df['Glucosa'] - glucosa_val) <= 20) &
        (np.abs(df['IMC'] - imc_val) <= 5)
    ]
    
    if len(casos_similares) > 0:
        diabetes_similares = casos_similares['Diabetes'].mean() * 100
        print(f"\n📊 En casos similares del dataset:")
        print(f"   • {len(casos_similares)} casos encontrados")
        print(f"   • {diabetes_similares:.1f}% tienen diabetes")
    
    print("\n" + "="*70)
    
    # Preguntar si desea hacer otra predicción
    continuar = input("\n¿Deseas realizar otro diagnóstico? (s/n): ").lower()
    
    if continuar != 's':
        print("\n" + "="*70)
        print("   ⚕️  ¡Gracias por usar el sistema de diagnóstico!")
        print("   ⚠️  Recuerda: Esta es solo una herramienta de apoyo.")
        print("       Consulta siempre con un profesional médico.")
        print("="*70)
        break

# ============================================
# 8. GUARDAR VISUALIZACIONES (OPCIONAL)
# ============================================
guardar = input("\n¿Deseas guardar las visualizaciones del modelo? (s/n): ").lower()

if guardar == 's':
    print("\nGenerando visualizaciones...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Pérdida y Exactitud
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Exactitud del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 3: Matriz de Confusión
    plt.subplot(2, 3, 3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Diabetes', 'Con Diabetes'],
                yticklabels=['Sin Diabetes', 'Con Diabetes'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico 4: Distribución de Glucosa
    plt.subplot(2, 3, 4)
    plt.hist(df[df['Diabetes']==0]['Glucosa'], alpha=0.5, label='Sin Diabetes', bins=30)
    plt.hist(df[df['Diabetes']==1]['Glucosa'], alpha=0.5, label='Con Diabetes', bins=30)
    plt.xlabel('Nivel de Glucosa (mg/dL)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Glucosa')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 5: Distribución de IMC
    plt.subplot(2, 3, 5)
    plt.hist(df[df['Diabetes']==0]['IMC'], alpha=0.5, label='Sin Diabetes', bins=30)
    plt.hist(df[df['Diabetes']==1]['IMC'], alpha=0.5, label='Con Diabetes', bins=30)
    plt.xlabel('IMC')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de IMC')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 6: Distribución de Edad
    plt.subplot(2, 3, 6)
    plt.hist(df[df['Diabetes']==0]['Edad'], alpha=0.5, label='Sin Diabetes', bins=30)
    plt.hist(df[df['Diabetes']==1]['Edad'], alpha=0.5, label='Con Diabetes', bins=30)
    plt.xlabel('Edad (años)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Edad')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('diagnostico_diabetes.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficos guardados en 'diagnostico_diabetes.png'")
    plt.show()

print("\n✓ Programa finalizado")