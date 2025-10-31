# Manual de Usuario - Sistema de Diagnóstico de Diabetes

## Índice
1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Inicio del Programa](#inicio-del-programa)
4. [Menú Principal](#menú-principal)
5. [Ingreso de Datos](#ingreso-de-datos)
6. [Interpretación de Resultados](#interpretación-de-resultados)
7. [Visualizaciones](#visualizaciones)
8. [Recomendaciones](#recomendaciones)

## Introducción

El Sistema de Diagnóstico de Diabetes es una herramienta de apoyo que utiliza inteligencia artificial para evaluar el riesgo de diabetes en pacientes. El sistema utiliza una red neuronal entrenada con datos reales y proporciona una evaluación detallada del riesgo junto con recomendaciones personalizadas.

## Requisitos del Sistema

- Python 3.x
- Bibliotecas necesarias:
  - TensorFlow
  - NumPy
  - Pandas
  - Matplotlib

## Inicio del Programa

1. Abra una terminal o línea de comandos
2. Navegue hasta la carpeta del proyecto
3. Ejecute el comando:
   ```
   python diabetes.py
   ```

## Menú Principal

El sistema presenta tres opciones principales:

1. **Ingresar datos manualmente**
   - Para pacientes nuevos
   - Permite introducir todos los parámetros médicos

2. **Seleccionar un paciente existente del dataset**
   - Para ver ejemplos o casos anteriores
   - Permite seleccionar de una base de datos existente

3. **Salir**
   - Finaliza el programa

## Ingreso de Datos

### Opción 1: Ingreso Manual

1. **Selección de Género**
   - Masculino
   - Femenino

2. **Datos Requeridos**
   - Número de embarazos (solo para mujeres)
   - Nivel de glucosa en sangre (0-200 mg/dL)
   - Presión arterial (0-122 mm Hg)
   - Grosor del pliegue cutáneo (0-100 mm)
   - Insulina sérica (0-846 mu U/ml)
   - Índice de Masa Corporal - IMC (0-70)
   - Edad (21-90 años)

### Opción 2: Selección de Paciente Existente

1. Se muestra una lista de 10 pacientes a la vez
2. Puede ver más pacientes ingresando -1
3. Seleccione el índice del paciente deseado

## Interpretación de Resultados

El sistema muestra:

1. **Datos del Paciente**
   - Resumen de todos los valores ingresados
   - Género del paciente

2. **Diagnóstico**
   - Predicción de diabetes
   - Probabilidad en porcentaje
   - Nivel de riesgo (BAJO 🟢, MODERADO 🟡, ALTO 🔴)

3. **Análisis de Factores de Riesgo**
   - Glucosa alta
   - Sobrepeso/Obesidad
   - Edad
   - Presión arterial
   - Nivel de insulina

4. **Recomendaciones Personalizadas**
   - Basadas en el nivel de riesgo
   - Sugerencias médicas específicas

5. **Comparación con Casos Similares**
   - Número de casos similares encontrados
   - Porcentaje de casos con diabetes

## Visualizaciones

Al finalizar, el sistema ofrece la opción de guardar visualizaciones:

1. **Gráficos de Entrenamiento**
   - Pérdida del modelo
   - Exactitud del modelo

2. **Matriz de Confusión**
   - Muestra la precisión del modelo

3. **Distribuciones**
   - Glucosa
   - IMC
   - Edad

Para guardar las visualizaciones:
1. Responda 's' cuando se le pregunte
2. Los gráficos se guardarán como 'diagnostico_diabetes.png'

## Recomendaciones

- Ingrese los datos con la mayor precisión posible
- Consulte siempre con un profesional médico
- Use este sistema solo como herramienta de apoyo
- Realice chequeos médicos regulares
- Mantenga un registro de sus resultados

## ⚠️ Importante

Este sistema es una herramienta de apoyo y no reemplaza el diagnóstico médico profesional. Siempre consulte con un profesional de la salud para un diagnóstico definitivo y tratamiento adecuado.

---
*Sistema desarrollado con fines académicos y de investigación.*