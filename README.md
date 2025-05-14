# 🛰️ Simulación de Monitoreo de Zonas Contaminadas

Sistema de simulación para comparar algoritmos de aprendizaje por refuerzo (Q-Learning, SARSA y Monte Carlo) en un entorno de monitoreo de drones.

## 📋 Descripción

Este proyecto implementa un entorno de simulación donde un dron debe navegar por una cuadrícula para monitorear zonas contaminadas mientras evita áreas peligrosas. Se implementan y comparan tres algoritmos clásicos de aprendizaje por refuerzo:

- **Q-Learning**: Algoritmo off-policy que aprende el valor óptimo de cada par estado-acción.
- **SARSA**: Algoritmo on-policy que aprende el valor de la política que está siguiendo.
- **Monte Carlo**: Algoritmo basado en episodios completos que aprende de experiencias completas.

## 🚀 Instalación

1. Clona este repositorio:
   ```
   git clone <url-del-repositorio>
   cd <nombre-directorio>
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## 🎮 Uso

Puedes ejecutar la aplicación Streamlit con:

```
streamlit run main.py
```

### Interfaz Web

La interfaz te permite:
- Seleccionar el algoritmo a utilizar
- Ajustar parámetros como tasa de aprendizaje (α), factor de descuento (γ) y exploración (ε)
- Visualizar el entorno, la política aprendida y el recorrido del dron
- Comparar el rendimiento entre algoritmos
- Analizar la robustez ante cambios en parámetros

## 📊 Métricas de Comparación

Las métricas implementadas para comparar los algoritmos incluyen:

- **Recompensa total promedio**: Promedio de recompensa acumulada por episodio
- **Tasa de convergencia**: Número de episodios necesarios para estabilizar el aprendizaje
- **Estabilidad**: Varianza de la recompensa a lo largo del tiempo
- **Tiempo de entrenamiento**: Tiempo computacional requerido por algoritmo
- **Robustez**: Sensibilidad ante cambios en ε o α

## 🧪 Estructura del Proyecto

- `main.py`: Aplicación principal con Streamlit
- `env/`: Implementación del entorno de simulación
- `agents/`: Implementación de los algoritmos de RL
- `utils/`: Utilidades para métricas, visualizaciones, etc.
- `tests/`: Scripts para pruebas unitarias

## 📝 Autores

- Javiera Cerda
- Cristobal Ricciardi.
