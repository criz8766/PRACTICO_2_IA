import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
import json
import sys

# Añadir el directorio actual al path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar clases necesarias
from env.drone_env import DroneEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.monte_carlo import MonteCarloAgent

st.set_page_config(page_title="Visualizador de Recorrido - Dron", layout="centered")

st.title("🛸 Visualizador de Recorrido del Dron")
st.write("Esta herramienta independiente permite visualizar el recorrido del dron paso a paso.")

# Función para cargar un archivo JSON
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error al cargar el archivo JSON: {e}")
        return None

# Función para visualizar el recorrido
def visualize_path(grid, trajectory, step):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Colores para cada tipo de celda
    colors = {
        'safe': [0.8, 0.8, 0.8],      # Gris claro
        'contaminated': [0.0, 0.8, 0.0],  # Verde
        'danger': [0.8, 0.0, 0.0],    # Rojo
        'goal': [1.0, 0.84, 0.0],      # Dorado
    }
    
    # Símbolos para los tipos de celda
    symbols = {
        'safe': ' ',
        'contaminated': '🌿',
        'danger': '⚠️',
        'goal': '🏁',
        'dron': '🛸'
    }
    
    # Crear matriz de colores
    color_matrix = np.zeros((grid.shape[0], grid.shape[1], 3))
    
    # Colorear según tipo de celda
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell_type = grid[i, j]
            color_matrix[i, j] = colors.get(cell_type, colors['safe'])
    
    # Mostrar matriz base
    ax.imshow(color_matrix)
    
    # Marcar el recorrido hasta el paso actual
    visited_cells = trajectory[:step+1]
    if len(visited_cells) > 1:
        # Convertir coordenadas para el gráfico
        path_y = [pos[1] for pos in visited_cells]
        path_x = [pos[0] for pos in visited_cells]
        ax.plot(path_x, path_y, 'o-', color='blue', linewidth=2)
    
    # Marcar posición actual del dron
    if step < len(trajectory):
        dron_x, dron_y = trajectory[step]
        circle = plt.Circle((dron_y, dron_x), 0.4, color='blue', alpha=0.5)
        ax.add_patch(circle)
        ax.text(dron_y, dron_x, symbols['dron'], ha='center', va='center', fontsize=18)
    
    # Añadir información a cada celda
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell_type = grid[i, j]
            symbol = symbols.get(cell_type, ' ')
            ax.text(j, i, symbol, ha='center', va='center', fontsize=14)
    
    # Configurar ejes
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels(np.arange(grid.shape[1]))
    ax.set_yticklabels(np.arange(grid.shape[0]))
    
    # Añadir líneas de cuadrícula
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.tick_params(which='minor', size=0)
    
    plt.tight_layout()
    return fig

# Función para crear un recorrido de ejemplo
def create_sample_trajectory():
    env = DroneEnv(seed=42)
    agent = QLearningAgent(actions=env.action_space, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # Generar un episodio
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
    
    return env.grid, env.get_trajectory()

# Comprobar si hay modelos guardados
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
has_models = os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0

# Pestaña de opciones
tab1, tab2 = st.tabs(["Usar modelo guardado" if has_models else "Generar ejemplo", 
                     "Cargar trayectoria personalizada"])

with tab1:
    if has_models:
        st.subheader("Cargar un modelo guardado")
        
        # Buscar algoritmos disponibles
        algorithms = []
        for algo_dir in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, algo_dir)):
                algorithms.append(algo_dir.replace("_", " ").title())
        
        if algorithms:
            # Elegir algoritmo
            selected_algo = st.selectbox("Algoritmo", algorithms)
            algo_dir = selected_algo.lower().replace(" ", "_")
            
            # Cargar modelo y generar trayectoria
            if st.button("Cargar y visualizar"):
                with st.spinner("Cargando modelo y generando trayectoria..."):
                    try:
                        env = DroneEnv(seed=42)
                        
                        # Mapeo de nombres de algoritmos a clases
                        agent_classes = {
                            "q learning": QLearningAgent,
                            "sarsa": SARSAAgent,
                            "monte carlo": MonteCarloAgent
                        }
                        
                        agent_class = agent_classes[algo_dir]
                        
                        # Cargar tabla Q
                        q_table_path = os.path.join(models_dir, algo_dir, "q_table.json")
                        if os.path.exists(q_table_path):
                            with open(q_table_path, 'r') as f:
                                q_data = json.load(f)
                                
                            # Crear agente y asignar tabla Q
                            if algo_dir == "monte carlo":
                                agent = agent_class(actions=env.action_space, gamma=0.9, epsilon=0.1)
                            else:
                                agent = agent_class(actions=env.action_space, alpha=0.1, gamma=0.9, epsilon=0.1)
                            
                            # Convertir claves de string a tupla
                            for state_str, actions in q_data.items():
                                state = eval(state_str)  # Convertir string a tupla
                                agent.q_table[state] = actions
                            
                            # Generar episodio
                            state = env.reset()
                            done = False
                            while not done:
                                action = agent.choose_action(state)
                                state, _, done = env.step(action)
                            
                            # Guardar grid y trayectoria
                            st.session_state.grid = env.grid
                            st.session_state.trajectory = env.get_trajectory()
                            st.success(f"Modelo {selected_algo} cargado correctamente")
                        else:
                            st.error(f"No se encontró la tabla Q para {selected_algo}")
                    except Exception as e:
                        st.error(f"Error al cargar el modelo: {e}")
        else:
            st.warning("No hay modelos guardados en el directorio 'models'")
            # Generar ejemplo
            if st.button("Generar ejemplo"):
                with st.spinner("Generando ejemplo..."):
                    grid, trajectory = create_sample_trajectory()
                    st.session_state.grid = grid
                    st.session_state.trajectory = trajectory
                    st.success("Ejemplo generado correctamente")
    else:
        st.subheader("Generar ejemplo")
        if st.button("Generar trayectoria de ejemplo"):
            with st.spinner("Generando ejemplo..."):
                grid, trajectory = create_sample_trajectory()
                st.session_state.grid = grid
                st.session_state.trajectory = trajectory
                st.success("Ejemplo generado correctamente")

with tab2:
    st.subheader("Cargar trayectoria personalizada")
    st.write("Sube un archivo JSON con la trayectoria y la configuración de la grilla")
    
    # Opción para cargar un archivo
    uploaded_file = st.file_uploader("Cargar archivo de trayectoria (JSON)", type=['json'])
    
    if uploaded_file is not None:
        try:
            # Cargar datos
            data = json.load(uploaded_file)
            if 'grid' in data and 'trajectory' in data:
                st.session_state.grid = np.array(data['grid'])
                st.session_state.trajectory = data['trajectory']
                st.success("Trayectoria cargada correctamente")
            else:
                st.error("El archivo debe contener 'grid' y 'trajectory'")
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

# Visualizador
if 'grid' in st.session_state and 'trajectory' in st.session_state:
    st.subheader("Visualizador de recorrido")
    
    # Información básica
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Tamaño de la grilla: {st.session_state.grid.shape}")
    with col2:
        st.write(f"Pasos en la trayectoria: {len(st.session_state.trajectory)}")
    
    # Control deslizante para seleccionar el paso
    step = st.slider("Paso:", 0, len(st.session_state.trajectory)-1, 0)
    
    # Mostrar información del paso actual
    pos = st.session_state.trajectory[step]
    cell_type = st.session_state.grid[pos]
    
    # Información del paso
    st.write(f"**Posición:** ({pos[0]}, {pos[1]})")
    st.write(f"**Tipo de celda:** {cell_type}")
    
    # Recompensa del paso
    reward = {
        'safe': -1,
        'contaminated': 10,
        'danger': -10,
        'goal': 20
    }.get(cell_type, -1)
    st.write(f"**Recompensa:** {reward}")
    
    # Visualización
    fig = visualize_path(st.session_state.grid, st.session_state.trajectory, step)
    st.pyplot(fig)
    
    # Opción para reproducir la secuencia
    if st.button("▶️ Reproducir secuencia"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Bucle de reproducción
        for i in range(step, len(st.session_state.trajectory)):
            # Actualizar progreso
            progress = int((i / (len(st.session_state.trajectory)-1)) * 100)
            progress_bar.progress(progress)
            status_text.write(f"Reproduciendo paso {i+1} de {len(st.session_state.trajectory)}")
            
            # Mostrar visualización del paso actual
            fig = visualize_path(st.session_state.grid, st.session_state.trajectory, i)
            st.pyplot(fig)
            
            # Pausa entre pasos
            time.sleep(0.5)
        
        # Marcar finalización
        progress_bar.progress(100)
        status_text.write("Reproducción completa")
else:
    st.info("Selecciona un modelo o genera un ejemplo para visualizar el recorrido.")

# Leyenda
with st.expander("Leyenda", expanded=False):
    st.markdown("""
    | Símbolo | Tipo | Recompensa |
    |---------|------|------------|
    | 🛸 | Dron | - |
    | 🌿 | Zona contaminada | +10 |
    | ⚠️ | Zona peligrosa | -10 |
    | 🏁 | Meta | +20 |
    """)

# Instrucciones
with st.expander("Instrucciones", expanded=False):
    st.write("""
    1. Usa el control deslizante para ver cada paso del recorrido.
    2. Haz clic en el botón "Reproducir secuencia" para ver la animación automática.
    3. Puedes cargar un modelo guardado o generar un ejemplo.
    """)

st.markdown("---")
st.caption("Visualizador de Recorrido del Dron - IA 2025") 