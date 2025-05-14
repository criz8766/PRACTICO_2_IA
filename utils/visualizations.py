import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import streamlit as st
import time
import matplotlib.patches as patches

def plot_rewards(reward_history, title="Recompensa por Episodio"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reward_history, label="Recompensa")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa Total")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig  # ✅ retorna la figura para usar en st.pyplot(fig)

def compare_rewards(histories, labels, title="Comparación de Algoritmos"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for rewards, label in zip(histories, labels):
        ax.plot(rewards, label=label)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa Total")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig  # ✅ retorna la figura para usar en st.pyplot(fig)

def visualize_grid(grid, title="Entorno de simulación"):
    """Visualiza el entorno como una grilla coloreada."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Definir colores para cada tipo de celda
    colors = {
        'safe': 'lightgray',
        'contaminated': 'green',
        'danger': 'red',
        'goal': 'gold',
        'D': 'blue'  # Dron
    }
    
    # Crear una matriz para colorear
    color_matrix = np.zeros((grid.shape[0], grid.shape[1], 3))
    
    # Asignar colores según el tipo de celda
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 'safe':
                color_matrix[i, j] = [0.8, 0.8, 0.8]  # lightgray
            elif grid[i, j] == 'contaminated':
                color_matrix[i, j] = [0.0, 0.8, 0.0]  # green
            elif grid[i, j] == 'danger':
                color_matrix[i, j] = [0.8, 0.0, 0.0]  # red
            elif grid[i, j] == 'goal':
                color_matrix[i, j] = [1.0, 0.84, 0.0]  # gold
            elif grid[i, j] == 'D':
                color_matrix[i, j] = [0.0, 0.0, 0.8]  # blue
    
    ax.imshow(color_matrix)
    
    # Añadir etiquetas a las celdas
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, grid[i, j], 
                    ha="center", va="center", 
                    color="white" if grid[i, j] in ['contaminated', 'danger', 'D'] else "black")
    
    # Configurar ejes
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels(np.arange(grid.shape[1]))
    ax.set_yticklabels(np.arange(grid.shape[0]))
    ax.set_title(title)
    
    # Añadir líneas de cuadrícula
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.tick_params(which='minor', size=0)
    plt.setp(ax.get_xticklabels(), rotation=0)
    
    fig.tight_layout()
    return fig

def visualize_policy(q_table, grid_shape, title="Política Aprendida"):
    """Visualiza la política aprendida como flechas en una grilla."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Crear matriz de acciones y valores
    action_matrix = np.zeros(grid_shape, dtype=object)
    value_matrix = np.zeros(grid_shape)
    
    # Mapeo de acciones a flechas
    action_symbols = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→'
    }
    
    # Para cada estado, encontrar la mejor acción
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            state = (i, j)
            if state in q_table:
                best_action = max(q_table[state], key=q_table[state].get)
                action_matrix[i, j] = action_symbols.get(best_action, 'o')
                value_matrix[i, j] = max(q_table[state].values())
            else:
                action_matrix[i, j] = 'o'
                value_matrix[i, j] = 0
    
    # Normalizar valores para colorear
    if np.max(value_matrix) > np.min(value_matrix):
        norm_values = (value_matrix - np.min(value_matrix)) / (np.max(value_matrix) - np.min(value_matrix))
    else:
        norm_values = np.zeros(grid_shape)
    
    # Crear mapa de calor
    cmap = plt.cm.Blues
    im = ax.imshow(norm_values, cmap=cmap)
    
    # Añadir flechas de dirección
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ax.text(j, i, action_matrix[i, j], 
                    ha="center", va="center", 
                    color="black", fontsize=20)
    
    # Configurar ejes
    ax.set_xticks(np.arange(grid_shape[1]))
    ax.set_yticks(np.arange(grid_shape[0]))
    ax.set_xticklabels(np.arange(grid_shape[1]))
    ax.set_yticklabels(np.arange(grid_shape[0]))
    ax.set_title(title)
    
    # Añadir barra de color para valores Q
    cbar = fig.colorbar(im)
    cbar.set_label('Valor Q estimado')
    
    # Añadir líneas de cuadrícula
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_xticks(np.arange(-.5, grid_shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_shape[0], 1), minor=True)
    ax.tick_params(which='minor', size=0)
    
    fig.tight_layout()
    return fig

def visualize_trajectory(env, trajectory, title="Recorrido del Dron"):
    """Visualiza el recorrido del dron en el entorno."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Copia de la grilla original
    grid_copy = np.copy(env.grid)
    
    # Definir colores para cada tipo de celda
    colors = {
        'safe': [0.8, 0.8, 0.8],  # lightgray
        'contaminated': [0.0, 0.8, 0.0],  # green
        'danger': [0.8, 0.0, 0.0],  # red
        'goal': [1.0, 0.84, 0.0],  # gold
        'visited': [0.0, 0.5, 1.0]  # azul claro para celdas visitadas
    }
    
    # Crear matriz para colorear
    color_matrix = np.zeros((grid_copy.shape[0], grid_copy.shape[1], 3))
    
    # Colorear según tipo de celda
    for i in range(grid_copy.shape[0]):
        for j in range(grid_copy.shape[1]):
            color_matrix[i, j] = colors.get(grid_copy[i, j], [0.8, 0.8, 0.8])
    
    # Marcar trayectoria
    for idx, position in enumerate(trajectory):
        x, y = position
        # No recolorear la meta
        if grid_copy[x, y] != 'goal':
            color_matrix[x, y] = colors['visited']
    
    ax.imshow(color_matrix)
    
    # Añadir números a la trayectoria
    for idx, position in enumerate(trajectory):
        x, y = position
        ax.text(y, x, str(idx), ha="center", va="center", color="white", fontweight='bold')
    
    # Añadir línea de trayectoria
    traj_y, traj_x = zip(*[(pos[1], pos[0]) for pos in trajectory])
    ax.plot(traj_x, traj_y, 'o-', color='blue', linewidth=2, markersize=10)
    
    # Configurar ejes
    ax.set_xticks(np.arange(grid_copy.shape[1]))
    ax.set_yticks(np.arange(grid_copy.shape[0]))
    ax.set_xticklabels(np.arange(grid_copy.shape[1]))
    ax.set_yticklabels(np.arange(grid_copy.shape[0]))
    ax.set_title(title)
    
    # Añadir líneas de cuadrícula
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_xticks(np.arange(-.5, grid_copy.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_copy.shape[0], 1), minor=True)
    ax.tick_params(which='minor', size=0)
    
    fig.tight_layout()
    return fig

def display_step_by_step_visualization(env, trajectory, speed=500):
    """
    Visualización paso a paso simple y robusta del recorrido del dron.
    
    Args:
        env: Entorno de simulación (DroneEnv)
        trajectory: Lista de posiciones (coordenadas x,y) del recorrido
        speed: Velocidad de reproducción (ms)
    """
    # Inicializar variables
    if 'step_idx' not in st.session_state:
        st.session_state.step_idx = 0
    
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    
    # Contenedor para mensajes de estado
    status_container = st.empty()
    
    # Estilo CSS para mejorar la apariencia de los controles
    st.markdown("""
    <style>
    .big-button {
        font-size: 24px !important;
        height: 60px !important;
        font-weight: bold !important;
    }
    .slider-container .stSlider {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    div.row-widget.stButton > button {
        font-size: 20px;
        font-weight: bold;
        height: 3em;
        border-radius: 10px;
    }
    /* Estilo para el botón de reproducción */
    div.element-container:has(button:contains("REPRODUCIR")) button {
        background-color: #4CAF50;
        color: white;
    }
    /* Estilo para el botón de detener */
    div.element-container:has(button:contains("DETENER")) button {
        background-color: #f44336;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Contenedor para el slider con estilo mejorado
    with st.container():
        st.markdown('<div class="slider-container">', unsafe_allow_html=True)
        # Control deslizante para seleccionar el paso
        step = st.slider(
            f"Paso: {st.session_state.step_idx+1} de {len(trajectory)}", 
            0, len(trajectory)-1, 
            st.session_state.step_idx, 
            key="step_slider"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state.step_idx = step
    
    # Botones de control en columnas con estilos mejorados
    col1, col2 = st.columns(2)
    with col1:
        play_button = st.button("▶️ REPRODUCIR", key="play_button", use_container_width=True, 
                               help="Inicia la reproducción automática paso a paso")
        if play_button:
            st.session_state.is_playing = True
            st.rerun()  # Forzar actualización para iniciar reproducción
    
    with col2:
        stop_button = st.button("⏹️ DETENER", key="stop_button", use_container_width=True,
                              help="Detiene la reproducción automática")
        if stop_button:
            st.session_state.is_playing = False
            st.rerun()  # Forzar actualización para detener reproducción
    
    # Mostrar información del paso actual
    position = trajectory[step]
    cell_type = env.grid[position]
    
    # Recompensa para este paso
    reward = {
        'safe': -1,
        'contaminated': 10,
        'danger': -10,
        'goal': 20
    }.get(cell_type, -1)
    
    # Mostrar información en un cuadro destacado
    st.info(f"""
    📍 **Posición:** ({position[0]}, {position[1]})
    🏷️ **Tipo de celda:** {cell_type}
    💰 **Recompensa:** {reward}
    """)
    
    # Crear visualización para el paso actual
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colores para cada tipo de celda 
    colors = {
        'safe': [0.9, 0.9, 0.9],      # Gris claro
        'contaminated': [0.0, 0.6, 0.0],  # Verde
        'danger': [0.8, 0.0, 0.0],    # Rojo
        'goal': [1.0, 0.84, 0.0],      # Dorado
        'visited': [0.0, 0.5, 1.0]     # Azul claro
    }
    
    # Crear matriz de colores para el fondo
    grid_copy = np.copy(env.grid)
    color_matrix = np.zeros((grid_copy.shape[0], grid_copy.shape[1], 3))
    
    # Colorear celdas según tipo
    for i in range(grid_copy.shape[0]):
        for j in range(grid_copy.shape[1]):
            cell_type = grid_copy[i, j]
            color_matrix[i, j] = colors.get(cell_type, colors['safe'])
    
    # Mostrar matriz de fondo
    ax.imshow(color_matrix)
    
    # Añadir líneas de cuadrícula
    for i in range(grid_copy.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(grid_copy.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)
    
    # Añadir información sobre las celdas
    for i in range(grid_copy.shape[0]):
        for j in range(grid_copy.shape[1]):
            cell_type = grid_copy[i, j]
            
            # Texto para cada tipo de celda
            if cell_type == 'contaminated':
                text = "C"  # Contaminada
            elif cell_type == 'danger':
                text = "P"  # Peligrosa
            elif cell_type == 'goal':
                text = "M"  # Meta
            else:
                text = ""  # Celda segura (sin texto)
                
            # Añadir texto
            if text:
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=14, fontweight='bold',
                       color='white' if cell_type in ['contaminated', 'danger'] else 'black')
    
    # Trazar el recorrido hasta el paso actual
    visited_cells = trajectory[:step+1]
    
    # Colorear celdas visitadas
    for idx, pos in enumerate(visited_cells[:-1]):  # Excluir la última (posición actual)
        i, j = pos
        # No recolorear las celdas especiales
        if grid_copy[i, j] not in ['goal']:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=colors['visited'], alpha=0.3)
            ax.add_patch(rect)
            # Añadir número de paso
            ax.text(j, i, str(idx), ha='center', va='center', color='blue', fontweight='bold')
    
    # Linea y flechas para el recorrido
    if len(visited_cells) > 1:
        # Preparar coordenadas (invertidas x,y para matplotlib)
        path_y = [pos[1] for pos in visited_cells]
        path_x = [pos[0] for pos in visited_cells]
        
        # Dibujar línea principal
        ax.plot(path_y, path_x, '-', color='blue', linewidth=2, alpha=0.7)
        
        # Añadir flechas de dirección
        for i in range(1, len(visited_cells)):
            prev = visited_cells[i-1]
            curr = visited_cells[i]
            
            # Calcular dirección (invertir coordenadas para matplotlib)
            dx = curr[1] - prev[1]  
            dy = curr[0] - prev[0]
            
            # Añadir flecha
            if dx != 0 or dy != 0:  # Evitar división por cero
                # Punto medio
                mid_x = (prev[0] + curr[0]) / 2
                mid_y = (prev[1] + curr[1]) / 2
                
                # Añadir flecha como anotación
                ax.annotate('', 
                           xy=(curr[1], curr[0]),        # Punta de la flecha
                           xytext=(prev[1], prev[0]),    # Base de la flecha
                           arrowprops=dict(arrowstyle='->', color='blue', linewidth=2),
                           alpha=0.8)
    
    # Destacar posición actual del dron
    if step < len(trajectory):
        dron_x, dron_y = trajectory[step]
        
        # Círculo destacado para la posición actual
        circle = plt.Circle((dron_y, dron_x), 0.4, color='blue', alpha=0.6)
        ax.add_patch(circle)
        
        # Texto para el dron
        ax.text(dron_y, dron_x, "D", ha='center', va='center', 
               color='white', fontsize=16, fontweight='bold')
        
        # Destacar la celda actual con un borde amarillo
        rect = plt.Rectangle((dron_y-0.5, dron_x-0.5), 1, 1, fill=False, 
                           edgecolor='yellow', linewidth=3)
        ax.add_patch(rect)
    
    # Configurar ejes
    ax.set_xticks(np.arange(grid_copy.shape[1]))
    ax.set_yticks(np.arange(grid_copy.shape[0]))
    ax.set_xticklabels(np.arange(grid_copy.shape[1]))
    ax.set_yticklabels(np.arange(grid_copy.shape[0]))
    
    # Título
    ax.set_title(f"Paso {step+1} de {len(trajectory)}", fontsize=14)
    
    # Leyenda
    legend_elements = [
        patches.Patch(facecolor=colors['safe'], edgecolor='black', label='Seguro (-1)'),
        patches.Patch(facecolor=colors['contaminated'], edgecolor='black', label='Contaminado (+10)'),
        patches.Patch(facecolor=colors['danger'], edgecolor='black', label='Peligroso (-10)'),
        patches.Patch(facecolor=colors['goal'], edgecolor='black', label='Meta (+20)'),
        patches.Patch(facecolor=colors['visited'], edgecolor='black', alpha=0.3, label='Visitado'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Trayectoria'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3)
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Mostrar figura
    st.pyplot(fig)
    
    # Modo reproducción automática
    if st.session_state.is_playing and step < len(trajectory) - 1:
        # Mostrar información sobre reproducción
        status_container.success(f"▶️ Reproduciendo paso {step+1} de {len(trajectory)}")
        
        # Añadir barra de progreso
        progress = (step + 1) / len(trajectory)
        st.progress(progress)
        
        # Pausa breve antes de avanzar al siguiente paso
        time.sleep(speed / 1000.0)
        
        # Avanzar al siguiente paso
        st.session_state.step_idx = step + 1
        
        # Recargar la página para mostrar el siguiente paso
        st.rerun()
    elif st.session_state.is_playing and step == len(trajectory) - 1:
        # Final de la reproducción
        status_container.success("✅ Reproducción completada")
        st.progress(1.0)
        st.session_state.is_playing = False
    elif not st.session_state.is_playing:
        status_container.info("⏸️ Pulsa '▶️ REPRODUCIR' para iniciar la animación")
        
    # Mostrar leyenda explicativa
    with st.expander("ℹ️ Leyenda y explicación", expanded=False):
        st.markdown("""
        | Símbolo | Significado | Recompensa |
        |---------|-------------|------------|
        | D | Dron | - |
        | C | Zona contaminada | +10 |
        | P | Zona peligrosa | -10 |
        | M | Meta | +20 |
        | Números | Orden de pasos | - |
        | Flecha → | Dirección del movimiento | - |
        """)
        
        st.markdown("""
        **Instrucciones:**
        1. Mueve el control deslizante para ver cada paso individualmente
        2. Usa los botones de Reproducir/Detener para la animación automática
        """)

def auto_play(env, trajectory, speed):
    """Función para reproducir automáticamente el recorrido"""
    total_steps = len(trajectory)
    
    # Crear contenedor para mensajes
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Comenzar desde el paso actual
    start_step = st.session_state.visualization_step
    
    # Reproducir cada paso
    for step in range(start_step, total_steps):
        # Actualizar el paso actual
        st.session_state.visualization_step = step
        
        # Actualizar progreso
        progress = int((step / (total_steps-1)) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Reproduciendo paso {step+1} de {total_steps}")
        
        # Crear visualización
        grid_copy = np.copy(env.grid)
        fig = create_step_visualization(grid_copy, trajectory, step)
        
        # Mostrar visualización actualizada
        st.pyplot(fig)
        plt.close(fig)
        
        # Pausa controlada
        time.sleep(speed / 1000.0)
    
    # Marcar finalización
    progress_bar.progress(100)
    status_text.text("¡Reproducción completa!")

def create_step_visualization(grid, trajectory, step):
    """Crea una visualización para un paso específico"""
    # Colores para cada tipo de celda
    colors = {
        'safe': [0.8, 0.8, 0.8],      # Gris claro
        'contaminated': [0.0, 0.8, 0.0],  # Verde
        'danger': [0.8, 0.0, 0.0],    # Rojo
        'goal': [1.0, 0.84, 0.0],      # Dorado
        'dron': [0.0, 0.5, 1.0]       # Azul
    }
    
    # Símbolos para cada tipo de celda
    symbols = {
        'safe': ' ',
        'contaminated': '🌿',
        'danger': '⚠️',
        'goal': '🏁',
        'dron': '🛸'
    }
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f"Paso: {step+1} de {len(trajectory)}", fontsize=16)
    
    # Matriz de colores
    color_matrix = np.zeros((grid.shape[0], grid.shape[1], 3))
    
    # Colorear según tipo de celda
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell_type = grid[i, j]
            color_matrix[i, j] = colors.get(cell_type, colors['safe'])
    
    # Mostrar matriz base
    ax.imshow(color_matrix)
    
    # Marcar celdas visitadas
    visited_cells = trajectory[:step+1]
    
    # Dibujar trayectoria
    if step > 0:
        # Convertir coordenadas para el gráfico
        path_y = [pos[1] for pos in visited_cells]
        path_x = [pos[0] for pos in visited_cells]
        
        # Dibujar línea de trayectoria
        ax.plot(path_x, path_y, 'o-', color='blue', linewidth=2, alpha=0.7)
        
        # Añadir flechas de dirección
        for i in range(1, len(visited_cells)):
            # Posiciones actual y anterior
            prev_x, prev_y = visited_cells[i-1]
            curr_x, curr_y = visited_cells[i]
            
            # Dibujar flecha de dirección
            dx = curr_y - prev_y  # Intercambiamos para la visualización
            dy = curr_x - prev_x
            
            # Determinar símbolo de dirección
            if dx > 0 and dy == 0:
                direction = "→"
            elif dx < 0 and dy == 0:
                direction = "←"
            elif dx == 0 and dy > 0:
                direction = "↓"
            elif dx == 0 and dy < 0:
                direction = "↑"
            else:
                direction = "•"
            
            # Añadir texto de dirección en la celda intermedia
            mid_x = (prev_y + curr_y) / 2
            mid_y = (prev_x + curr_x) / 2
            ax.text(mid_y, mid_x, direction, ha='center', va='center', fontsize=16, color='blue', fontweight='bold')
    
    # Destacar posición actual del dron
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
            
            # No mostrar el símbolo del dron en las celdas (se muestra separado)
            if symbol != symbols['dron']:
                # Destacar celda actual
                if (i, j) == trajectory[step]:
                    bbox_props = dict(boxstyle="round,pad=0.3", fc='yellow', ec='red', alpha=0.7)
                    ax.text(j, i, symbol, ha='center', va='center', fontsize=14, 
                           bbox=bbox_props)
                else:
                    ax.text(j, i, symbol, ha='center', va='center', fontsize=14)
    
    # Añadir números a la trayectoria
    for idx, position in enumerate(visited_cells):
        x, y = position
        # No mostrar número en la posición actual del dron
        if idx < step:
            ax.text(y, x, str(idx), ha='center', va='center', 
                   color='white', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="circle,pad=0.2", fc='blue', ec='blue', alpha=0.7))
    
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
    
    # Leyenda en la parte inferior
    legend_elements = [
        patches.Patch(facecolor=colors['safe'], edgecolor='black', label='Seguro (-1)'),
        patches.Patch(facecolor=colors['contaminated'], edgecolor='black', label='Contaminado (+10)'),
        patches.Patch(facecolor=colors['danger'], edgecolor='black', label='Peligroso (-10)'),
        patches.Patch(facecolor=colors['goal'], edgecolor='black', label='Meta (+20)'),
        patches.Patch(facecolor=colors['dron'], edgecolor='black', label='Dron'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
             fancybox=True, shadow=True, ncol=5)
    
    plt.tight_layout()
    
    return fig

def create_step_by_step_animation(env, trajectory, interval=500, title="Recorrido Paso a Paso"):
    """
    Crea una animación del recorrido del dron paso a paso.
    [Mantiene este método por compatibilidad, pero usamos display_step_by_step_visualization para Streamlit]
    """
    # ... código existente sin cambios ...
