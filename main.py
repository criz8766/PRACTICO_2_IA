# main.py

# Importaciones de la biblioteca estándar
import os
import time

# Importaciones de terceros
import numpy as np
import pandas as pd
import streamlit as st

# Importaciones locales
from env.drone_env import DroneEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.monte_carlo import MonteCarloAgent
from utils.metrics import compute_metrics
from utils.visualizations import (
    plot_rewards,
    compare_rewards,
    visualize_grid,
    visualize_policy,
    visualize_trajectory,
    display_step_by_step_visualization
)
from utils.model_io import (
    save_q_table,
    load_q_table,
    save_model,
    load_model,
    load_agent
)
from utils.robustness import (
    test_epsilon_robustness,
    test_alpha_robustness,
    plot_robustness_results
)

st.set_page_config(page_title="Simulación con Dron - TIA 2025", layout="centered")

st.title("🛰️ Simulación de Monitoreo de Zonas Contaminadas")
st.subheader("Comparación de algoritmos: Q-Learning, SARSA y Monte Carlo")

# Crear directorios para modelos si no existen
if not os.path.exists("models"):
    os.makedirs("models")

# Secciones de la aplicación
tab1, tab2, tab3 = st.tabs(["Entrenamiento", "Visualización", "Análisis de Robustez"])

with tab1:
    # Parámetros de control
    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox("Selecciona un algoritmo", 
                                ["Q-Learning", "SARSA", "Monte Carlo", "Comparar todos"])
        episodes = st.slider("Cantidad de episodios", 100, 1000, 500, step=100)
    
    with col2:
        alpha = st.slider("Tasa de aprendizaje (α)", 0.01, 1.0, 0.1)
        gamma = st.slider("Factor de descuento (γ)", 0.1, 1.0, 0.9)
        epsilon = st.slider("Exploración (ε)", 0.0, 1.0, 0.2)
    
    max_steps = st.slider("Pasos máximos por episodio", 50, 300, 100, step=10, 
                          help="Límite de pasos para evitar episodios infinitos")
    seed = st.number_input("Semilla aleatoria (para reproducibilidad)", min_value=1, max_value=10000, value=42)
    save_model_option = st.checkbox("Guardar modelo entrenado", value=True)

    def train_agent(agent_name, agent_class, **kwargs):
        env = DroneEnv(seed=seed, max_steps=max_steps)
        agent = agent_class(actions=env.action_space, **kwargs)

        reward_history = []
        trajectory = None
        start = time.time()
        
        for ep in range(episodes):
            total_reward = 0

            if agent_name == "Monte Carlo":
                # Aseguramos que el episodio termine en un número finito de pasos
                episode = agent.generate_episode(env, max_steps=max_steps)
                agent.learn(episode)
                total_reward = sum([r for _, _, r in episode])
                
                # Guardar trayectoria del último episodio
                if ep == episodes - 1:
                    trajectory = env.get_trajectory()

            elif agent_name == "SARSA":
                state = env.reset_with_same_grid() if ep > 0 else env.reset()
                action = agent.choose_action(state)
                done = False
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.choose_action(next_state)
                    agent.learn(state, action, reward, next_state, next_action, done)
                    state, action = next_state, next_action
                    total_reward += reward
                    
                # Guardar trayectoria del último episodio
                if ep == episodes - 1:
                    trajectory = env.get_trajectory()

            else:  # Q-Learning
                state = env.reset_with_same_grid() if ep > 0 else env.reset()
                done = False
                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, done = env.step(action)
                    agent.learn(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    
                # Guardar trayectoria del último episodio
                if ep == episodes - 1:
                    trajectory = env.get_trajectory()

            reward_history.append(total_reward)
            # Mostrar progreso para episodios largos
            if ep % 50 == 0 and ep > 0:
                st.text(f"Completado {ep}/{episodes} episodios...")

        end = time.time()
        metrics = compute_metrics(reward_history, start, end)
        
        # Guardar modelo si está habilitada la opción
        if save_model_option:
            model_dir = os.path.join("models", agent_name.lower().replace(" ", "_"))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Guardar tabla Q
            save_q_table(agent.q_table, os.path.join(model_dir, "q_table.json"))
            
            # Guardar agente completo
            save_model(agent, os.path.join(model_dir, "agent.pkl"))
        
        return agent, env, reward_history, metrics, trajectory

    if st.button("Entrenar agente(s) 🚀"):
        st.session_state.training_complete = False
        st.session_state.results = {}
        
        if algorithm == "Comparar todos":
            with st.spinner("Entrenando Q-Learning..."):
                q_agent, q_env, q_rewards, q_metrics, q_trajectory = train_agent("Q-Learning", QLearningAgent, alpha=alpha, gamma=gamma, epsilon=epsilon)
                st.session_state.results["Q-Learning"] = {
                    "agent": q_agent,
                    "env": q_env,
                    "rewards": q_rewards,
                    "metrics": q_metrics,
                    "trajectory": q_trajectory
                }
            
            with st.spinner("Entrenando SARSA..."):
                sarsa_agent, sarsa_env, sarsa_rewards, sarsa_metrics, sarsa_trajectory = train_agent("SARSA", SARSAAgent, alpha=alpha, gamma=gamma, epsilon=epsilon)
                st.session_state.results["SARSA"] = {
                    "agent": sarsa_agent,
                    "env": sarsa_env,
                    "rewards": sarsa_rewards,
                    "metrics": sarsa_metrics,
                    "trajectory": sarsa_trajectory
                }
            
            with st.spinner("Entrenando Monte Carlo..."):
                mc_agent, mc_env, mc_rewards, mc_metrics, mc_trajectory = train_agent("Monte Carlo", MonteCarloAgent, gamma=gamma, epsilon=epsilon)
                st.session_state.results["Monte Carlo"] = {
                    "agent": mc_agent,
                    "env": mc_env,
                    "rewards": mc_rewards,
                    "metrics": mc_metrics,
                    "trajectory": mc_trajectory
                }

            st.success("Entrenamiento completo ✅")
            
            # Guardar resultados en session_state para acceder desde otras pestañas
            st.session_state.all_rewards = {
                "Q-Learning": q_rewards,
                "SARSA": sarsa_rewards,
                "Monte Carlo": mc_rewards
            }
            
            # Graficar comparación
            st.pyplot(compare_rewards(
                histories=[q_rewards, sarsa_rewards, mc_rewards],
                labels=["Q-Learning", "SARSA", "Monte Carlo"],
                title="Comparación de Recompensa por Episodio"
            ))

            # Mostrar métricas
            st.markdown("### 📊 Métricas:")
            metrics_df = pd.DataFrame({
                "Métrica": ["Recompensa Promedio", "Varianza", "Tiempo (s)", "Episodio de Convergencia"],
                "Q-Learning": [
                    f"{q_metrics['avg_reward']:.2f}",
                    f"{q_metrics['variance']:.2f}",
                    f"{q_metrics['training_time']:.2f}",
                    f"{q_metrics['convergence_episode']}"
                ],
                "SARSA": [
                    f"{sarsa_metrics['avg_reward']:.2f}",
                    f"{sarsa_metrics['variance']:.2f}",
                    f"{sarsa_metrics['training_time']:.2f}",
                    f"{sarsa_metrics['convergence_episode']}"
                ],
                "Monte Carlo": [
                    f"{mc_metrics['avg_reward']:.2f}",
                    f"{mc_metrics['variance']:.2f}",
                    f"{mc_metrics['training_time']:.2f}",
                    f"{mc_metrics['convergence_episode']}"
                ]
            })
            st.table(metrics_df)
            
        else:
            # Entrenamiento individual
            if algorithm == "Q-Learning":
                agent_class = QLearningAgent
                kwargs = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}
            elif algorithm == "SARSA":
                agent_class = SARSAAgent
                kwargs = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}
            else:  # Monte Carlo
                agent_class = MonteCarloAgent
                kwargs = {"gamma": gamma, "epsilon": epsilon}

            with st.spinner(f"Entrenando {algorithm}..."):
                agent, env, rewards, metrics, trajectory = train_agent(algorithm, agent_class, **kwargs)
                st.session_state.results[algorithm] = {
                    "agent": agent,
                    "env": env,
                    "rewards": rewards,
                    "metrics": metrics,
                    "trajectory": trajectory
                }
            
            st.success("Entrenamiento completo ✅")
            
            # Guardar resultados en session_state
            st.session_state.current_algorithm = algorithm
            
            # Mostrar gráfico de recompensas
            st.pyplot(plot_rewards(rewards, title=f"{algorithm} - Recompensa por Episodio"))
            
            # Mostrar métricas
            st.markdown("### 📊 Métricas:")
            st.write(f"**Recompensa Promedio:** {metrics['avg_reward']:.2f}")
            st.write(f"**Varianza:** {metrics['variance']:.2f}")
            st.write(f"**Tiempo de Entrenamiento:** {metrics['training_time']:.2f} segundos")
            st.write(f"**Episodio de Convergencia:** {metrics['convergence_episode']}")
        
        st.session_state.training_complete = True

    # Cargar modelo pre-entrenado
    st.markdown("### 💾 Cargar modelo pre-entrenado")
    
    load_col1, load_col2 = st.columns(2)
    with load_col1:
        load_algorithm = st.selectbox("Algoritmo", ["Q-Learning", "SARSA", "Monte Carlo"])
    
    with load_col2:
        model_dir = os.path.join("models", load_algorithm.lower().replace(" ", "_"))
        q_table_path = os.path.join(model_dir, "q_table.json")
        agent_path = os.path.join(model_dir, "agent.json")
        can_load = os.path.exists(q_table_path) and os.path.exists(agent_path)
        
        if can_load:
            if st.button("Cargar modelo"):
                with st.spinner(f"Cargando modelo {load_algorithm}..."):
                    env = DroneEnv(seed=seed, max_steps=max_steps)
                    
                    # Cargar el agente usando la función correcta
                    agent = load_agent(agent_path, q_table_path)
                    
                    if agent:
                        st.session_state.results[load_algorithm] = {
                            "agent": agent,
                            "env": env,
                            "loaded": True
                        }
                        
                        st.success(f"Modelo {load_algorithm} cargado correctamente")
                    else:
                        st.error(f"Error al cargar el modelo {load_algorithm}")
        else:
            st.warning(f"No hay modelo guardado para {load_algorithm}")


with tab2:
    if 'training_complete' not in st.session_state or not st.session_state.training_complete:
        st.info("👆 Primero entrena o carga un modelo en la pestaña 'Entrenamiento'")
    else:
        # Mostrar visualizaciones
        st.markdown("### 🖼️ Visualizaciones")
        
        visualization_algorithm = st.selectbox(
            "Selecciona el algoritmo a visualizar", 
            list(st.session_state.results.keys())
        )
        
        if visualization_algorithm in st.session_state.results:
            result = st.session_state.results[visualization_algorithm]
            
            if 'loaded' in result and result['loaded']:
                # Caso especial para modelos cargados
                agent = result['agent']
                env = result['env']
                
                # Generar un episodio de prueba
                state = env.reset()
                done = False
                while not done:
                    action = agent.choose_action(state)
                    state, _, done = env.step(action)
                
                trajectory = env.get_trajectory()
            else:
                agent = result['agent']
                env = result['env']
                trajectory = result['trajectory']
            
            # Opciones de visualización
            viz_options = st.multiselect(
                "Selecciona las visualizaciones", 
                ["Entorno", "Política aprendida", "Recorrido", "Recorrido paso a paso"],
                default=["Entorno", "Política aprendida", "Recorrido"]
            )
            
            col1, col2 = st.columns(2)
            
            if "Entorno" in viz_options:
                with col1:
                    st.markdown("#### Entorno")
                    grid_copy = np.copy(env.grid)
                    st.pyplot(visualize_grid(grid_copy))
            
            if "Política aprendida" in viz_options:
                with col2:
                    st.markdown("#### Política Aprendida")
                    st.pyplot(visualize_policy(agent.q_table, env.grid_size, title=f"Política {visualization_algorithm}"))
            
            if "Recorrido" in viz_options:
                st.markdown("#### Recorrido del Dron")
                st.pyplot(visualize_trajectory(env, trajectory, title=f"Recorrido con {visualization_algorithm}"))
                
                # Tabla de recorrido
                steps_df = pd.DataFrame({
                    "Paso": list(range(len(trajectory))),
                    "Posición": [f"({pos[0]}, {pos[1]})" for pos in trajectory],
                    "Tipo de Celda": [env.grid[pos] for pos in trajectory]
                })
                st.dataframe(steps_df)
            
            # Nueva visualización paso a paso
            if "Recorrido paso a paso" in viz_options:
                st.markdown("#### 🚀 Visualización interactiva del recorrido")
                
                # Caja informativa con instrucciones claras
                st.info("""
                **Instrucciones:**
                1. 🔍 Utiliza el **control deslizante** para navegar manualmente por cada paso
                2. ▶️ Presiona el botón **REPRODUCIR** para ver la animación automática
                3. ⏹️ Usa **DETENER** para pausar la animación en cualquier momento
                """)
                
                # Añadir un separador visual para destacar esta sección
                st.markdown("---")
                
                # Contenedor dedicado para la visualización paso a paso
                step_viz_container = st.container()
                with step_viz_container:
                    # Llamar a la función de visualización paso a paso con un identificador único
                    display_step_by_step_visualization(env, trajectory, speed=500)
                
                # Añadir un separador visual al final
                st.markdown("---")
        
        # Si hay comparación de algoritmos
        if 'all_rewards' in st.session_state:
            st.markdown("### 📊 Comparación de Algoritmos")
            st.pyplot(compare_rewards(
                histories=[st.session_state.all_rewards[alg] for alg in st.session_state.all_rewards],
                labels=list(st.session_state.all_rewards.keys()),
                title="Comparación de Recompensa por Episodio"
            ))

with tab3:
    st.markdown("### 🔍 Análisis de Robustez")
    st.write("Evalúa cómo afecta el cambio de parámetros al rendimiento de los algoritmos.")
    
    param_to_test = st.radio("Parámetro a evaluar", ["Epsilon (ε)", "Alpha (α)"])
    algorithm_for_robustness = st.selectbox(
        "Algoritmo a evaluar", 
        ["Q-Learning", "SARSA", "Monte Carlo"], 
        key="algo_robust"
    )
    
    # Parámetros de experimentación
    test_episodes = st.slider("Episodios por prueba", 100, 500, 200, step=50)
    runs_per_param = st.slider("Ejecuciones por valor de parámetro", 1, 5, 3)
    
    # Configurar valores a probar según el parámetro
    if param_to_test == "Epsilon (ε)":
        param_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        param_name = "epsilon"
    else:  # Alpha
        param_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        param_name = "alpha"
    
    # Mensajes especiales para Monte Carlo y alpha
    if algorithm_for_robustness == "Monte Carlo" and param_to_test == "Alpha (α)":
        st.warning("El algoritmo Monte Carlo no utiliza el parámetro Alpha (α). Selecciona Epsilon (ε) para este algoritmo.")
    
    # Mapeo de nombres
    agent_class_map = {
        "Q-Learning": QLearningAgent,
        "SARSA": SARSAAgent,
        "Monte Carlo": MonteCarloAgent
    }
    
    if st.button("Ejecutar análisis de robustez"):
        if algorithm_for_robustness == "Monte Carlo" and param_to_test == "Alpha (α)":
            st.error("El algoritmo Monte Carlo no utiliza el parámetro Alpha (α). Por favor, selecciona Epsilon (ε) para este algoritmo.")
        else:
            with st.spinner(f"Analizando robustez para {algorithm_for_robustness} con diferentes valores de {param_name}..."):
                agent_class = agent_class_map[algorithm_for_robustness]
                
                if param_to_test == "Epsilon (ε)":
                    results_df = test_epsilon_robustness(
                        agent_class, 
                        param_values, 
                        episodes=test_episodes, 
                        alpha=alpha if algorithm_for_robustness != "Monte Carlo" else None,
                        gamma=gamma, 
                        seed=seed,
                        runs=runs_per_param,
                        max_steps=max_steps
                    )
                else:  # Alpha
                    results_df = test_alpha_robustness(
                        agent_class, 
                        param_values, 
                        episodes=test_episodes, 
                        epsilon=epsilon,
                        gamma=gamma, 
                        seed=seed,
                        runs=runs_per_param,
                        max_steps=max_steps
                    )
                
                st.success("Análisis completado")
                
                # Guardar en estado de sesión
                st.session_state.robustness_results = results_df
                st.session_state.robustness_param = param_name
                
                # Mostrar tabla de resultados
                st.markdown("#### Resultados")
                st.dataframe(results_df)
                
                # Mostrar gráficos
                st.markdown("#### Gráficos")
                
                metric_options = ["avg_reward", "convergence_episode", "training_time"]
                metric_labels = ["Recompensa Promedio", "Episodio de Convergencia", "Tiempo de Entrenamiento"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(plot_robustness_results(
                        results_df, 
                        param_name=param_name, 
                        metric="avg_reward",
                        title=f"Efecto de {param_name.capitalize()} en Recompensa Promedio"
                    ))
                
                with col2:
                    st.pyplot(plot_robustness_results(
                        results_df, 
                        param_name=param_name, 
                        metric="convergence_episode",
                        title=f"Efecto de {param_name.capitalize()} en Convergencia"
                    ))
                
                st.pyplot(plot_robustness_results(
                    results_df, 
                    param_name=param_name, 
                    metric="training_time",
                    title=f"Efecto de {param_name.capitalize()} en Tiempo de Entrenamiento"
                ))
    
    # Mostrar resultados previos
    if 'robustness_results' in st.session_state:
        st.markdown("#### Resultados del análisis anterior")
        st.dataframe(st.session_state.robustness_results)
        
        # Mostrar gráficos
        param_name = st.session_state.robustness_param
        results_df = st.session_state.robustness_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(plot_robustness_results(
                results_df, 
                param_name=param_name, 
                metric="avg_reward",
                title=f"Efecto de {param_name.capitalize()} en Recompensa Promedio"
            ))
        
        with col2:
            st.pyplot(plot_robustness_results(
                results_df, 
                param_name=param_name, 
                metric="convergence_episode",
                title=f"Efecto de {param_name.capitalize()} en Convergencia"
            ))
        
        st.pyplot(plot_robustness_results(
            results_df, 
            param_name=param_name, 
            metric="training_time",
            title=f"Efecto de {param_name.capitalize()} en Tiempo de Entrenamiento"
        ))
