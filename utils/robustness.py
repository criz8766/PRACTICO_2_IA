# utils/robustness.py

import numpy as np
import matplotlib.pyplot as plt
from env.drone_env import DroneEnv
from utils.metrics import compute_metrics
import time
import pandas as pd

def test_epsilon_robustness(agent_class, epsilon_values, episodes=500, alpha=0.1, gamma=0.9, seed=42, runs=3, max_steps=100):
    """
    Evalúa la robustez de un algoritmo ante cambios en epsilon.
    
    Args:
        agent_class: Clase del agente a evaluar
        epsilon_values (list): Lista de valores de epsilon a probar
        episodes (int): Número de episodios para entrenar
        alpha (float): Tasa de aprendizaje
        gamma (float): Factor de descuento
        seed (int): Semilla para reproducibilidad
        runs (int): Número de ejecuciones para promediar
        max_steps (int): Límite de pasos por episodio
        
    Returns:
        pandas.DataFrame: Resultados del análisis
    """
    results = []
    
    for epsilon in epsilon_values:
        # Recopilar métricas de múltiples ejecuciones
        run_metrics = []
        
        for run in range(runs):
            # Usar una semilla diferente para cada ejecución basada en la semilla principal
            run_seed = seed + run
            
            # Crear entorno y agente
            env = DroneEnv(seed=run_seed, max_steps=max_steps)
            if agent_class.__name__ == "MonteCarloAgent":
                agent = agent_class(actions=env.action_space, gamma=gamma, epsilon=epsilon)
            else:
                agent = agent_class(actions=env.action_space, alpha=alpha, gamma=gamma, epsilon=epsilon)
            
            # Entrenar
            reward_history = []
            start = time.time()
            
            for ep in range(episodes):
                if agent_class.__name__ == "MonteCarloAgent":
                    # Usamos el límite de pasos para evitar bucles infinitos
                    episode = agent.generate_episode(env, max_steps=max_steps)
                    agent.learn(episode)
                    reward = sum([r for _, _, r in episode])
                    reward_history.append(reward)
                
                elif agent_class.__name__ == "SARSAAgent":
                    state = env.reset_with_same_grid() if ep > 0 else env.reset()
                    action = agent.choose_action(state)
                    done = False
                    total_reward = 0
                    
                    while not done:
                        next_state, reward, done = env.step(action)
                        next_action = agent.choose_action(next_state)
                        agent.learn(state, action, reward, next_state, next_action, done)
                        state, action = next_state, next_action
                        total_reward += reward
                    
                    reward_history.append(total_reward)
                
                else:  # QLearningAgent
                    state = env.reset_with_same_grid() if ep > 0 else env.reset()
                    done = False
                    total_reward = 0
                    
                    while not done:
                        action = agent.choose_action(state)
                        next_state, reward, done = env.step(action)
                        agent.learn(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward
                    
                    reward_history.append(total_reward)
            
            end = time.time()
            
            # Calcular métricas
            metrics = compute_metrics(reward_history, start, end)
            run_metrics.append(metrics)
        
        # Calcular promedio y desviación estándar de las métricas
        avg_reward = np.mean([m['avg_reward'] for m in run_metrics])
        std_reward = np.std([m['avg_reward'] for m in run_metrics])
        avg_conv = np.mean([m['convergence_episode'] for m in run_metrics])
        std_conv = np.std([m['convergence_episode'] for m in run_metrics])
        avg_time = np.mean([m['training_time'] for m in run_metrics])
        
        # Guardar resultados
        results.append({
            'epsilon': epsilon,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'convergence_episode': avg_conv,
            'std_convergence': std_conv,
            'training_time': avg_time
        })
    
    return pd.DataFrame(results)

def test_alpha_robustness(agent_class, alpha_values, episodes=500, epsilon=0.2, gamma=0.9, seed=42, runs=3, max_steps=100):
    """
    Evalúa la robustez de un algoritmo ante cambios en alpha.
    
    Args:
        agent_class: Clase del agente a evaluar
        alpha_values (list): Lista de valores de alpha a probar
        episodes (int): Número de episodios para entrenar
        epsilon (float): Factor de exploración
        gamma (float): Factor de descuento
        seed (int): Semilla para reproducibilidad
        runs (int): Número de ejecuciones para promediar
        max_steps (int): Límite de pasos por episodio
        
    Returns:
        pandas.DataFrame: Resultados del análisis
    """
    # Omitir esta prueba para Monte Carlo que no usa alpha
    if agent_class.__name__ == "MonteCarloAgent":
        print("Esta prueba no aplica para Monte Carlo ya que no utiliza alpha")
        return pd.DataFrame()
    
    results = []
    
    for alpha in alpha_values:
        # Recopilar métricas de múltiples ejecuciones
        run_metrics = []
        
        for run in range(runs):
            # Usar una semilla diferente para cada ejecución basada en la semilla principal
            run_seed = seed + run
            
            # Crear entorno y agente
            env = DroneEnv(seed=run_seed, max_steps=max_steps)
            agent = agent_class(actions=env.action_space, alpha=alpha, gamma=gamma, epsilon=epsilon)
            
            # Entrenar
            reward_history = []
            start = time.time()
            
            for ep in range(episodes):
                if agent_class.__name__ == "SARSAAgent":
                    state = env.reset_with_same_grid() if ep > 0 else env.reset()
                    action = agent.choose_action(state)
                    done = False
                    total_reward = 0
                    
                    while not done:
                        next_state, reward, done = env.step(action)
                        next_action = agent.choose_action(next_state)
                        agent.learn(state, action, reward, next_state, next_action, done)
                        state, action = next_state, next_action
                        total_reward += reward
                    
                    reward_history.append(total_reward)
                
                else:  # QLearningAgent
                    state = env.reset_with_same_grid() if ep > 0 else env.reset()
                    done = False
                    total_reward = 0
                    
                    while not done:
                        action = agent.choose_action(state)
                        next_state, reward, done = env.step(action)
                        agent.learn(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward
                    
                    reward_history.append(total_reward)
            
            end = time.time()
            
            # Calcular métricas
            metrics = compute_metrics(reward_history, start, end)
            run_metrics.append(metrics)
        
        # Calcular promedio y desviación estándar de las métricas
        avg_reward = np.mean([m['avg_reward'] for m in run_metrics])
        std_reward = np.std([m['avg_reward'] for m in run_metrics])
        avg_conv = np.mean([m['convergence_episode'] for m in run_metrics])
        std_conv = np.std([m['convergence_episode'] for m in run_metrics])
        avg_time = np.mean([m['training_time'] for m in run_metrics])
        
        # Guardar resultados
        results.append({
            'alpha': alpha,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'convergence_episode': avg_conv,
            'std_convergence': std_conv,
            'training_time': avg_time
        })
    
    return pd.DataFrame(results)

def plot_robustness_results(df, param_name='epsilon', metric='avg_reward', title=None):
    """
    Grafica los resultados del análisis de robustez.
    
    Args:
        df (pandas.DataFrame): DataFrame con los resultados
        param_name (str): Nombre del parámetro analizado ('epsilon' o 'alpha')
        metric (str): Métrica a visualizar
        title (str): Título para el gráfico
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar valores promedio
    ax.plot(df[param_name], df[metric], 'o-', linewidth=2, markersize=8)
    
    # Añadir barras de error si hay desviación estándar
    if f'std_{metric.split("_")[1]}' in df.columns:
        ax.errorbar(df[param_name], df[metric], 
                    yerr=df[f'std_{metric.split("_")[1]}'], 
                    fmt='o', capsize=5, alpha=0.7)
    
    # Personalizar gráfico
    ax.set_xlabel(f'Valor de {param_name.capitalize()}')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Efecto de {param_name.capitalize()} en {metric.replace("_", " ").title()}')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    return fig 