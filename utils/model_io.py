# utils/model_io.py

import json
import os
import pickle
from collections import defaultdict

def save_q_table(q_table, filepath):
    """
    Guarda una tabla Q en un archivo.
    
    Args:
        q_table (defaultdict): La tabla Q a guardar.
        filepath (str): Ruta donde guardar el archivo.
    """
    # Convertir defaultdict a diccionario regular
    serializable_q_table = {}
    for state, actions in q_table.items():
        # Convertir las tuplas (estado) a strings para serialización
        serializable_q_table[str(state)] = actions
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Guardar como json
    with open(filepath, 'w') as f:
        json.dump(serializable_q_table, f, indent=2)
    
    print(f"Tabla Q guardada en {filepath}")

def load_q_table(filepath, actions):
    """
    Carga una tabla Q desde un archivo.
    
    Args:
        filepath (str): Ruta del archivo a cargar.
        actions (list): Lista de acciones disponibles.
        
    Returns:
        defaultdict: La tabla Q cargada.
    """
    if not os.path.exists(filepath):
        print(f"No se encontró el archivo {filepath}")
        # Importamos las clases aquí para evitar problemas de importación circular
        from agents.q_learning import DefaultDict
        return DefaultDict(actions)
    
    with open(filepath, 'r') as f:
        serialized_q_table = json.load(f)
    
    # Importamos las clases aquí para evitar problemas de importación circular
    from agents.q_learning import DefaultDict
    
    # Convertir de vuelta a defaultdict con tuplas como claves
    q_table = DefaultDict(actions)
    for state_str, actions_dict in serialized_q_table.items():
        # Convertir string a tupla: "(0, 1)" -> (0, 1)
        state_tuple = eval(state_str)
        q_table[state_tuple] = actions_dict
    
    print(f"Tabla Q cargada desde {filepath}")
    return q_table

def save_agent_data(agent, filepath):
    """
    Guarda los datos importantes del agente en un formato seguro.
    
    Args:
        agent: El agente a guardar.
        filepath (str): Ruta donde guardar el archivo.
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Extraer los datos relevantes del agente
    agent_data = {
        'class_name': agent.__class__.__name__,
        'params': {
            'actions': agent.actions,
            'alpha': getattr(agent, 'alpha', None),
            'gamma': agent.gamma,
            'epsilon': agent.epsilon
        }
    }
    
    # Guardar datos como json
    with open(filepath, 'w') as f:
        json.dump(agent_data, f, indent=2)
    
    print(f"Datos del agente guardados en {filepath}")

def load_agent(data_filepath, q_table_filepath):
    """
    Carga un agente a partir de sus datos y tabla Q.
    
    Args:
        data_filepath (str): Ruta del archivo con los datos del agente.
        q_table_filepath (str): Ruta del archivo con la tabla Q.
        
    Returns:
        El agente cargado.
    """
    if not os.path.exists(data_filepath) or not os.path.exists(q_table_filepath):
        print(f"No se encontraron los archivos necesarios para cargar el agente")
        return None
    
    # Cargar datos del agente
    with open(data_filepath, 'r') as f:
        agent_data = json.load(f)
    
    # Importar las clases de agente
    from agents.q_learning import QLearningAgent
    from agents.sarsa import SARSAAgent
    from agents.monte_carlo import MonteCarloAgent
    
    # Crear el agente según la clase
    class_name = agent_data['class_name']
    params = agent_data['params']
    
    if class_name == 'QLearningAgent':
        agent = QLearningAgent(
            actions=params['actions'],
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon=params['epsilon']
        )
    elif class_name == 'SARSAAgent':
        agent = SARSAAgent(
            actions=params['actions'],
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon=params['epsilon']
        )
    elif class_name == 'MonteCarloAgent':
        agent = MonteCarloAgent(
            actions=params['actions'],
            gamma=params['gamma'],
            epsilon=params['epsilon']
        )
    else:
        print(f"Tipo de agente desconocido: {class_name}")
        return None
    
    # Cargar la tabla Q
    agent.q_table = load_q_table(q_table_filepath, params['actions'])
    
    print(f"Agente {class_name} cargado correctamente")
    return agent

# Mantener compatibilidad con el código existente
def save_model(agent, filepath):
    """
    Guarda los datos del agente (mantiene compatibilidad con el código existente).
    
    Args:
        agent: El agente a guardar.
        filepath (str): Ruta donde guardar el archivo.
    """
    # Cambiar la extensión a .json
    data_filepath = filepath.replace('.pkl', '.json')
    
    # Guardar también la tabla Q, asegurándonos de que el directorio existe
    q_table_filepath = os.path.join(os.path.dirname(filepath), 'q_table.json')
    save_q_table(agent.q_table, q_table_filepath)
    
    # Guardar datos del agente
    save_agent_data(agent, data_filepath)

def load_model(filepath):
    """
    Carga un agente (mantiene compatibilidad con el código existente).
    
    Args:
        filepath (str): Ruta del archivo a cargar.
        
    Returns:
        El agente cargado.
    """
    # Cambiar la extensión a .json
    data_filepath = filepath.replace('.pkl', '.json')
    # Buscar el archivo de la tabla Q en el mismo directorio
    q_table_filepath = os.path.join(os.path.dirname(filepath), 'q_table.json')
    
    # Verificar si los archivos existen
    if not os.path.exists(data_filepath):
        print(f"No se encontró el archivo de datos del agente: {data_filepath}")
        return None
    
    if not os.path.exists(q_table_filepath):
        print(f"No se encontró el archivo de la tabla Q: {q_table_filepath}")
        return None
    
    return load_agent(data_filepath, q_table_filepath) 