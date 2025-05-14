# config.py

"""Configuración global del proyecto."""

# Parámetros del entorno
ENV_CONFIG = {
    'grid_size': (5, 5),
    'start': (0, 0),
    'goal': (4, 4),
    'max_steps': 100
}

# Parámetros por defecto de los agentes
AGENT_CONFIG = {
    'q_learning': {
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon': 0.2
    },
    'sarsa': {
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon': 0.2
    },
    'monte_carlo': {
        'gamma': 0.9,
        'epsilon': 0.2
    }
}

# Parámetros de entrenamiento
TRAINING_CONFIG = {
    'default_episodes': 500,
    'evaluation_episodes': 100,
    'save_frequency': 100
}

# Parámetros de visualización
VISUALIZATION_CONFIG = {
    'cell_colors': {
        'safe': 'white',
        'contaminated': 'yellow',
        'dangerous': 'red',
        'start': 'green',
        'goal': 'blue'
    },
    'animation_speed': 500  # milisegundos
}

# Directorios del proyecto
PATHS = {
    'models': 'models',
    'results': 'results',
    'logs': 'logs'
}

# Semilla para reproducibilidad
RANDOM_SEED = 42
