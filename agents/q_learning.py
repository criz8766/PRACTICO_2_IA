# agents/q_learning.py

import numpy as np
import random
from collections import defaultdict
from .base_agent import BaseAgent

class DefaultDict(defaultdict):
    """Clase personalizada para reemplazar defaultdict con lambda."""
    def __init__(self, actions):
        super().__init__(dict)
        self.actions = actions
    
    def __missing__(self, key):
        self[key] = {a: 0.0 for a in self.actions}
        return self[key]

class QLearningAgent(BaseAgent):
    """Agente que implementa el algoritmo Q-Learning.
    
    Q-Learning es un algoritmo de aprendizaje por refuerzo off-policy que aprende
    la política óptima independientemente de la política que está siguiendo.
    """
    
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        Args:
            actions (list): Lista de acciones posibles
            alpha (float, optional): Tasa de aprendizaje. Por defecto 0.1
            gamma (float, optional): Factor de descuento. Por defecto 0.9
            epsilon (float, optional): Factor de exploración. Por defecto 0.2
        """
        super().__init__(actions, gamma, epsilon)
        self.alpha = alpha
        self.q_table = DefaultDict(actions)

    def choose_action(self, state):
        """Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: El estado actual del entorno
            
        Returns:
            action: La acción seleccionada
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # exploración
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)  # explotación con desempate aleatorio

    def learn(self, state, action, reward, next_state, done):
        """Actualiza la tabla Q usando la ecuación de Q-Learning.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Indica si el episodio ha terminado
        """
        # Obtener el máximo valor Q para el siguiente estado
        next_q = max(self.q_table[next_state].values()) if not done else 0
        
        # Actualizar el valor Q usando la ecuación de Q-Learning
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )
