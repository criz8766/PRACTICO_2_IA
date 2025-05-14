# agents/sarsa.py

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

class SARSAAgent(BaseAgent):
    """Agente que implementa el algoritmo SARSA.
    
    SARSA (State-Action-Reward-State-Action) es un algoritmo de aprendizaje por refuerzo
    on-policy que aprende valores Q basándose en las acciones que realmente toma.
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
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_action, done):
        """Actualiza la tabla Q usando la ecuación de SARSA.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            next_action: Siguiente acción
            done: Indica si el episodio ha terminado
        """
        # Obtener el valor Q para la siguiente acción-estado
        next_q = self.q_table[next_state][next_action] if not done else 0
        
        # Actualizar el valor Q usando la ecuación de SARSA
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )
