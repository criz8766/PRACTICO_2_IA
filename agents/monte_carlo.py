# agents/monte_carlo.py

import random
from collections import defaultdict
from .base_agent import BaseAgent

class QTableDict(defaultdict):
    """Clase personalizada para reemplazar defaultdict de q_table."""
    def __init__(self, actions):
        super().__init__(dict)
        self.actions = actions
    
    def __missing__(self, key):
        self[key] = {a: 0.0 for a in self.actions}
        return self[key]

class ReturnsDict(defaultdict):
    """Clase personalizada para reemplazar defaultdict de returns."""
    def __init__(self, actions):
        super().__init__(dict)
        self.actions = actions
    
    def __missing__(self, key):
        self[key] = {a: [] for a in self.actions}
        return self[key]

class MonteCarloAgent(BaseAgent):
    """Agente que implementa el algoritmo Monte Carlo.
    
    El método Monte Carlo aprende de episodios completos de experiencia,
    actualizando los valores Q basándose en el retorno promedio observado.
    """
    
    def __init__(self, actions, gamma=0.9, epsilon=0.2):
        """
        Args:
            actions (list): Lista de acciones posibles
            gamma (float, optional): Factor de descuento. Por defecto 0.9
            epsilon (float, optional): Factor de exploración. Por defecto 0.2
        """
        super().__init__(actions, gamma, epsilon)
        self.q_table = QTableDict(actions)
        self.returns = ReturnsDict(actions)

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

    def learn(self, episode):
        """Actualiza la política basándose en el episodio completo.
        
        Args:
            episode (list): Lista de tuplas (estado, acción, recompensa)
        """
        # Calcular retornos para cada paso
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            
            # Actualizar promedio de retornos para el par estado-acción
            self.returns[state][action].append(G)
            self.q_table[state][action] = sum(self.returns[state][action]) / len(self.returns[state][action])

    def generate_episode(self, env, max_steps=100):
        """Genera un episodio usando la política actual.
        
        Args:
            env: El entorno
            max_steps (int): Número máximo de pasos
            
        Returns:
            list: Lista de tuplas (estado, acción, recompensa)
        """
        episode = []
        state = env.reset()
        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode
