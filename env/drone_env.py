# env/drone_env.py

import numpy as np
import random

class DroneEnv:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4), seed=None, max_steps=100):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.position = start
        self.state_space = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]
        self.action_space = ['up', 'down', 'left', 'right']
        self.max_steps = max_steps
        self.steps_taken = 0
        
        # Establecer semilla para reproducibilidad
        if seed is not None:
            random.seed(seed)

        # Matriz de tipo de celdas
        self.grid = self._create_grid()
        self.initial_grid = np.copy(self.grid)
        
        # Registrar trayectoria
        self.trajectory = [self.position]

    def _create_grid(self):
        """Crea una grilla con zonas contaminadas y peligrosas asegurando que existe un camino viable."""
        grid = np.full(self.grid_size, 'safe', dtype=object)
        
        # Garantizamos que exista al menos un camino seguro a la meta
        # Primero identificamos un camino viable (simple, derecha y abajo)
        path_cells = []
        x, y = self.start
        while (x, y) != self.goal:
            if x < self.goal[0]:
                x += 1
            elif y < self.goal[1]:
                y += 1
            path_cells.append((x, y))
        
        # Definimos las celdas disponibles para contaminadas y peligrosas (excluyendo el camino seguro)
        available_cells = [cell for cell in self.state_space if cell != self.start and cell not in path_cells]
        
        # Zonas contaminadas con alto impacto ambiental (recompensa alta)
        num_contaminated = min(4, len(available_cells) // 2)
        contaminated = random.sample(available_cells, k=num_contaminated)
        for cell in contaminated:
            grid[cell] = 'contaminated'
            available_cells.remove(cell)

        # Zonas peligrosas (castigan)
        num_dangerous = min(3, len(available_cells) // 2)
        dangerous = random.sample(available_cells, k=num_dangerous)
        for cell in dangerous:
            grid[cell] = 'danger'

        grid[self.goal] = 'goal'
        return grid

    def reset(self):
        """Reinicia el entorno a su estado inicial con una nueva grilla."""
        self.position = self.start
        self.steps_taken = 0
        self.trajectory = [self.position]
        # Generar nueva grilla
        self.grid = self._create_grid()
        self.initial_grid = np.copy(self.grid)
        return self.position
    
    def reset_with_same_grid(self):
        """Reinicia la posición pero mantiene la misma configuración de la grilla."""
        self.position = self.start
        self.steps_taken = 0
        self.trajectory = [self.position]
        return self.position

    def step(self, action):
        self.steps_taken += 1
        
        x, y = self.position
        if action == 'up':
            x = max(x - 1, 0)
        elif action == 'down':
            x = min(x + 1, self.grid_size[0] - 1)
        elif action == 'left':
            y = max(y - 1, 0)
        elif action == 'right':
            y = min(y + 1, self.grid_size[1] - 1)

        self.position = (x, y)
        cell_type = self.grid[self.position]
        
        # Registrar nueva posición en la trayectoria
        self.trajectory.append(self.position)

        # Recompensas
        if cell_type == 'contaminated':
            reward = 10
        elif cell_type == 'danger':
            reward = -10
        elif cell_type == 'goal':
            reward = 20
        else:
            reward = -1  # paso normal
            
        # Verificar si se alcanzó el número máximo de pasos
        if self.steps_taken >= self.max_steps and self.position != self.goal:
            done = True
            reward = -15  # Penalización por exceder el máximo de pasos
        else:
            done = self.position == self.goal
            
        return self.position, reward, done

    def render(self):
        display_grid = np.copy(self.grid)
        x, y = self.position
        display_grid[x, y] = 'D'
        print(display_grid)
        
    def get_trajectory(self):
        """Devuelve la trayectoria actual del dron."""
        return self.trajectory
