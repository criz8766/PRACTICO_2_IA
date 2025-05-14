from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Clase base abstracta para todos los agentes de aprendizaje por refuerzo.
    
    Esta clase define la interfaz común que todos los agentes deben implementar.
    
    Attributes:
        actions (list): Lista de acciones posibles que el agente puede tomar
        gamma (float): Factor de descuento para recompensas futuras
        epsilon (float): Factor de exploración para la política epsilon-greedy
    """
    
    def __init__(self, actions, gamma=0.9, epsilon=0.2):
        """
        Args:
            actions (list): Lista de acciones posibles
            gamma (float, optional): Factor de descuento. Por defecto 0.9
            epsilon (float, optional): Factor de exploración. Por defecto 0.2
        """
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon

    @abstractmethod
    def choose_action(self, state):
        """Selecciona una acción para un estado dado.
        
        Args:
            state: El estado actual del entorno
            
        Returns:
            action: La acción seleccionada
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """Actualiza la política del agente basándose en la experiencia.
        
        Los argumentos varían según el algoritmo específico.
        """
        pass

    def save(self, filepath):
        """Guarda el modelo del agente.
        
        Args:
            filepath (str): Ruta donde guardar el modelo
        """
        pass

    def load(self, filepath):
        """Carga el modelo del agente.
        
        Args:
            filepath (str): Ruta del modelo a cargar
        """
        pass
