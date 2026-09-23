import numpy as np

class RoundRobin:
    """
    Round Robin Baseline
    Stateless cyclic routing, simply routes requests in a 0, 1, 2, 3 cycle.
    """
    def __init__(self, num_backends=4):
        self.num_backends = num_backends
        self.current = 0
        
    def select_backend(self, obs: np.ndarray) -> int:
        """
        Select the next backend.
        obs: numpy array of shape (23,)
        Returns: int in {0, 1, 2, 3}
        """
        selected = self.current
        self.current = (self.current + 1) % self.num_backends
        return selected
