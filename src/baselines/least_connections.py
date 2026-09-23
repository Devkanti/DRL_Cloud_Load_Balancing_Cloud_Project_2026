import numpy as np

class LeastConnections:
    """
    Least Connections Baseline
    Always routes to the backend with the fewest active connections.
    """
    def __init__(self, num_backends=4):
        self.num_backends = num_backends
        
    def select_backend(self, obs: np.ndarray) -> int:
        """
        Select the next backend based on active connections.
        obs: numpy array of shape (23,)
        Returns: int in {0, 1, 2, 3}
        """
        # Extract active connections for each backend (indices 1, 6, 11, 16)
        conns = [obs[i * 5 + 1] for i in range(self.num_backends)]
        
        # Select the backend with the minimum active connections
        return int(np.argmin(conns))
