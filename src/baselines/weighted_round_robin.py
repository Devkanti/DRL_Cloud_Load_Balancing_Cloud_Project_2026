import numpy as np

class WeightedRoundRobin:
    """
    Weighted Round Robin Baseline
    Routes requests proportionally to 1 / cpu_utilization.
    """
    def __init__(self, num_backends=4):
        self.num_backends = num_backends
        
    def select_backend(self, obs: np.ndarray) -> int:
        """
        Select the next backend based on CPU utilization.
        obs: numpy array of shape (23,)
        Returns: int in {0, 1, 2, 3}
        """
        # Extract CPU utilization for each backend (indices 0, 5, 10, 15)
        cpu_utils = [obs[i * 5] for i in range(self.num_backends)]
        
        # Calculate weights (inverse of CPU utilization)
        # Add epsilon to prevent division by zero
        eps = 1e-4
        weights = [1.0 / (cpu + eps) for cpu in cpu_utils]
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Probabilistic selection acts as a stateless Weighted Round Robin
        selected = np.random.choice(self.num_backends, p=probs)
        return int(selected)
