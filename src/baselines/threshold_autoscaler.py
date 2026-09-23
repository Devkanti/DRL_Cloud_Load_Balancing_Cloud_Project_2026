import numpy as np

class ThresholdAutoscaler:
    """
    Threshold-based Autoscaler Baseline
    Simulates auto-scaling by adding/removing available backends 
    when average CPU > 70% or < 30%.
    Routes requests among active backends using Round Robin.
    """
    def __init__(self, num_backends=4, min_active=2):
        self.num_backends = num_backends
        self.active_count = min_active
        self.current = 0
        self.min_active = min_active
        
    def select_backend(self, obs: np.ndarray) -> int:
        """
        Select the next backend based on CPU thresholds and active count.
        obs: numpy array of shape (23,)
        Returns: int in {0, 1, 2, 3}
        """
        # Compute mean CPU of ACTIVE backends (cpu is at indices 0, 5, 10, 15)
        cpu_utils = [obs[i * 5] for i in range(self.active_count)]
        mean_cpu = np.mean(cpu_utils) if cpu_utils else 0.0
        
        # Scale out if CPU > 70%
        if mean_cpu > 0.70 and self.active_count < self.num_backends:
            self.active_count += 1
            
        # Scale in if CPU < 30%
        elif mean_cpu < 0.30 and self.active_count > self.min_active:
            self.active_count -= 1
            
        # Route using Round Robin among the currently active backends
        selected = self.current % self.active_count
        self.current = (self.current + 1) % self.active_count
        
        return selected
