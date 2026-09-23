import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os

class SimulatedBackend:
    def __init__(self, max_connections=100, base_latency_ms=50.0):
        self.max_connections = max_connections
        self.base_latency_ms = base_latency_ms
        
        self.active_requests = []  # [(finish_t, latency_ms)]
        self.queue = []            # [arrival_t]
        self.response_time_ema = base_latency_ms
        self.health_status = 1.0
        
        self.total_processed = 0
        self.sla_violations = 0
        
    @property
    def active_connections(self):
        return len(self.active_requests)
        
    @property
    def queue_depth(self):
        return len(self.queue)
        
    @property
    def cpu_util(self):
        return self.active_connections / self.max_connections
        
    def advance_time(self, current_t):
        self.total_processed = 0
        self.sla_violations = 0
        while self.active_requests:
            self.active_requests.sort(key=lambda x: x[0])
            finish_t, lat_ms = self.active_requests[0]
            if finish_t <= current_t:
                self.active_requests.pop(0)
                # Update EMA
                self.response_time_ema = 0.1 * lat_ms + 0.9 * self.response_time_ema
                self.total_processed += 1
                
                if self.queue:
                    arr_t = self.queue.pop(0)
                    cpu = min(1.0, (len(self.active_requests) + 1) / self.max_connections)
                    new_lat_ms = self.base_latency_ms * (1.0 + cpu)
                    new_finish_t = finish_t + new_lat_ms / 1000.0
                    self.active_requests.append((new_finish_t, new_lat_ms))
                    
                    total_time_ms = (new_finish_t - arr_t) * 1000.0
                    if total_time_ms > 200.0:
                        self.sla_violations += 1
            else:
                break
                
    def route_request(self, current_t):
        if self.active_connections < self.max_connections:
            cpu = min(1.0, (self.active_connections + 1) / self.max_connections)
            lat_ms = self.base_latency_ms * (1.0 + cpu)
            finish_t = current_t + lat_ms / 1000.0
            self.active_requests.append((finish_t, lat_ms))
            
            if lat_ms > 200.0:
                self.sla_violations += 1
        else:
            self.queue.append(current_t)

class FlashSaleEnv(gym.Env):
    """
    Simulates the load balancing MDP.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, seed=None, **kwargs):
        super(FlashSaleEnv, self).__init__()
        
        # Accept kwargs like render_mode if passed by gym
        self.render_mode = kwargs.get('render_mode')
        
        self.num_backends = 4
        self.action_space = spaces.Discrete(self.num_backends)
        
        # 5N + 3 = 23 dims
        self.observation_space = spaces.Box(
            low=0.0, high=1e5, shape=(23,), dtype=np.float32
        )
        
        self.max_steps = 7900
        self.step_duration = 0.1  # 100ms
        
        self.weights = {"w_lat": 0.4, "w_util": 0.2, "w_tput": 0.2, "w_sla": 0.2}
        
        # Attempt to load from config
        try:
            config_path = os.path.join(os.path.dirname(__file__), '../../configs/ppo_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'reward_weights' in config:
                        self.weights = config['reward_weights']
        except Exception:
            pass
            
        if seed is not None:
            self.reset(seed=seed)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.backends = [SimulatedBackend() for _ in range(self.num_backends)]
        self.current_step = 0
        self.current_time = 0.0
        
        self.arrival_rate = 100.0 
        self.burst_indicator = 0.0
        self.time_since_last_spike = 0.0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = []
        for b in self.backends:
            obs.extend([
                b.cpu_util,
                b.active_connections,
                b.queue_depth,
                b.response_time_ema,
                b.health_status
            ])
            
        obs.append(self.arrival_rate / 1000.0)
        obs.append(self.burst_indicator)
        obs.append(self.time_since_last_spike)
        
        return np.array(obs, dtype=np.float32)
        
    def step(self, action):
        self.current_time += self.step_duration
        self.current_step += 1
        
        # Simple dummy traffic model for self-sufficiency
        if 1000 <= self.current_step <= 4000:
            self.arrival_rate = 1000.0
            self.burst_indicator = 1.0
            self.time_since_last_spike = 0.0
        else:
            self.arrival_rate = 100.0
            self.burst_indicator = 0.0
            self.time_since_last_spike += self.step_duration
            
        total_processed = 0
        total_violations = 0
        for b in self.backends:
            b.advance_time(self.current_time)
            total_processed += b.total_processed
            total_violations += b.sla_violations
            
        # Route requests arriving in this step
        # Poisson distribution based on arrival rate
        # For simplicity and determinism (unless seeded np.random is used), 
        # just use expected value
        num_requests = int(self.arrival_rate * self.step_duration)
        if num_requests < 1:
            num_requests = 1
            
        for _ in range(num_requests):
            self.backends[action].route_request(self.current_time)
            
        # Compute Reward
        avg_ema = np.mean([b.response_time_ema for b in self.backends])
        r_lat = max(0.0, 1.0 - (avg_ema / 200.0))
        
        avg_cpu = np.mean([b.cpu_util for b in self.backends])
        r_util = avg_cpu
        
        # Tput: normalise against max expected throughput per step
        max_expected_tput = 100.0 
        r_tput = min(1.0, total_processed / max_expected_tput)
        
        sla_rate = total_violations / max(1, total_processed)
        r_sla = 1.0 - sla_rate
        
        reward = (
            self.weights.get('w_lat', 0.4) * r_lat +
            self.weights.get('w_util', 0.2) * r_util +
            self.weights.get('w_tput', 0.2) * r_tput +
            self.weights.get('w_sla', 0.2) * r_sla
        )
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), float(reward), terminated, truncated, {}
        
    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Rate: {self.arrival_rate}")
            for i, b in enumerate(self.backends):
                print(f" Backend {i}: CPU={b.cpu_util:.2f}, Conns={b.active_connections}, Queue={b.queue_depth}")
        return None

# Register the env
gym.envs.registration.register(
    id='FlashSaleEnv-v1',
    entry_point='environment.flash_sale_env:FlashSaleEnv',
    max_episode_steps=7900,
)
