import pytest
from environment.flash_sale_env import FlashSaleEnv

def test_reward_range():
    env = FlashSaleEnv()
    env.reset()
    
    # Take a bunch of actions to see if reward stays in [-1, 1]
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert -1.0 <= reward <= 1.0

def test_reward_weights():
    env = FlashSaleEnv()
    
    # Force mock weights
    env.weights = {"w_lat": 0.5, "w_util": 0.5, "w_tput": 0.0, "w_sla": 0.0}
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Since tput and sla are 0, reward should only depend on lat and util
    assert -1.0 <= reward <= 1.0
    
def test_sla_violation_penalty():
    env = FlashSaleEnv()
    env.reset()
    
    # Overload backend 0 directly to bypass step logic overwriting arrival_rate
    for _ in range(400):
        env.backends[0].route_request(env.current_time)
        
    # Queue depth should be > 0 since max_connections is 100
    assert env.backends[0].queue_depth > 0
    
    # Advance time to force SLA violations (wait time > 200ms)
    env.backends[0].advance_time(env.current_time + 1.0)
    
    assert env.backends[0].sla_violations > 0
