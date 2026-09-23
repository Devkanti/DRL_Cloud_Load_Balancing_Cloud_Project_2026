import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from environment.flash_sale_env import FlashSaleEnv

def test_check_env():
    # Instantiate using gym.make to avoid spec warnings
    env = gym.make('FlashSaleEnv-v1')
    unwrapped_env = env.unwrapped
    unwrapped_env.spec = env.spec
    # verify it passes gym's standard checker without warnings
    check_env(unwrapped_env)

def test_observation_shape():
    env = FlashSaleEnv()
    obs, info = env.reset()
    assert obs.shape == (23,)
    assert env.observation_space.contains(obs)
    
def test_step_logic():
    env = FlashSaleEnv()
    obs, info = env.reset()
    
    # Step 0
    obs, reward, terminated, truncated, info = env.step(0)
    
    # action 0 means backend 0 gets the requests
    # Active connections on backend 0 should increase
    assert obs[1] > 0 # active_connections is the 2nd index of backend 0
    assert obs[6] == 0 # backend 1 active_connections is 0
    
    assert -1.0 <= reward <= 1.0
    assert not terminated
    
def test_termination():
    env = FlashSaleEnv()
    env.reset()
    env.current_step = 7899
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
