import numpy as np
import pytest
from baselines.round_robin import RoundRobin
from baselines.weighted_round_robin import WeightedRoundRobin
from baselines.least_connections import LeastConnections
from baselines.threshold_autoscaler import ThresholdAutoscaler

def test_round_robin():
    rr = RoundRobin()
    dummy_obs = np.zeros(23)
    
    # Verify return types and bounds
    for _ in range(10):
        action = rr.select_backend(dummy_obs)
        assert isinstance(action, int)
        assert 0 <= action <= 3
        
    # Verify even distribution over 100 steps
    rr = RoundRobin()
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(100):
        counts[rr.select_backend(dummy_obs)] += 1
        
    for i in range(4):
        assert counts[i] == 25

def test_weighted_round_robin():
    wrr = WeightedRoundRobin()
    # Mock observation: backends 0, 1, 2 have high CPU (0.9), backend 3 has low CPU (0.1)
    obs = np.zeros(23)
    obs[0] = 0.9
    obs[5] = 0.9
    obs[10] = 0.9
    obs[15] = 0.1
    
    action = wrr.select_backend(obs)
    assert isinstance(action, int)
    assert 0 <= action <= 3
    
    # Over 1000 steps, backend 3 should be selected much more often
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(1000):
        counts[wrr.select_backend(obs)] += 1
    
    assert counts[3] > counts[0]
    assert counts[3] > counts[1]
    assert counts[3] > counts[2]

def test_least_connections():
    lc = LeastConnections()
    obs = np.zeros(23)
    # Mock active connections (indices 1, 6, 11, 16)
    obs[1] = 50
    obs[6] = 30
    obs[11] = 10  # Minimum connections
    obs[16] = 20
    
    for _ in range(5):
        action = lc.select_backend(obs)
        assert isinstance(action, int)
        assert action == 2

def test_threshold_autoscaler():
    scaler = ThresholdAutoscaler(num_backends=4, min_active=2)
    
    # Test scale out
    # Active backends (0 and 1) have CPU > 0.7
    obs = np.zeros(23)
    obs[0] = 0.8
    obs[5] = 0.8
    obs[10] = 0.8
    obs[15] = 0.8
    
    # Initially active=2
    action = scaler.select_backend(obs)
    assert isinstance(action, int)
    assert 0 <= action <= 3
    
    # After one call with high CPU, active_count should increase from 2 to 3
    assert scaler.active_count == 3
    
    # One more call -> increases to 4
    scaler.select_backend(obs)
    assert scaler.active_count == 4
    
    # Test scale in
    # CPU < 0.3
    obs_low = np.zeros(23)
    obs_low[0] = 0.2
    obs_low[5] = 0.2
    obs_low[10] = 0.2
    obs_low[15] = 0.2
    
    scaler.select_backend(obs_low)
    # Reduces from 4 to 3
    assert scaler.active_count == 3
    
    scaler.select_backend(obs_low)
    # Reduces from 3 to 2
    assert scaler.active_count == 2
    
    # Should not drop below min_active (2)
    scaler.select_backend(obs_low)
    assert scaler.active_count == 2
