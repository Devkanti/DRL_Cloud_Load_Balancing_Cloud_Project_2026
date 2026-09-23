import pytest
import threading
from backend.app import app, MAX_CONNECTIONS, BASE_LATENCY_MS

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert 'instance_id' in data

def test_metrics_fields(client):
    response = client.get('/metrics')
    assert response.status_code == 200
    data = response.get_json()
    assert 'cpu_util' in data
    assert 'active_connections' in data
    assert 'queue_depth' in data
    assert 'response_time_ema' in data

def test_single_request_latency(client):
    response = client.post('/request')
    assert response.status_code == 200
    data = response.get_json()
    assert 'latency_ms' in data
    
    # Since it's the only request, active_connections becomes 1 when processed
    expected_latency = BASE_LATENCY_MS * (1.0 + (1.0 / MAX_CONNECTIONS))
    assert abs(data['latency_ms'] - expected_latency) < 1e-5

def test_latency_scales_linearly(monkeypatch):
    """
    Test that latency scales linearly with load (active connections).
    We mock time.sleep to block using a barrier, ensuring multiple threads
    are inside the processing block concurrently.
    """
    import time
    
    # We want 3 concurrent requests
    NUM_REQUESTS = 3
    barrier = threading.Barrier(NUM_REQUESTS)
    
    def mock_sleep(seconds):
        try:
            # Wait for all threads to reach the sleep phase (processing)
            barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            pass

    # Monkeypatch time.sleep so we don't actually wait the latencies,
    # but we DO block until all threads have incremented active_connections.
    monkeypatch.setattr(time, "sleep", mock_sleep)
    
    results = []
    
    # Use the test client for the threads
    test_client = app.test_client()

    def worker():
        resp = test_client.post('/request')
        if resp.status_code == 200:
            results.append(resp.get_json()['latency_ms'])
            
    threads = [threading.Thread(target=worker) for _ in range(NUM_REQUESTS)]
    
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(results) == NUM_REQUESTS
    
    # Because of the barrier, threads enter the processing block one by one.
    # The first thread sees active_connections = 1
    # The second thread sees active_connections = 2
    # The third thread sees active_connections = 3
    # They all calculate their latency based on the active_connections AT THAT MOMENT.
    # So we expect the latencies to correspond exactly to 1, 2, and 3 active connections.
    
    expected_latencies = {
        BASE_LATENCY_MS * (1.0 + (i / MAX_CONNECTIONS)) for i in range(1, NUM_REQUESTS + 1)
    }
    
    for res in results:
        matched = False
        for exp in expected_latencies:
            if abs(res - exp) < 1e-5:
                matched = True
                expected_latencies.remove(exp)
                break
        assert matched, f"Latency {res} did not match any expected latencies"
    
    assert len(expected_latencies) == 0, "Not all expected latencies were generated"
