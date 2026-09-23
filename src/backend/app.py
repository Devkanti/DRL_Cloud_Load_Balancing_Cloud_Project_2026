"""
Flask backend server simulating an EC2 instance.
Tracks CPU utilization, active connections, and request queue.
"""
import time
import threading
import uuid
import os
from flask import Flask, jsonify

app = Flask(__name__)

# Configurable settings
MAX_CONNECTIONS = int(os.environ.get('MAX_CONNECTIONS', 100))
BASE_LATENCY_MS = float(os.environ.get('BASE_LATENCY_MS', 50.0))
INSTANCE_ID = os.environ.get('INSTANCE_ID', str(uuid.uuid4()))

# Global state
state_lock = threading.Lock()
active_connections = 0
queue_depth = 0
response_time_ema = BASE_LATENCY_MS
EMA_ALPHA = 0.1

# Semaphore to restrict max concurrent active connections
connection_semaphore = threading.Semaphore(MAX_CONNECTIONS)

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Returns:
        JSON response with health status and instance ID.
    """
    return jsonify({
        "status": "healthy",
        "instance_id": INSTANCE_ID
    }), 200

@app.route('/request', methods=['POST'])
def handle_request():
    """
    Simulates a backend request processing.
    The latency increases linearly as CPU utilization (active connections) goes up.
    Returns:
        JSON response with latency_ms and status code 200.
    """
    global queue_depth, active_connections, response_time_ema
    
    with state_lock:
        queue_depth += 1
        
    # Wait to acquire a connection slot
    acquired = connection_semaphore.acquire(timeout=10.0)
    
    with state_lock:
        queue_depth -= 1
        
    if not acquired:
        return jsonify({"error": "Service Unavailable", "status": 503}), 503
        
    try:
        with state_lock:
            active_connections += 1
            cpu_util = active_connections / MAX_CONNECTIONS
            
        latency_ms = BASE_LATENCY_MS * (1.0 + cpu_util)
        
        # Simulate processing delay
        time.sleep(latency_ms / 1000.0)
        
        with state_lock:
            response_time_ema = (EMA_ALPHA * latency_ms) + ((1 - EMA_ALPHA) * response_time_ema)
            
        return jsonify({"latency_ms": latency_ms, "status": 200}), 200
        
    finally:
        with state_lock:
            active_connections -= 1
        connection_semaphore.release()

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Metrics endpoint exposing current state.
    Returns:
        JSON response with cpu_util, active_connections, queue_depth, response_time_ema.
    """
    with state_lock:
        cpu_util = active_connections / MAX_CONNECTIONS
        return jsonify({
            "cpu_util": cpu_util,
            "active_connections": active_connections,
            "queue_depth": queue_depth,
            "response_time_ema": response_time_ema
        }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
