import os
import boto3
import zipfile
from flask import Flask, request, jsonify
from stable_baselines3 import PPO

app = Flask(__name__)

# Global model variable
model = None

def download_and_load_model(bucket, key):
    global model
    local_zip = "/tmp/model.zip"
    local_dir = "/tmp/model_dir"
    
    print(f"Downloading s3://{bucket}/{key} to {local_zip}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_zip)
    
    # Wait, stable_baselines3 .load() takes the zip path directly
    print(f"Loading PPO model from {local_zip}")
    model = PPO.load(local_zip)
    print("Model loaded successfully.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    }), 200

@app.route("/action", methods=["POST"])
def get_action():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    data = request.json
    state_vector = data.get("state_vector")
    
    if not state_vector:
        return jsonify({"error": "Missing state_vector"}), 400
        
    # PPO model predict expects a numpy array. 
    import numpy as np
    obs = np.array(state_vector, dtype=np.float32)
    
    action, _states = model.predict(obs, deterministic=True)
    
    # action is usually a numpy scalar for discrete spaces
    action_val = int(action)
    
    return jsonify({
        "action": action_val
    }), 200

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()
    
    # The key is fixed or could be passed. We'll use the one from config: models/ppo_flash_v1.zip
    # Download the model before starting the server
    try:
        download_and_load_model(args.bucket, "models/ppo_flash_v1.zip")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        
    app.run(host="0.0.0.0", port=args.port, threaded=True)
