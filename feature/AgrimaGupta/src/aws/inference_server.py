"""
inference_server.py — FlashBalanceAI PPO Inference Server
==========================================================
Runs on a t2.micro EC2 instance. Loads a trained PPO model from S3 and
exposes a lightweight HTTP endpoint that the Lambda InferenceCoordinator
calls to select a routing action.

Architecture context (ADR-001 D14, ADR-002):
  JMeter → ALB → Backend EC2 × 4
  Lambda StateCollector → DynamoDB
  Lambda InferenceCoordinator → THIS SERVER (port 6000) → Lambda ScalingTrigger

Why EC2 and not Lambda?
  Stable-Baselines3 + PyTorch ≈ 300 MB — exceeds Lambda 250 MB deployment limit.
  A persistent t2.micro (Free Tier) avoids cold-start latency on every request.

API (HTTP/JSON on 0.0.0.0:6000):
  POST /action
    Request:  {"state": [float × 23]}
    Response: {"action": int, "q_values": [float × 4]}   # action ∈ {0,1,2,3}

  GET  /health
    Response: {"status": "ok", "model_loaded": bool}

Usage:
  python inference_server.py --model-key models/ppo_flash_v1.zip \
                              --bucket flashbalanceai-<account_id> \
                              --port 6000

Author: Agrima Gupta (24BIT0253)
Phase:  5 — AWS Integration (Issue #22)
"""

from __future__ import annotations

import argparse
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# TODO (Issue #22): implement the following
# ---------------------------------------------------------------------------
# 1. Download model from S3 using boto3:
#       s3.download_file(bucket, model_key, "/tmp/model.zip")
# 2. Load model with stable_baselines3.PPO.load("/tmp/model.zip")
# 3. In POST /action handler:
#       obs = np.array(request_body["state"], dtype=np.float32)
#       action, _ = model.predict(obs, deterministic=True)
#       return {"action": int(action)}
# 4. Wrap in Flask or use built-in HTTPServer (keep dependencies minimal)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Placeholder — will be replaced with actual model in Issue #22
_model = None
_model_loaded = False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the inference server."""
    parser = argparse.ArgumentParser(description="FlashBalanceAI PPO Inference Server")
    parser.add_argument("--model-key", default="models/ppo_flash_v1.zip",
                        help="S3 key of the trained PPO model zip")
    parser.add_argument("--bucket", required=True,
                        help="S3 bucket name (e.g. flashbalanceai-123456789012)")
    parser.add_argument("--port", type=int, default=6000,
                        help="TCP port to listen on (default: 6000)")
    return parser.parse_args()


class InferenceHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — to be fully implemented in Issue #22."""

    def do_GET(self) -> None:  # noqa: N802
        """Health-check endpoint: GET /health"""
        if self.path == "/health":
            self._respond(200, {"status": "ok", "model_loaded": _model_loaded})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        """Action endpoint: POST /action  body: {"state": [...23 floats...]}"""
        if self.path == "/action":
            # TODO (Issue #22): parse body, call model.predict(), return action
            self._respond(501, {"error": "not implemented yet — Issue #22"})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code: int, body: dict) -> None:
        import json
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:  # suppress default access log
        logger.debug(fmt, *args)


def main() -> None:
    """Entry point — parse args, (eventually) load model, start server."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    logger.info("Starting inference server on port %d", args.port)
    logger.info("Model key: s3://%s/%s", args.bucket, args.model_key)
    # TODO (Issue #22): load model here
    server = HTTPServer(("0.0.0.0", args.port), InferenceHandler)
    logger.info("Listening on 0.0.0.0:%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
