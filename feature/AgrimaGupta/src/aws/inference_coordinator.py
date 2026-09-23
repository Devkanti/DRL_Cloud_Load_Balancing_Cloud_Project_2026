"""
inference_coordinator.py — FlashBalanceAI Lambda Inference Coordinator
=======================================================================
AWS Lambda function that acts as the bridge between the state-collection
pipeline and the PPO inference server running on EC2.

Architecture context (ADR-001 D14, ADR-002):
  Lambda StateCollector  →  DynamoDB (current state)
  Lambda InferenceCoordinator  (THIS FILE)
      - Reads latest 23-dim state from DynamoDB
      - Calls EC2 inference server POST /action
      - Receives action ∈ {0,1,2,3}
      - Writes routing decision back to DynamoDB
      - Optionally triggers Lambda ScalingTrigger

Why Lambda (not EC2) for this coordinator?
  This function is stateless, event-driven, and < 5 MB — well within Lambda
  limits. The heavy model is kept on EC2 (inference_server.py).

Trigger:
  CloudWatch Events rule — every 500 ms (configurable via INVOKE_INTERVAL_MS)
  OR direct invocation by StateCollector after each batch write.

Environment variables (set in CloudFormation / deploy.py):
  INFERENCE_SERVER_URL   e.g. http://<ec2-private-ip>:6000
  DYNAMODB_TABLE         routing_decisions
  AWS_REGION             us-east-1

Lambda handler signature:
  def handler(event: dict, context: LambdaContext) -> dict

Author: Agrima Gupta (24BIT0253)
Phase:  5 — AWS Integration (Issue #22)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Environment variables (injected by CloudFormation at deploy time)
# ---------------------------------------------------------------------------
INFERENCE_SERVER_URL: str = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:6000")
DYNAMODB_TABLE: str = os.environ.get("DYNAMODB_TABLE", "routing_decisions")


# ---------------------------------------------------------------------------
# TODO (Issue #22): implement the following
# ---------------------------------------------------------------------------
# 1. Read latest state vector from DynamoDB:
#       dynamodb = boto3.resource("dynamodb")
#       table = dynamodb.Table(DYNAMODB_TABLE)
#       item = table.get_item(Key={"pk": "current_state"})["Item"]
#       state = item["state_vector"]   # list of 23 floats
#
# 2. Call inference server:
#       payload = json.dumps({"state": state}).encode()
#       req = urllib.request.Request(
#           f"{INFERENCE_SERVER_URL}/action",
#           data=payload,
#           headers={"Content-Type": "application/json"},
#           method="POST",
#       )
#       with urllib.request.urlopen(req, timeout=2) as resp:
#           result = json.loads(resp.read())
#       action = result["action"]
#
# 3. Write routing decision back to DynamoDB:
#       table.put_item(Item={"pk": "latest_action", "action": action,
#                            "timestamp": datetime.utcnow().isoformat()})
#
# 4. Optionally invoke ScalingTrigger Lambda if action requires scale-out/in.
# ---------------------------------------------------------------------------


def _read_state_from_dynamodb() -> list[float]:
    """
    Read the current 23-dimensional state vector from DynamoDB.
    Returns a placeholder zero-vector until Issue #22 is implemented.
    """
    # TODO (Issue #22): replace with real DynamoDB read
    logger.warning("_read_state_from_dynamodb: placeholder — returning zeros")
    return [0.0] * 23


def _call_inference_server(state: list[float]) -> int:
    """
    POST state to EC2 inference server and return the chosen action index.
    Returns -1 if the server is unavailable (fallback to Round Robin).
    """
    # TODO (Issue #22): implement with proper error handling / retries
    logger.warning("_call_inference_server: placeholder — returning action 0")
    return 0


def _write_action_to_dynamodb(action: int) -> None:
    """Write the chosen routing action back to DynamoDB for the backend to consume."""
    # TODO (Issue #22): implement DynamoDB write
    logger.info("_write_action_to_dynamodb: action=%d (placeholder — no-op)", action)


def handler(event: dict, context) -> dict:  # type: ignore[type-arg]
    """
    Lambda entry point.

    Parameters
    ----------
    event : dict
        CloudWatch Events payload (contents ignored — we pull state from DynamoDB).
    context : LambdaContext
        AWS Lambda context object (unused beyond logging).

    Returns
    -------
    dict
        {"statusCode": 200, "action": int}
    """
    logger.info("InferenceCoordinator invoked")

    state = _read_state_from_dynamodb()
    action = _call_inference_server(state)
    _write_action_to_dynamodb(action)

    logger.info("Routing action selected: %d", action)
    return {"statusCode": 200, "action": action}
