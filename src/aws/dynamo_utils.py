#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #19: DynamoDB Utility Functions
======================================================

Provides read/write helpers for the ``routing_decisions`` DynamoDB table.

Table schema
------------
- pk (String, HASH)  : "STATE#<session_id>" or "ACTION#<session_id>"
- sk (String, RANGE) : ISO-8601 timestamp   e.g. "2026-10-02T14:30:00.123456"
- expires_at (Number): epoch seconds, TTL = now + 24 h

Functions
---------
- write_state(session_id, state_vector, metadata=None)
- read_latest_state(session_id) -> dict | None
- write_action(session_id, action, q_values=None, metadata=None)
- read_latest_action(session_id) -> dict | None

Usage:
    from src.aws.dynamo_utils import write_state, read_latest_state
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import boto3
    import yaml
    from boto3.dynamodb.conditions import Key
    from botocore.exceptions import ClientError
except ImportError as e:
    raise ImportError(f"Missing dependency -- {e}. Run: pip install boto3 pyyaml") from e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AWS_CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"
REGION = "us-east-1"
TTL_SECONDS = 24 * 60 * 60  # 24 hours

_table_resource = None  # cached


def _load_config() -> dict:
    with open(AWS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _get_table():
    """Return a cached boto3 Table resource."""
    global _table_resource
    if _table_resource is None:
        cfg = _load_config()
        dynamo = boto3.resource("dynamodb", region_name=REGION)
        _table_resource = dynamo.Table(cfg["dynamodb_table_name"])
    return _table_resource


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 with microseconds for sort key uniqueness."""
    return datetime.now(timezone.utc).isoformat()


def _ttl_epoch() -> int:
    """Return epoch seconds 24 hours from now."""
    return int(time.time()) + TTL_SECONDS


def _floats_to_decimals(values: List[float]) -> List[Decimal]:
    """Convert float list to Decimal list (DynamoDB requires Decimal for numbers)."""
    return [Decimal(str(v)) for v in values]


def _decimals_to_floats(values: list) -> List[float]:
    """Convert Decimal list back to float list."""
    return [float(v) for v in values]


def _sanitize_for_dynamo(obj):
    """Recursively convert floats to Decimal in dicts/lists (DynamoDB rejects raw floats)."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _sanitize_for_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_dynamo(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------

def write_state(
    session_id: str,
    state_vector: List[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Write a state observation to DynamoDB.

    Parameters
    ----------
    session_id : str
        Experiment session identifier (e.g. "exp-001").
    state_vector : list[float]
        The 23-dimensional state vector from the environment.
    metadata : dict, optional
        Extra key-value pairs to store alongside the state.

    Returns
    -------
    dict with "pk" and "sk" of the written item.
    """
    table = _get_table()
    pk = f"STATE#{session_id}"
    sk = _now_iso()

    item = {
        "pk": pk,
        "sk": sk,
        "state_vector": _floats_to_decimals(state_vector),
        "dim": len(state_vector),
        "expires_at": _ttl_epoch(),
    }
    if metadata:
        item["metadata"] = _sanitize_for_dynamo(metadata)

    table.put_item(Item=item)
    return {"pk": pk, "sk": sk}


def read_latest_state(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Read the most recent state for a session.

    Returns
    -------
    dict with keys: pk, sk, state_vector (list[float]), dim, metadata, expires_at.
    None if no state found.
    """
    table = _get_table()
    pk = f"STATE#{session_id}"

    resp = table.query(
        KeyConditionExpression=Key("pk").eq(pk),
        ScanIndexForward=False,  # descending by sk (latest first)
        Limit=1,
    )

    items = resp.get("Items", [])
    if not items:
        return None

    item = items[0]
    item["state_vector"] = _decimals_to_floats(item["state_vector"])
    item["dim"] = int(item["dim"])
    item["expires_at"] = int(item["expires_at"])
    return item


def write_action(
    session_id: str,
    action: int,
    q_values: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Write a routing action (decision) to DynamoDB.

    Parameters
    ----------
    session_id : str
        Experiment session identifier.
    action : int
        The chosen backend server index (0..N-1).
    q_values : list[float], optional
        Q-values or action probabilities from the DRL agent.
    metadata : dict, optional
        Extra info (algorithm name, epsilon, etc.).

    Returns
    -------
    dict with "pk" and "sk" of the written item.
    """
    table = _get_table()
    pk = f"ACTION#{session_id}"
    sk = _now_iso()

    item = {
        "pk": pk,
        "sk": sk,
        "action": action,
        "expires_at": _ttl_epoch(),
    }
    if q_values is not None:
        item["q_values"] = _floats_to_decimals(q_values)
    if metadata:
        item["metadata"] = _sanitize_for_dynamo(metadata)

    table.put_item(Item=item)
    return {"pk": pk, "sk": sk}


def read_latest_action(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Read the most recent routing action for a session.

    Returns
    -------
    dict with keys: pk, sk, action (int), q_values (list[float]|None), metadata, expires_at.
    None if no action found.
    """
    table = _get_table()
    pk = f"ACTION#{session_id}"

    resp = table.query(
        KeyConditionExpression=Key("pk").eq(pk),
        ScanIndexForward=False,
        Limit=1,
    )

    items = resp.get("Items", [])
    if not items:
        return None

    item = items[0]
    item["action"] = int(item["action"])
    if "q_values" in item:
        item["q_values"] = _decimals_to_floats(item["q_values"])
    item["expires_at"] = int(item["expires_at"])
    return item
