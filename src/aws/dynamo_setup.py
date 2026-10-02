#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #19: DynamoDB Table Setup
=================================================

Creates and configures the routing_decisions DynamoDB table.

Usage:
    python src/aws/dynamo_setup.py create     # Create table + enable TTL
    python src/aws/dynamo_setup.py verify     # Verify table config
    python src/aws/dynamo_setup.py test       # Round-trip integration test
    python src/aws/dynamo_setup.py delete     # Delete table

Prerequisites:
    - Issue #15 IAM roles created
    - pip install boto3 pyyaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import boto3
    import yaml
    from botocore.exceptions import ClientError
except ImportError as e:
    print(f"ERROR: Missing dependency -- {e}. Run: pip install boto3 pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AWS_CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"
REGION = "us-east-1"
SSM_PREFIX = "/flashbalanceai"


def _load_config() -> dict:
    with open(AWS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _dynamo_client():
    return boto3.client("dynamodb", region_name=REGION)


def _dynamo_resource():
    return boto3.resource("dynamodb", region_name=REGION)


def _table_exists(client, table_name: str) -> bool:
    try:
        resp = client.describe_table(TableName=table_name)
        return resp["Table"]["TableStatus"] in ("ACTIVE", "CREATING")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return False
        raise


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_create(args):
    """Create DynamoDB table with TTL."""
    cfg = _load_config()
    client = _dynamo_client()
    table_name = cfg["dynamodb_table_name"]

    print(f"Issue #19: DynamoDB Table Setup")
    print(f"  Table    : {table_name}")
    print(f"  Region   : {REGION}")
    print(f"  Billing  : PAY_PER_REQUEST (on-demand, free tier)")
    print()

    # ---- 1. Create table ----
    if _table_exists(client, table_name):
        print(f"[SKIP] Table '{table_name}' already exists")
    else:
        print(f"[1/3] Creating table '{table_name}' ...")
        client.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
            Tags=[
                {"Key": "Project", "Value": "FlashBalanceAI"},
                {"Key": "Issue", "Value": "19"},
                {"Key": "Phase", "Value": "4"},
            ],
        )
        print("  [OK] Table creation initiated")

        # Wait for ACTIVE
        print("  Waiting for table to become ACTIVE ...", end="", flush=True)
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=table_name, WaiterConfig={"Delay": 3, "MaxAttempts": 30})
        print(" ACTIVE")

    # ---- 2. Enable TTL ----
    print(f"\n[2/3] Enabling TTL on 'expires_at' attribute ...")
    try:
        ttl_resp = client.describe_time_to_live(TableName=table_name)
        ttl_status = ttl_resp["TimeToLiveDescription"]["TimeToLiveStatus"]
        if ttl_status in ("ENABLED", "ENABLING"):
            print(f"  [SKIP] TTL already {ttl_status}")
        else:
            client.update_time_to_live(
                TableName=table_name,
                TimeToLiveSpecification={
                    "Enabled": True,
                    "AttributeName": "expires_at",
                },
            )
            print("  [OK] TTL enabled on 'expires_at'")
    except ClientError as e:
        if "already exists" in str(e).lower() or "ValidationException" in str(e):
            print(f"  [SKIP] TTL already configured")
        else:
            raise

    # ---- 3. Store in SSM ----
    print(f"\n[3/3] Storing table info in SSM Parameter Store ...")
    ssm = boto3.client("ssm", region_name=REGION)
    ssm_params = {
        f"{SSM_PREFIX}/dynamodb/table-name": (table_name, "DynamoDB table name"),
        f"{SSM_PREFIX}/dynamodb/table-arn": (
            f"arn:aws:dynamodb:{REGION}:{boto3.client('sts').get_caller_identity()['Account']}:table/{table_name}",
            "DynamoDB table ARN",
        ),
    }
    for param_name, (value, description) in ssm_params.items():
        try:
            ssm.put_parameter(
                Name=param_name, Value=value, Description=description,
                Type="String",
                Tags=[{"Key": "Project", "Value": "FlashBalanceAI"}, {"Key": "Issue", "Value": "19"}],
            )
        except ssm.exceptions.ParameterAlreadyExists:
            ssm.put_parameter(
                Name=param_name, Value=value, Description=description,
                Type="String", Overwrite=True,
            )
        print(f"  [OK] {param_name} = {value}")

    print(f"\n[DONE] Issue #19 -- DynamoDB table setup complete.")
    print(f"   Run 'python src/aws/dynamo_setup.py verify' to confirm.")
    print(f"   Run 'python src/aws/dynamo_setup.py test' for round-trip test.")


def cmd_verify(args):
    """Verify table exists with correct config."""
    cfg = _load_config()
    client = _dynamo_client()
    table_name = cfg["dynamodb_table_name"]
    all_ok = True

    print(f"Verifying DynamoDB table: {table_name}\n")

    # Table exists
    if not _table_exists(client, table_name):
        print(f"[FAIL] Table '{table_name}' does not exist")
        return False

    resp = client.describe_table(TableName=table_name)
    table = resp["Table"]

    print(f"[OK] Table exists (status: {table['TableStatus']})")

    # Key schema
    keys = {k["AttributeName"]: k["KeyType"] for k in table["KeySchema"]}
    if keys.get("pk") == "HASH" and keys.get("sk") == "RANGE":
        print(f"[OK] Key schema: pk (HASH) + sk (RANGE)")
    else:
        print(f"[FAIL] Unexpected key schema: {keys}")
        all_ok = False

    # Billing mode
    billing = table.get("BillingModeSummary", {}).get("BillingMode", "PROVISIONED")
    if billing == "PAY_PER_REQUEST":
        print(f"[OK] Billing mode: PAY_PER_REQUEST (on-demand)")
    else:
        print(f"[WARN] Billing mode: {billing} (expected PAY_PER_REQUEST)")
        all_ok = False

    # TTL
    ttl_resp = client.describe_time_to_live(TableName=table_name)
    ttl_status = ttl_resp["TimeToLiveDescription"]["TimeToLiveStatus"]
    ttl_attr = ttl_resp["TimeToLiveDescription"].get("AttributeName", "N/A")
    if ttl_status in ("ENABLED", "ENABLING") and ttl_attr == "expires_at":
        print(f"[OK] TTL: {ttl_status} on '{ttl_attr}'")
    else:
        print(f"[FAIL] TTL status={ttl_status}, attribute={ttl_attr}")
        all_ok = False

    # Item count
    print(f"\n  Item count  : {table.get('ItemCount', 0)}")
    print(f"  Table size  : {table.get('TableSizeBytes', 0)} bytes")

    # SSM
    print()
    ssm = boto3.client("ssm", region_name=REGION)
    for param in [f"{SSM_PREFIX}/dynamodb/table-name", f"{SSM_PREFIX}/dynamodb/table-arn"]:
        try:
            val = ssm.get_parameter(Name=param)["Parameter"]["Value"]
            print(f"[OK] {param} = {val}")
        except Exception:
            print(f"[FAIL] Missing SSM param: {param}")
            all_ok = False

    print()
    if all_ok:
        print("[PASS] All DynamoDB checks passed.")
    else:
        print("[WARN] Some checks failed -- review above.")
    return all_ok


def cmd_test(args):
    """Integration test: write state, read it back, write action, read it back."""
    # Import the utility module
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from aws.dynamo_utils import write_state, read_latest_state, write_action, read_latest_action

    print("DynamoDB Integration Test\n")

    # Mock 23-dim state vector
    mock_state = [
        0.45, 0.62, 0.38, 0.71,   # CPU utilization (4 servers)
        0.30, 0.55, 0.22, 0.68,   # Memory utilization (4 servers)
        0.20, 0.40, 0.15, 0.50,   # Active connections (4 servers, normalized)
        0.35, 0.60, 0.25, 0.70,   # Response times (4 servers, normalized)
        5.2,                       # Request rate
        0.45,                      # Avg CPU
        0.44,                      # Avg memory
        0.31,                      # Avg connections
        0.48,                      # Avg response time
        0.15,                      # Std CPU
        0.18,                      # Std memory
    ]

    test_session = "test-session-001"

    # Test 1: write_state + read_latest_state
    print("[TEST 1] write_state + read_latest_state")
    write_state(session_id=test_session, state_vector=mock_state, metadata={"source": "integration_test"})
    print("  [OK] write_state succeeded")

    time.sleep(0.5)  # Brief pause for consistency

    result = read_latest_state(session_id=test_session)
    if result is None:
        print("  [FAIL] read_latest_state returned None")
        return

    read_back = result["state_vector"]
    if len(read_back) == len(mock_state):
        print(f"  [OK] read_latest_state returned {len(read_back)}-dim vector")
    else:
        print(f"  [FAIL] Expected {len(mock_state)}-dim, got {len(read_back)}-dim")
        return

    # Check values match (within float precision)
    max_diff = max(abs(a - b) for a, b in zip(mock_state, read_back))
    if max_diff < 1e-6:
        print(f"  [OK] Values match (max diff: {max_diff:.2e})")
    else:
        print(f"  [FAIL] Values differ (max diff: {max_diff:.2e})")
        return

    # Test 2: write_action + read_latest_action
    print("\n[TEST 2] write_action + read_latest_action")
    mock_action = 2  # Route to server index 2
    write_action(
        session_id=test_session,
        action=mock_action,
        q_values=[0.12, 0.34, 0.89, 0.56],
        metadata={"algorithm": "PPO", "epsilon": 0.05},
    )
    print("  [OK] write_action succeeded")

    time.sleep(0.5)

    action_result = read_latest_action(session_id=test_session)
    if action_result is None:
        print("  [FAIL] read_latest_action returned None")
        return

    if action_result["action"] == mock_action:
        print(f"  [OK] Action matches: {action_result['action']}")
    else:
        print(f"  [FAIL] Expected action={mock_action}, got {action_result['action']}")
        return

    if action_result.get("q_values"):
        print(f"  [OK] Q-values present: {action_result['q_values']}")
    else:
        print(f"  [WARN] Q-values missing")

    print(f"\n[PASS] All integration tests passed!")


def cmd_delete(args):
    """Delete the DynamoDB table."""
    cfg = _load_config()
    client = _dynamo_client()
    table_name = cfg["dynamodb_table_name"]

    if not _table_exists(client, table_name):
        print(f"Table '{table_name}' does not exist.")
        return

    print(f"Deleting table '{table_name}' ...")
    client.delete_table(TableName=table_name)
    print(f"  [OK] Table deletion initiated")

    # Clean SSM
    ssm = boto3.client("ssm", region_name=REGION)
    for param in [f"{SSM_PREFIX}/dynamodb/table-name", f"{SSM_PREFIX}/dynamodb/table-arn"]:
        try:
            ssm.delete_parameter(Name=param)
            print(f"  [OK] Deleted SSM param: {param}")
        except ssm.exceptions.ParameterNotFound:
            pass

    print(f"\n[DONE] DynamoDB table deleted.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashBalanceAI Issue #19 -- DynamoDB Table Setup",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Create table + enable TTL")
    sub.add_parser("verify", help="Verify table configuration")
    sub.add_parser("test", help="Run integration test (write/read round-trip)")
    sub.add_parser("delete", help="Delete table")

    args = parser.parse_args()
    {"create": cmd_create, "verify": cmd_verify, "test": cmd_test, "delete": cmd_delete}[args.command](args)


if __name__ == "__main__":
    main()
