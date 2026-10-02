#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #20: Lambda Deployment Script
=====================================================

Packages and deploys the StateCollector Lambda function with EventBridge trigger.

Usage:
    python src/aws/lambda_deploy.py deploy-collector   # Package + deploy + create trigger
    python src/aws/lambda_deploy.py invoke-collector    # Manually invoke and show result
    python src/aws/lambda_deploy.py logs-collector      # Show recent CloudWatch logs
    python src/aws/lambda_deploy.py delete-collector    # Delete Lambda + EventBridge rule
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
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


def _ssm_get(name: str) -> str:
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception:
        return ""


def _ssm_put(name: str, value: str, desc: str = ""):
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        ssm.put_parameter(Name=name, Value=value, Description=desc, Type="String",
                          Tags=[{"Key": "Project", "Value": "FlashBalanceAI"}])
    except ssm.exceptions.ParameterAlreadyExists:
        ssm.put_parameter(Name=name, Value=value, Description=desc, Type="String", Overwrite=True)


def _get_account_id() -> str:
    return boto3.client("sts").get_caller_identity()["Account"]


# ---------------------------------------------------------------------------
# Package Lambda
# ---------------------------------------------------------------------------

def _package_lambda(source_file: Path) -> bytes:
    """Create a zip file containing the Lambda source code."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add the main handler file as lambda_function.py
        zf.write(source_file, "lambda_function.py")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Deploy Collector
# ---------------------------------------------------------------------------

def cmd_deploy_collector(args):
    """Package and deploy the StateCollector Lambda."""
    cfg = _load_config()
    account_id = _get_account_id()
    lambda_client = boto3.client("lambda", region_name=REGION)
    events_client = boto3.client("events", region_name=REGION)

    func_name = cfg["lambda_state_collector_name"]
    role_name = cfg["lambda_role_name"]
    memory_mb = cfg["lambda_memory_mb"]
    timeout_sec = cfg["lambda_timeout_seconds"]
    runtime = cfg["lambda_runtime"]
    table_name = cfg["dynamodb_table_name"]
    asg_name = cfg["asg_name"]
    cw_namespace = cfg["cloudwatch_namespace"]
    tg_arn = _ssm_get(f"{SSM_PREFIX}/alb/target-group-arn")
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

    print(f"Issue #20: Deploy StateCollector Lambda")
    print(f"  Function : {func_name}")
    print(f"  Runtime  : {runtime}")
    print(f"  Memory   : {memory_mb} MB")
    print(f"  Timeout  : {timeout_sec}s")
    print(f"  Role     : {role_arn}")
    print(f"  Table    : {table_name}")
    print(f"  ASG      : {asg_name}")
    print(f"  TG ARN   : {tg_arn or '(not set -- ALB not created yet)'}")
    print()

    # 1. Package
    print("[1/4] Packaging Lambda function ...")
    source_file = PROJECT_ROOT / "src" / "aws" / "cloudwatch_collector.py"
    zip_bytes = _package_lambda(source_file)
    print(f"  [OK] Package size: {len(zip_bytes):,} bytes")

    # 2. Create or update Lambda function
    print("\n[2/4] Creating/updating Lambda function ...")
    env_vars = {
        "DYNAMODB_TABLE": table_name,
        "ASG_NAME": asg_name,
        "TARGET_GROUP_ARN": tg_arn,
        "CW_NAMESPACE": cw_namespace,
        "SESSION_ID": "live",
        "N_SERVERS": str(cfg["n_backend_instances"]),
    }

    try:
        # Try to update existing
        lambda_client.update_function_code(
            FunctionName=func_name,
            ZipFile=zip_bytes,
        )
        print(f"  [OK] Code updated")

        # Wait for update to complete
        time.sleep(2)

        lambda_client.update_function_configuration(
            FunctionName=func_name,
            Runtime=runtime,
            Handler="lambda_function.lambda_handler",
            MemorySize=memory_mb,
            Timeout=timeout_sec,
            Environment={"Variables": env_vars},
        )
        print(f"  [OK] Configuration updated")

    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new
        lambda_client.create_function(
            FunctionName=func_name,
            Runtime=runtime,
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Description="FlashBalanceAI Issue #20: Collects 23-dim state vector every 30s",
            Timeout=timeout_sec,
            MemorySize=memory_mb,
            Environment={"Variables": env_vars},
            Tags={
                "Project": "FlashBalanceAI",
                "Issue": "20",
                "Phase": "5",
            },
        )
        print(f"  [OK] Function created")

    # Wait for function to become Active
    print("  Waiting for function to become Active ...", end="", flush=True)
    for _ in range(30):
        resp = lambda_client.get_function(FunctionName=func_name)
        state = resp["Configuration"]["State"]
        if state == "Active":
            break
        print(".", end="", flush=True)
        time.sleep(2)
    print(f" {state}")

    func_arn = resp["Configuration"]["FunctionArn"]

    # 3. Create EventBridge rule (every 30 seconds = rate(30 seconds) is not supported,
    #    so use rate(1 minute) as the minimum. We'll note this constraint.)
    print("\n[3/4] Creating EventBridge trigger ...")
    rule_name = "FlashBalanceAI-StateCollector-Trigger"

    events_client.put_rule(
        Name=rule_name,
        ScheduleExpression="rate(1 minute)",
        State="ENABLED",
        Description="Trigger StateCollector Lambda every 1 minute (Issue #20)",
        Tags=[
            {"Key": "Project", "Value": "FlashBalanceAI"},
            {"Key": "Issue", "Value": "20"},
        ],
    )
    print(f"  [OK] Rule: {rule_name} (rate: 1 minute)")

    # Add Lambda as target
    events_client.put_targets(
        Rule=rule_name,
        Targets=[{
            "Id": "StateCollectorTarget",
            "Arn": func_arn,
        }],
    )
    print(f"  [OK] Target: {func_name}")

    # Grant EventBridge permission to invoke Lambda
    try:
        lambda_client.add_permission(
            FunctionName=func_name,
            StatementId="EventBridgeInvoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=f"arn:aws:events:{REGION}:{account_id}:rule/{rule_name}",
        )
        print(f"  [OK] Permission granted to EventBridge")
    except lambda_client.exceptions.ResourceConflictException:
        print(f"  [SKIP] Permission already exists")

    # 4. Store in SSM
    print("\n[4/4] Storing in SSM ...")
    _ssm_put(f"{SSM_PREFIX}/lambda/state-collector-arn", func_arn,
             "StateCollector Lambda ARN")
    _ssm_put(f"{SSM_PREFIX}/lambda/state-collector-rule", rule_name,
             "EventBridge rule name for StateCollector")
    print(f"  [OK] {SSM_PREFIX}/lambda/state-collector-arn")
    print(f"  [OK] {SSM_PREFIX}/lambda/state-collector-rule")

    print(f"\n[DONE] StateCollector Lambda deployed and scheduled.")
    print(f"  ARN : {func_arn}")
    print(f"  Rule: {rule_name} (every 1 minute)")
    print(f"\n  Test: python src/aws/lambda_deploy.py invoke-collector")
    print(f"  Logs: python src/aws/lambda_deploy.py logs-collector")


# ---------------------------------------------------------------------------
# Invoke Collector (manual test)
# ---------------------------------------------------------------------------

def cmd_invoke_collector(args):
    """Manually invoke the StateCollector and print results."""
    cfg = _load_config()
    func_name = cfg["lambda_state_collector_name"]
    lambda_client = boto3.client("lambda", region_name=REGION)

    print(f"Invoking {func_name} ...")
    resp = lambda_client.invoke(
        FunctionName=func_name,
        InvocationType="RequestResponse",
        LogType="Tail",
    )

    status = resp["StatusCode"]
    payload = json.loads(resp["Payload"].read().decode("utf-8"))

    print(f"\n  Status: {status}")
    print(f"  Response:")
    print(json.dumps(payload, indent=4, default=str))

    # Decode and print execution log
    import base64
    log_result = resp.get("LogResult", "")
    if log_result:
        log_text = base64.b64decode(log_result).decode("utf-8", errors="replace")
        print(f"\n  Execution Log:")
        for line in log_text.strip().split("\n"):
            print(f"    {line}")

    if payload.get("statusCode") == 200:
        body = payload.get("body", {})
        print(f"\n[PASS] State vector dim={body.get('state_dim')}, "
              f"instances={body.get('instances')}, "
              f"custom_metrics={body.get('custom_metrics')}")
    else:
        print(f"\n[FAIL] Lambda returned error")


# ---------------------------------------------------------------------------
# View Logs
# ---------------------------------------------------------------------------

def cmd_logs_collector(args):
    """Show recent CloudWatch log entries for the StateCollector."""
    cfg = _load_config()
    func_name = cfg["lambda_state_collector_name"]
    logs = boto3.client("logs", region_name=REGION)
    log_group = f"/aws/lambda/{func_name}"

    print(f"Recent logs for {func_name}:\n")

    try:
        # Get latest log stream
        streams = logs.describe_log_streams(
            logGroupName=log_group,
            orderBy="LastEventTime",
            descending=True,
            limit=1,
        )
        if not streams.get("logStreams"):
            print("  No log streams found.")
            return

        stream_name = streams["logStreams"][0]["logStreamName"]
        events = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            limit=20,
            startFromHead=False,
        )

        for event in events.get("events", []):
            msg = event["message"].strip()
            if msg:
                print(f"  {msg}")

    except ClientError as e:
        if "ResourceNotFoundException" in str(e):
            print(f"  Log group '{log_group}' not found. Lambda may not have been invoked yet.")
        else:
            raise


# ---------------------------------------------------------------------------
# Delete Collector
# ---------------------------------------------------------------------------

def cmd_delete_collector(args):
    """Delete the StateCollector Lambda and EventBridge rule."""
    cfg = _load_config()
    func_name = cfg["lambda_state_collector_name"]
    rule_name = "FlashBalanceAI-StateCollector-Trigger"
    lambda_client = boto3.client("lambda", region_name=REGION)
    events_client = boto3.client("events", region_name=REGION)

    print(f"Deleting StateCollector ...")

    # Remove EventBridge targets and rule
    try:
        events_client.remove_targets(Rule=rule_name, Ids=["StateCollectorTarget"])
        print(f"  [OK] Removed EventBridge target")
    except ClientError:
        pass

    try:
        events_client.delete_rule(Name=rule_name)
        print(f"  [OK] Deleted EventBridge rule: {rule_name}")
    except ClientError:
        pass

    # Delete Lambda
    try:
        lambda_client.delete_function(FunctionName=func_name)
        print(f"  [OK] Deleted Lambda: {func_name}")
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"  [SKIP] Lambda '{func_name}' not found")

    # Clean SSM
    ssm = boto3.client("ssm", region_name=REGION)
    for param in [f"{SSM_PREFIX}/lambda/state-collector-arn",
                  f"{SSM_PREFIX}/lambda/state-collector-rule"]:
        try:
            ssm.delete_parameter(Name=param)
            print(f"  [OK] Deleted SSM: {param}")
        except Exception:
            pass

    print(f"\n[DONE] StateCollector deleted.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashBalanceAI Issue #20 -- Lambda Deployment",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deploy-collector", help="Deploy StateCollector Lambda + EventBridge trigger")
    sub.add_parser("invoke-collector", help="Manually invoke StateCollector and show result")
    sub.add_parser("logs-collector", help="Show recent CloudWatch logs")
    sub.add_parser("delete-collector", help="Delete Lambda + EventBridge rule")

    args = parser.parse_args()
    commands = {
        "deploy-collector": cmd_deploy_collector,
        "invoke-collector": cmd_invoke_collector,
        "logs-collector": cmd_logs_collector,
        "delete-collector": cmd_delete_collector,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
