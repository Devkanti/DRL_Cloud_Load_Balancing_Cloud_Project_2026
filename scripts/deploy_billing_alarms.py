#!/usr/bin/env python3
"""
FlashBalanceAI — Issue #14: Deploy / Verify / Teardown Billing Alarms
=====================================================================

Usage:
    python scripts/deploy_billing_alarms.py deploy   --email1 a@x.com --email2 b@x.com [--email3 c@x.com] [--email4 d@x.com]
    python scripts/deploy_billing_alarms.py verify
    python scripts/deploy_billing_alarms.py test     # sets $0.01 threshold, waits, then restores
    python scripts/deploy_billing_alarms.py delete

Prerequisites:
    - AWS CLI configured (aws configure) with us-east-1 default region
    - Billing alerts enabled in account settings (Billing → Billing preferences → "Receive Billing Alerts")
    - pip install boto3 pyyaml

References:
    - ADR-001 D22 (billing safeguards)
    - configs/aws_config.yaml lines 65-66 (threshold values)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import boto3
    import yaml
except ImportError as e:
    print(f"ERROR: Missing dependency — {e}. Run: pip install boto3 pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AWS_CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"
TEMPLATE_PATH = PROJECT_ROOT / "infra" / "cloudformation" / "billing-alarms.yaml"
STACK_NAME = "FlashBalanceAI-BillingAlarms"
REGION = "us-east-1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_aws_config() -> dict:
    """Load configs/aws_config.yaml."""
    with open(AWS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _cfn_client():
    return boto3.client("cloudformation", region_name=REGION)


def _cw_client():
    return boto3.client("cloudwatch", region_name=REGION)


def _wait_for_stack(cfn, stack_name: str, target_status: str, timeout: int = 300):
    """Poll stack until it reaches target_status or times out."""
    print(f"  Waiting for stack '{stack_name}' → {target_status} ...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = cfn.describe_stacks(StackName=stack_name)
            status = resp["Stacks"][0]["StackStatus"]
        except cfn.exceptions.ClientError:
            if "DELETE_COMPLETE" in target_status:
                print(" done.")
                return
            raise
        if status == target_status:
            print(" done.")
            return
        if "FAILED" in status or "ROLLBACK" in status:
            reason = resp["Stacks"][0].get("StackStatusReason", "unknown")
            print(f"\n  ERROR: Stack reached {status} — {reason}")
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(10)
    print(f"\n  ERROR: Timed out after {timeout}s (last status: {status})")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_deploy(args):
    """Deploy the billing-alarms CloudFormation stack."""
    cfg = _load_aws_config()
    warning_usd = cfg.get("billing_alarm_warning_usd", 5)
    ceiling_usd = cfg.get("billing_alarm_action_usd", 15)

    # Read template body
    template_body = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Build parameters
    params = [
        {"ParameterKey": "WarningThresholdUSD", "ParameterValue": str(warning_usd)},
        {"ParameterKey": "CeilingThresholdUSD", "ParameterValue": str(ceiling_usd)},
        {"ParameterKey": "Email1", "ParameterValue": args.email1},
        {"ParameterKey": "Email2", "ParameterValue": args.email2},
    ]
    if args.email3:
        params.append({"ParameterKey": "Email3", "ParameterValue": args.email3})
    if args.email4:
        params.append({"ParameterKey": "Email4", "ParameterValue": args.email4})

    cfn = _cfn_client()

    # Check if stack already exists
    try:
        cfn.describe_stacks(StackName=STACK_NAME)
        stack_exists = True
    except cfn.exceptions.ClientError:
        stack_exists = False

    print(f"{'Updating' if stack_exists else 'Creating'} stack '{STACK_NAME}' ...")
    print(f"  Warning alarm  : ${warning_usd}")
    print(f"  Ceiling alarm  : ${ceiling_usd}")
    print(f"  Email1         : {args.email1}")
    print(f"  Email2         : {args.email2}")
    if args.email3:
        print(f"  Email3         : {args.email3}")
    if args.email4:
        print(f"  Email4         : {args.email4}")

    common_kwargs = dict(
        StackName=STACK_NAME,
        TemplateBody=template_body,
        Parameters=params,
        Tags=[
            {"Key": "Project", "Value": "FlashBalanceAI"},
            {"Key": "Issue", "Value": "14"},
            {"Key": "Phase", "Value": "4"},
        ],
    )

    if stack_exists:
        try:
            cfn.update_stack(**common_kwargs)
            _wait_for_stack(cfn, STACK_NAME, "UPDATE_COMPLETE")
        except cfn.exceptions.ClientError as e:
            if "No updates" in str(e):
                print("  Stack is already up-to-date.")
            else:
                raise
    else:
        cfn.create_stack(**common_kwargs)
        _wait_for_stack(cfn, STACK_NAME, "CREATE_COMPLETE")

    # Print outputs
    resp = cfn.describe_stacks(StackName=STACK_NAME)
    outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0].get("Outputs", [])}
    print("\n--- Stack Outputs ---")
    for key, val in outputs.items():
        print(f"  {key}: {val}")

    print("\n[ACTION REQUIRED] Check all team member inboxes and CONFIRM the SNS subscription emails.")
    print("  Alarms will NOT notify unconfirmed addresses.\n")

    # Write ARNs as comments into aws_config.yaml
    _update_config_comments(outputs)


def _update_config_comments(outputs: dict):
    """Append alarm ARN documentation to configs/aws_config.yaml."""
    lines = AWS_CONFIG_PATH.read_text(encoding="utf-8").splitlines()

    # Remove any previously written billing alarm ARN comments
    cleaned = [l for l in lines if not l.startswith("# billing_alarm_")]

    sns_arn = outputs.get("SNSTopicARN", "UNKNOWN")
    warning_arn = outputs.get("WarningAlarmARN", "UNKNOWN")
    ceiling_arn = outputs.get("CeilingAlarmARN", "UNKNOWN")

    cleaned.append("")
    cleaned.append("# --- Billing Alarm ARNs (auto-populated by deploy_billing_alarms.py — Issue #14) ---")
    cleaned.append(f"# billing_alarm_sns_topic_arn: \"{sns_arn}\"")
    cleaned.append(f"# billing_alarm_warning_arn:   \"{warning_arn}\"")
    cleaned.append(f"# billing_alarm_ceiling_arn:   \"{ceiling_arn}\"")

    AWS_CONFIG_PATH.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    print(f"  Alarm ARNs documented in {AWS_CONFIG_PATH.relative_to(PROJECT_ROOT)}")


def cmd_verify(args):
    """Verify that both alarms exist and the SNS topic has subscriptions."""
    cfn = _cfn_client()
    cw = _cw_client()
    sns = boto3.client("sns", region_name=REGION)

    # 1. Stack exists?
    try:
        resp = cfn.describe_stacks(StackName=STACK_NAME)
        status = resp["Stacks"][0]["StackStatus"]
        print(f"✓ Stack '{STACK_NAME}' exists — status: {status}")
    except cfn.exceptions.ClientError:
        print(f"✗ Stack '{STACK_NAME}' does NOT exist. Run 'deploy' first.")
        sys.exit(1)

    outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0].get("Outputs", [])}

    # 2. Alarms exist in CloudWatch?
    for label, key in [("Warning", "WarningAlarmName"), ("Ceiling", "CeilingAlarmName")]:
        name = outputs.get(key)
        if not name:
            print(f"✗ {label} alarm name not found in stack outputs.")
            continue
        alarms = cw.describe_alarms(AlarmNames=[name])
        if alarms["MetricAlarms"]:
            a = alarms["MetricAlarms"][0]
            print(f"✓ {label} alarm '{name}' — state: {a['StateValue']}, threshold: ${a['Threshold']}")
        else:
            print(f"✗ {label} alarm '{name}' NOT found in CloudWatch!")

    # 3. SNS subscriptions
    topic_arn = outputs.get("SNSTopicARN")
    if topic_arn:
        subs = sns.list_subscriptions_by_topic(TopicArn=topic_arn)["Subscriptions"]
        print(f"\n  SNS subscriptions for {topic_arn}:")
        for s in subs:
            print(f"    {s['Endpoint']} — {s['SubscriptionArn']}")
        confirmed = sum(1 for s in subs if "arn:aws:sns:" in s.get("SubscriptionArn", ""))
        pending = len(subs) - confirmed
        print(f"  {confirmed} confirmed, {pending} pending confirmation")
    else:
        print("✗ SNS Topic ARN not found in stack outputs.")


def cmd_test(args):
    """
    Test alarm trigger: temporarily set warning threshold to $0.01,
    wait for alarm state change, then restore original threshold.
    """
    cfg = _load_aws_config()
    original_threshold = cfg.get("billing_alarm_warning_usd", 5)
    test_threshold = 0.01

    cfn = _cfn_client()

    # Get current stack parameters
    try:
        resp = cfn.describe_stacks(StackName=STACK_NAME)
    except cfn.exceptions.ClientError:
        print(f"✗ Stack '{STACK_NAME}' does NOT exist. Run 'deploy' first.")
        sys.exit(1)

    current_params = resp["Stacks"][0].get("Parameters", [])

    # Update warning threshold to $0.01
    print(f"Setting warning threshold to ${test_threshold} for testing ...")
    new_params = []
    for p in current_params:
        if p["ParameterKey"] == "WarningThresholdUSD":
            new_params.append({"ParameterKey": "WarningThresholdUSD", "ParameterValue": str(test_threshold)})
        else:
            new_params.append({"ParameterKey": p["ParameterKey"], "UsePreviousValue": True})

    cfn.update_stack(
        StackName=STACK_NAME,
        UsePreviousTemplate=True,
        Parameters=new_params,
    )
    _wait_for_stack(cfn, STACK_NAME, "UPDATE_COMPLETE")

    # Wait a bit for billing metric to trigger the alarm
    print("  Waiting 60s for alarm evaluation cycle ...")
    time.sleep(60)

    # Check alarm state
    cw = _cw_client()
    alarms = cw.describe_alarms(AlarmNamePrefix="FlashBalanceAI-Billing-Warning")
    if alarms["MetricAlarms"]:
        state = alarms["MetricAlarms"][0]["StateValue"]
        print(f"  Warning alarm state: {state}")
    else:
        print("  Warning alarm not found!")

    # Restore original threshold
    print(f"Restoring warning threshold to ${original_threshold} ...")
    restore_params = []
    for p in current_params:
        if p["ParameterKey"] == "WarningThresholdUSD":
            restore_params.append({"ParameterKey": "WarningThresholdUSD", "ParameterValue": str(original_threshold)})
        else:
            restore_params.append({"ParameterKey": p["ParameterKey"], "UsePreviousValue": True})

    cfn.update_stack(
        StackName=STACK_NAME,
        UsePreviousTemplate=True,
        Parameters=restore_params,
    )
    _wait_for_stack(cfn, STACK_NAME, "UPDATE_COMPLETE")
    print("✓ Test complete — threshold restored.")


def cmd_delete(args):
    """Delete the billing-alarms stack (teardown)."""
    cfn = _cfn_client()
    print(f"Deleting stack '{STACK_NAME}' ...")
    try:
        cfn.delete_stack(StackName=STACK_NAME)
        _wait_for_stack(cfn, STACK_NAME, "DELETE_COMPLETE")
        print("✓ Stack deleted.")
    except cfn.exceptions.ClientError as e:
        if "does not exist" in str(e):
            print("  Stack already deleted or never created.")
        else:
            raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashBalanceAI Issue #14 — Billing Alarm Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # deploy
    p_deploy = sub.add_parser("deploy", help="Deploy billing alarm CloudFormation stack")
    p_deploy.add_argument("--email1", required=True, help="Team member 1 email (Devkanti)")
    p_deploy.add_argument("--email2", required=True, help="Team member 2 email (Agrima)")
    p_deploy.add_argument("--email3", default="", help="Team member 3 email (Mohar)")
    p_deploy.add_argument("--email4", default="", help="Team member 4 email")

    # verify
    sub.add_parser("verify", help="Verify alarms and SNS subscriptions")

    # test
    sub.add_parser("test", help="Test alarm with $0.01 threshold then restore")

    # delete
    sub.add_parser("delete", help="Delete billing alarm stack")

    args = parser.parse_args()

    commands = {
        "deploy": cmd_deploy,
        "verify": cmd_verify,
        "test": cmd_test,
        "delete": cmd_delete,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
