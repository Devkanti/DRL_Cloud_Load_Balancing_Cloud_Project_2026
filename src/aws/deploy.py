#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #17: EC2 ASG Deployment Script
=====================================================

Manages the backend Auto Scaling Group via CloudFormation.
Supports: deploy, status, stop (ASG -> 0), start (ASG -> 4), delete, health-check.

Usage:
    python src/aws/deploy.py deploy              # Deploy VPC + ASG stack
    python src/aws/deploy.py deploy --key-pair my-key  # Deploy with SSH key
    python src/aws/deploy.py status              # Show ASG instance status
    python src/aws/deploy.py stop                # Set ASG to 0 (cost saving)
    python src/aws/deploy.py start               # Set ASG to desired=4
    python src/aws/deploy.py health-check        # Curl /health on all instances
    python src/aws/deploy.py delete              # Delete entire stack

Prerequisites:
    - Issue #14 billing alarms deployed
    - Issue #15 IAM roles created (SSM params available)
    - pip install boto3 pyyaml requests

References:
    - ADR-001 D11 (instance config), ADR-002 (cost discipline)
    - configs/aws_config.yaml (ASG parameters)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import boto3
    import yaml
    from botocore.exceptions import ClientError
except ImportError as e:
    print(f"ERROR: Missing dependency -- {e}. Run: pip install boto3 pyyaml requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AWS_CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"
TEMPLATE_PATH = PROJECT_ROOT / "infra" / "cloudformation" / "backend-asg.yaml"
STACK_NAME = "FlashBalanceAI-Backend-ASG"
REGION = "us-east-1"
SSM_PREFIX = "/flashbalanceai"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(AWS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _cfn_client():
    return boto3.client("cloudformation", region_name=REGION)


def _ec2_client():
    return boto3.client("ec2", region_name=REGION)


def _asg_client():
    return boto3.client("autoscaling", region_name=REGION)


def _ssm_client():
    return boto3.client("ssm", region_name=REGION)


def _get_ssm_param(ssm, name: str) -> str:
    """Get a value from SSM Parameter Store."""
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception as e:
        print(f"[FAIL] Cannot read SSM param '{name}': {e}")
        print("       Run 'python src/aws/iam_setup.py create' first (Issue #15).")
        sys.exit(1)


def _wait_for_stack(cfn, stack_name: str, target_statuses: list, timeout: int = 600):
    """Poll stack until it reaches one of target_statuses or times out."""
    print(f"  Waiting for stack '{stack_name}' ...", end="", flush=True)
    start = time.time()
    last_status = ""
    while time.time() - start < timeout:
        try:
            resp = cfn.describe_stacks(StackName=stack_name)
            status = resp["Stacks"][0]["StackStatus"]
            last_status = status
        except ClientError:
            if "DELETE_COMPLETE" in target_statuses:
                print(" done.")
                return
            raise

        if status in target_statuses:
            print(f" {status}")
            return

        if "FAILED" in status or "ROLLBACK_COMPLETE" == status:
            reason = resp["Stacks"][0].get("StackStatusReason", "unknown")
            print(f"\n  [FAIL] Stack reached {status} -- {reason}")
            # Print failed resource events
            _print_stack_errors(cfn, stack_name)
            sys.exit(1)

        print(".", end="", flush=True)
        time.sleep(15)

    print(f"\n  [FAIL] Timed out after {timeout}s (last status: {last_status})")
    sys.exit(1)


def _print_stack_errors(cfn, stack_name: str):
    """Print CloudFormation stack events that contain failures."""
    try:
        events = cfn.describe_stack_events(StackName=stack_name)["StackEvents"]
        print("\n  Recent failure events:")
        for ev in events[:20]:
            status = ev.get("ResourceStatus", "")
            if "FAILED" in status:
                reason = ev.get("ResourceStatusReason", "")
                resource = ev.get("LogicalResourceId", "")
                print(f"    {resource}: {reason}")
    except Exception:
        pass


def _stack_exists(cfn, stack_name: str) -> bool:
    try:
        resp = cfn.describe_stacks(StackName=stack_name)
        status = resp["Stacks"][0]["StackStatus"]
        return status != "DELETE_COMPLETE"
    except ClientError as e:
        if "does not exist" in str(e):
            return False
        raise


def _get_stack_outputs(cfn, stack_name: str) -> dict:
    """Return stack outputs as a dict."""
    resp = cfn.describe_stacks(StackName=stack_name)
    return {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0].get("Outputs", [])}


def _get_asg_instances(asg_client, asg_name: str) -> list:
    """Return list of instance dicts from the ASG."""
    resp = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
    if not resp["AutoScalingGroups"]:
        return []
    return resp["AutoScalingGroups"][0].get("Instances", [])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_deploy(args):
    """Deploy the backend ASG CloudFormation stack."""
    cfg = _load_config()
    cfn = _cfn_client()
    ssm = _ssm_client()

    # Read EC2 instance profile ARN from SSM (created by Issue #15)
    ec2_profile_arn = _get_ssm_param(ssm, f"{SSM_PREFIX}/iam/ec2-instance-profile-arn")
    bucket_name = _get_ssm_param(ssm, f"{SSM_PREFIX}/iam/s3-bucket-name")

    template_body = TEMPLATE_PATH.read_text(encoding="utf-8")

    params = [
        {"ParameterKey": "InstanceType", "ParameterValue": cfg.get("instance_type", "t2.micro")},
        {"ParameterKey": "ASGMinSize", "ParameterValue": str(cfg.get("asg_min", 2))},
        {"ParameterKey": "ASGDesiredCapacity", "ParameterValue": str(cfg.get("asg_desired", 4))},
        {"ParameterKey": "ASGMaxSize", "ParameterValue": str(cfg.get("asg_max", 8))},
        {"ParameterKey": "ASGCooldown", "ParameterValue": str(cfg.get("asg_cooldown_seconds", 180))},
        {"ParameterKey": "EC2InstanceProfileArn", "ParameterValue": ec2_profile_arn},
        {"ParameterKey": "S3BucketName", "ParameterValue": bucket_name},
    ]

    if args.key_pair:
        params.append({"ParameterKey": "KeyPairName", "ParameterValue": args.key_pair})

    print(f"Deploying stack '{STACK_NAME}' ...")
    print(f"  Instance type : {cfg.get('instance_type', 't2.micro')}")
    print(f"  ASG           : min={cfg.get('asg_min')}, desired={cfg.get('asg_desired')}, max={cfg.get('asg_max')}")
    print(f"  EC2 Profile   : {ec2_profile_arn}")
    print(f"  S3 Bucket     : {bucket_name}")
    if args.key_pair:
        print(f"  Key Pair      : {args.key_pair}")

    common_kwargs = dict(
        StackName=STACK_NAME,
        TemplateBody=template_body,
        Parameters=params,
        Tags=[
            {"Key": "Project", "Value": "FlashBalanceAI"},
            {"Key": "Issue", "Value": "17"},
            {"Key": "Phase", "Value": "4"},
        ],
        Capabilities=["CAPABILITY_IAM"],
    )

    if _stack_exists(cfn, STACK_NAME):
        resp = cfn.describe_stacks(StackName=STACK_NAME)
        status = resp["Stacks"][0]["StackStatus"]
        if status == "ROLLBACK_COMPLETE":
            print("  Stack is in ROLLBACK_COMPLETE state. Deleting it first ...")
            cfn.delete_stack(StackName=STACK_NAME)
            _wait_for_stack(cfn, STACK_NAME, ["DELETE_COMPLETE"])
            print("  Creating new stack ...")
            cfn.create_stack(**common_kwargs)
            _wait_for_stack(cfn, STACK_NAME, ["CREATE_COMPLETE"])
        else:
            print("  Stack exists -- updating ...")
            try:
                cfn.update_stack(**common_kwargs)
                _wait_for_stack(cfn, STACK_NAME, ["UPDATE_COMPLETE"])
            except ClientError as e:
                if "No updates" in str(e):
                    print("  Stack is already up-to-date.")
                else:
                    raise
    else:
        print("  Creating new stack ...")
        cfn.create_stack(**common_kwargs)
        _wait_for_stack(cfn, STACK_NAME, ["CREATE_COMPLETE"])

    # Print outputs
    outputs = _get_stack_outputs(cfn, STACK_NAME)
    print("\n--- Stack Outputs ---")
    for key, val in sorted(outputs.items()):
        print(f"  {key}: {val}")

    # Store VPC/SG IDs in SSM for other issues (ALB, etc.)
    ssm_mappings = {
        "VPCID": f"{SSM_PREFIX}/vpc/vpc-id",
        "PublicSubnet1ID": f"{SSM_PREFIX}/vpc/public-subnet-1",
        "PublicSubnet2ID": f"{SSM_PREFIX}/vpc/public-subnet-2",
        "ALBSecurityGroupID": f"{SSM_PREFIX}/vpc/alb-sg-id",
        "BackendSecurityGroupID": f"{SSM_PREFIX}/vpc/backend-sg-id",
        "LaunchTemplateID": f"{SSM_PREFIX}/asg/launch-template-id",
        "ASGName": f"{SSM_PREFIX}/asg/asg-name",
    }
    print("\n[SSM] Storing stack outputs in Parameter Store ...")
    for output_key, ssm_path in ssm_mappings.items():
        val = outputs.get(output_key, "")
        if val:
            try:
                ssm.put_parameter(
                    Name=ssm_path,
                    Value=val,
                    Type="String",
                    Tags=[
                        {"Key": "Project", "Value": "FlashBalanceAI"},
                        {"Key": "Issue", "Value": "17"},
                    ]
                )
            except ssm.exceptions.ParameterAlreadyExists:
                ssm.put_parameter(
                    Name=ssm_path,
                    Value=val,
                    Type="String",
                    Overwrite=True,
                )
            print(f"  [OK] {ssm_path} = {val}")

    print(f"\n[DONE] Stack deployed. Run 'python src/aws/deploy.py status' to check instances.")
    print("[IMPORTANT] Remember to run 'python src/aws/deploy.py stop' after experiments to save costs!")


def cmd_status(args):
    """Show ASG instance status."""
    cfg = _load_config()
    asg = _asg_client()
    ec2 = _ec2_client()
    asg_name = cfg["asg_name"]

    resp = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
    if not resp["AutoScalingGroups"]:
        print(f"[FAIL] ASG '{asg_name}' not found. Run 'deploy' first.")
        return

    group = resp["AutoScalingGroups"][0]
    print(f"ASG: {asg_name}")
    print(f"  Status       : {group.get('Status', 'active')}")
    print(f"  Min/Desired/Max : {group['MinSize']}/{group['DesiredCapacity']}/{group['MaxSize']}")
    print(f"  Instances    : {len(group.get('Instances', []))}")
    print()

    instances = group.get("Instances", [])
    if not instances:
        print("  No instances running.")
        return

    # Get detailed instance info
    instance_ids = [i["InstanceId"] for i in instances]
    ec2_resp = ec2.describe_instances(InstanceIds=instance_ids)

    print(f"  {'Instance ID':<22} {'State':<12} {'AZ':<15} {'Public IP':<16} {'Health'}")
    print(f"  {'-'*22} {'-'*12} {'-'*15} {'-'*16} {'-'*10}")

    for reservation in ec2_resp["Reservations"]:
        for inst in reservation["Instances"]:
            iid = inst["InstanceId"]
            state = inst["State"]["Name"]
            az = inst["Placement"]["AvailabilityZone"]
            pub_ip = inst.get("PublicIpAddress", "N/A")

            # Get ASG health
            asg_health = "N/A"
            for ai in instances:
                if ai["InstanceId"] == iid:
                    asg_health = ai.get("HealthStatus", "N/A")
                    break

            print(f"  {iid:<22} {state:<12} {az:<15} {pub_ip:<16} {asg_health}")


def cmd_stop(args):
    """Stop all ASG instances by setting desired/min to 0 (cost saving)."""
    cfg = _load_config()
    asg = _asg_client()
    asg_name = cfg["asg_name"]

    print(f"Stopping ASG '{asg_name}' (setting min=0, desired=0) ...")
    try:
        asg.update_auto_scaling_group(
            AutoScalingGroupName=asg_name,
            MinSize=0,
            DesiredCapacity=0,
        )
        print("[OK] ASG updated. Instances will terminate within ~2 minutes.")
        print("[TIP] Run 'python src/aws/deploy.py status' to monitor.")
    except ClientError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)


def cmd_start(args):
    """Start ASG instances by restoring desired capacity."""
    cfg = _load_config()
    asg = _asg_client()
    asg_name = cfg["asg_name"]

    desired = args.desired if args.desired else cfg.get("asg_desired", 4)
    min_size = cfg.get("asg_min", 2)

    print(f"Starting ASG '{asg_name}' (min={min_size}, desired={desired}) ...")
    try:
        asg.update_auto_scaling_group(
            AutoScalingGroupName=asg_name,
            MinSize=min_size,
            DesiredCapacity=desired,
        )
        print(f"[OK] ASG updated. {desired} instances will launch within ~2 minutes.")
        print("[TIP] Run 'python src/aws/deploy.py status' to monitor.")
        print("[TIP] Run 'python src/aws/deploy.py health-check' once instances are 'InService'.")
    except ClientError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)


def cmd_health_check(args):
    """Check /health endpoint on all running backend instances."""
    try:
        import requests
    except ImportError:
        print("[FAIL] 'requests' not installed. Run: pip install requests")
        sys.exit(1)

    cfg = _load_config()
    asg = _asg_client()
    ec2 = _ec2_client()
    asg_name = cfg["asg_name"]

    instances = _get_asg_instances(asg, asg_name)
    if not instances:
        print(f"[FAIL] No instances in ASG '{asg_name}'.")
        return

    instance_ids = [i["InstanceId"] for i in instances if i.get("LifecycleState") == "InService"]
    if not instance_ids:
        print("[WARN] No instances in 'InService' state yet. Wait and retry.")
        return

    ec2_resp = ec2.describe_instances(InstanceIds=instance_ids)

    all_healthy = True
    print(f"Health check on {len(instance_ids)} instances:\n")

    for reservation in ec2_resp["Reservations"]:
        for inst in reservation["Instances"]:
            iid = inst["InstanceId"]
            pub_ip = inst.get("PublicIpAddress")

            if not pub_ip:
                print(f"  {iid}: [SKIP] No public IP")
                continue

            url = f"http://{pub_ip}:5000/health"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"  {iid}: [OK] {data}")
                else:
                    print(f"  {iid}: [FAIL] HTTP {resp.status_code}")
                    all_healthy = False
            except requests.exceptions.ConnectionError:
                print(f"  {iid}: [FAIL] Connection refused (app may still be starting)")
                all_healthy = False
            except requests.exceptions.Timeout:
                print(f"  {iid}: [FAIL] Timeout")
                all_healthy = False
            except Exception as e:
                print(f"  {iid}: [FAIL] {e}")
                all_healthy = False

    print()
    if all_healthy:
        print("[PASS] All instances healthy.")
    else:
        print("[WARN] Some instances failed health check.")
        print("       If just deployed, wait 2-3 minutes for UserData to finish.")


def cmd_delete(args):
    """Delete the entire backend ASG CloudFormation stack."""
    cfn = _cfn_client()
    ssm = _ssm_client()

    if not _stack_exists(cfn, STACK_NAME):
        print(f"Stack '{STACK_NAME}' does not exist.")
        return

    print(f"Deleting stack '{STACK_NAME}' ...")
    print("  This will terminate all backend instances and remove the VPC.")

    cfn.delete_stack(StackName=STACK_NAME)
    _wait_for_stack(cfn, STACK_NAME, ["DELETE_COMPLETE"])

    # Clean up SSM parameters
    print("\nCleaning SSM parameters ...")
    ssm_keys = [
        f"{SSM_PREFIX}/vpc/vpc-id",
        f"{SSM_PREFIX}/vpc/public-subnet-1",
        f"{SSM_PREFIX}/vpc/public-subnet-2",
        f"{SSM_PREFIX}/vpc/alb-sg-id",
        f"{SSM_PREFIX}/vpc/backend-sg-id",
        f"{SSM_PREFIX}/asg/launch-template-id",
        f"{SSM_PREFIX}/asg/asg-name",
    ]
    for key in ssm_keys:
        try:
            ssm.delete_parameter(Name=key)
            print(f"  [OK] Deleted: {key}")
        except ssm.exceptions.ParameterNotFound:
            pass

    print("\n[DONE] Stack deleted. All instances terminated.")


def stop_all_instances():
    """
    Convenience function for teardown scripts (Issue #46).
    Sets ASG to 0 instances and waits for termination.
    """
    cfg = _load_config()
    asg = _asg_client()
    asg_name = cfg["asg_name"]

    print(f"[TEARDOWN] Stopping all instances in '{asg_name}' ...")
    try:
        asg.update_auto_scaling_group(
            AutoScalingGroupName=asg_name,
            MinSize=0,
            DesiredCapacity=0,
        )
    except ClientError as e:
        if "AutoScalingGroup name not found" in str(e):
            print(f"  [SKIP] ASG '{asg_name}' not found -- already deleted?")
            return
        raise

    # Wait for instances to terminate
    print("  Waiting for instances to terminate ...", end="", flush=True)
    for _ in range(30):
        instances = _get_asg_instances(asg, asg_name)
        if not instances:
            print(" done.")
            return
        print(".", end="", flush=True)
        time.sleep(10)
    print(" (some instances may still be terminating)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashBalanceAI Issue #17 -- EC2 ASG Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # deploy
    p_deploy = sub.add_parser("deploy", help="Deploy VPC + ASG CloudFormation stack")
    p_deploy.add_argument("--key-pair", default="", help="EC2 key pair name for SSH access")

    # status
    sub.add_parser("status", help="Show ASG instance status")

    # stop
    sub.add_parser("stop", help="Set ASG to 0 instances (cost saving)")

    # start
    p_start = sub.add_parser("start", help="Restore ASG to desired capacity")
    p_start.add_argument("--desired", type=int, default=0,
                         help="Override desired capacity (default: from config)")

    # health-check
    sub.add_parser("health-check", help="Check /health on all running instances")

    # delete
    sub.add_parser("delete", help="Delete entire stack (VPC, ASG, instances)")

    args = parser.parse_args()

    commands = {
        "deploy": cmd_deploy,
        "status": cmd_status,
        "stop": cmd_stop,
        "start": cmd_start,
        "health-check": cmd_health_check,
        "delete": cmd_delete,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
