#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #17/#18: EC2 ASG & ALB Deployment Script
===============================================================

Manages the backend Auto Scaling Group (CloudFormation) and
Application Load Balancer (boto3 -- separate lifecycle per ADR-002).

ASG commands:
    python src/aws/deploy.py deploy              # Deploy VPC + ASG stack
    python src/aws/deploy.py deploy --key-pair my-key  # Deploy with SSH key
    python src/aws/deploy.py status              # Show ASG instance status
    python src/aws/deploy.py stop                # Set ASG to 0 (cost saving)
    python src/aws/deploy.py start               # Set ASG to desired=4
    python src/aws/deploy.py health-check        # Curl /health on all instances
    python src/aws/deploy.py delete              # Delete entire stack

ALB commands (Issue #18 -- separate lifecycle, delete between sessions):
    python src/aws/deploy.py create-alb          # Create ALB + target group
    python src/aws/deploy.py alb-status          # Show ALB + target health
    python src/aws/deploy.py delete-alb          # DELETE ALB (required after each session!)

Full teardown:
    python src/aws/deploy.py teardown            # delete-alb + stop ASG

Prerequisites:
    - Issue #14 billing alarms deployed
    - Issue #15 IAM roles created (SSM params available)
    - Issue #16 S3 bucket created
    - Issue #17 ASG deployed (for ALB commands)
    - pip install boto3 pyyaml requests

References:
    - ADR-001 D11 (instance config), ADR-002 (cost discipline, ALB delete-per-session)
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


def _elbv2_client():
    return boto3.client("elbv2", region_name=REGION)


def _get_ssm_param(ssm, name: str) -> str:
    """Get a value from SSM Parameter Store."""
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception as e:
        print(f"[FAIL] Cannot read SSM param '{name}': {e}")
        print("       Run 'python src/aws/iam_setup.py create' first (Issue #15).")
        sys.exit(1)


def _get_ssm_param_optional(ssm, name: str) -> str | None:
    """Get SSM param, return None if not found (non-fatal)."""
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception:
        return None


def _store_ssm_param(ssm, name: str, value: str, description: str = ""):
    """Store a value in SSM Parameter Store (creates or overwrites)."""
    try:
        ssm.put_parameter(
            Name=name,
            Value=value,
            Description=description,
            Type="String",
            Tags=[
                {"Key": "Project", "Value": "FlashBalanceAI"},
                {"Key": "Issue", "Value": "18"},
            ],
        )
    except ssm.exceptions.ParameterAlreadyExists:
        ssm.put_parameter(
            Name=name,
            Value=value,
            Description=description,
            Type="String",
            Overwrite=True,
        )


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


# ---------------------------------------------------------------------------
# ALB Commands (Issue #18 -- separate lifecycle, boto3-managed)
# ---------------------------------------------------------------------------

ALB_NAME = "FlashBalanceAI-ALB"
TG_NAME = "FlashBalanceAI-TG"


def create_alb(vpc_id: str, subnet_ids: list, alb_sg_id: str,
               asg_name: str) -> tuple[str, str, str]:
    """
    Create ALB + target group + register ASG instances.
    Returns (alb_arn, alb_dns, tg_arn).
    Callable from other modules (e.g. teardown scripts).
    """
    elbv2 = _elbv2_client()
    ec2 = _ec2_client()
    asg = _asg_client()

    # ---- 1. Create Target Group ----
    print("[1/4] Creating target group ...")
    tg_resp = elbv2.create_target_group(
        Name=TG_NAME,
        Protocol="HTTP",
        Port=5000,
        VpcId=vpc_id,
        HealthCheckProtocol="HTTP",
        HealthCheckPort="5000",
        HealthCheckPath="/health",
        HealthCheckIntervalSeconds=30,
        HealthCheckTimeoutSeconds=5,
        HealthyThresholdCount=2,
        UnhealthyThresholdCount=3,
        Matcher={"HttpCode": "200"},
        TargetType="instance",
        Tags=[
            {"Key": "Project", "Value": "FlashBalanceAI"},
            {"Key": "Issue", "Value": "18"},
            {"Key": "Phase", "Value": "4"},
        ],
    )
    tg_arn = tg_resp["TargetGroups"][0]["TargetGroupArn"]
    print(f"  [OK] Target group: {TG_NAME}")
    print(f"       ARN: {tg_arn}")

    # ---- 2. Register ASG instances in target group ----
    print("\n[2/4] Registering instances ...")
    instances = _get_asg_instances(asg, asg_name)
    in_service = [i for i in instances if i.get("LifecycleState") == "InService"]

    if not in_service:
        print("  [WARN] No InService instances found. Register later with alb-status.")
    else:
        targets = [{"Id": i["InstanceId"], "Port": 5000} for i in in_service]
        elbv2.register_targets(TargetGroupArn=tg_arn, Targets=targets)
        for t in targets:
            print(f"  [OK] Registered: {t['Id']}:5000")

    # ---- 3. Create ALB ----
    print("\n[3/4] Creating Application Load Balancer ...")
    alb_resp = elbv2.create_load_balancer(
        Name=ALB_NAME,
        Subnets=subnet_ids,
        SecurityGroups=[alb_sg_id],
        Scheme="internet-facing",
        Type="application",
        IpAddressType="ipv4",
        Tags=[
            {"Key": "Project", "Value": "FlashBalanceAI"},
            {"Key": "Issue", "Value": "18"},
            {"Key": "Phase", "Value": "4"},
        ],
    )
    alb = alb_resp["LoadBalancers"][0]
    alb_arn = alb["LoadBalancerArn"]
    alb_dns = alb["DNSName"]
    print(f"  [OK] ALB created: {ALB_NAME}")
    print(f"       DNS: {alb_dns}")
    print(f"       ARN: {alb_arn}")

    # Wait for ALB to become active
    print("  Waiting for ALB to become active ...", end="", flush=True)
    waiter = elbv2.get_waiter("load_balancer_available")
    try:
        waiter.wait(
            LoadBalancerArns=[alb_arn],
            WaiterConfig={"Delay": 15, "MaxAttempts": 20},
        )
        print(" active.")
    except Exception as e:
        print(f"\n  [WARN] Waiter error: {e}")
        print("  ALB may still be provisioning. Check with 'alb-status'.")

    # ---- 4. Create HTTP listener ----
    print("\n[4/4] Creating HTTP listener (port 80 -> target group) ...")
    elbv2.create_listener(
        LoadBalancerArn=alb_arn,
        Protocol="HTTP",
        Port=80,
        DefaultActions=[
            {
                "Type": "forward",
                "TargetGroupArn": tg_arn,
            }
        ],
    )
    print("  [OK] Listener: HTTP:80 -> target group:5000")

    return alb_arn, alb_dns, tg_arn


def delete_alb() -> bool:
    """
    Delete ALB, listener, and target group.
    Callable from other modules (e.g. teardown scripts, Issue #46).
    Returns True if deleted, False if nothing to delete.
    """
    elbv2 = _elbv2_client()
    ssm = _ssm_client()

    # Find ALB by name
    alb_arn = None
    try:
        resp = elbv2.describe_load_balancers(Names=[ALB_NAME])
        if resp["LoadBalancers"]:
            alb_arn = resp["LoadBalancers"][0]["LoadBalancerArn"]
    except ClientError as e:
        if "LoadBalancerNotFound" in str(e):
            pass
        else:
            raise

    if not alb_arn:
        print(f"[SKIP] ALB '{ALB_NAME}' not found -- already deleted?")
        return False

    # Delete listeners first
    print(f"Deleting ALB '{ALB_NAME}' ...")
    listeners = elbv2.describe_listeners(LoadBalancerArn=alb_arn)
    for listener in listeners.get("Listeners", []):
        elbv2.delete_listener(ListenerArn=listener["ListenerArn"])
        print(f"  [OK] Deleted listener: {listener['Protocol']}:{listener['Port']}")

    # Delete ALB
    elbv2.delete_load_balancer(LoadBalancerArn=alb_arn)
    print(f"  [OK] ALB deletion initiated: {ALB_NAME}")

    # Wait for ALB to be fully deleted
    print("  Waiting for ALB to be fully deleted ...", end="", flush=True)
    for _ in range(40):
        try:
            resp = elbv2.describe_load_balancers(Names=[ALB_NAME])
            if not resp["LoadBalancers"]:
                break
            state = resp["LoadBalancers"][0].get("State", {}).get("Code", "")
            if state == "active":
                # Still deleting
                pass
        except ClientError as e:
            if "LoadBalancerNotFound" in str(e):
                break
            raise
        print(".", end="", flush=True)
        time.sleep(10)
    print(" done.")

    # Delete target group (must wait until ALB is gone)
    try:
        tg_resp = elbv2.describe_target_groups(Names=[TG_NAME])
        if tg_resp["TargetGroups"]:
            tg_arn = tg_resp["TargetGroups"][0]["TargetGroupArn"]
            elbv2.delete_target_group(TargetGroupArn=tg_arn)
            print(f"  [OK] Deleted target group: {TG_NAME}")
    except ClientError as e:
        if "TargetGroupNotFound" in str(e):
            pass
        else:
            raise

    # Clean SSM params
    for param in [
        f"{SSM_PREFIX}/alb/alb-arn",
        f"{SSM_PREFIX}/alb/alb-dns",
        f"{SSM_PREFIX}/alb/target-group-arn",
    ]:
        try:
            ssm.delete_parameter(Name=param)
        except ssm.exceptions.ParameterNotFound:
            pass

    print(f"\n[DONE] ALB deleted. No more ALB charges accumulating.")
    return True


def cmd_create_alb(args):
    """Create ALB + target group + listener (Issue #18)."""
    ssm = _ssm_client()
    cfg = _load_config()

    # Read VPC/subnet/SG from SSM (stored by Issue #17 deploy)
    vpc_id = _get_ssm_param_optional(ssm, f"{SSM_PREFIX}/vpc/vpc-id")
    subnet1 = _get_ssm_param_optional(ssm, f"{SSM_PREFIX}/vpc/public-subnet-1")
    subnet2 = _get_ssm_param_optional(ssm, f"{SSM_PREFIX}/vpc/public-subnet-2")
    alb_sg = _get_ssm_param_optional(ssm, f"{SSM_PREFIX}/vpc/alb-sg-id")

    if not all([vpc_id, subnet1, subnet2, alb_sg]):
        print("[FAIL] VPC/subnet/SG SSM parameters not found.")
        print("       Deploy the ASG stack first: python src/aws/deploy.py deploy")
        sys.exit(1)

    # Check if ALB already exists
    elbv2 = _elbv2_client()
    try:
        resp = elbv2.describe_load_balancers(Names=[ALB_NAME])
        if resp["LoadBalancers"]:
            alb = resp["LoadBalancers"][0]
            print(f"[SKIP] ALB already exists:")
            print(f"  DNS: {alb['DNSName']}")
            print(f"  State: {alb['State']['Code']}")
            print("  Run 'delete-alb' first if you want to recreate it.")
            return
    except ClientError as e:
        if "LoadBalancerNotFound" not in str(e):
            raise

    asg_name = cfg["asg_name"]

    print("Issue #18: ALB Setup (Flash-Session Only)")
    print(f"  VPC       : {vpc_id}")
    print(f"  Subnets   : {subnet1}, {subnet2}")
    print(f"  ALB SG    : {alb_sg}")
    print(f"  ASG       : {asg_name}")
    print()

    alb_arn, alb_dns, tg_arn = create_alb(
        vpc_id=vpc_id,
        subnet_ids=[subnet1, subnet2],
        alb_sg_id=alb_sg,
        asg_name=asg_name,
    )

    # Store in SSM
    print("\n[SSM] Storing ALB info ...")
    _store_ssm_param(ssm, f"{SSM_PREFIX}/alb/alb-arn", alb_arn, "FlashBalanceAI ALB ARN")
    print(f"  [OK] /flashbalanceai/alb/alb-arn")
    _store_ssm_param(ssm, f"{SSM_PREFIX}/alb/alb-dns", alb_dns, "FlashBalanceAI ALB DNS")
    print(f"  [OK] /flashbalanceai/alb/alb-dns = {alb_dns}")
    _store_ssm_param(ssm, f"{SSM_PREFIX}/alb/target-group-arn", tg_arn, "FlashBalanceAI target group ARN")
    print(f"  [OK] /flashbalanceai/alb/target-group-arn")

    print(f"\n[DONE] ALB is live!")
    print(f"  Test: curl http://{alb_dns}/health")
    print(f"\n[IMPORTANT] Delete ALB after each session to avoid charges:")
    print(f"  python src/aws/deploy.py delete-alb")


def cmd_alb_status(args):
    """Show ALB status and target health."""
    elbv2 = _elbv2_client()

    # Find ALB
    try:
        resp = elbv2.describe_load_balancers(Names=[ALB_NAME])
        if not resp["LoadBalancers"]:
            print(f"[FAIL] ALB '{ALB_NAME}' not found.")
            return
    except ClientError as e:
        if "LoadBalancerNotFound" in str(e):
            print(f"ALB '{ALB_NAME}' does not exist. Nothing running (no ALB charges).")
            return
        raise

    alb = resp["LoadBalancers"][0]
    print(f"ALB: {ALB_NAME}")
    print(f"  DNS   : {alb['DNSName']}")
    print(f"  State : {alb['State']['Code']}")
    print(f"  Scheme: {alb['Scheme']}")
    print(f"  ARN   : {alb['LoadBalancerArn']}")

    # Find target group
    try:
        tg_resp = elbv2.describe_target_groups(Names=[TG_NAME])
        if tg_resp["TargetGroups"]:
            tg = tg_resp["TargetGroups"][0]
            tg_arn = tg["TargetGroupArn"]
            print(f"\nTarget Group: {TG_NAME}")
            print(f"  Port     : {tg['Port']}")
            print(f"  Protocol : {tg['Protocol']}")
            print(f"  Health   : {tg['HealthCheckPath']} (interval: {tg['HealthCheckIntervalSeconds']}s)")

            # Target health
            health = elbv2.describe_target_health(TargetGroupArn=tg_arn)
            targets = health.get("TargetHealthDescriptions", [])
            if targets:
                print(f"\n  Targets ({len(targets)}):")
                all_healthy = True
                for t in targets:
                    tid = t["Target"]["Id"]
                    port = t["Target"]["Port"]
                    state = t["TargetHealth"]["State"]
                    reason = t["TargetHealth"].get("Reason", "")
                    desc = t["TargetHealth"].get("Description", "")
                    status_str = f"[OK]" if state == "healthy" else f"[{state.upper()}]"
                    extra = f" ({reason}: {desc})" if reason else ""
                    print(f"    {tid}:{port} {status_str}{extra}")
                    if state != "healthy":
                        all_healthy = False

                print()
                if all_healthy:
                    print("[PASS] All targets healthy.")
                else:
                    print("[WARN] Some targets not healthy. They may still be initializing.")
            else:
                print("\n  No targets registered.")
    except ClientError as e:
        if "TargetGroupNotFound" in str(e):
            print(f"\n[WARN] Target group '{TG_NAME}' not found.")
        else:
            raise


def cmd_delete_alb(args):
    """Delete ALB -- REQUIRED after each experiment session (ADR-002)."""
    delete_alb()


def cmd_teardown(args):
    """Full teardown: delete ALB + stop all ASG instances."""
    print("=" * 60)
    print("FlashBalanceAI Full Teardown")
    print("=" * 60)

    # 1. Delete ALB (most expensive -- do first)
    print("\n--- Step 1: Delete ALB ---")
    delete_alb()

    # 2. Stop ASG instances
    print("\n--- Step 2: Stop ASG instances ---")
    stop_all_instances()

    print("\n" + "=" * 60)
    print("[DONE] Teardown complete. No more charges accumulating.")
    print("=" * 60)


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
        description="FlashBalanceAI Issue #17/#18 -- EC2 ASG & ALB Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ASG commands (Issue #17) ---
    p_deploy = sub.add_parser("deploy", help="Deploy VPC + ASG CloudFormation stack")
    p_deploy.add_argument("--key-pair", default="", help="EC2 key pair name for SSH access")

    sub.add_parser("status", help="Show ASG instance status")
    sub.add_parser("stop", help="Set ASG to 0 instances (cost saving)")

    p_start = sub.add_parser("start", help="Restore ASG to desired capacity")
    p_start.add_argument("--desired", type=int, default=0,
                         help="Override desired capacity (default: from config)")

    sub.add_parser("health-check", help="Check /health on all running instances")
    sub.add_parser("delete", help="Delete entire stack (VPC, ASG, instances)")

    # --- ALB commands (Issue #18) ---
    sub.add_parser("create-alb", help="Create ALB + target group + listener")
    sub.add_parser("alb-status", help="Show ALB and target health status")
    sub.add_parser("delete-alb", help="DELETE ALB (required after each session!)")

    # --- Full teardown ---
    sub.add_parser("teardown", help="Full teardown: delete ALB + stop ASG")

    args = parser.parse_args()

    commands = {
        "deploy": cmd_deploy,
        "status": cmd_status,
        "stop": cmd_stop,
        "start": cmd_start,
        "health-check": cmd_health_check,
        "delete": cmd_delete,
        "create-alb": cmd_create_alb,
        "alb-status": cmd_alb_status,
        "delete-alb": cmd_delete_alb,
        "teardown": cmd_teardown,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
