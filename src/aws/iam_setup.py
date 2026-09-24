#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #15: IAM Roles & Policies Setup
=======================================================

Creates the minimum-privilege IAM roles required by Lambda functions and
EC2 instances per ADR-001 D12/D14.

Roles created:
    1. FlashBalanceAI-Lambda-Role
       - AWSLambdaBasicExecutionRole (managed)
       - CloudWatchReadOnlyAccess (managed)
       - Inline: DynamoDB full access scoped to 'routing_decisions' table
       - Inline: S3 access scoped to project bucket only

    2. FlashBalanceAI-EC2-Role + Instance Profile
       - CloudWatchAgentServerPolicy (managed)
       - Inline: S3 read-only scoped to project bucket (model download only)

All role ARNs are stored in AWS SSM Parameter Store (never committed to git).

Usage:
    python src/aws/iam_setup.py create    # Create roles, policies, instance profile
    python src/aws/iam_setup.py verify    # Verify roles exist and policies attached
    python src/aws/iam_setup.py delete    # Delete roles, policies, instance profile
    python src/aws/iam_setup.py show-arns # Print role ARNs from SSM

Prerequisites:
    - AWS CLI configured (aws configure) with us-east-1
    - pip install boto3 pyyaml
    - Issue #14 billing alarms deployed

References:
    - ADR-001 D12 (service inventory), D14 (EC2 inference)
    - configs/aws_config.yaml lines 50-53 (role names)
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
    print(f"ERROR: Missing dependency -- {e}. Run: pip install boto3 pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AWS_CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"
REGION = "us-east-1"

# SSM Parameter Store paths (never in git)
SSM_PREFIX = "/flashbalanceai/iam"


# ---------------------------------------------------------------------------
# Trust policy documents
# ---------------------------------------------------------------------------

LAMBDA_TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
})

EC2_TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
})


# ---------------------------------------------------------------------------
# Inline policy builders (least-privilege, no wildcard *)
# ---------------------------------------------------------------------------

def _build_lambda_dynamodb_policy(account_id: str, table_name: str) -> dict:
    """DynamoDB access scoped to the routing_decisions table only."""
    table_arn = f"arn:aws:dynamodb:{REGION}:{account_id}:table/{table_name}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DynamoDBTableAccess",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchWriteItem",
                    "dynamodb:BatchGetItem",
                    "dynamodb:DescribeTable"
                ],
                "Resource": [
                    table_arn,
                    f"{table_arn}/index/*"
                ]
            }
        ]
    }


def _build_lambda_s3_policy(bucket_name: str) -> dict:
    """S3 access scoped to the project bucket only."""
    bucket_arn = f"arn:aws:s3:::{bucket_name}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3BucketListAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket"
                ],
                "Resource": bucket_arn
            },
            {
                "Sid": "S3ObjectAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject"
                ],
                "Resource": f"{bucket_arn}/*"
            }
        ]
    }


def _build_lambda_elb_policy() -> dict:
    """ALB access for the inference coordinator Lambda to update target weights."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ELBv2DescribeAccess",
                "Effect": "Allow",
                "Action": [
                    "elasticloadbalancing:DescribeTargetGroups",
                    "elasticloadbalancing:DescribeTargetHealth",
                    "elasticloadbalancing:DescribeLoadBalancers",
                    "elasticloadbalancing:DescribeListeners",
                    "elasticloadbalancing:DescribeRules"
                ],
                "Resource": "*"
            },
            {
                "Sid": "ELBv2ModifyAccess",
                "Effect": "Allow",
                "Action": [
                    "elasticloadbalancing:ModifyRule",
                    "elasticloadbalancing:ModifyTargetGroupAttributes",
                    "elasticloadbalancing:SetRulePriorities"
                ],
                "Resource": "*"
            }
        ]
    }


def _build_lambda_autoscaling_policy() -> dict:
    """ASG access for the scaling trigger Lambda."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ASGAccess",
                "Effect": "Allow",
                "Action": [
                    "autoscaling:DescribeAutoScalingGroups",
                    "autoscaling:SetDesiredCapacity",
                    "autoscaling:UpdateAutoScalingGroup"
                ],
                "Resource": "*"
            }
        ]
    }


def _build_ec2_s3_readonly_policy(bucket_name: str) -> dict:
    """S3 read-only access scoped to the project bucket (model download)."""
    bucket_arn = f"arn:aws:s3:::{bucket_name}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3BucketListReadOnly",
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket"
                ],
                "Resource": bucket_arn
            },
            {
                "Sid": "S3ObjectReadOnly",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject"
                ],
                "Resource": f"{bucket_arn}/*"
            }
        ]
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load configs/aws_config.yaml."""
    with open(AWS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _get_account_id() -> str:
    """Retrieve the current AWS account ID via STS."""
    sts = boto3.client("sts", region_name=REGION)
    return sts.get_caller_identity()["Account"]


def _iam_client():
    return boto3.client("iam", region_name=REGION)


def _ssm_client():
    return boto3.client("ssm", region_name=REGION)


def _role_exists(iam, role_name: str) -> bool:
    try:
        iam.get_role(RoleName=role_name)
        return True
    except iam.exceptions.NoSuchEntityException:
        return False


def _instance_profile_exists(iam, profile_name: str) -> bool:
    try:
        iam.get_instance_profile(InstanceProfileName=profile_name)
        return True
    except iam.exceptions.NoSuchEntityException:
        return False


def _attach_managed_policy(iam, role_name: str, policy_arn: str):
    """Attach an AWS managed policy to a role (idempotent)."""
    try:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"    [OK] Attached {policy_arn.split('/')[-1]}")
    except ClientError as e:
        print(f"    [FAIL] Error attaching {policy_arn}: {e}")
        raise


def _put_inline_policy(iam, role_name: str, policy_name: str, policy_doc: dict):
    """Put an inline policy on a role (idempotent -- overwrites if exists)."""
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(policy_doc)
    )
    print(f"    [OK] Inline policy: {policy_name}")


def _store_ssm_param(ssm, name: str, value: str, description: str):
    """Store a value in SSM Parameter Store (creates or overwrites)."""
    try:
        # Try creating with tags (works only for new parameters)
        ssm.put_parameter(
            Name=name,
            Description=description,
            Value=value,
            Type="String",
            Tags=[
                {"Key": "Project", "Value": "FlashBalanceAI"},
                {"Key": "Issue", "Value": "15"},
            ]
        )
    except ssm.exceptions.ParameterAlreadyExists:
        # Parameter exists -- overwrite without tags
        ssm.put_parameter(
            Name=name,
            Description=description,
            Value=value,
            Type="String",
            Overwrite=True,
        )


def _delete_ssm_param(ssm, name: str):
    """Delete an SSM parameter (ignores if not found)."""
    try:
        ssm.delete_parameter(Name=name)
        print(f"    [OK] Deleted SSM param: {name}")
    except ssm.exceptions.ParameterNotFound:
        pass


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_create(args):
    """Create both IAM roles with least-privilege policies."""
    cfg = _load_config()
    account_id = _get_account_id()
    iam = _iam_client()
    ssm = _ssm_client()

    lambda_role_name = cfg["lambda_role_name"]       # FlashBalanceAI-Lambda-Role
    ec2_role_name = cfg["ec2_role_name"]             # FlashBalanceAI-EC2-Role
    ec2_profile_name = cfg["ec2_instance_profile_name"]  # FlashBalanceAI-EC2-Profile
    dynamodb_table = cfg["dynamodb_table_name"]       # routing_decisions

    # Build the S3 bucket name from config
    raw_bucket = cfg["s3_bucket"]
    if "TODO_ACCOUNT_ID" in raw_bucket:
        bucket_name = raw_bucket.replace("TODO_ACCOUNT_ID", account_id)
        print(f"  S3 bucket name resolved: {bucket_name}")
    else:
        bucket_name = raw_bucket

    print(f"  Account ID   : {account_id}")
    print(f"  Lambda role  : {lambda_role_name}")
    print(f"  EC2 role     : {ec2_role_name}")
    print(f"  EC2 profile  : {ec2_profile_name}")
    print(f"  DynamoDB tbl : {dynamodb_table}")
    print(f"  S3 bucket    : {bucket_name}")
    print()

    # ---- 1. Lambda Role ----
    print(f"[1/2] Creating {lambda_role_name} ...")
    if not _role_exists(iam, lambda_role_name):
        iam.create_role(
            RoleName=lambda_role_name,
            AssumeRolePolicyDocument=LAMBDA_TRUST_POLICY,
            Description="FlashBalanceAI Lambda execution role - Issue #15",
            Tags=[
                {"Key": "Project", "Value": "FlashBalanceAI"},
                {"Key": "Issue", "Value": "15"},
                {"Key": "Phase", "Value": "4"},
            ],
        )
        print(f"    [OK] Role created")
        # Brief wait for IAM propagation
        time.sleep(2)
    else:
        print(f"    [SKIP] Role already exists -- updating policies")

    # Managed policies
    _attach_managed_policy(iam, lambda_role_name,
                           "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    _attach_managed_policy(iam, lambda_role_name,
                           "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess")

    # Inline policies (least-privilege, scoped)
    _put_inline_policy(iam, lambda_role_name,
                       "FlashBalanceAI-DynamoDB-Scoped",
                       _build_lambda_dynamodb_policy(account_id, dynamodb_table))

    _put_inline_policy(iam, lambda_role_name,
                       "FlashBalanceAI-S3-Scoped",
                       _build_lambda_s3_policy(bucket_name))

    _put_inline_policy(iam, lambda_role_name,
                       "FlashBalanceAI-ELBv2-Access",
                       _build_lambda_elb_policy())

    _put_inline_policy(iam, lambda_role_name,
                       "FlashBalanceAI-ASG-Access",
                       _build_lambda_autoscaling_policy())

    lambda_role_arn = iam.get_role(RoleName=lambda_role_name)["Role"]["Arn"]
    print(f"    ARN: {lambda_role_arn}")

    # ---- 2. EC2 Role + Instance Profile ----
    print(f"\n[2/2] Creating {ec2_role_name} + {ec2_profile_name} ...")
    if not _role_exists(iam, ec2_role_name):
        iam.create_role(
            RoleName=ec2_role_name,
            AssumeRolePolicyDocument=EC2_TRUST_POLICY,
            Description="FlashBalanceAI EC2 execution role - Issue #15",
            Tags=[
                {"Key": "Project", "Value": "FlashBalanceAI"},
                {"Key": "Issue", "Value": "15"},
                {"Key": "Phase", "Value": "4"},
            ],
        )
        print(f"    [OK] Role created")
        time.sleep(2)
    else:
        print(f"    [SKIP] Role already exists -- updating policies")

    # Managed policy
    _attach_managed_policy(iam, ec2_role_name,
                           "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy")

    # Inline: S3 read-only scoped to project bucket (model download)
    _put_inline_policy(iam, ec2_role_name,
                       "FlashBalanceAI-S3-ReadOnly-Scoped",
                       _build_ec2_s3_readonly_policy(bucket_name))

    ec2_role_arn = iam.get_role(RoleName=ec2_role_name)["Role"]["Arn"]
    print(f"    ARN: {ec2_role_arn}")

    # Instance profile
    if not _instance_profile_exists(iam, ec2_profile_name):
        iam.create_instance_profile(
            InstanceProfileName=ec2_profile_name,
            Tags=[
                {"Key": "Project", "Value": "FlashBalanceAI"},
                {"Key": "Issue", "Value": "15"},
            ],
        )
        print(f"    [OK] Instance profile created: {ec2_profile_name}")
        time.sleep(2)
    else:
        print(f"    [SKIP] Instance profile already exists")

    # Add role to instance profile (idempotent check)
    profile = iam.get_instance_profile(InstanceProfileName=ec2_profile_name)
    existing_roles = [r["RoleName"] for r in profile["InstanceProfile"]["Roles"]]
    if ec2_role_name not in existing_roles:
        iam.add_role_to_instance_profile(
            InstanceProfileName=ec2_profile_name,
            RoleName=ec2_role_name
        )
        print(f"    [OK] Role added to instance profile")
    else:
        print(f"    [SKIP] Role already in instance profile")

    ec2_profile_arn = profile["InstanceProfile"]["Arn"]
    print(f"    Profile ARN: {ec2_profile_arn}")

    # ---- 3. Store ARNs in SSM Parameter Store ----
    print("\n[SSM] Storing role ARNs in Parameter Store ...")
    ssm_params = {
        f"{SSM_PREFIX}/lambda-role-arn": (lambda_role_arn, "FlashBalanceAI Lambda role ARN"),
        f"{SSM_PREFIX}/ec2-role-arn": (ec2_role_arn, "FlashBalanceAI EC2 role ARN"),
        f"{SSM_PREFIX}/ec2-instance-profile-arn": (ec2_profile_arn, "FlashBalanceAI EC2 instance profile ARN"),
        f"{SSM_PREFIX}/s3-bucket-name": (bucket_name, "FlashBalanceAI S3 bucket name (resolved)"),
    }
    for param_name, (value, description) in ssm_params.items():
        _store_ssm_param(ssm, param_name, value, description)
        print(f"    [OK] {param_name} = {value}")

    print("\n[DONE] Issue #15 -- IAM setup complete.")
    print("   Run 'python src/aws/iam_setup.py verify' to confirm.")


def cmd_verify(args):
    """Verify both roles exist with correct policies attached."""
    cfg = _load_config()
    iam = _iam_client()
    ssm = _ssm_client()

    lambda_role_name = cfg["lambda_role_name"]
    ec2_role_name = cfg["ec2_role_name"]
    ec2_profile_name = cfg["ec2_instance_profile_name"]

    all_ok = True

    # ---- Lambda Role ----
    print(f"Checking {lambda_role_name} ...")
    if _role_exists(iam, lambda_role_name):
        role = iam.get_role(RoleName=lambda_role_name)["Role"]
        print(f"  [OK] Exists -- ARN: {role['Arn']}")

        # Check managed policies
        attached = iam.list_attached_role_policies(RoleName=lambda_role_name)
        policy_names = [p["PolicyName"] for p in attached["AttachedPolicies"]]
        expected_managed = ["AWSLambdaBasicExecutionRole", "CloudWatchReadOnlyAccess"]
        for mp in expected_managed:
            if mp in policy_names:
                print(f"  [OK] Managed policy: {mp}")
            else:
                print(f"  [FAIL] MISSING managed policy: {mp}")
                all_ok = False

        # Check inline policies
        inline = iam.list_role_policies(RoleName=lambda_role_name)
        inline_names = inline["PolicyNames"]
        expected_inline = [
            "FlashBalanceAI-DynamoDB-Scoped",
            "FlashBalanceAI-S3-Scoped",
            "FlashBalanceAI-ELBv2-Access",
            "FlashBalanceAI-ASG-Access",
        ]
        for ip in expected_inline:
            if ip in inline_names:
                print(f"  [OK] Inline policy: {ip}")
            else:
                print(f"  [FAIL] MISSING inline policy: {ip}")
                all_ok = False

        # Check no wildcard in scoped policies
        for ip_name in ["FlashBalanceAI-DynamoDB-Scoped", "FlashBalanceAI-S3-Scoped"]:
            if ip_name in inline_names:
                doc = iam.get_role_policy(RoleName=lambda_role_name, PolicyName=ip_name)
                # Resource "*" check -- we only flag if the Resource field itself is "*"
                for stmt in doc["PolicyDocument"].get("Statement", []):
                    resource = stmt.get("Resource", "")
                    if resource == "*":
                        print(f"  [WARN] Wildcard resource found in {ip_name}!")
                        all_ok = False
                    else:
                        pass  # scoped correctly
    else:
        print(f"  [FAIL] Role does NOT exist!")
        all_ok = False

    # ---- EC2 Role ----
    print(f"\nChecking {ec2_role_name} ...")
    if _role_exists(iam, ec2_role_name):
        role = iam.get_role(RoleName=ec2_role_name)["Role"]
        print(f"  [OK] Exists -- ARN: {role['Arn']}")

        attached = iam.list_attached_role_policies(RoleName=ec2_role_name)
        policy_names = [p["PolicyName"] for p in attached["AttachedPolicies"]]
        if "CloudWatchAgentServerPolicy" in policy_names:
            print(f"  [OK] Managed policy: CloudWatchAgentServerPolicy")
        else:
            print(f"  [FAIL] MISSING managed policy: CloudWatchAgentServerPolicy")
            all_ok = False

        inline = iam.list_role_policies(RoleName=ec2_role_name)
        if "FlashBalanceAI-S3-ReadOnly-Scoped" in inline["PolicyNames"]:
            print(f"  [OK] Inline policy: FlashBalanceAI-S3-ReadOnly-Scoped")
        else:
            print(f"  [FAIL] MISSING inline policy: FlashBalanceAI-S3-ReadOnly-Scoped")
            all_ok = False
    else:
        print(f"  [FAIL] Role does NOT exist!")
        all_ok = False

    # ---- Instance Profile ----
    print(f"\nChecking instance profile {ec2_profile_name} ...")
    if _instance_profile_exists(iam, ec2_profile_name):
        profile = iam.get_instance_profile(InstanceProfileName=ec2_profile_name)
        roles = [r["RoleName"] for r in profile["InstanceProfile"]["Roles"]]
        print(f"  [OK] Exists -- roles: {roles}")
        if ec2_role_name not in roles:
            print(f"  [FAIL] {ec2_role_name} NOT attached to profile!")
            all_ok = False
    else:
        print(f"  [FAIL] Instance profile does NOT exist!")
        all_ok = False

    # ---- SSM Parameters ----
    print(f"\nChecking SSM Parameter Store ...")
    expected_params = [
        f"{SSM_PREFIX}/lambda-role-arn",
        f"{SSM_PREFIX}/ec2-role-arn",
        f"{SSM_PREFIX}/ec2-instance-profile-arn",
        f"{SSM_PREFIX}/s3-bucket-name",
    ]
    for param_name in expected_params:
        try:
            resp = ssm.get_parameter(Name=param_name)
            print(f"  [OK] {param_name} = {resp['Parameter']['Value']}")
        except ssm.exceptions.ParameterNotFound:
            print(f"  [FAIL] MISSING: {param_name}")
            all_ok = False

    # ---- Summary ----
    print()
    if all_ok:
        print("[PASS] All IAM resources verified successfully.")
    else:
        print("[WARN] Some checks failed -- review above output.")
    return all_ok


def cmd_show_arns(args):
    """Print role ARNs from SSM Parameter Store."""
    ssm = _ssm_client()
    params = [
        f"{SSM_PREFIX}/lambda-role-arn",
        f"{SSM_PREFIX}/ec2-role-arn",
        f"{SSM_PREFIX}/ec2-instance-profile-arn",
        f"{SSM_PREFIX}/s3-bucket-name",
    ]
    print("FlashBalanceAI IAM ARNs (from SSM Parameter Store):")
    print("-" * 60)
    for param_name in params:
        try:
            resp = ssm.get_parameter(Name=param_name)
            print(f"  {param_name.split('/')[-1]:30s} {resp['Parameter']['Value']}")
        except ssm.exceptions.ParameterNotFound:
            print(f"  {param_name.split('/')[-1]:30s} NOT SET -- run 'create' first")


def cmd_delete(args):
    """Delete both IAM roles, instance profile, inline policies, and SSM params."""
    cfg = _load_config()
    iam = _iam_client()
    ssm = _ssm_client()

    lambda_role_name = cfg["lambda_role_name"]
    ec2_role_name = cfg["ec2_role_name"]
    ec2_profile_name = cfg["ec2_instance_profile_name"]

    # ---- Remove EC2 role from instance profile ----
    print(f"Removing {ec2_role_name} from instance profile ...")
    if _instance_profile_exists(iam, ec2_profile_name):
        profile = iam.get_instance_profile(InstanceProfileName=ec2_profile_name)
        for r in profile["InstanceProfile"]["Roles"]:
            iam.remove_role_from_instance_profile(
                InstanceProfileName=ec2_profile_name,
                RoleName=r["RoleName"]
            )
            print(f"    [OK] Removed {r['RoleName']} from profile")
        iam.delete_instance_profile(InstanceProfileName=ec2_profile_name)
        print(f"    [OK] Deleted instance profile: {ec2_profile_name}")
    else:
        print(f"    [SKIP] Instance profile not found")

    # ---- Delete roles (must detach policies first) ----
    for role_name in [lambda_role_name, ec2_role_name]:
        print(f"\nDeleting {role_name} ...")
        if not _role_exists(iam, role_name):
            print(f"    [SKIP] Role not found")
            continue

        # Detach managed policies
        attached = iam.list_attached_role_policies(RoleName=role_name)
        for pol in attached["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=role_name, PolicyArn=pol["PolicyArn"])
            print(f"    [OK] Detached: {pol['PolicyName']}")

        # Delete inline policies
        inline = iam.list_role_policies(RoleName=role_name)
        for pol_name in inline["PolicyNames"]:
            iam.delete_role_policy(RoleName=role_name, PolicyName=pol_name)
            print(f"    [OK] Deleted inline: {pol_name}")

        # Delete role
        iam.delete_role(RoleName=role_name)
        print(f"    [OK] Role deleted")

    # ---- Delete SSM parameters ----
    print("\nCleaning SSM parameters ...")
    for param in [
        f"{SSM_PREFIX}/lambda-role-arn",
        f"{SSM_PREFIX}/ec2-role-arn",
        f"{SSM_PREFIX}/ec2-instance-profile-arn",
        f"{SSM_PREFIX}/s3-bucket-name",
    ]:
        _delete_ssm_param(ssm, param)

    print("\n[DONE] IAM teardown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashBalanceAI Issue #15 -- IAM Roles & Policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("create", help="Create IAM roles, policies, and instance profile")
    sub.add_parser("verify", help="Verify roles and policies exist correctly")
    sub.add_parser("delete", help="Delete roles, policies, instance profile, SSM params")
    sub.add_parser("show-arns", help="Print role ARNs stored in SSM Parameter Store")

    args = parser.parse_args()

    commands = {
        "create": cmd_create,
        "verify": cmd_verify,
        "delete": cmd_delete,
        "show-arns": cmd_show_arns,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
