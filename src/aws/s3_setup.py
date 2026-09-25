#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #16: S3 Bucket Setup
============================================

Creates and configures the project S3 bucket with:
  - Public access blocked
  - Folder structure: models/, state/, results/, dataset/
  - Lifecycle rule: delete objects in state/ after 7 days
  - Versioning disabled (cost control)

Usage:
    python src/aws/s3_setup.py create        # Create bucket + lifecycle + folders
    python src/aws/s3_setup.py verify        # Verify bucket config
    python src/aws/s3_setup.py upload-model  # Upload trained model (after Issue #11)
    python src/aws/s3_setup.py delete        # Delete bucket and all contents
    python src/aws/s3_setup.py list          # List bucket contents

Prerequisites:
    - AWS CLI configured (aws configure) with us-east-1
    - pip install boto3 pyyaml
    - Issue #15 IAM roles created

References:
    - configs/aws_config.yaml lines 14-18 (S3 config)
    - ADR-002 (cost optimization)
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

# Folder prefixes to create in the bucket
FOLDER_PREFIXES = [
    "models/",
    "state/",
    "results/",
    "dataset/",
]

# Lifecycle rules
LIFECYCLE_RULES = [
    {
        "ID": "DeleteStateAfter7Days",
        "Filter": {"Prefix": "state/"},
        "Status": "Enabled",
        "Expiration": {"Days": 7},
    },
    {
        "ID": "DeleteResultsAfter30Days",
        "Filter": {"Prefix": "results/"},
        "Status": "Enabled",
        "Expiration": {"Days": 30},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(AWS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _s3_client():
    return boto3.client("s3", region_name=REGION)


def _s3_resource():
    return boto3.resource("s3", region_name=REGION)


def _get_account_id() -> str:
    sts = boto3.client("sts", region_name=REGION)
    return sts.get_caller_identity()["Account"]


def _get_bucket_name(cfg: dict) -> str:
    """Resolve bucket name from config."""
    raw = cfg.get("s3_bucket", "")
    if "TODO_ACCOUNT_ID" in raw:
        account_id = _get_account_id()
        return raw.replace("TODO_ACCOUNT_ID", account_id)
    return raw


def _bucket_exists(s3, bucket_name: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            return False
        if code == "403":
            # Bucket exists but we don't own it
            print(f"[FAIL] Bucket '{bucket_name}' exists but access denied.")
            print("       The bucket name may be taken by another AWS account.")
            sys.exit(1)
        raise


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_create(args):
    """Create the S3 bucket with security, folders, and lifecycle."""
    cfg = _load_config()
    s3 = _s3_client()
    bucket_name = _get_bucket_name(cfg)

    print(f"Issue #16: S3 Bucket Setup")
    print(f"  Bucket : {bucket_name}")
    print(f"  Region : {REGION}")
    print()

    # ---- 1. Create bucket ----
    if _bucket_exists(s3, bucket_name):
        print(f"[SKIP] Bucket already exists")
    else:
        print(f"[1/4] Creating bucket '{bucket_name}' ...")
        # us-east-1 does NOT use LocationConstraint
        s3.create_bucket(Bucket=bucket_name)
        print(f"  [OK] Bucket created")
        time.sleep(2)

    # ---- 2. Block all public access ----
    print(f"\n[2/4] Blocking all public access ...")
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"  [OK] Public access blocked")

    # ---- 3. Create folder structure ----
    print(f"\n[3/4] Creating folder structure ...")
    for prefix in FOLDER_PREFIXES:
        try:
            # Check if folder marker already exists
            s3.head_object(Bucket=bucket_name, Key=prefix)
            print(f"  [SKIP] {prefix} (already exists)")
        except ClientError:
            s3.put_object(Bucket=bucket_name, Key=prefix, Body=b"")
            print(f"  [OK] {prefix}")

    # ---- 4. Set lifecycle rules ----
    print(f"\n[4/4] Setting lifecycle rules ...")
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration={"Rules": LIFECYCLE_RULES},
    )
    for rule in LIFECYCLE_RULES:
        prefix = rule["Filter"]["Prefix"]
        days = rule["Expiration"]["Days"]
        print(f"  [OK] {rule['ID']}: {prefix} -> expire after {days} days")

    # ---- Store in SSM ----
    print(f"\n[SSM] Storing bucket info in Parameter Store ...")
    ssm = boto3.client("ssm", region_name=REGION)
    ssm_params = {
        f"{SSM_PREFIX}/s3/bucket-name": (bucket_name, "FlashBalanceAI S3 bucket name"),
        f"{SSM_PREFIX}/s3/bucket-arn": (f"arn:aws:s3:::{bucket_name}", "FlashBalanceAI S3 bucket ARN"),
    }
    for param_name, (value, description) in ssm_params.items():
        try:
            ssm.put_parameter(
                Name=param_name,
                Value=value,
                Description=description,
                Type="String",
                Tags=[
                    {"Key": "Project", "Value": "FlashBalanceAI"},
                    {"Key": "Issue", "Value": "16"},
                ],
            )
        except ssm.exceptions.ParameterAlreadyExists:
            ssm.put_parameter(
                Name=param_name,
                Value=value,
                Description=description,
                Type="String",
                Overwrite=True,
            )
        print(f"  [OK] {param_name} = {value}")

    print(f"\n[DONE] Issue #16 -- S3 bucket setup complete.")
    print(f"   Run 'python src/aws/s3_setup.py verify' to confirm.")


def cmd_verify(args):
    """Verify bucket exists with correct config."""
    cfg = _load_config()
    s3 = _s3_client()
    bucket_name = _get_bucket_name(cfg)
    all_ok = True

    print(f"Verifying S3 bucket: {bucket_name}\n")

    # ---- Bucket exists ----
    if _bucket_exists(s3, bucket_name):
        print(f"[OK] Bucket exists")
    else:
        print(f"[FAIL] Bucket does NOT exist")
        return False

    # ---- Public access block ----
    try:
        pab = s3.get_public_access_block(Bucket=bucket_name)
        config = pab["PublicAccessBlockConfiguration"]
        if all([
            config.get("BlockPublicAcls"),
            config.get("IgnorePublicAcls"),
            config.get("BlockPublicPolicy"),
            config.get("RestrictPublicBuckets"),
        ]):
            print(f"[OK] All public access blocked")
        else:
            print(f"[WARN] Public access not fully blocked: {config}")
            all_ok = False
    except ClientError:
        print(f"[FAIL] Cannot read public access block")
        all_ok = False

    # ---- Folder structure ----
    print()
    for prefix in FOLDER_PREFIXES:
        resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
        if resp.get("KeyCount", 0) > 0:
            print(f"[OK] Prefix: {prefix}")
        else:
            print(f"[FAIL] Missing prefix: {prefix}")
            all_ok = False

    # ---- Lifecycle rules ----
    print()
    try:
        lc = s3.get_bucket_lifecycle_configuration(Bucket=bucket_name)
        rules = lc.get("Rules", [])
        rule_ids = [r["ID"] for r in rules]
        for expected in LIFECYCLE_RULES:
            if expected["ID"] in rule_ids:
                print(f"[OK] Lifecycle rule: {expected['ID']}")
            else:
                print(f"[FAIL] Missing lifecycle rule: {expected['ID']}")
                all_ok = False
    except ClientError as e:
        if "NoSuchLifecycleConfiguration" in str(e):
            print(f"[FAIL] No lifecycle rules configured")
            all_ok = False
        else:
            raise

    # ---- SSM Parameters ----
    print()
    ssm = boto3.client("ssm", region_name=REGION)
    for param in [f"{SSM_PREFIX}/s3/bucket-name", f"{SSM_PREFIX}/s3/bucket-arn"]:
        try:
            resp = ssm.get_parameter(Name=param)
            print(f"[OK] {param} = {resp['Parameter']['Value']}")
        except ssm.exceptions.ParameterNotFound:
            print(f"[FAIL] Missing SSM param: {param}")
            all_ok = False

    # ---- Summary ----
    print()
    if all_ok:
        print("[PASS] All S3 checks passed.")
    else:
        print("[WARN] Some checks failed -- review above.")
    return all_ok


def cmd_list(args):
    """List bucket contents."""
    cfg = _load_config()
    s3 = _s3_client()
    bucket_name = _get_bucket_name(cfg)

    if not _bucket_exists(s3, bucket_name):
        print(f"[FAIL] Bucket '{bucket_name}' does not exist.")
        return

    print(f"Contents of s3://{bucket_name}/\n")
    resp = s3.list_objects_v2(Bucket=bucket_name)
    objects = resp.get("Contents", [])
    if not objects:
        print("  (empty)")
        return

    total_size = 0
    for obj in objects:
        key = obj["Key"]
        size = obj["Size"]
        total_size += size
        modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M")
        if size == 0:
            print(f"  {key:50s} (folder marker)")
        else:
            print(f"  {key:50s} {size:>10,} bytes  {modified}")

    print(f"\n  Total: {len(objects)} objects, {total_size:,} bytes")


def cmd_upload_model(args):
    """Upload trained model to S3."""
    cfg = _load_config()
    s3 = _s3_client()
    bucket_name = _get_bucket_name(cfg)

    # Look for model files
    model_paths = [
        (PROJECT_ROOT / "models" / "ppo" / "best_model.zip", "models/ppo_flash_v1.zip"),
        (PROJECT_ROOT / "models" / "dqn" / "best_model.zip", "models/dqn_flash_v1.zip"),
    ]

    if not _bucket_exists(s3, bucket_name):
        print(f"[FAIL] Bucket '{bucket_name}' does not exist. Run 'create' first.")
        return

    uploaded = False
    for local_path, s3_key in model_paths:
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"Uploading {local_path.name} ({size_mb:.1f} MB) -> s3://{bucket_name}/{s3_key} ...")
            s3.upload_file(str(local_path), bucket_name, s3_key)
            print(f"  [OK] Uploaded: {s3_key}")
            uploaded = True
        else:
            print(f"  [SKIP] {local_path} not found (train model first)")

    if uploaded:
        print(f"\n[DONE] Model upload complete.")
    else:
        print(f"\n[WARN] No model files found. Train first (Issue #11/#12).")
        print(f"  Expected locations:")
        for lp, _ in model_paths:
            print(f"    {lp}")


def cmd_delete(args):
    """Delete the S3 bucket and all its contents."""
    cfg = _load_config()
    s3 = _s3_client()
    s3_res = _s3_resource()
    bucket_name = _get_bucket_name(cfg)

    if not _bucket_exists(s3, bucket_name):
        print(f"Bucket '{bucket_name}' does not exist.")
        return

    print(f"Deleting bucket '{bucket_name}' and all contents ...")

    # Must empty bucket before deleting
    bucket = s3_res.Bucket(bucket_name)
    deleted = 0
    for obj in bucket.objects.all():
        obj.delete()
        deleted += 1
    if deleted:
        print(f"  [OK] Deleted {deleted} objects")

    # Delete bucket
    s3.delete_bucket(Bucket=bucket_name)
    print(f"  [OK] Bucket deleted")

    # Clean SSM
    ssm = boto3.client("ssm", region_name=REGION)
    for param in [f"{SSM_PREFIX}/s3/bucket-name", f"{SSM_PREFIX}/s3/bucket-arn"]:
        try:
            ssm.delete_parameter(Name=param)
            print(f"  [OK] Deleted SSM param: {param}")
        except ssm.exceptions.ParameterNotFound:
            pass

    print(f"\n[DONE] S3 bucket teardown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashBalanceAI Issue #16 -- S3 Bucket Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("create", help="Create S3 bucket with folders and lifecycle rules")
    sub.add_parser("verify", help="Verify bucket configuration")
    sub.add_parser("list", help="List bucket contents")
    sub.add_parser("upload-model", help="Upload trained model to S3")
    sub.add_parser("delete", help="Delete bucket and all contents")

    args = parser.parse_args()

    commands = {
        "create": cmd_create,
        "verify": cmd_verify,
        "list": cmd_list,
        "upload-model": cmd_upload_model,
        "delete": cmd_delete,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
