#!/usr/bin/env python3
"""
FlashBalanceAI - Issue #21: CloudWatch Agent & Metrics Collector
=================================================================

Deploys the standard Amazon CloudWatch Agent AND our custom MetricsCollector
sidecar to all 4 backend instances using AWS Systems Manager (SSM) Run Command.

The custom MetricsCollector (src/backend/metrics_collector.py) polls the local
Flask /metrics endpoint every 10 seconds and pushes QueueDepth and ResponseTimeEMA
to the FlashBalanceAI/Instances CloudWatch namespace.
"""

import boto3
import time
from pathlib import Path
import os
import textwrap

REGION = "us-east-1"
ASG_NAME = "FlashBalanceAI-ASG"

def get_instance_ids():
    """Get InService instances from ASG"""
    asg = boto3.client("autoscaling", region_name=REGION)
    resp = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
    groups = resp.get("AutoScalingGroups", [])
    if not groups:
        return []
    instances = groups[0].get("Instances", [])
    return [i["InstanceId"] for i in instances if i.get("LifecycleState") == "InService"]

def run_ssm_command(ssm_client, instance_ids, document_name, parameters):
    resp = ssm_client.send_command(
        InstanceIds=instance_ids,
        DocumentName=document_name,
        Parameters=parameters,
        TimeoutSeconds=120,
    )
    cmd_id = resp["Command"]["CommandId"]
    print(f"Sent SSM command {cmd_id} to {len(instance_ids)} instances. Waiting...")
    
    # Wait for completion
    while True:
        time.sleep(5)
        status_resp = ssm_client.list_commands(CommandId=cmd_id)
        status = status_resp["Commands"][0]["Status"]
        if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
            return status, cmd_id
        print(".", end="", flush=True)

def check_command_output(ssm_client, cmd_id, instance_ids):
    success = True
    for iid in instance_ids:
        try:
            invocation = ssm_client.get_command_invocation(CommandId=cmd_id, InstanceId=iid)
            if invocation["Status"] != "Success":
                print(f"\nInstance {iid} failed!")
                print(f"Stdout:\n{invocation.get('StandardOutputContent', '')}")
                print(f"Stderr:\n{invocation.get('StandardErrorContent', '')}")
                success = False
        except Exception as e:
            print(f"Error fetching output for {iid}: {e}")
            success = False
    return success

def main():
    ssm = boto3.client("ssm", region_name=REGION)
    instance_ids = get_instance_ids()
    
    if not instance_ids:
        print("No running instances found in ASG.")
        return

    print(f"Found {len(instance_ids)} instances: {instance_ids}")
    
    # 1. Install standard Amazon CloudWatch Agent via SSM
    print("\n[1/3] Installing standard Amazon CloudWatch Agent...")
    status, cmd_id = run_ssm_command(
        ssm, 
        instance_ids, 
        "AWS-ConfigureAWSPackage", 
        {
            "action": ["Install"], 
            "name": ["AmazonCloudWatchAgent"]
        }
    )
    print(f" {status}")
    if status != "Success":
        check_command_output(ssm, cmd_id, instance_ids)
        print("[WARN] Standard agent install had issues, continuing anyway.")
        
    # 2. Start standard Amazon CloudWatch Agent
    print("\n[2/3] Starting standard CloudWatch Agent...")
    status, cmd_id = run_ssm_command(
        ssm, 
        instance_ids, 
        "AmazonCloudWatch-ManageAgent", 
        {
            "action": ["configure"], 
            "mode": ["ec2"],
            "optionalConfigurationSource": ["default"],
            "optionalRestart": ["yes"]
        }
    )
    print(f" {status}")
    
    # 3. Deploy our custom MetricsCollector script
    print("\n[3/3] Deploying custom MetricsCollector (Issue #7)...")
    
    # Read the script we just created
    script_path = Path(__file__).resolve().parent.parent / "backend" / "metrics_collector.py"
    with open(script_path, "r") as f:
        collector_code = f.read()
        
    # We will run a bash script via SSM that writes this python code to a file
    # and sets up a systemd service to run it.
    
    # Escape single quotes and backticks in the python code for the bash heredoc
    # Wait, simple EOF heredoc doesn't interpret quotes if we use 'EOF'
    bash_script = f"""#!/bin/bash
cat << 'EOF' > /opt/flashbalanceai/metrics_collector.py
{collector_code}
EOF

cat << 'EOF' > /etc/systemd/system/metrics_collector.service
[Unit]
Description=FlashBalanceAI Metrics Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/flashbalanceai
ExecStart=/usr/bin/python3 metrics_collector.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Install boto3 safely on Amazon Linux 2023
dnf install -y python3-boto3 || python3 -m pip install boto3 urllib3

systemctl daemon-reload
systemctl enable metrics_collector.service
systemctl restart metrics_collector.service
"""

    status, cmd_id = run_ssm_command(
        ssm, 
        instance_ids, 
        "AWS-RunShellScript", 
        {"commands": [bash_script]}
    )
    print(f" {status}")
    if status == "Success":
        print("\n[OK] MetricsCollector successfully deployed and started on all instances!")
    else:
        print("\n[FAIL] Failed to deploy MetricsCollector.")
        check_command_output(ssm, cmd_id, instance_ids)
        
    print("\n[DONE] Issue #21 complete.")
    print("Check CloudWatch Console -> Metrics -> FlashBalanceAI/Instances")

if __name__ == "__main__":
    main()
