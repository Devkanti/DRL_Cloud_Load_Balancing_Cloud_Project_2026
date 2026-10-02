import boto3
import time
import zipfile
import io
import os
import yaml

def get_config():
    with open("configs/aws_config.yaml") as f:
        return yaml.safe_load(f)

def get_vpc_info():
    ec2 = boto3.client("ec2", region_name="us-east-1")
    vpcs = ec2.describe_vpcs(Filters=[{"Name":"tag:Project", "Values":["FlashBalanceAI"]}])["Vpcs"]
    if not vpcs: raise Exception("VPC not found")
    vpc_id = vpcs[0]["VpcId"]
    
    subnets = ec2.describe_subnets(Filters=[{"Name":"vpc-id", "Values":[vpc_id]}])["Subnets"]
    subnet_ids = [s["SubnetId"] for s in subnets]
    
    return vpc_id, subnet_ids

def create_security_group(vpc_id):
    ec2 = boto3.client("ec2", region_name="us-east-1")
    sg_name = "FlashBalanceAI-Inference-SG"
    try:
        resp = ec2.describe_security_groups(Filters=[{"Name":"group-name","Values":[sg_name]}, {"Name":"vpc-id","Values":[vpc_id]}])
        if resp["SecurityGroups"]:
            return resp["SecurityGroups"][0]["GroupId"]
    except: pass
    
    resp = ec2.create_security_group(GroupName=sg_name, Description="Inference server and lambda", VpcId=vpc_id)
    sg_id = resp["GroupId"]
    
    # Allow port 6000 within SG
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 6000,
            "ToPort": 6000,
            "UserIdGroupPairs": [{"GroupId": sg_id}]
        }]
    )
    
    # Actually, allow 6000 from VPC CIDR for simplicity
    vpc = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 6000,
            "ToPort": 6000,
            "IpRanges": [{"CidrIp": vpc["CidrBlock"]}]
        }]
    )
    return sg_id

def deploy_inference_server(subnet_id, sg_id, role_name, bucket):
    ec2 = boto3.client("ec2", region_name="us-east-1")
    
    # Read the script
    with open("src/aws/inference_server.py") as f:
        script_code = f.read()
        
    user_data = f"""#!/bin/bash
exec > >(tee /var/log/inference_setup.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Creating 2GB swap file..."
dd if=/dev/zero of=/swapfile bs=1M count=2048
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

echo "Setting up virtual environment..."
python3 -m venv /opt/venv
/opt/venv/bin/pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
/opt/venv/bin/pip install --no-cache-dir flask boto3 stable-baselines3 numpy

cat << 'EOF' > /opt/inference_server.py
{script_code}
EOF

cat << 'EOF' > /etc/systemd/system/inference_server.service
[Unit]
Description=FlashBalanceAI Inference Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt
ExecStart=/opt/venv/bin/python inference_server.py --bucket {bucket} --port 6000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable inference_server.service
systemctl start inference_server.service
echo "Setup complete."
"""
    
    # Find AL2023 AMI
    ssm = boto3.client("ssm", region_name="us-east-1")
    ami_id = ssm.get_parameter(Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64")["Parameter"]["Value"]
    
    print("Launching inference server EC2 instance...")
    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType="t3.micro",
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Name": "FlashBalanceAI-EC2-Profile"},
        UserData=user_data,
        NetworkInterfaces=[{
            "DeviceIndex": 0,
            "AssociatePublicIpAddress": True,
            "SubnetId": subnet_id,
            "Groups": [sg_id]
        }],
        BlockDeviceMappings=[{
            "DeviceName": "/dev/xvda",
            "Ebs": {"VolumeSize": 12, "VolumeType": "gp3"}
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "FlashBalanceAI-InferenceServer"}, {"Key": "Project", "Value": "FlashBalanceAI"}]
        }]
    )
    inst = resp["Instances"][0]
    inst_id = inst["InstanceId"]
    private_ip = inst["PrivateIpAddress"]
    print(f"Launched {inst_id} at private IP {private_ip}")
    return private_ip

def deploy_lambda(subnet_ids, sg_id, private_ip):
    lam = boto3.client("lambda", region_name="us-east-1")
    iam = boto3.client("iam")
    
    # 1. Ensure Lambda role has VPC permissions
    role_name = "FlashBalanceAI-Lambda-Role"
    iam.attach_role_policy(RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole")
    
    # 2. Package
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write("src/aws/inference_coordinator.py", "lambda_function.py")
    buf.seek(0)
    zip_bytes = buf.read()
    
    func_name = "FlashBalanceAI-InferenceCoordinator"
    env_vars = {
        "DYNAMODB_TABLE": "routing_decisions",
        "INFERENCE_SERVER_URL": f"http://{private_ip}:6000",
        "SESSION_ID": "live"
    }
    
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    
    try:
        lam.update_function_code(FunctionName=func_name, ZipFile=zip_bytes)
        time.sleep(2)
        lam.update_function_configuration(
            FunctionName=func_name,
            Environment={"Variables": env_vars},
            VpcConfig={
                "SubnetIds": subnet_ids,
                "SecurityGroupIds": [sg_id]
            }
        )
        print("Updated existing Lambda")
    except lam.exceptions.ResourceNotFoundException:
        print("Creating Lambda...")
        # Wait a sec for IAM policy propagation if just added
        time.sleep(10)
        lam.create_function(
            FunctionName=func_name,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=2,
            MemorySize=128,
            Environment={"Variables": env_vars},
            VpcConfig={
                "SubnetIds": subnet_ids,
                "SecurityGroupIds": [sg_id]
            }
        )
        print("Lambda created")

def stop_inference_server():
    ec2 = boto3.client("ec2", region_name="us-east-1")
    instances = ec2.describe_instances(Filters=[{"Name":"tag:Name", "Values":["FlashBalanceAI-InferenceServer"]}, {"Name":"instance-state-name", "Values":["running"]}])
    if instances["Reservations"]:
        iid = instances["Reservations"][0]["Instances"][0]["InstanceId"]
        ec2.stop_instances(InstanceIds=[iid])
        print(f"Stopped Inference Server {iid}")
    else:
        print("No running Inference Server found.")

def start_inference_server():
    ec2 = boto3.client("ec2", region_name="us-east-1")
    instances = ec2.describe_instances(Filters=[{"Name":"tag:Name", "Values":["FlashBalanceAI-InferenceServer"]}, {"Name":"instance-state-name", "Values":["stopped"]}])
    if instances["Reservations"]:
        iid = instances["Reservations"][0]["Instances"][0]["InstanceId"]
        ec2.start_instances(InstanceIds=[iid])
        print(f"Started Inference Server {iid}")
        
        # Wait a bit and get the new IP to update Lambda
        print("Waiting for new IP...")
        time.sleep(10)
        res = ec2.describe_instances(InstanceIds=[iid])
        new_ip = res["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
        print(f"New private IP: {new_ip}")
        
        # Update Lambda
        lam = boto3.client("lambda", region_name="us-east-1")
        env = lam.get_function_configuration(FunctionName="FlashBalanceAI-InferenceCoordinator")["Environment"]["Variables"]
        env["INFERENCE_SERVER_URL"] = f"http://{new_ip}:6000"
        lam.update_function_configuration(FunctionName="FlashBalanceAI-InferenceCoordinator", Environment={"Variables": env})
        print("Lambda updated with new Inference Server IP.")
    else:
        print("No stopped Inference Server found.")

def main():
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "stop":
            stop_inference_server()
            return
        elif sys.argv[1] == "start":
            start_inference_server()
            return
            
    cfg = get_config()
    vpc_id, subnet_ids = get_vpc_info()
    sg_id = create_security_group(vpc_id)
    
    # Find existing inference server to avoid launching duplicates
    ec2 = boto3.client("ec2", region_name="us-east-1")
    instances = ec2.describe_instances(Filters=[{"Name":"tag:Name", "Values":["FlashBalanceAI-InferenceServer"]}, {"Name":"instance-state-name", "Values":["pending", "running"]}])
    if instances["Reservations"]:
        private_ip = instances["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
        print(f"Found existing Inference Server at {private_ip}")
    else:
        private_ip = deploy_inference_server(subnet_ids[0], sg_id, cfg["ec2_role_name"], cfg["s3_bucket"])
        
    deploy_lambda(subnet_ids, sg_id, private_ip)
    print("Deployment complete.")

if __name__ == "__main__":
    main()
