import boto3
import time
import zipfile
import io
import json

def update_iam_role():
    iam = boto3.client("iam")
    role_name = "FlashBalanceAI-Lambda-Role"
    
    # Policy to allow describing ASG, setting desired capacity, and describing scaling activities
    # AND invoking Lambda (for InferenceCoordinator to invoke ScalingTrigger)
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "autoscaling:DescribeAutoScalingGroups",
                    "autoscaling:SetDesiredCapacity",
                    "autoscaling:DescribeScalingActivities",
                    "lambda:InvokeFunction"
                ],
                "Resource": "*"
            }
        ]
    }
    
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="FlashBalanceAI-Scaling-Permissions",
            PolicyDocument=json.dumps(policy_doc)
        )
        print("Updated Lambda IAM role with ASG and Invoke permissions.")
        time.sleep(5) # wait for IAM propagation
    except Exception as e:
        print(f"Failed to update IAM role: {e}")

def deploy_lambda():
    lam = boto3.client("lambda", region_name="us-east-1")
    iam = boto3.client("iam")
    
    # 1. Ensure IAM permissions are updated
    update_iam_role()
    
    # 2. Package
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write("src/aws/scaling_trigger.py", "lambda_function.py")
    buf.seek(0)
    zip_bytes = buf.read()
    
    func_name = "FlashBalanceAI-ScalingTrigger"
    env_vars = {
        "DYNAMODB_TABLE": "routing_decisions",
        "SESSION_ID": "live",
        "ASG_NAME": "FlashBalanceAI-ASG"
    }
    
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    role_name = "FlashBalanceAI-Lambda-Role"
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    
    try:
        lam.update_function_code(FunctionName=func_name, ZipFile=zip_bytes)
        time.sleep(2)
        lam.update_function_configuration(
            FunctionName=func_name,
            Environment={"Variables": env_vars},
            Timeout=5
        )
        print("Updated existing ScalingTrigger Lambda")
    except lam.exceptions.ResourceNotFoundException:
        print("Creating ScalingTrigger Lambda...")
        lam.create_function(
            FunctionName=func_name,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=5,
            MemorySize=128,
            Environment={"Variables": env_vars}
        )
        print("ScalingTrigger Lambda created")
        
    # Also update InferenceCoordinator code so it triggers ScalingTrigger
    try:
        buf2 = io.BytesIO()
        with zipfile.ZipFile(buf2, "w") as zf2:
            zf2.write("src/aws/inference_coordinator.py", "lambda_function.py")
        buf2.seek(0)
        lam.update_function_code(FunctionName="FlashBalanceAI-InferenceCoordinator", ZipFile=buf2.read())
        print("Updated InferenceCoordinator code.")
    except Exception as e:
        print(f"Could not update InferenceCoordinator code: {e}")

if __name__ == "__main__":
    deploy_lambda()
