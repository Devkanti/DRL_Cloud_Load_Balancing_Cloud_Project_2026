import os
import time
import boto3
from boto3.dynamodb.conditions import Key

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "routing_decisions")
SESSION_ID = os.environ.get("SESSION_ID", "live")
ASG_NAME = os.environ.get("ASG_NAME", "FlashBalanceAI-ASG")
COOLDOWN_SECONDS = 180

def lambda_handler(event, context):
    dynamo = boto3.resource('dynamodb', region_name=REGION)
    table = dynamo.Table(TABLE_NAME)
    asg_client = boto3.client('autoscaling', region_name=REGION)
    
    # 1. Read current action from DynamoDB
    action_resp = table.query(
        KeyConditionExpression=Key("pk").eq(f"ACTION#{SESSION_ID}"),
        ScanIndexForward=False,
        Limit=1
    )
    if not action_resp.get("Items"):
        print("No action found in DynamoDB.")
        return {"statusCode": 404, "body": "No action found"}
    latest_action_item = action_resp["Items"][0]
    
    # 2. Read the latest 2 states to check for sustained periods
    state_resp = table.query(
        KeyConditionExpression=Key("pk").eq(f"STATE#{SESSION_ID}"),
        ScanIndexForward=False,
        Limit=2
    )
    if not state_resp.get("Items") or len(state_resp["Items"]) == 0:
        print("No state found.")
        return {"statusCode": 404, "body": "No state found"}
    
    # Analyze latest state for generic info
    latest_state = state_resp["Items"][0]
    
    # Helper to analyze a single state
    def analyze_state(state_item):
        vec = state_item.get("state_vector", [])
        if len(vec) < 23: return False, 0.0
        
        cpu_util = [float(x) for x in vec[0:4]]
        burst = bool(float(vec[21]) > 0.5)
        
        metadata = state_item.get("metadata", {})
        instances_found = int(metadata.get("instances_found", 4))
        
        if instances_found == 0: return False, 0.0
            
        active_cpus = cpu_util[:instances_found]
        avg_cpu = sum(active_cpus) / len(active_cpus)
        all_above_70 = all(c > 70.0 for c in active_cpus)
        return burst and all_above_70, avg_cpu
        
    latest_high, latest_avg = analyze_state(latest_state)
    
    # Check if sustained over 2 periods
    sustained_high = False
    if len(state_resp["Items"]) == 2:
        prev_high, _ = analyze_state(state_resp["Items"][1])
        sustained_high = latest_high and prev_high
    else:
        # If there's only 1 state ever, assume not sustained yet
        sustained_high = False
        
    # Re-extract burst_indicator for scale in logic from latest
    vec = latest_state.get("state_vector", [])
    burst_indicator = bool(float(vec[21]) > 0.5) if len(vec) >= 22 else False

    print(f"Action: {latest_action_item.get('action')}, Burst: {burst_indicator}, CPU Avg: {latest_avg:.1f}%, Sustained High: {sustained_high}")
    
    # Check ASG scaling cooldown
    activities = asg_client.describe_scaling_activities(
        AutoScalingGroupName=ASG_NAME,
        MaxRecords=5
    )
    
    last_scale_time = 0
    now = time.time()
    for activity in activities.get('Activities', []):
        act_time = activity['StartTime'].timestamp()
        if act_time > last_scale_time:
            last_scale_time = act_time
            
    time_since_last_scale = now - last_scale_time
    
    # Get current capacity
    asg_info = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
    groups = asg_info.get("AutoScalingGroups", [])
    if not groups:
        return {"statusCode": 404, "body": "ASG not found"}
        
    asg = groups[0]
    current_capacity = asg["DesiredCapacity"]
    max_size = asg["MaxSize"]
    min_size = asg["MinSize"]
    
    target_capacity = current_capacity
    
    if sustained_high:
        if current_capacity < max_size:
            target_capacity = current_capacity + 1
            print("Decision: SCALE OUT")
        else:
            print("Decision: SCALE OUT (Already at max size)")
    elif not burst_indicator and latest_avg < 30.0:
        if current_capacity > min_size:
            target_capacity = current_capacity - 1
            print("Decision: SCALE IN")
        else:
            print("Decision: SCALE IN (Already at min size)")
    else:
        print("Decision: NO ACTION")
        
    if target_capacity != current_capacity:
        if time_since_last_scale < COOLDOWN_SECONDS:
            print(f"Cooldown active. {time_since_last_scale:.1f}s < {COOLDOWN_SECONDS}s. Not scaling.")
        else:
            print(f"Executing scaling: {current_capacity} -> {target_capacity}")
            asg_client.set_desired_capacity(
                AutoScalingGroupName=ASG_NAME,
                DesiredCapacity=target_capacity,
                HonorCooldown=False # We handle cooldown manually
            )
            
    return {
        "statusCode": 200,
        "body": {
            "current_capacity": current_capacity,
            "target_capacity": target_capacity,
            "time_since_last_scale": time_since_last_scale
        }
    }
