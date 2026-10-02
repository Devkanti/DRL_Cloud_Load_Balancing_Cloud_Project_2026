import os
import json
import time
import urllib.request
import urllib.error
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

# Configuration from Environment Variables
REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "routing_decisions")
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL") # e.g. http://10.0.1.5:6000
SESSION_ID = os.environ.get("SESSION_ID", "live")
TTL_SECONDS = 24 * 3600

_dynamo = None

def get_table():
    global _dynamo
    if _dynamo is None:
        _dynamo = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
    return _dynamo

def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [decimal_to_float(v) for v in obj]
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    return obj

def lambda_handler(event, context):
    start_time = time.time()
    table = get_table()
    
    # 1. Read latest state from DynamoDB
    pk = f"STATE#{SESSION_ID}"
    print(f"Reading state from DynamoDB: {pk}")
    resp = table.query(
        KeyConditionExpression=Key("pk").eq(pk),
        ScanIndexForward=False,
        Limit=1
    )
    items = resp.get("Items", [])
    if not items:
        return {"statusCode": 404, "body": json.dumps({"error": "No state found"})}
        
    latest_state = items[0]
    state_vector = decimal_to_float(latest_state.get("state_vector", []))
    
    if not state_vector:
        return {"statusCode": 400, "body": json.dumps({"error": "Empty state vector"})}
        
    # 2. Call EC2 Inference Server
    url = f"{INFERENCE_SERVER_URL}/action"
    print(f"POSTing to {url}")
    req = urllib.request.Request(
        url, 
        data=json.dumps({"state_vector": state_vector}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=1.5) as http_resp:
            result = json.loads(http_resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"statusCode": 502, "body": json.dumps({"error": f"Failed to reach inference server: {e}"})}
        
    action = result.get("action")
    if action is None:
        return {"statusCode": 500, "body": json.dumps({"error": "Invalid response from server"})}
        
    # 3. Write action to DynamoDB
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    expires_at = int(time.time()) + TTL_SECONDS
    
    item = {
        "pk": f"ACTION#{SESSION_ID}",
        "sk": now_iso,
        "action": action,
        "expires_at": expires_at,
        "state_sk": latest_state["sk"], # Link back to the state used
        "latency_ms": int((time.time() - start_time) * 1000)
    }
    table.put_item(Item=item)
    
    # 4. Trigger ScalingTrigger asynchronously
    try:
        lam = boto3.client("lambda", region_name=REGION)
        lam.invoke(
            FunctionName="FlashBalanceAI-ScalingTrigger",
            InvocationType="Event",
            Payload=json.dumps({"action": action})
        )
    except Exception as e:
        print(f"Failed to trigger ScalingTrigger: {e}")
    
    return {
        "statusCode": 200,
        "body": json.dumps(item)
    }
