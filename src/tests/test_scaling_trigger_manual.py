import boto3
import time
from decimal import Decimal

# Write 2 mock states with High CPU
dynamo = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamo.Table("routing_decisions")

# Prev state
prev_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - 30))
state_vector = [Decimal("75.0")] * 4 + [Decimal("0")] * 17 + [Decimal("1.0"), Decimal("0.0")]

mock_state_1 = {
    "pk": "STATE#live",
    "sk": prev_iso,
    "state_vector": state_vector,
    "expires_at": int(time.time()) + 3600,
    "metadata": {"instances_found": Decimal("4")}
}
table.put_item(Item=mock_state_1)

# Latest state
now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
state_sk = now_iso

mock_state_2 = {
    "pk": "STATE#live",
    "sk": state_sk,
    "state_vector": state_vector,
    "expires_at": int(time.time()) + 3600,
    "metadata": {"instances_found": Decimal("4")}
}
table.put_item(Item=mock_state_2)

mock_action = {
    "pk": "ACTION#live",
    "sk": now_iso,
    "action": 1,
    "state_sk": state_sk,
    "expires_at": int(time.time()) + 3600
}
table.put_item(Item=mock_action)

print("2 Mock states inserted.")

# Invoke scaling trigger
lam = boto3.client("lambda", region_name="us-east-1")
resp = lam.invoke(FunctionName="FlashBalanceAI-ScalingTrigger", InvocationType="RequestResponse", LogType="Tail")

import json, base64
print("Response:", json.loads(resp["Payload"].read()))
print("Logs:\n", base64.b64decode(resp["LogResult"]).decode())
