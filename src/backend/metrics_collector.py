import time
import json
import urllib.request
import urllib.error
import boto3
import sys

NAMESPACE = "FlashBalanceAI/Instances"
METRICS_URL = "http://localhost:5000/metrics"
REGION = "us-east-1"

def get_instance_id():
    try:
        # Try IMDSv2 first
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT")
        req.add_header("X-aws-ec2-metadata-token-ttl-seconds", "21600")
        with urllib.request.urlopen(req, timeout=2) as response:
            token = response.read().decode()
            
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-id")
        req.add_header("X-aws-ec2-metadata-token", token)
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.read().decode()
    except Exception as e:
        print(f"Failed to get instance ID from IMDSv2: {e}")
        return "unknown"

def main():
    cw = boto3.client('cloudwatch', region_name=REGION)
    instance_id = get_instance_id()
    print(f"Starting metrics collector for instance: {instance_id}")

    while True:
        try:
            with urllib.request.urlopen(METRICS_URL, timeout=2) as req:
                data = json.loads(req.read().decode())
            
            qd = float(data.get("queue_depth", 0))
            rt = float(data.get("response_time_ema", 50.0))
            
            cw.put_metric_data(
                Namespace=NAMESPACE,
                MetricData=[
                    {
                        "MetricName": "QueueDepth",
                        "Dimensions": [{"Name": "InstanceId", "Value": instance_id}],
                        "Value": qd,
                        "Unit": "Count"
                    },
                    {
                        "MetricName": "ResponseTimeEMA",
                        "Dimensions": [{"Name": "InstanceId", "Value": instance_id}],
                        "Value": rt,
                        "Unit": "Milliseconds"
                    }
                ]
            )
            print(f"Pushed QueueDepth={qd}, ResponseTimeEMA={rt}")
        except Exception as e:
            print(f"Error pushing metrics: {e}")
        
        # Sleep for 10 seconds
        time.sleep(10)

if __name__ == "__main__":
    main()
