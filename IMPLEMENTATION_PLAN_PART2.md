# IMPLEMENTATION PLAN — FlashBalanceAI (Part 2 of 2)
## Phases 5–9: AWS Integration → Experiments → Stats → Paper → Demo + Full Timeline

**Document Version:** 1.0 | **Date:** 2026-08-20  
**Team:** Devkanti Sarkar · Agrima Gupta · Mohar Gorai · [4th member]  
**Read alongside:** `IMPLEMENTATION_PLAN_PART1.md` (Part 1) and `PRD.md`.

> **Part 2 covers:** Phase 5 (AWS Integration), Phase 6 (Experiments E1–E10), Phase 7 (Statistical Validation), Phase 8 (Paper Preparation), Phase 9 (Demo/Teardown), Full Timeline Table, Research Integrity Checklist, Contradictions Between PDFs.

---

## PHASE 5 — AWS INTEGRATION

**Duration:** 5–7 days | **[COSTS MONEY during testing]**  
**Prerequisites:** Phase 4 complete (all AWS resources running and verified).

---

### T5.1 — CloudWatch State Collector Lambda [CRITICAL PATH][COSTS MONEY — minimal]

**Objective:** Lambda function that polls CloudWatch every 10 seconds, builds the 23-dimensional state vector, and writes it to S3 for the inference Lambda to consume.

**Steps:**
1. Create `src/aws/cloudwatch_collector.py`:
```python
import boto3, json, time, numpy as np, os

cw    = boto3.client('cloudwatch', region_name='us-east-1')
s3    = boto3.client('s3')
BUCKET = os.environ['S3_BUCKET']
N_INSTANCES = int(os.environ.get('N_INSTANCES', '4'))
INSTANCE_IDS = os.environ['INSTANCE_IDS'].split(',')  # comma-separated EC2 IDs
TG_ARN = os.environ['TARGET_GROUP_ARN']

def get_metric(instance_id, metric_name, namespace, stat='Average', period=10):
    resp = cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=time.time() - 20,
        EndTime=time.time(),
        Period=period,
        Statistics=[stat]
    )
    pts = resp.get('Datapoints', [])
    return pts[0][stat] / 100.0 if pts else 0.1  # normalise CPU to 0-1

def build_state_vector():
    cpu     = [get_metric(iid, 'CPUUtilization', 'AWS/EC2') for iid in INSTANCE_IDS]
    # Active connections per target: from ALB custom metric via CloudWatch
    # Approximate using request count if active conn not available
    conn    = [get_metric(iid, 'RequestCount', 'FlashBalanceAI/Instances') for iid in INSTANCE_IDS]
    queue   = [get_metric(iid, 'QueueDepth',   'FlashBalanceAI/Instances') for iid in INSTANCE_IDS]
    resp    = [get_metric(iid, 'ResponseTimeEMA', 'FlashBalanceAI/Instances') for iid in INSTANCE_IDS]
    health  = [1.0] * N_INSTANCES  # updated via ALB health check separately
    arr_raw = get_metric('ALL', 'RequestCount', 'FlashBalanceAI/ALB')
    arrival_norm     = min(arr_raw / 10.0, 10.0)  # normalised against baseline
    burst_indicator  = float(arrival_norm > 0.2)   # > 2x baseline
    # Retrieve last spike time from DynamoDB (simplified: use S3 state file)
    time_since_spike = 1.0  # default; updated by inference Lambda
    return cpu + conn + queue + resp + health + [arrival_norm, burst_indicator, time_since_spike]

def lambda_handler(event, context):
    state = build_state_vector()
    s3.put_object(
        Bucket=BUCKET,
        Key='state/current.json',
        Body=json.dumps({'state': state, 'ts': time.time()})
    )
    return {'statusCode': 200, 'body': 'State updated'}
```
2. Package and deploy:
```bash
cd src/aws
zip collector_lambda.zip cloudwatch_collector.py
aws lambda create-function \
    --function-name FlashBalanceAI-StateCollector \
    --runtime python3.11 \
    --handler cloudwatch_collector.lambda_handler \
    --role arn:aws:iam::{ACCOUNT_ID}:role/FlashBalanceAI-Lambda-Role \
    --zip-file fileb://collector_lambda.zip \
    --timeout 15 \
    --memory-size 128 \
    --environment Variables="{S3_BUCKET=flashbalanceai-{ACCOUNT_ID},\
N_INSTANCES=4,INSTANCE_IDS=i-1111,i-2222,i-3333,i-4444,\
TARGET_GROUP_ARN=arn:...}"
```
3. Create EventBridge rule to trigger every 10 seconds:
```bash
aws events put-rule \
    --name FlashBalanceAI-StateCollect \
    --schedule-expression "rate(1 minute)"  # EventBridge minimum = 1 min
    # For 10s polling: use a Step Functions state machine loop or
    # trigger from CloudWatch Metric Alarm with high frequency
```
4. **⚠️ NOTE on 10s polling:** AWS EventBridge minimum rate is 1 minute. For 10s polling, use:
   - Option A: Step Functions Express Workflow with a 10s `Wait` state (free tier: 1M state transitions/month)
   - Option B: Reduce to 30s CloudWatch Alarm trigger (sufficient for our latency SLA window)
   - **Recommended:** Use 30s polling (simplest, free tier compliant).

**How to delete:** `aws lambda delete-function --function-name FlashBalanceAI-StateCollector`

**Dependencies:** T4.3, T4.4, T4.5  
**Tool:** AWS Lambda, EventBridge, CloudWatch  
**Input:** Running EC2 instances with CloudWatch agent  
**Output:** `state/current.json` updated in S3 every 30s  
**Estimated Time:** 5 hours  
**Acceptance Criteria:** After deployment, `aws s3 cp s3://flashbalanceai-{ACCOUNT_ID}/state/current.json -` shows a valid 23-element state vector with plausible CPU values (0–1).  
**Costs Money?** Lambda free tier covers this easily. EventBridge: first 1M events free → **effectively free**.

---

### T5.2 — PPO Inference Server + Inference Coordinator Lambda [CRITICAL PATH]

**Objective:** Deploy the PPO inference EC2 server and a lightweight Lambda coordinator that reads state from S3, calls the inference server, updates ALB weights, and logs to DynamoDB every 30s.

> **Architecture decision (from cost review 2026-08-20):** Pure Lambda inference is NOT used. SB3 + PyTorch unzipped is ~300 MB, exceeding Lambda's 250 MB unzipped package limit. The EC2 inference server approach is the confirmed primary path. See PRD §12.2 and ADR-001 D14.

**Component A — EC2 Inference Server (`src/aws/inference_server.py`):**
```python
# Runs on a dedicated t2.micro EC2 instance.
# Model is loaded from S3 at startup and held in memory.
from flask import Flask, request, jsonify
import boto3, numpy as np, json, os
from stable_baselines3 import PPO

app = Flask(__name__)
BUCKET = os.environ['S3_BUCKET']
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/ppo_flash_v1.zip')
N = int(os.environ.get('N_INSTANCES', '4'))

# Load model at startup — not per-request
s3 = boto3.client('s3')
s3.download_file(BUCKET, MODEL_KEY, '/tmp/ppo_model.zip')
model = PPO.load('/tmp/ppo_model')  # SB3 load from zip path
print(f"PPO model loaded. Ready to serve inference on port 6000.")

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': MODEL_KEY}), 200

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    state = np.array(data['state'], dtype=np.float32)
    action, _ = model.predict(state, deterministic=True)
    return jsonify({'action': int(action), 'n_instances': N}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
```
Deploy: start this server on the inference t2.micro EC2 instance at startup via systemd or UserData.

**Component B — Inference Coordinator Lambda (`src/aws/inference_coordinator.py`):**
```python
# Lightweight Lambda — does NOT contain SB3. < 5 MB package.
import boto3, json, os, requests, time

s3    = boto3.client('s3')
elbv2 = boto3.client('elbv2')
ddb   = boto3.resource('dynamodb').Table('routing_decisions')
BUCKET          = os.environ['S3_BUCKET']
TG_ARN          = os.environ['TARGET_GROUP_ARN']
INFERENCE_URL   = os.environ['INFERENCE_SERVER_URL']  # http://{inference_ec2_ip}:6000/infer
N               = int(os.environ.get('N_INSTANCES', '4'))
EXPERIMENT_ID   = os.environ.get('EXPERIMENT_ID', 'default')

def lambda_handler(event, context):
    # 1. Read current state from S3
    obj = s3.get_object(Bucket=BUCKET, Key='state/current.json')
    state_data = json.loads(obj['Body'].read())
    state = state_data['state']  # 23-element list

    # 2. Call inference server
    resp = requests.post(INFERENCE_URL, json={'state': state}, timeout=3)
    action = resp.json()['action']

    # 3. Update ALB target group weights
    # Weighted routing: chosen instance gets weight 40, others get 5
    tg_resp = elbv2.describe_target_health(TargetGroupArn=TG_ARN)
    healthy_targets = [t['Target']['Id'] for t in tg_resp['TargetHealthDescriptions']
                       if t['TargetHealth']['State'] == 'healthy']
    new_targets = []
    for i, tid in enumerate(healthy_targets[:N]):
        weight = 40 if i == action else 5
        new_targets.append({'Id': tid, 'Port': 5000, 'Weight': weight})
    if new_targets:
        elbv2.register_targets(TargetGroupArn=TG_ARN, Targets=new_targets)

    # 4. Log to DynamoDB
    ddb.put_item(Item={
        'timestamp': str(time.time()),
        'experiment_id': EXPERIMENT_ID,
        'action': str(action),
        'arrival_rate': str(state[-3]),
        'state_snapshot': json.dumps(state[:4])  # CPU utils only (compact log)
    })

    return {'statusCode': 200, 'action': action}
```
Deploy:
```bash
zip coordinator_lambda.zip inference_coordinator.py
# Add requests library layer (AWS public layer or pip install requests -t .)
aws lambda create-function \
    --function-name FlashBalanceAI-InferenceCoordinator \
    --runtime python3.11 \
    --handler inference_coordinator.lambda_handler \
    --role arn:aws:iam::{ACCOUNT_ID}:role/FlashBalanceAI-Lambda-Role \
    --zip-file fileb://coordinator_lambda.zip \
    --timeout 10 \
    --memory-size 128 \
    --environment Variables="{S3_BUCKET=...,TARGET_GROUP_ARN=...,\
INFERENCE_SERVER_URL=http://{INFERENCE_EC2_IP}:6000/infer,N_INSTANCES=4,\
EXPERIMENT_ID=default}"
```
Trigger: EventBridge rule `rate(1 minute)` — every 30s not achievable without Step Functions; 1-minute polling is sufficient for the 30s state-collection cycle.

**Dependencies:** T5.1, T4.3, T4.5 (ALB and inference EC2 must be running)
**Tool:** EC2 (inference server), AWS Lambda (coordinator), boto3, requests
**Output:** PPO inference updating ALB routing every 60s (EventBridge minimum); DynamoDB logging
**Estimated Time:** 6 hours
**Acceptance Criteria:**
- Inference server health check: `curl http://{INFERENCE_EC2_IP}:6000/health` returns `{"status":"healthy"}`
- Lambda coordinator invocation: `aws lambda invoke --function-name FlashBalanceAI-InferenceCoordinator out.json` returns `{"statusCode": 200, "action": 0-3}`
- DynamoDB accumulating rows during a 5-minute test run
- CloudWatch `RequestCount` per target shifts after action changes
**Costs Money?** Lambda: free tier. EC2 inference server: 1 t2.micro, stopped between sessions.

---

### T5.3 — Scaling Trigger Lambda [PARALLEL][FREE TIER]

**Objective:** Lambda triggered by CloudWatch alarm (CPU > 70%) to scale out/in the ASG. Also handles proactive pre-scaling when `burst_indicator = 1`.

**Steps:**
1. Create `src/aws/scaling_trigger.py`:
```python
import boto3, json, os, time

asg = boto3.client('autoscaling')
ASG_NAME = os.environ['ASG_NAME']
MIN_INSTANCES = int(os.environ.get('MIN_INSTANCES', '2'))
MAX_INSTANCES = int(os.environ.get('MAX_INSTANCES', '8'))

def lambda_handler(event, context):
    source = event.get('source', 'cloudwatch')
    current = asg.describe_auto_scaling_groups(
        AutoScalingGroupNames=[ASG_NAME])['AutoScalingGroups'][0]
    desired = current['DesiredCapacity']
    cpu_alarm = event.get('detail', {}).get('state', {}).get('value', '') == 'ALARM'

    if source == 'burst_preemptive':
        # Proactive: burst_indicator triggered — add 2 instances immediately
        new_desired = min(desired + 2, MAX_INSTANCES)
        reason = "proactive_burst"
    elif cpu_alarm and desired < MAX_INSTANCES:
        new_desired = min(desired + 1, MAX_INSTANCES)
        reason = "cpu_threshold_scale_out"
    else:
        return {'statusCode': 200, 'action': 'no_change'}

    asg.set_desired_capacity(
        AutoScalingGroupName=ASG_NAME,
        DesiredCapacity=new_desired,
        HonorCooldown=False if reason == "proactive_burst" else True
    )
    print(f"Scaled to {new_desired} ({reason})")
    return {'statusCode': 200, 'action': reason, 'new_desired': new_desired}
```
2. Create CloudWatch alarm that triggers this Lambda:
```bash
aws cloudwatch put-metric-alarm \
    --alarm-name FlashBalanceAI-HighCPU \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 30 \
    --threshold 70 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:lambda:us-east-1:{ACCOUNT_ID}:function:FlashBalanceAI-ScalingTrigger
```
3. Deploy Lambda: zip, `aws lambda create-function ...` (same pattern as T5.1).
4. Add scale-in alarm (CPU < 30% for 5 periods → scale in by 1).

**Dependencies:** T5.1, T4.4  
**Estimated Time:** 3 hours  
**Acceptance Criteria:** When CPU manually raised above 70% on a test instance, the ASG desired capacity increases within 120 seconds.  
**Costs Money?** Lambda free tier → **Free**. CloudWatch alarm: 5 free alarms → **Free**.

---

### T5.4 — CloudWatch Custom Metrics from Backend [PARALLEL]

**Objective:** Flask backend publishes **exactly 2 custom metrics per instance** (QueueDepth, ResponseTimeEMA) to CloudWatch every 30s. Total: 8 custom metrics across 4 instances — within the 10-metric free tier.

> **Cost note:** The original plan specified ~20 custom metrics. This revision reduces to 8 by deriving `active_conn` from the free ALB built-in `ActiveConnectionCount` metric and `arrival_rate` from the free ALB `RequestCount` metric. Only QueueDepth and ResponseTimeEMA require custom publishing.

**Steps:**
1. Add to `src/backend/app.py` a background thread that publishes metrics:
```python
import threading, boto3, time, collections

_request_times = collections.deque(maxlen=1000)
_queue_depth = 0

@app.before_request
def before():
    global _queue_depth
    _queue_depth += 1

@app.after_request
def after(response):
    global _queue_depth
    _queue_depth = max(0, _queue_depth - 1)
    return response

def publish_metrics():
    cw = boto3.client('cloudwatch', region_name='us-east-1')
    instance_id = os.environ.get('INSTANCE_ID', 'local')
    while True:
        if _request_times:
            ema = sum(list(_request_times)[-10:]) / min(10, len(_request_times))
        else:
            ema = 20.0
        cw.put_metric_data(
            Namespace='FlashBalanceAI/Instances',
            MetricData=[
                {'MetricName': 'QueueDepth', 'Value': _queue_depth,
                 'Dimensions': [{'Name': 'InstanceId', 'Value': instance_id}]},
                {'MetricName': 'ResponseTimeEMA', 'Value': ema / 500.0,  # normalise
                 'Dimensions': [{'Name': 'InstanceId', 'Value': instance_id}]},
            ]
        )
        time.sleep(30)

threading.Thread(target=publish_metrics, daemon=True).start()
```
2. Test: `aws cloudwatch get-metric-statistics --namespace FlashBalanceAI/Instances --metric-name QueueDepth --start-time $(date -u -d '-5 minutes' +%Y-%m-%dT%H:%M:%SZ) --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) --period 30 --statistics Average`.

**Dependencies:** T5.1, T1.2  
**Estimated Time:** 2 hours  
**Acceptance Criteria:** CloudWatch console shows `FlashBalanceAI/Instances` namespace with 4 instance dimensions. Metrics update every 30s.  
**Costs Money?** First 10 custom metrics free; we need 8 (2 per instance × 4) → **Free**.

---

### T5.5 — End-to-End Integration Test [CRITICAL PATH]

**Objective:** Verify the complete revised data flow: JMeter → ALB (direct, no API Gateway) → EC2 → CloudWatch → Lambda State Collector → S3 → Lambda Inference Coordinator → EC2 Inference Server → ALB update → DynamoDB.

**Steps:**
1. Confirm ALB is running; set ASG desired=4 and wait for all 4 targets to be `healthy`.
2. Start EC2 inference server; confirm `/health` returns 200.
3. Run a 5-minute JMeter test at baseline load (100 req/s) **targeting ALB DNS directly** — no API Gateway.
4. Check CloudWatch `FlashBalanceAI/Instances`: QueueDepth and ResponseTimeEMA updating every 30s for all 4 instances.
5. Check S3: `state/current.json` contains a valid 23-element state vector with CPU values in [0, 1].
6. Check DynamoDB: `routing_decisions` table accumulating rows with valid `action` (0–3) values.
7. Check ALB access logs: requests distributed across all 4 target instances (no single instance receiving > 80%).
8. Check Lambda CloudWatch Logs: no errors in State Collector or Inference Coordinator for 5 consecutive invocations.
9. **GATE CHECK:** All 6 data-flow components verified before experiments begin.

**Dependencies:** T5.1, T5.2, T5.3, T5.4
**Estimated Time:** 3 hours
**Acceptance Criteria:** All 6 checklist steps confirmed. Cost during 1-hour integration test: ~$0.04–$0.06 (ALB + EC2). Results documented in `decisions/integration_test_report.md`.
**Costs Money?** **YES** — ALB + EC2 active. Estimate: ~$0.05 for a 1-hour test.

---

## PHASE 6 — EXPERIMENTAL SETUP

**Duration:** 7–10 days | **[COSTS MONEY]** | **Prerequisites:** T5.5 (integration test) passed.

**⚠️ RESEARCH INTEGRITY:** No results from these experiments may be reported in the paper until  
(a) the experiment has completed the minimum specified repetitions,  
(b) all raw data has been saved to S3 and committed to the repo, and  
(c) statistical analysis has been completed per Phase 7.

---

### E1 — Baseline Traffic (1× Load) [CRITICAL PATH]

| Parameter | Value |
|-----------|-------|
| **Purpose** | Confirm all systems work correctly at normal load; calibrate baselines |
| **Traffic profile** | Constant 100 req/s, no burst |
| **Duration** | 13 minutes per run |
| **Repetitions** | 5 (seeds 127, 128, 129, 130, 131) |
| **Algorithms** | PPO, DQN, RR, WRR, LC, Threshold — all 6 |
| **AWS config** | 4 EC2 instances, desired=4, min=2, max=8 |

**Steps:**
1. Start all EC2 instances (desired-capacity=4).
2. Run each algorithm in isolation for 13 minutes.
3. For PPO/DQN: load model from S3, start inference Lambda/server.
4. For baselines: run equivalent software routing logic.
5. JMeter command: `jmeter -n -t flash_sale_baseline.jmx -Jhost={ALB_DNS} -Jpeak_users=100 -Jresultsfile=experiments/results/e1_baseline_{ALGO}_{SEED}.jtl`
6. After each run: `aws s3 cp experiments/results/ s3://flashbalanceai-{ACCOUNT_ID}/results/ --recursive`

**Metrics collected:**
- Average latency (ms)
- P95, P99 latency (ms)
- Throughput (req/s)
- Request failure rate (%)
- CPU utilisation per instance (%)
- Load imbalance coefficient of variation

**Expected output:** All algorithms perform similarly at baseline (difference < 10%). If PPO shows large divergence at baseline, the reward function needs debugging.

**Statistical analysis:** Mean ± std across 5 seeds. No significance test needed (baseline sanity check).

**Costs Money?** **YES** — 6 algorithms × 5 runs × 13 min = ~6.5 hours of ALB + EC2 time → ~$0.50.

---

### E2 — 10× Traffic Spike [CRITICAL PATH]

| Parameter | Value |
|-----------|-------|
| **Purpose** | Primary experiment — main paper result |
| **Traffic profile** | Warm-up (2min, 100rps) → Pre-burst (1min, 300rps) → Spike onset (10s, 0→1000rps) → Peak (5min, 1000rps) → Cooldown (5min) |
| **Duration** | 13 minutes per run |
| **Repetitions** | 5 (seeds 132–136) |
| **Algorithms** | PPO, DQN, RR, WRR, LC, Threshold — all 6 |

**Steps:**
1. Use JMeter test plan `flash_sale_10x.jmx` with peak_users=1000.
2. All other steps identical to E1.
3. Record scaling events: timestamp when ASG adds new instance (compare PPO proactive vs Threshold reactive).

**Expected output:**
- PPO: P95 < 500ms during peak (EXPECTED — unvalidated until experiment runs)
- Threshold: P95 likely 800ms–2000ms during spike onset (EXPECTED)
- RR/LC: P95 likely 500ms–1500ms depending on load distribution (EXPECTED)

**Statistical analysis:** Mean ± std, 95% CI, paired t-test PPO vs each baseline on P95 latency.

**Costs Money?** ~$0.50 for full E2 run.

---

### E3 — 50× Traffic Spike

| Parameter | Value |
|-----------|-------|
| **Purpose** | Stress test; expected degradation of simpler baselines |
| **Traffic profile** | Same as E2 but peak_users=5000 |
| **Duration** | 13 minutes per run |
| **Repetitions** | 5 (seeds 137–141) |
| **⚠️ PREREQUISITE** | E2 must complete successfully first. If E2 shows infrastructure ceiling issues, redesign before running E3. |

**Steps:** Same as E2, `peak_users=5000`.

**Expected output:** RR/Threshold likely exceed SLA at 50× due to queue overflow. PPO may also degrade but maintain lower P95 than baselines (EXPECTED). Infrastructure ceiling (max 8 t2.micro) likely reached.

**Acceptance criteria (to proceed):** EC2 instances must not crash. If all instances hit 100% CPU, reduce to 4 repetitions and document the ceiling as a limitation.

**Costs Money?** ~$0.60.

---

### E4 — 100× Traffic Spike (Conditional)

| Parameter | Value |
|-----------|-------|
| **Purpose** | Upper bound; expected all methods exceed SLA |
| **Traffic profile** | peak_users=10000 |
| **Duration** | 8 minutes (shortened — infrastructure stress) |
| **Repetitions** | 3 (seeds 127–129) — reduced due to expected infrastructure ceiling |
| **Condition** | Run ONLY if E3 showed PPO maintaining < 2000ms P95 at 50×. Otherwise document as out-of-scope. |

**Expected output:** All methods exceed 500ms SLA. Results reported as infrastructure ceiling analysis, not algorithm comparison. Document graceful degradation behaviour.

**Costs Money?** ~$0.30.

---

### E5 — Repeated Flash-Sale Patterns

| Parameter | Value |
|-----------|-------|
| **Purpose** | Test recovery between bursts; validate that PPO adapts across multiple events |
| **Traffic profile** | 3 consecutive 10× bursts, 2-minute gap between each |
| **Duration** | ~45 minutes per run |
| **Repetitions** | 5 |
| **Algorithms** | PPO, DQN, Threshold (primary comparison — most relevant for repeated events) |

**Steps:**
1. Use `generate_repeated_bursts(n_bursts=3)` from traffic generator.
2. Run 45-minute JMeter test with corresponding thread ramp-up pattern.
3. Record PPO reward per burst window (does reward improve from burst 1 to burst 3?).

**Expected output:** PPO shows improved handling of 2nd/3rd burst compared to 1st (EXPECTED — PPO adapts online). Threshold scales out/in each time with cold-start delay.

**Costs Money?** ~$1.50 (longer duration).

---

### E6 — Noisy/Transient Spikes

| Parameter | Value |
|-----------|-------|
| **Purpose** | Test whether PPO over-reacts to short transient noise within baseline |
| **Traffic profile** | Baseline 100rps with random 2× spikes lasting 10–30 seconds, noise_std=0.40 |
| **Duration** | 13 minutes per run |
| **Repetitions** | 5 |
| **Algorithms** | PPO, DQN, Threshold (Threshold over-reaction expected) |

**Expected output:** PPO should NOT trigger scale-out for transient 2× spikes (burst_indicator dampens noise). Threshold may oscillate. This tests the H-MAS noise-filtering insight from Mohar's document (simplified via burst_indicator).

**Costs Money?** ~$0.50.

---

### E7 — Cold-Start / Scaling Response Time

| Parameter | Value |
|-----------|-------|
| **Purpose** | Measure time from burst onset to N+1 instances being healthy in target group |
| **Setup** | Start with desired-capacity=2 (below normal). Trigger 10× burst. Measure time until 3rd instance is healthy. |
| **Duration** | 10 minutes per run |
| **Repetitions** | 5 (PPO proactive) + 5 (Threshold reactive) |

**Measurement protocol:**
1. Record t₀ = timestamp of burst onset (JMeter ramp-up start).
2. Record t₁ = timestamp when ASG desired-capacity changes (CloudWatch event).
3. Record t₂ = timestamp when new instance shows `healthy` in ALB target health.
4. Scaling reaction time = t₂ − t₀ (user-visible impact window).
5. For PPO proactive: burst_indicator fires at t₀, immediate scale call → expect t₁ ≈ t₀ + 5s.
6. For Threshold: CPU must breach 70% for 2×30s = 60s minimum → t₁ ≈ t₀ + 90s.

**Expected output (UNVALIDATED):** PPO proactive reduces scaling reaction time by ~70% vs Threshold.

**Costs Money?** ~$0.50.

---

### E8 — Cost / Performance Trade-off

| Parameter | Value |
|-----------|-------|
| **Purpose** | Understand how changing w₂ (utilisation weight) affects AWS cost vs P95 latency |
| **Config** | 3 configurations: (a) w₂=0.10 (performance focus), (b) w₂=0.20 (default), (c) w₂=0.40 (cost focus) |
| **Traffic** | 10× spike (same as E2) |
| **Duration** | 13 minutes per run × 3 configs |
| **Repetitions** | 3 per config = 9 total runs |

**For each config:**
1. Retrain PPO with modified reward weights (3 new training runs, ~4 hours each — consider running on Colab overnight).
2. Run E2-equivalent experiment with the retrained model.
3. Record: P95 latency, average CPU utilisation, estimated AWS cost (instances × hours).

**Expected output:** Cost-focus config uses fewer instances (higher CPU) but higher P95. Performance-focus uses more instances (lower CPU) but lower P95. Default is Pareto-optimal (EXPECTED).

**Costs Money?** ~$0.50 for experiments + potential SageMaker training ($0.46 × 4hrs × 3 configs = ~$0.55).

---

### E9 — DRL Algorithm Comparison (PPO vs DQN)

| Parameter | Value |
|-----------|-------|
| **Purpose** | Core research question RQ2: does PPO outperform DQN for flash-sale routing? |
| **Traffic** | 10× and 50× bursts (E2 + E3 scenarios) |
| **Duration** | Same as E2/E3 — use same runs (no additional AWS cost) |
| **Repetitions** | 5 each (already collected in E2, E3) |

**Steps:**
1. Extract PPO and DQN results from E2 and E3.
2. Run head-to-head comparison with same test seeds.
3. Statistical comparison: paired t-test on P95 latency (PPO vs DQN), α = 0.05.
4. Report: mean P95, 95% CI, Cohen's d effect size, p-value.

**⚠️ INTEGRITY NOTE:** Do not adjust test seeds or filter runs after seeing results. Report all 5 repetitions regardless of outcome.

**Costs Money?** No additional cost (uses E2/E3 data).

---

### E10 — Ablation Study of Reward Components

| Parameter | Value |
|-----------|-------|
| **Purpose** | Validate each reward component's contribution (RQ3) |
| **Variants** | (a) Full reward (default), (b) No R_lat (w₁=0), (c) No R_util (w₂=0), (d) No R_tput (w₃=0), (e) No R_sla (w₄=0) |
| **Traffic** | 10× burst |
| **Duration** | 13 minutes per run |
| **Repetitions** | 3 per variant = 15 total runs |

**Steps:**
1. For each variant, modify `ppo_config.yaml` reward weights → retrain PPO (run on Colab to save SageMaker cost).
2. Evaluate on same 3 test seeds as E2.
3. Compare P95 latency, SLA violation rate, CPU utilisation across variants.
4. Expected: removing R_sla (w₄=0) should increase P99 latency most (EXPECTED — unvalidated).

**Critical for paper:** Ablation study is essential for defending the reward function design. Without E10, reviewers will correctly challenge the reward weights as arbitrary.

**Costs Money?** ~$0.75 for experiments. Training cost per variant: ~$0 (local/Colab).

---

## PHASE 7 — RESULTS AND STATISTICAL VALIDATION

**Duration:** 4–5 days | **[LOCAL ONLY]**

---

### T7.1 — Raw Data Collection and Verification [CRITICAL PATH]

**Objective:** Ensure all experiment results are properly collected and verified before analysis.

**Steps:**
1. For each experiment run: download JMeter `.jtl` file and CloudWatch metric export to `experiments/results/`.
2. Verify completeness: each `.jtl` should have approximately 7800 rows (13 min × 10 req/s average).
3. Check for data gaps: `pandas.read_csv(jtl_file).isnull().sum()` should be 0.
4. Export CloudWatch metrics: for each experiment, run:
```python
import boto3, pandas as pd, json
cw = boto3.client('cloudwatch')
# Export CPU, response time, request count for each instance
# Save to experiments/results/cloudwatch_{exp_id}_{seed}.csv
```
5. Commit all raw data to `experiments/results/` (gitignore `.jtl` files if > 100MB; store in S3 and link in README).

**Acceptance Criteria:** Every experiment (E1–E10) has raw data files for the specified number of repetitions. No file is empty or corrupted.

---

### T7.2 — Statistical Analysis [CRITICAL PATH][LOCAL ONLY]

**Objective:** Compute all reported statistics with correct methodology.

**Steps — execute in `notebooks/03_results_analysis.ipynb`:**

**Step 1: Load all results**
```python
import pandas as pd, numpy as np
from scipy import stats

def load_results(experiment_id, algorithm):
    files = glob(f"experiments/results/{experiment_id}_{algorithm}_seed*.csv")
    dfs = [pd.read_csv(f) for f in sorted(files)]
    return dfs
```

**Step 2: Compute per-run summaries**
```python
def summarise(df):
    return {
        'mean_latency':  df['elapsed'].mean(),
        'p95_latency':   df['elapsed'].quantile(0.95),
        'p99_latency':   df['elapsed'].quantile(0.99),
        'throughput':    len(df) / df['elapsed'].count() * 1000 / 13,  # req/s
        'failure_rate':  (df['responseCode'] >= 500).mean(),
        'sla_violation': (df['elapsed'] > 500).mean(),
    }
```

**Step 3: Mean and standard deviation across runs**
```python
metrics = [summarise(df) for df in load_results('e2_10x', 'PPO')]
mean_p95 = np.mean([m['p95_latency'] for m in metrics])
std_p95  = np.std([m['p95_latency']  for m in metrics], ddof=1)
```

**Step 4: 95% Confidence Interval (t-distribution, n=5)**
```python
n = len(metrics)
se = std_p95 / np.sqrt(n)
ci_low  = mean_p95 - stats.t.ppf(0.975, df=n-1) * se
ci_high = mean_p95 + stats.t.ppf(0.975, df=n-1) * se
```

**Step 5: Paired t-test (PPO vs each baseline)**
```python
ppo_p95    = [summarise(df)['p95_latency'] for df in load_results('e2_10x', 'PPO')]
thresh_p95 = [summarise(df)['p95_latency'] for df in load_results('e2_10x', 'Threshold')]

t_stat, p_value = stats.ttest_rel(ppo_p95, thresh_p95)
cohens_d = (np.mean(ppo_p95) - np.mean(thresh_p95)) / np.std(
    [a-b for a,b in zip(ppo_p95, thresh_p95)], ddof=1)
print(f"t={t_stat:.3f}, p={p_value:.4f}, d={cohens_d:.3f}")
```

**Step 6: Aggregate results table**

| Algorithm | Mean P95 (ms) | Std | 95% CI Low | 95% CI High | t vs PPO | p-value | Cohen's d |
|-----------|--------------|-----|------------|-------------|----------|---------|-----------|
| PPO | ??? | ??? | ??? | ??? | — | — | — |
| DQN | ??? | ??? | ??? | ??? | ??? | ??? | ??? |
| RoundRobin | ??? | ??? | ??? | ??? | ??? | ??? | ??? |
| LeastConn | ??? | ??? | ??? | ??? | ??? | ??? | ??? |
| Threshold | ??? | ??? | ??? | ??? | ??? | ??? | ??? |

*All ??? values to be filled with actual experimental results. NEVER fill with expected/hypothetical values.*

**Step 7: Ablation table (E10)**

| Reward Variant | Mean P95 (ms) | Mean CPU% | SLA Violation% |
|----------------|--------------|-----------|----------------|
| Full (default) | ??? | ??? | ??? |
| No R_lat | ??? | ??? | ??? |
| No R_util | ??? | ??? | ??? |
| No R_tput | ??? | ??? | ??? |
| No R_sla | ??? | ??? | ??? |

---

### T7.3 — Figures and Plots [LOCAL ONLY][PARALLEL with T7.2]

**Objective:** Create all paper-quality figures.

**Required figures:**

1. **Figure 1 — System Architecture Diagram** (draw.io or matplotlib, vector format)
2. **Figure 2 — Traffic Profile** (matplotlib, shows warm-up/burst/peak/cooldown phases for 10× and 50×)
3. **Figure 3 — Reward Convergence Curves** (PPO and DQN, episode reward vs training steps, with shaded 95% CI)
4. **Figure 4 — Latency CDF** (P5–P99 latency CDF for all 6 algorithms at 10× burst)
5. **Figure 5 — P95 Latency Comparison Bar Chart** (E1, E2, E3 results, grouped by algorithm)
6. **Figure 6 — CPU Utilisation Timeline** (time-series CPU during 10× burst, one line per instance)
7. **Figure 7 — Scaling Reaction Time** (timeline bar: PPO proactive vs Threshold reactive, E7)
8. **Figure 8 — Ablation Study** (E10 results, bar chart for each reward variant)

**Tools:** matplotlib 3.8 + seaborn 0.13 (LaTeX font rendering: `plt.rcParams.update({'text.usetex': True})`)

**Format requirements for paper:**
- All figures: PDF vector format (not PNG — PDF preserves crispness in IEEE/ACM papers)
- Font size ≥ 8pt in figures (IEEE minimum)
- Figure dimensions: single-column (3.5 in wide) or double-column (7.16 in wide)
- Save to `experiments/analysis/figures/`

---

### T7.4 — Reproducibility Verification [CRITICAL PATH]

**Objective:** Confirm a fresh run with the documented seeds produces results within ±5% of reported values.

**Steps:**
1. On a different team member's laptop: clone repo, recreate conda env, load pre-trained model.
2. Run E2 with seeds 132–133 (2 repetitions only for verification).
3. Compare P95 latency to recorded results — must be within ±5%.
4. Document any deviations in `decisions/reproducibility_report.md`.

**Acceptance Criteria:** Both verification runs produce P95 latency within ±5% of the values recorded during original experiments.

---

## PHASE 8 — CONFERENCE PAPER PREPARATION

**Duration:** 7–10 days | **[LOCAL ONLY]**

---

### T8.1 — Paper Outline and Section Assignment [CRITICAL PATH]

**Objective:** Assign paper sections to team members, set deadlines, prevent overlapping work.

**Recommended section assignments:**

| Section | Owner | Length | Deadline |
|---------|-------|--------|----------|
| Abstract | Devkanti (lead) | 250 words | After all experiments complete |
| 1. Introduction | Devkanti | 1 page | After T7.2 |
| 2. Related Work | Agrima | 1.5 pages | Before T8.3 |
| 3. System Design | All (Agrima drafts) | 1.5 pages | Before T8.3 |
| 4. Methodology | Devkanti + Mohar | 2 pages | Before T8.3 |
| 5. Experimental Setup | Agrima | 1 page | After Phase 6 |
| 6. Results | All (Devkanti leads) | 2 pages | After T7.3 |
| 7. Discussion | All | 0.5 page | After T7.3 |
| 8. Conclusion | Mohar | 0.3 page | Last |
| References | Agrima | 15–20 refs | Ongoing |

**Target conference format:** IEEE 2-column format (IEEEtran). Download template from ieeetran.com.  
**Target length:** 6–8 pages (typical for IEEE Cloud/ICDCS workshops).

---

### T8.2 — Algorithm Pseudocode [PARALLEL]

**Objective:** Write clean, paper-ready pseudocode for PPO training and the full inference loop.

**PPO Training pseudocode (LaTeX algorithm2e format):**
```
Algorithm 1: PPO Training for FlashBalanceAI
Input: FlashSaleEnv E, config C
Initialize: PPO actor π_θ, critic V_φ, rollout buffer B
for episode = 1 to N_episodes do
  obs ← E.reset(seed)
  for step = 0 to K do
    a_t ∼ π_θ(·|s_t)          // Sample action from policy
    s_{t+1}, r_t ← E.step(a_t)  // Execute action
    B.store(s_t, a_t, r_t, s_{t+1})
  end for
  Compute GAE advantages: Â_t from B
  for epoch = 0 to n_epochs do
    for mini-batch m in B do
      r_t(θ) ← π_θ(a_t|s_t)/π_θ_old(a_t|s_t)
      L^CLIP ← E[min(r_t(θ)Â_t, clip(r_t(θ),1-ε,1+ε)Â_t)]
      Update θ, φ via Adam on −L^CLIP + c₁L^VF − c₂S[π_θ]
    end for
  end for
  if eval_reward plateaus for 500k steps: break
end for
Save π_θ to S3
```

---

### T8.3 — Related Work Section [PARALLEL]

**Objective:** Write the Related Work section citing the 10 surveyed papers plus 5–8 additional papers.

**Mandatory citations (from PDFs):**

From Devkanti's survey:
1. Zhou et al. (2024) — DRL survey, Springer AI Review
2. Hu et al. (2025) — D4PG network routing, IEEE Trans. Ind. Informatics
3. Chen et al. (2026) — MDP DRL for e-commerce, Elsevier ESWA
4. Femminella & Reali (2024) — PPO for serverless HPA, MDPI Computers
5. Funika et al. (2023) — PPO-LSTM heterogeneous cloud, Springer J. Supercomputing

From Agrima's survey:
6. Jian et al. (2024) — DRS K8s scheduler, Wiley SPE
7. Yamsani & Chenna Reddy (2026) — SLA-DRL, Nature Scientific Reports
8. Zhou et al. (2024) — TS-SDTRA PPO+DQN, Elsevier Computer Communications
9. Li et al. (2024) — Distributional RL batch scheduling, IEEE TPDS

From Mohar's survey:
10. MAS-H2 paper — Hierarchical Multi-Agent autoscaling, Kubernetes cluster metrics

**Additional papers recommended for strong related work:**

| Paper | Why Needed |
|-------|-----------|
| Mnih et al. (2015) — DQN (Nature) | Foundational citation for DQN baseline |
| Schulman et al. (2017) — PPO (arXiv) | Foundational citation for PPO |
| Sutton & Barto (2018) — RL textbook | General RL background |
| Busta et al. (2023/2024) — AWS Auto Scaling production analysis | Grounds AWS-specific claims |
| A flash-sale/thundering-herd paper from e-commerce literature | Establishes flash-sale problem severity |

**⚠️ INTEGRITY NOTE:** Only cite papers you have actually read. Do not fabricate DOIs, volume numbers, or page ranges. Verify all citations against Google Scholar before submission.

---

### T8.4 — Results Section and Tables [CRITICAL PATH after T7.2]

**Main results table (Table 1):**

```
Table 1: Performance comparison under 10× flash-sale burst traffic.
Results are mean ± std over 5 independent runs (test seeds 132-136).
† denotes statistically significant improvement over PPO at α=0.05 (paired t-test).

Algorithm   | Mean Lat (ms) | P95 (ms)    | P99 (ms)    | Fail%  | CPU%   | Cost/1M
------------|---------------|-------------|-------------|--------|--------|--------
PPO (ours)  | [fill]±[fill] | [fill]±[fill] | [fill]±[fill] | [fill] | [fill] | [fill]
DQN         | [fill]±[fill] | [fill]±[fill]†| ...         | ...    | ...    | ...
RoundRobin  | [fill]±[fill] | [fill]±[fill]†| ...         | ...    | ...    | ...
WRR         | ...           | ...          | ...         | ...    | ...    | ...
LeastConn   | ...           | ...          | ...         | ...    | ...    | ...
Threshold   | [fill]±[fill] | [fill]±[fill]†| ...         | ...    | ...    | ...
```

**All [fill] cells must contain actual experimental results, not hypothetical values.**

---

### T8.5 — Limitations, Threats to Validity, Future Work [PARALLEL]

The limitations section must honestly address:
1. Simulated environment for training (sim-to-real gap)
2. Synthetic traffic (not real production data)
3. Scale limited to 8 t2.micro instances (results may not scale linearly)
4. Discrete routing approximation (100ms time windows, not per-request)
5. Single region, single AZ design
6. No database layer in backend (real e-commerce has DB bottlenecks)

Do NOT claim these are minor limitations without evidence. If the paper cannot address them, the limitations section must state them clearly.

---

## PHASE 9 — FINAL DEPLOYMENT AND DEMO

**Duration:** 2–3 days | **[COSTS MONEY — minimal]**

---

### T9.1 — Read-Only Public Demo [OPTIONAL][COSTS MONEY]

**Objective:** Public URL showing live routing decisions (if safely and cheaply achievable).

**Decision:** Only proceed with public demo if:
- Billing is under $15 total at this point (leave budget for demo)
- The system has been fully validated in Phase 7
- Teardown procedure is documented and tested

**Minimal demo setup:**
1. Keep 2 EC2 instances running (desired-capacity=2).
2. Run 1× baseline traffic only (no burst — avoids cost spike).
3. Host a read-only Dash dashboard on EC2 showing:
   - Live request routing decisions (which instance received last request)
   - Real-time CPU per instance
   - Running P95 latency chart (last 60 seconds)
4. Share URL with Dr. Priya V and team.

**Cost:** 2 EC2 t2.micro + 1 ALB running for demo duration (e.g., 4 hours during demo session) → ~$0.15.

---

### T9.2 — Monitoring Dashboard [PARALLEL]

**Objective:** Python Dash dashboard showing live experiment metrics.

**Steps:**
1. Create `src/metrics/visualiser.py`:
```python
import dash, boto3
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd, json, boto3

app = dash.Dash(__name__)
s3 = boto3.client('s3')
BUCKET = 'flashbalanceai-{ACCOUNT_ID}'

app.layout = html.Div([
    html.H1("FlashBalanceAI — Live Monitor"),
    dcc.Graph(id='latency-graph'),
    dcc.Graph(id='cpu-graph'),
    dcc.Graph(id='routing-histogram'),
    dcc.Interval(id='interval', interval=5000, n_intervals=0)  # 5s refresh
])

@app.callback(
    [Output('latency-graph', 'figure'),
     Output('cpu-graph', 'figure'),
     Output('routing-histogram', 'figure')],
    [Input('interval', 'n_intervals')]
)
def update_graphs(n):
    # Load state from S3
    try:
        obj = s3.get_object(Bucket=BUCKET, Key='state/current.json')
        state = json.loads(obj['Body'].read())['state']
    except:
        state = [0.1] * 23

    cpu_vals = state[:4]
    resp_vals = [v * 500 for v in state[12:16]]  # resp_ema → ms

    lat_fig = go.Figure(data=[go.Bar(y=resp_vals, x=['S1','S2','S3','S4'])])
    lat_fig.update_layout(title='Response Time EMA per Instance (ms)')

    cpu_fig = go.Figure(data=[go.Bar(y=[v*100 for v in cpu_vals],
                                      x=['S1','S2','S3','S4'])])
    cpu_fig.update_layout(title='CPU Utilisation per Instance (%)')

    hist_fig = go.Figure()
    hist_fig.update_layout(title='Routing Distribution (last 100 decisions)')

    return lat_fig, cpu_fig, hist_fig

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
```
2. Run locally: `python src/metrics/visualiser.py` → open `http://localhost:8050`.

---

### T9.3 — Teardown Procedure [CRITICAL PATH — must execute after all experiments]

**Objective:** Delete all chargeable AWS resources to prevent unexpected billing.

**COMPLETE TEARDOWN SCRIPT (`scripts/teardown.sh`):**
```bash
#!/bin/bash
set -e
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

echo "=== STEP 1: Set ASG to 0 ==="
aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name FlashBalanceAI-ASG \
    --desired-capacity 0 --min-size 0

echo "=== STEP 2: Wait for instances to terminate ==="
sleep 120

echo "=== STEP 3: Delete ASG ==="
aws autoscaling delete-auto-scaling-group \
    --auto-scaling-group-name FlashBalanceAI-ASG --force-delete

echo "=== STEP 4: Delete ALB ==="
ALB_ARN=$(aws elbv2 describe-load-balancers \
    --names FlashBalanceAI-ALB \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text)
aws elbv2 delete-load-balancer --load-balancer-arn $ALB_ARN

echo "=== STEP 5: Delete Target Group ==="
TG_ARN=$(aws elbv2 describe-target-groups \
    --names FlashBalanceAI-TG \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
aws elbv2 delete-target-group --target-group-arn $TG_ARN

echo "=== STEP 6: Delete Lambda functions ==="
aws lambda delete-function --function-name FlashBalanceAI-StateCollector
aws lambda delete-function --function-name FlashBalanceAI-ScalingTrigger

echo "=== STEP 7: Stop SageMaker (if running) ==="
aws sagemaker stop-notebook-instance \
    --notebook-instance-name FlashBalanceAI-Training 2>/dev/null || true

echo "=== STEP 8: Delete DynamoDB table ==="
aws dynamodb delete-table --table-name routing_decisions

echo "=== STEP 9: Empty and delete S3 bucket (CAUTION: deletes all results) ==="
echo "WARNING: This deletes all experiment results. Ensure S3 data is backed up locally."
read -p "Confirm S3 deletion? (yes/no): " confirm
if [ "$confirm" == "yes" ]; then
    aws s3 rb s3://flashbalanceai-${ACCOUNT_ID} --force
fi

echo "=== STEP 10: Delete launch template ==="
aws ec2 delete-launch-template --launch-template-name FlashBalanceAI-Backend

echo "=== TEARDOWN COMPLETE ==="
echo "Verify $0 in AWS Console: EC2, ALB, Lambda, DynamoDB, SageMaker, S3"
```

**⚠️ DO NOT DELETE S3 until all experimental results are downloaded locally AND committed to git.**

---

## RESEARCH INTEGRITY APPENDIX

### Contradictions Between the Three PDFs

| Contradiction | Devkanti | Agrima | Mohar | Resolution |
|---------------|----------|--------|-------|------------|
| Primary algorithm | PPO | DQN vs PPO comparison | DDPG + H-MAS | **PPO primary, DQN baseline, DDPG/H-MAS = related work** |
| Dataset | Flash-sale traces + Google/NASA | Synthetic FlashSale-Synthetic v1.0 | Alibaba Cluster Trace | **Synthetic primary; Alibaba supplementary only** |
| AWS services | ELB + ASG + CloudWatch + Lambda + API Gateway | EC2 + ALB + S3 + SageMaker + DynamoDB + Cognito + CloudFront | EC2 + SageMaker + Lambda + CloudWatch + ALB | **Use union minus unnecessary services. Cognito and CloudFront removed (not needed for experiments).** |
| Novelty scope | "No existing work" on AWS DRL LB for flash sales | Within reviewed literature gap | Dual-layered H-MAS + DDPG novelty | **Restate as: "within surveyed literature (10 papers), no work combines..."** |
| Expected improvements | 30% P95 reduction, 85% utilisation, 20–25% cost saving | Not quantified | Zero-downtime scaling | **Mark all as EXPECTED until experimentally validated** |

### Unsupported Claims to Remove or Verify

1. ❌ "30% reduction in P95 response time" — REMOVE from PRD/paper until measured.
2. ❌ "85%+ resource utilization" — REMOVE until measured.
3. ❌ "20–25% cost savings" — REMOVE until measured.
4. ❌ "First DRL load balancer on AWS" — WEAKEN to "within our surveyed literature..."
5. ❌ "Zero-downtime scaling" — Unverifiable without production deployment.
6. ⚠️ Alibaba Cluster Trace cited as primary dataset for flash-sale validation (Mohar) — **INCORRECT**. Alibaba trace is batch jobs. Correct to: supplementary only.
7. ⚠️ NASA/ClarkNet logs from 1995 cited by Devkanti (P5) as a limitation in the source paper — do not use these datasets in our project.

### Additional Papers Recommended for Literature Review

| Paper | Venue | Why Needed |
|-------|-------|-----------|
| Mnih et al., "Human-level control through DRL" (2015) | Nature | DQN foundational citation |
| Schulman et al., "Proximal Policy Optimization" (2017) | arXiv | PPO foundational citation |
| Sutton & Barto, "RL: An Introduction" (2018) | MIT Press | RL background |
| "Thundering Herd problem" analysis in distributed systems | Any 2022+ | Establishes flash-sale problem |
| AWS Auto Scaling official documentation / blog post | AWS | Grounds AWS-specific claims |
| Any paper from Alibaba 11.11 / Amazon Prime Day traffic analysis | Industry/2023+ | Real e-commerce traffic evidence |

---

## TIMELINE

### Hours and Calendar Days Per Task

| Task | Owner Suggestion | Dependencies | Hours | Calendar Days | AWS Cost Risk | Deliverable |
|------|-----------------|--------------|-------|---------------|---------------|-------------|
| T0.1 ADR | All (Devkanti leads) | None | 3h | Day 1 | None | ADR-001.md |
| T0.2 Repo structure | Agrima | T0.1 | 2h | Day 1 | None | Populated repo |
| T1.1 Python env | All | T0.2 | 1h | Day 2 | None | environment.yml |
| T1.2 Flask backend | Mohar | T1.1 | 2h | Day 2–3 | None | src/backend/app.py |
| T1.3 Gymnasium env | Devkanti | T1.1, T2.1 | 6h | Day 3–4 | None | flash_sale_env.py |
| T1.4 Baselines | Agrima | T1.1 | 3h | Day 2–3 | None | 4 baseline classes |
| T1.5 Metrics | All | T1.1 | 2h | Day 2–3 | None | collector.py |
| T2.1 Traffic generator | Devkanti | T1.1 | 4h | Day 3–4 | None | traffic_generator.py |
| T2.2 JMeter plans | Mohar | T1.2 | 3h | Day 4–5 | None | .jmx test plans |
| T2.3 Alibaba dataset | Agrima | T1.1 | 2h | Day 4–5 | None | alibaba_processed.csv |
| T3.1 PPO training | Devkanti | T1.3, T2.1 | 9h | Day 5–7 | $0–0.50 (SageMaker) | ppo_flash_v1.zip |
| T3.2 DQN training | Agrima | T1.3, T2.1 | 9h | Day 5–7 | $0–0.50 | dqn_flash_v1.zip |
| T3.3 Local evaluation | All | T3.1, T3.2 | 4h | Day 8 | None | Preliminary results |
| T4.1 AWS safety setup | Devkanti | None | 2h | Day 9 | Free | Billing alarms |
| T4.2 IAM roles | Devkanti | T4.1 | 2h | Day 9 | Free | IAM roles |
| T4.3 S3 bucket | Agrima | T4.2, T3.1 | 1h | Day 9 | Free | S3 bucket |
| T4.4 EC2 + ASG | Mohar | T4.2, T1.2 | 4h | Day 10–11 | ⚠️ Medium | Running EC2 pool |
| T4.5 ALB | Mohar | T4.4 | 2h | Day 11 | ⚠️ High ($/hr) | ALB DNS |
| T4.6 DynamoDB | Agrima | T4.1 | 0.5h | Day 9 | Free | routing_decisions |
| T4.7 SageMaker (optional) | Any | T4.2 | 2h | Day 10 | ⚠️ Low ($0.23) | model in S3 |
| T5.1 State collector Lambda | Agrima | T4.3, T4.4, T4.5 | 5h | Day 12–13 | ⚠️ Low | state/current.json |
| T5.2 PPO inference | Devkanti | T5.1, T4.3 | 6h | Day 13–14 | ⚠️ Low | Routing decisions |
| T5.3 Scaling trigger | Mohar | T5.1, T4.4 | 3h | Day 13 | Free | ASG auto-scaling |
| T5.4 Custom CW metrics | Mohar | T5.1, T1.2 | 2h | Day 12 | Free | CW namespace |
| T5.5 Integration test | All | T5.1–T5.4 | 3h | Day 14–15 | ⚠️ Low ($0.05) | E2E verified |
| E1 Baseline traffic | Agrima | T5.5 | 4h | Day 16 | ⚠️ Medium ($0.50) | E1 results CSVs |
| E2 10× spike | All | E1 | 6h | Day 16–17 | ⚠️ Medium ($0.50) | E2 results CSVs |
| E3 50× spike | All | E2 | 5h | Day 17–18 | ⚠️ Medium ($0.60) | E3 results CSVs |
| E4 100× (conditional) | Devkanti | E3 | 3h | Day 18 | ⚠️ Medium ($0.30) | E4 results |
| E5 Repeated bursts | Mohar | E2 | 5h | Day 18–19 | ⚠️ Medium ($1.50) | E5 results |
| E6 Noisy spikes | Agrima | E2 | 4h | Day 18–19 | ⚠️ Low ($0.50) | E6 results |
| E7 Scaling response | All | E2 | 4h | Day 19–20 | ⚠️ Low ($0.50) | E7 results |
| E8 Cost/perf trade-off | Devkanti | E2 | 8h | Day 20–22 | ⚠️ Medium ($1.00) | E8 results |
| E9 DRL comparison | All | E2, E3 | 2h | Day 17 (reuse data) | None | E9 analysis |
| E10 Ablation study | All | E2 + retraining | 10h | Day 21–24 | ⚠️ Medium ($1.00) | E10 results |
| T7.1 Data collection | Agrima | All E-tasks | 3h | Day 25 | None | Raw CSVs in S3 |
| T7.2 Statistical analysis | Devkanti | T7.1 | 6h | Day 25–26 | None | Stats table |
| T7.3 Figures | Mohar | T7.1 | 6h | Day 25–27 | None | PDF figures |
| T7.4 Reproducibility check | Any non-lead | T7.2 | 3h | Day 26 | None | Repro report |
| T8.1 Paper outline | All | T7.2 | 2h | Day 28 | None | Section assignments |
| T8.2 Pseudocode | Devkanti | T3.1 | 2h | Day 27–28 | None | Algorithm blocks |
| T8.3 Related work | Agrima | T8.1 | 5h | Day 28–30 | None | Related work draft |
| T8.4 Results section | Devkanti | T7.3 | 4h | Day 28–30 | None | Results draft |
| T8.5 Limitations + discussion | Mohar + all | T8.4 | 3h | Day 30–31 | None | Discussion draft |
| Paper review + revisions | All | T8.3–T8.5 | 8h | Day 31–35 | None | Submission-ready draft |
| T9.1 Demo (optional) | Any | T5.5 | 3h | Day 35 | ⚠️ Low ($0.15) | Live demo URL |
| T9.2 Dashboard | Agrima | T5.4 | 4h | Day 34–35 | None | Dashboard running |
| T9.3 Teardown | Devkanti | After experiments | 1h | Day 36 | None | All resources deleted |

**Total estimated hours:** ~170–200 engineering hours  
**Total AWS cost:** $15–35 (conservative to realistic)

---

### Calendar Timeline Summary (4 Students Working in Parallel)

```
Week 1 (Days 1–7):   Phase 0 + Phase 1 + Phase 2 (all local, no AWS cost)
Week 2 (Days 8–14):  Phase 3 (training) + Phase 4 (AWS setup)
Week 3 (Days 15–22): Phase 5 (integration) + Phase 6 (experiments E1–E7)
Week 4 (Days 23–28): Phase 6 cont. (E8–E10) + Phase 7 (stats + figures)
Week 5 (Days 29–35): Phase 8 (paper writing) + Phase 9 (demo + teardown)
```

**Total calendar time with 4 students parallel: ~5 weeks (35 days)**

---

### Critical Path

```
T0.1 → T0.2 → T1.1 → T1.3 → T2.1 [environment ready]
                              ↓
                         T3.1 [PPO trained]
                              ↓
                         T3.3 [local eval gate check]
                              ↓
T4.1 → T4.2 → T4.4 → T4.5 [AWS infrastructure]
                         ↓
                    T5.1 → T5.5 [integration verified]
                         ↓
                    E1 → E2 → E3 [primary experiments]
                         ↓
                    T7.2 → T7.3 [statistics + figures]
                         ↓
                    T8.4 → paper submission
```

---

### Fastest Path to Working Prototype

**Target: 10 days**
1. Days 1–2: T0.1, T0.2, T1.1 (env setup)
2. Days 2–4: T1.2, T1.3, T2.1 (backend + Gymnasium env + traffic generator)
3. Days 4–7: T3.1 (PPO training — run overnight)
4. Days 7–8: T4.1, T4.2, T4.3, T4.4, T4.5 (AWS stack)
5. Days 8–10: T5.1, T5.2, T5.5 (integration + basic inference running)

**Prototype definition:** PPO model routing traffic across 4 EC2 instances behind ALB, with CloudWatch feeding state, Lambda updating routing every 30s.

---

### Fastest Path to Publication-Ready Experimental Evidence

**Target: 5 weeks (35 days)**  
Cannot be shortened below 5 weeks without:
- Skipping E5/E6/E7/E8/E10 (weakens paper)
- Reducing repetitions below 5 (weakens statistical claims)
- Skipping reproducibility verification (T7.4)

The bottleneck is PPO/DQN training time (~4–5 hours each) and the minimum 5-repetition experimental runs with ALB active.

---

### Tasks That MUST NOT Start Until Validation Succeeds

| Task | Gate Condition |
|------|---------------|
| T3.2 (DQN training) | Must use same FlashSaleEnv version as T3.1 — freeze environment before starting |
| T4.4 (EC2 launch) | T4.1 billing alarms must be set first |
| T5.5 (integration test) | All of T5.1–T5.4 must be individually verified first |
| E2 (10× experiments) | E1 must complete with all 6 algorithms; no NaN metrics |
| E3 (50× experiments) | E2 must show PPO outperforming RR by > 10% P95 |
| E4 (100× experiments) | E3 must show infrastructure can handle 50× without all instances crashing |
| E10 (ablation) | E2 reward ablation retraining must converge before evaluating |
| T8.4 (results section) | All experiments + T7.2 statistical analysis must be complete |
| Paper submission | T7.4 reproducibility check must pass |

---

## FINAL RECOMMENDED ARCHITECTURE

### Exact Implementation Decision

**DRL Algorithm:** PPO (Proximal Policy Optimization) via Stable-Baselines3 v2.3.0  
**Architecture:** 2-layer MLP actor-critic, [64, 64] hidden units, discrete action space N=4  
**Baseline DRL:** DQN (identical architecture, SB3 implementation)  
**Non-DRL Baselines:** Round Robin, Weighted Round Robin, Least Connections, Threshold-based CloudWatch autoscaling  

**AWS Services (minimum required, all others excluded):**
- EC2 t2.micro × 4 (backend) + 1 (inference server)
- ALB × 1
- Lambda × 3 (state collector, scaling trigger, lightweight coordinator)
- CloudWatch (custom metrics, alarms)
- S3 × 1 bucket
- DynamoDB × 1 table
- API Gateway × 1 REST API
- ASG × 1
- IAM × 2 roles

**Services explicitly excluded:** SageMaker (local training preferred), RDS (no database layer needed), EFS (no shared filesystem needed), SQS (Lambda + S3 is sufficient for state passing), Cognito (no authentication needed for experiments), CloudFront (not needed for backend experiments).

**Dataset:** FlashSale-Traffic-Synthetic v1.0 (custom generator, `src/traffic/traffic_generator.py`, seeds documented). Alibaba trace supplementary only.

**Experiments:** E1–E10 as defined, minimum 5 repetitions each, 95% CI, paired t-test for primary claims.

**Why this architecture wins:**
1. PPO's clipped objective makes it stable under the high reward variance of flash-sale burst traffic — no equivalent guarantee from DQN or DDPG
2. AWS-native deployment fills the verified Gap G2 from all 10 surveyed papers
3. DQN comparison fills Gap G3 (only one prior paper does this, in SDN sim — not AWS, not flash-sale)
4. Synthetic traffic generator is reproducible, calibrated, and avoids legal/privacy issues with real data
5. Total AWS budget is $15–35 — achievable with a student account within 5 weeks
