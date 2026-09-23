# GitHub Project Plan — FlashBalanceAI
## BCSE355L Cloud Architecture Design | Deep Reinforcement Learning-Based Adaptive Cloud Load Balancing

**Project:** FlashBalanceAI: PPO Framework for Adaptive Cloud Load Balancing under Flash-Sale Burst Traffic  
**Repository:** `DRL_Cloud_Load_Balancing_Cloud_Project_2026`  
**Team:** Devkanti Sarkar (24BIT0162) · Agrima Gupta (24BIT0253) · Mohar Gorai · [4th member]  
**Branch Model:** `main` ← `develop` ← `feature/*` (already implemented — not redesigned here)  
**Document Version:** 1.1 | **Last Revised:** 2026-08-20 (cost-optimisation update — see ADR-002)

---

## Table of Contents

1. [Label Taxonomy](#1-label-taxonomy)
2. [Milestone Overview](#2-milestone-overview)
3. [Phase 0 — Design Lock & Repository Setup](#phase-0--design-lock--repository-setup)
4. [Phase 1 — Local Development Foundation](#phase-1--local-development-foundation)
5. [Phase 2 — Dataset & Traffic Generation](#phase-2--dataset--traffic-generation)
6. [Phase 3 — DRL Implementation](#phase-3--drl-implementation)
7. [Phase 4 — AWS Infrastructure Deployment](#phase-4--aws-infrastructure-deployment)
8. [Phase 5 — AWS Integration & End-to-End Pipeline](#phase-5--aws-integration--end-to-end-pipeline)
9. [Phase 6 — Experimental Campaign](#phase-6--experimental-campaign)
10. [Phase 7 — Results & Statistical Validation](#phase-7--results--statistical-validation)
11. [Phase 8 — Conference Paper Preparation](#phase-8--conference-paper-preparation)
12. [Phase 9 — Demo, Reproducibility & Teardown](#phase-9--demo-reproducibility--teardown)
13. [Critical Path](#13-critical-path)
14. [Parallelisable Work](#14-parallelisable-work)
15. [Release Checkpoints](#15-release-checkpoints)
16. [Contribution Tracking](#16-contribution-tracking)

---

## 1. Label Taxonomy

All GitHub Issues must carry at least one **type** label, one **phase** label, and one **component** label. Additional modifier labels are optional.

### Type Labels

| Label | Colour | Meaning |
|-------|--------|---------|
| `type:research` | `#0075ca` | Literature review, gap analysis, hypothesis formation |
| `type:implementation` | `#e4e669` | Source-code tasks (src/, tests/, configs/) |
| `type:experiment` | `#d93f0b` | Requires running code against AWS or local env |
| `type:analysis` | `#f9d0c4` | Statistical analysis, figure generation, notebook work |
| `type:documentation` | `#0e8a16` | README, ADR, PRD updates, paper sections |
| `type:infrastructure` | `#5319e7` | AWS setup, IAM, deploy scripts |
| `type:testing` | `#fbca04` | Unit tests, integration tests, reproducibility verification |
| `type:reproducibility` | `#bfd4f2` | Seed documentation, environment pinning, verification runs |

### Phase Labels

| Label | Meaning |
|-------|---------|
| `phase:0-design` | Architecture lock, ADR, repo scaffold |
| `phase:1-local-dev` | Local Python development (no AWS) |
| `phase:2-dataset` | Traffic generator, JMeter, Alibaba trace |
| `phase:3-drl` | PPO/DQN agent implementation and training |
| `phase:4-aws-deploy` | AWS resource provisioning |
| `phase:5-integration` | Lambda + EC2 inference integration, end-to-end pipeline |
| `phase:6-experiments` | Experiment runs E1–E10 |
| `phase:7-results` | Statistical analysis, figures |
| `phase:8-paper` | Conference paper writing |
| `phase:9-demo` | Demo, teardown, final release |

### Component Labels

| Label | Meaning |
|-------|---------|
| `component:environment` | `FlashSaleEnv` (Gymnasium) |
| `component:ppo-agent` | PPO training and inference code |
| `component:dqn-agent` | DQN training and inference code |
| `component:baselines` | RR, WRR, LC, Threshold baseline implementations |
| `component:traffic-gen` | Synthetic traffic generator and JMeter plans |
| `component:backend` | Flask EC2 backend server |
| `component:aws-lambda` | Lambda functions (state collector, inference coordinator, scaling trigger) |
| `component:aws-infra` | EC2, ALB, ASG, CloudWatch, S3, DynamoDB — stop/delete between sessions |
| `component:inference-server` | EC2 PPO inference server (`src/aws/inference_server.py`) |
| `component:metrics` | MetricsCollector and Dash visualiser |
| `component:paper` | Conference paper manuscript sections |
| `component:dataset` | Alibaba trace, synthetic dataset management |

### Modifier Labels

| Label | Meaning |
|-------|---------|
| `priority:critical-path` | Blocks downstream work — must not slip |
| `priority:high` | Should not slip more than 1 day |
| `priority:normal` | Default priority |
| `costs:aws-billing` | Will incur AWS charges when running |
| `costs:free` | No AWS billing |
| `costs:delete-after-use` | AWS resource must be deleted (not just stopped) after each session |
| `gate:required` | This issue is a gate check — work downstream cannot start until it is closed |
| `blocked` | Cannot proceed; blocking issue must be linked |
| `good-first-issue` | Suitable as a first task for a new contributor |

---

## 2. Milestone Overview

| # | Milestone | Phase(s) | Calendar Target | Exit Gate |
|---|-----------|----------|-----------------|-----------|
| M0 | Design Lock & Repo Scaffold | 0 | Week 1 Day 1–2 | ADR-001.md signed by all 4 members; ADR-002 acknowledged; repo structure verified |
| M1 | Local Development Foundation | 1 | Week 1 Day 2–7 | All unit tests pass; Gymnasium env validated |
| M2 | Dataset & Traffic Generation | 2 | Week 1 Day 3–7 | Traffic generator reproducible; JMeter plans run locally |
| M3 | DRL Agents Trained & Locally Validated | 3 | Week 2 Day 5–8 | PPO reward converges; local eval gate check passes |
| M4 | AWS Infrastructure Deployed | 4 | Week 2 Day 9–11 | All AWS resources live; billing alarm set; no quota errors |
| M5 | End-to-End Pipeline Operational | 5 | Week 3 Day 12–15 | Integration test T5.5 passes all 6 checks |
| M6 | Experimental Campaign Complete | 6 | Week 3–4 Day 16–24 | E1–E10 raw data committed; no missing repetitions |
| M7 | Statistical Validation & Figures | 7 | Week 4 Day 25–27 | Stats table complete; all 8 figures generated as PDF |
| M8 | Conference Paper Draft | 8 | Week 5 Day 28–31 | All sections drafted; reproducibility check passed |
| M9 | Final Release & Demo | 9 | Week 5 Day 32–35 | v1.0 GitHub release tagged; AWS teardown complete |

**Milestone dependencies:** M0 → M1, M0 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9

---

## Phase 0 — Design Lock & Repository Setup

**Milestone:** M0 | **Duration:** 1–2 days | **All costs: Free**

**Objective:** Lock all architecture decisions and create the canonical repository structure.

**Deliverables:** `decisions/ADR-001.md` (signed), `decisions/ADR-002.md` (acknowledged), fully scaffolded directories, `.gitignore`, branch structure.

**Exit Criteria:** All 4 members signed ADR-001. ADR-002 cost review acknowledged. `git ls-tree -r --name-only HEAD` shows all required directories.

---

### Issue #1 — Write Architecture Decision Record (ADR-001) and Acknowledge Cost Review (ADR-002)

**Labels:** `type:documentation` `phase:0-design` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar (lead) — all 4 members must sign off  
**Estimated Effort:** 3 hours (meeting + write-up)  
**Depends On:** Nothing  
**Blocks:** All other issues

**Description:**  
Hold a synchronous meeting to resolve design disagreements. Document all locked decisions in `decisions/ADR-001.md`. Acknowledge the cost-optimisation review in `decisions/ADR-002.md`.

**Key decisions to confirm:**
- Primary DRL: PPO via SB3 v2.3.0
- Baseline DRL: DQN (identical architecture)
- AWS: ALB (required, delete between sessions), Lambda × 3 (free tier), EC2 × 5 t2.micro
- **API Gateway: NOT in primary data flow** — JMeter → ALB DNS directly
- **CloudFront: REMOVED** — Dash dashboard runs locally
- **SageMaker: optional last resort** — local laptop or Google Colab first
- **Custom CloudWatch metrics: 8 only** (within free tier)
- Inference: EC2 inference server, not Lambda (SB3 size limit)
- Budget ceiling: $15 (to confirm)

**Acceptance Criteria:**
- [ ] `decisions/ADR-001.md` exists; all 4 sign-off rows show `[x]`
- [ ] `decisions/ADR-002.md` exists; cost-review acknowledgement ticked by all 4
- [ ] No unresolved design or cost questions remain open

---

### Issue #2 — Scaffold Repository Directory Structure

**Labels:** `type:implementation` `phase:0-design` `priority:critical-path` `costs:free` `good-first-issue`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #1  
**Blocks:** Issues #3–#7

**Description:**  
Create the full canonical directory tree per `IMPLEMENTATION_PLAN_PART1.md` T0.2. Add `decisions/` directory (already exists with ADR-001/ADR-002). Include `src/aws/inference_server.py` and `src/aws/inference_coordinator.py` as placeholder files.

**Acceptance Criteria:**
- [ ] All `src/`, `configs/`, `experiments/`, `notebooks/`, `decisions/` directories present
- [ ] `.gitignore` covers `*.pyc`, `__pycache__/`, `.env`, `*.jtl`, `models/*.zip`, `data/alibaba_trace/`
- [ ] PR merged into `develop` with at least one reviewer

---

## Phase 1 — Local Development Foundation

**Milestone:** M1 | **Duration:** 5–7 days | **All costs: Free (local only)**

---

### Issue #3 — Create Reproducible Python Environment

**Labels:** `type:implementation` `phase:1-local-dev` `type:reproducibility` `priority:critical-path` `costs:free` `good-first-issue`  
**Owner:** All 4 members  
**Estimated Effort:** 1 hour  
**Depends On:** Issue #2  
**Blocks:** Issues #4–#8

**Description:** Create conda env with pinned packages. Python 3.11 (not 3.12 — SB3 2.3.0 compatibility). Verify on all 4 machines.

**Acceptance Criteria:**
- [ ] `environment.yml` and `requirements.txt` committed
- [ ] `python -c "import stable_baselines3; import gymnasium; print('OK')"` succeeds on all 4 machines

---

### Issue #4 — Implement Flask Backend Server (EC2 Simulator)

**Labels:** `type:implementation` `phase:1-local-dev` `component:backend` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #9 (JMeter), Issue #17 (EC2 deployment)

**Description:** `src/backend/app.py` — Flask server with `/health`, `/product/<pid>`, `/metrics` endpoints and configurable `BASE_LATENCY_MS`. Background thread publishes **exactly 2 custom CloudWatch metrics** per instance: `QueueDepth` and `ResponseTimeEMA` (within 10-metric free tier across 4 instances).

**Acceptance Criteria:**
- [ ] All 3 endpoints return HTTP 200 with valid JSON
- [ ] 4 instances start on ports 5001–5004 without conflict
- [ ] Background metrics thread publishes only `QueueDepth` and `ResponseTimeEMA` (not `RequestCount` — that comes from ALB built-in)

---

### Issue #5 — Implement Custom Gymnasium Environment (FlashSaleEnv)

**Labels:** `type:implementation` `phase:1-local-dev` `component:environment` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 6 hours  
**Depends On:** Issue #3, Issue #8  
**Blocks:** Issues #11–#12 (training)

**Description:** `src/environment/flash_sale_env.py` — 23-dim state vector, discrete N=4 action space, four-component reward function. Unit tests: shape, action space, episode completion, reward range.

**Acceptance Criteria:**
- [ ] `pytest src/tests/test_environment.py -v` passes
- [ ] `check_env(FlashSaleEnv())` raises no errors
- [ ] Full episode (7800 steps) completes in < 10s

---

### Issue #6 — Implement Baseline Load Balancers

**Labels:** `type:implementation` `phase:1-local-dev` `component:baselines` `priority:high` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 3 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #12 (local evaluation)

**Description:** `src/baselines/` — RoundRobinLB, WeightedRoundRobinLB, LeastConnectionsLB, ThresholdAutoscaler. All expose `.select(state)` → int in [0, N-1].

**Acceptance Criteria:**
- [ ] `pytest src/tests/test_baselines.py -v` passes
- [ ] All run 7800 steps in FlashSaleEnv without error

---

### Issue #7 — Implement Metrics Collection Module

**Labels:** `type:implementation` `phase:1-local-dev` `component:metrics` `priority:high` `costs:free`  
**Owner:** Any available member  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #12

**Description:** `src/metrics/collector.py` — MetricsCollector with `record_step()`, `compute_summary()` (mean, P95, P99, throughput, failure_rate, mean_cpu, total_reward), `save(path)`.

**Acceptance Criteria:**
- [ ] `pytest src/tests/test_metrics.py -v` passes
- [ ] `save(path)` produces valid CSV readable by pandas

---

## Phase 2 — Dataset & Traffic Generation

**Milestone:** M2 | **Duration:** 3–4 days | **Parallel with Phase 1 after Issue #3**

---

### Issue #8 — Implement Synthetic Flash-Sale Traffic Generator

**Labels:** `type:implementation` `phase:2-dataset` `component:traffic-gen` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 4 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #5, Issues #11–#12

**Description:** `src/traffic/traffic_generator.py` — `TrafficConfig` dataclass, `generate_profile()`, `generate_repeated_bursts()`. Scenarios: e1_baseline (1×), e2_10x, e3_50x, e4_100x, e6_noisy (2×, noise_std=0.40).

**Acceptance Criteria:**
- [ ] All 5 unit tests pass (length, baseline range, burst peak, reproducibility, different seeds)
- [ ] Profile plots in `notebooks/01_traffic_analysis.ipynb` show correct phase transitions

---

### Issue #9 — Create JMeter Load-Injection Test Plans

**Labels:** `type:implementation` `phase:2-dataset` `component:traffic-gen` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 3 hours  
**Depends On:** Issue #4  
**Blocks:** Issues #25–#34 (experiment runs)

**Description:** `src/traffic/jmeter_configs/` — 4 parameterised `.jmx` plans (10×, 50×, repeated bursts, noisy). JMeter targets `{ALB_DNS}` directly via `-Jhost=` property — **no API Gateway in any test plan**.

**Acceptance Criteria:**
- [ ] Each plan completes a 13-min local run against 4 Flask servers on ports 5001–5004
- [ ] No hardcoded host/port; `-Jhost` property used throughout

---

### Issue #10 — Acquire and Preprocess Alibaba Cluster Trace (Supplementary)

**Labels:** `type:research` `phase:2-dataset` `component:dataset` `priority:normal` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #3  
**Blocks:** Nothing critical

**Description:** Download Alibaba Cluster Trace v2018; process to arrival-rate distribution; save `data/alibaba_processed.csv`. Supplementary only — NOT primary evaluation dataset.

**Acceptance Criteria:**
- [ ] Raw trace directory gitignored (not committed)
- [ ] Notebook cell explicitly states: "Alibaba trace = batch jobs, supplementary use only"

---

## Phase 3 — DRL Implementation

**Milestone:** M3 | **Duration:** 7–10 days | **Local only**

---

### Issue #11 — Train PPO Agent

**Labels:** `type:implementation` `phase:3-drl` `component:ppo-agent` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 9 hours (5h impl + 4h training)  
**Depends On:** Issue #5, Issue #8  
**Blocks:** Issue #13 (gate check), Issue #21 (inference server)

**Description:** `src/agents/ppo_agent.py` — SB3 PPO, 2M steps, 4 VecEnvs, EvalCallback, CheckpointCallback. Hyperparams from `configs/ppo_config.yaml`. Training: local laptop first (3–4 hrs), then Google Colab, then SageMaker as last resort.

**Acceptance Criteria:**
- [ ] TensorBoard shows reward converging (screenshot in `decisions/`)
- [ ] `model.predict(obs)` returns int in [0,3] within 5ms
- [ ] Final model saved as `models/ppo/ppo_flash_v1.zip`

---

### Issue #12 — Train DQN Agent

**Labels:** `type:implementation` `phase:3-drl` `component:dqn-agent` `priority:high` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 9 hours (4h impl + 5h training)  
**Depends On:** Issue #5 (env frozen), Issue #11 started  
**Blocks:** Issue #13

**Description:** `src/agents/dqn_agent.py` — SB3 DQN, identical architecture to PPO, same test seeds.

**Acceptance Criteria:**
- [ ] DQN training completes; final model `models/dqn/dqn_flash_v1.zip`
- [ ] Environment version matches PPO training commit exactly

---

### Issue #13 — Local Evaluation Gate Check (All 6 Algorithms)

**Labels:** `type:experiment` `phase:3-drl` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** All 4 members  
**Estimated Effort:** 4 hours  
**Depends On:** Issues #11, #12, #6, #7  
**Blocks:** Issue #14 (AWS deployment)

**Description:** Run PPO, DQN, RR, WRR, LC, Threshold on test seeds 127–131 in `FlashSaleEnv`. Gate pass: PPO ≥ 10% P95 improvement over RR; no NaN; DQN differs from random.

**Acceptance Criteria:**
- [ ] All gate conditions pass
- [ ] Preliminary results table committed to `experiments/analysis/`
- [ ] Team explicitly agrees in issue comment that Phase 4 may begin

---

## Phase 4 — AWS Infrastructure Deployment

**Milestone:** M4 | **Duration:** 3–4 days | **⚠️ Some tasks incur AWS charges**

> **Cost discipline (ADR-002):** API Gateway is NOT provisioned. CloudFront is NOT provisioned. SageMaker is a last resort for training. ALB must be deleted (not stopped) between experiment phases. EC2 must be stopped (ASG desired=0) between sessions.

---

### Issue #14 — AWS Account Safety Setup & Billing Alarm

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #13  
**Blocks:** All Phase 4 issues

**Description:** Configure billing protection before any chargeable resources are launched. Billing alarm at $5 (warning) and $15 (action). AWS Budgets alert at $12/month. Confirm EC2 t2.micro quota ≥ 5.

**Acceptance Criteria:**
- [ ] Two billing alarms confirmed active in CloudWatch console
- [ ] `aws sts get-caller-identity` returns correct account ID
- [ ] EC2 quota ≥ 5 confirmed

---

### Issue #15 — Create IAM Roles and Policies

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #14  
**Blocks:** Issues #16–#19

**Description:** `src/aws/iam_setup.py` — create `FlashBalanceAI-Lambda-Role` and `FlashBalanceAI-EC2-Role` with least-privilege policies. No wildcard `*` permissions. Policy JSON documents committed to `decisions/iam_policies/`.

**Acceptance Criteria:**
- [ ] Both roles exist; `iam_setup.py` runs idempotently
- [ ] No wildcard `*` permissions granted

---

### Issue #16 — Provision S3 Bucket and Upload Trained Models

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 1 hour  
**Depends On:** Issue #15, Issue #11  
**Blocks:** Issues #17, #21

**Description:** Create `flashbalanceai-{ACCOUNT_ID}` with versioning enabled, public access blocked. Upload `ppo_flash_v1.zip`, `dqn_flash_v1.zip`. Expected total: < 3 GB → free tier.

**Acceptance Criteria:**
- [ ] `aws s3 ls s3://flashbalanceai-{ACCOUNT_ID}/models/` shows both model zips

---

### Issue #17 — Launch EC2 Backend Instances and Auto Scaling Group

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:aws-billing`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 4 hours  
**Depends On:** Issue #15, Issue #4  
**Blocks:** Issue #18 (ALB needs targets)

**Description:** ASG `FlashBalanceAI-ASG`, min=2, desired=4, max=8. t2.micro instances with Flask backend via UserData. **Set desired=0, min=0 between sessions.** CloudWatch agent publishing `CPUUtilization`.

**Acceptance Criteria:**
- [ ] 4 instances healthy; `curl http://{IP}:5000/health` returns 200
- [ ] ASG scale-out policy verified: CPU > 70% for 2×30s → +1 instance

---

### Issue #18 — Deploy Application Load Balancer

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:aws-billing` `costs:delete-after-use`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #17  
**Blocks:** Issue #20 (state collector Lambda)

**Description:** Create `FlashBalanceAI-ALB` with target group `FlashBalanceAI-TG`. Enable ALB access logs to S3. **Delete ALB and target group after each experiment session** — $0.0225/hr applies even when idle.

The `src/aws/deploy.py` script must support idempotent ALB creation so it can be re-created quickly before each experiment session.

**How to delete (run after EVERY experiment session):**
```bash
aws elbv2 delete-load-balancer --load-balancer-arn <ARN>
aws elbv2 delete-target-group --target-group-arn <ARN>
aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name FlashBalanceAI-ASG \
    --desired-capacity 0 --min-size 0
```

**Acceptance Criteria:**
- [ ] ALB DNS resolves; all 4 targets `healthy`
- [ ] ALB access logs appearing in S3 within 5 minutes of first request
- [ ] `src/aws/deploy.py` recreates ALB idempotently in < 5 minutes
- [ ] Delete procedure tested successfully at least once

---

### Issue #19 — Create DynamoDB Routing Decisions Table

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:normal` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 0.5 hours  
**Depends On:** Issue #14  
**Blocks:** Issue #21

**Description:** `routing_decisions` table, PAY_PER_REQUEST, free tier. Partition key: `timestamp` (String), sort key: `experiment_id` (String).

**Acceptance Criteria:**
- [ ] `aws dynamodb describe-table --table-name routing_decisions` returns 200

---

## Phase 5 — AWS Integration & End-to-End Pipeline

**Milestone:** M5 | **Duration:** 5–7 days | **⚠️ Minimal charges during testing**

---

### Issue #20 — Deploy CloudWatch State Collector Lambda

**Labels:** `type:implementation` `phase:5-integration` `component:aws-lambda` `priority:critical-path` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 5 hours  
**Depends On:** Issues #16, #17, #18  
**Blocks:** Issue #21

**Description:** Lambda `FlashBalanceAI-StateCollector` (128 MB, 15s timeout, Python 3.11). Reads **CPUUtilization (built-in) + 8 custom metrics (QueueDepth × 4, ResponseTimeEMA × 4)** + ALB built-in `ActiveConnectionCount` + ALB built-in `RequestCount`. Builds 23-dim state vector; writes `state/current.json` to S3. EventBridge trigger: `rate(1 minute)`.

**Acceptance Criteria:**
- [ ] `state/current.json` shows valid 23-element array with CPU values in [0, 1]
- [ ] Lambda execution time < 5s; no errors in 5 consecutive invocations

---

### Issue #21 — Deploy PPO Inference Server and Inference Coordinator Lambda

**Labels:** `type:implementation` `phase:5-integration` `component:inference-server` `component:aws-lambda` `priority:critical-path` `costs:aws-billing`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 6 hours  
**Depends On:** Issues #16, #18, #20  
**Blocks:** Issue #24 (integration test)

**Description (two components — confirmed per ADR-002 D14):**

**Component A: EC2 Inference Server** (`src/aws/inference_server.py`)
- Runs on dedicated t2.micro; loads PPO model from S3 at startup
- Exposes `GET /health` and `POST /infer` (returns `{"action": int}`)
- Started ≥ 60s before experiments; stopped between sessions

**Component B: Lightweight Lambda Coordinator** (`src/aws/inference_coordinator.py`)
- No SB3 dependency (< 5 MB); reads S3 state → `POST /infer` to EC2 → updates ALB weights → logs DynamoDB
- EventBridge trigger: `rate(1 minute)`

> **Why EC2, not Lambda:** SB3 + PyTorch ≈ 300 MB unzipped, exceeding Lambda's 250 MB limit. Pure Lambda inference is not architecturally viable.

**Acceptance Criteria:**
- [ ] `curl http://{INFERENCE_EC2_IP}:6000/health` returns `{"status":"healthy"}`
- [ ] `aws lambda invoke --function-name FlashBalanceAI-InferenceCoordinator out.json` returns `{"action": 0-3}`
- [ ] DynamoDB accumulating rows during 5-min test

---

### Issue #22 — Deploy Scaling Trigger Lambda

**Labels:** `type:implementation` `phase:5-integration` `component:aws-lambda` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 3 hours  
**Depends On:** Issues #20, #17  
**Blocks:** Issue #24

**Description:** Lambda `FlashBalanceAI-ScalingTrigger` (128 MB). Handles proactive (`burst_indicator=1` → +2 instances) and reactive (CPU alarm → +1 instance) scaling. CloudWatch alarms: `HighCPU` (CPU > 70%, 2×30s) and `LowCPU` (CPU < 30%, 5×30s).

**Acceptance Criteria:**
- [ ] Manually raising CPU above 70% causes ASG desired capacity to increase within 120s

---

### Issue #23 — Add Custom CloudWatch Metrics Publishing to Flask Backend

**Labels:** `type:implementation` `phase:5-integration` `component:backend` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 2 hours  
**Depends On:** Issues #20, #4  
**Blocks:** Issue #24

**Description:** Flask backend background thread publishes **exactly 2 custom metrics per instance**: `QueueDepth` and `ResponseTimeEMA`. Total: 8 custom metrics across 4 instances — within the 10-metric free tier. `RequestCount` and `ActiveConnectionCount` are sourced from free ALB built-in metrics by the state collector; do NOT publish these as custom metrics.

**Acceptance Criteria:**
- [ ] CloudWatch `FlashBalanceAI/Instances` namespace shows exactly 2 metric names with 4 instance dimensions each
- [ ] Custom metric count ≤ 10 (within free tier)

---

### Issue #24 — End-to-End Integration Test (Gate Check)

**Labels:** `type:testing` `phase:5-integration` `priority:critical-path` `gate:required` `costs:aws-billing`  
**Owner:** All 4 members  
**Estimated Effort:** 3 hours  
**Depends On:** Issues #20–#23  
**Blocks:** All Phase 6 experiment issues

**Description:** 5-min JMeter test at 100 req/s targeting **ALB DNS directly** (no API Gateway). Verify 6 pipeline components simultaneously.

**Verification checklist:**
1. CloudWatch custom metrics updating every 30s
2. S3 `state/current.json` valid 23-element array
3. DynamoDB rows accumulating with valid actions (0–3)
4. ALB access logs: traffic distributed across all 4 instances
5. Lambda CloudWatch Logs: no errors in 5 consecutive invocations
6. No single instance receiving > 80% of requests

**Acceptance Criteria:**
- [ ] All 6 checklist items confirmed with evidence
- [ ] Results documented in `decisions/integration_test_report.md`
- [ ] Test cost: ~$0.05 for 1-hour test

---

## Phase 6 — Experimental Campaign

**Milestone:** M6 | **Duration:** 7–10 days | **⚠️ Primary cost phase (~$4–10 total with discipline)**

> **Cost discipline per session:** Start ALB + EC2 → run experiments → delete ALB → stop EC2 (desired=0). Never leave ALB running overnight.

---

### Issue #25 — E1: Baseline Traffic Calibration (1× Load)

**Labels:** `type:experiment` `phase:6-experiments` `priority:critical-path` `costs:aws-billing`  
**Owner:** Agrima Gupta | **AWS Cost:** ~$0.25 | **Depends On:** Issue #24 | **Blocks:** Issues #26–#28

**Description:** 100 req/s constant, 5 reps, seeds 127–131, 6 algorithms. JMeter targets ALB DNS directly. PPO and RR P95 difference < 20% (sanity check).

---

### Issue #26 — E2: 10× Traffic Spike (Primary Experiment)

**Labels:** `type:experiment` `phase:6-experiments` `priority:critical-path` `costs:aws-billing`  
**Owner:** All 4 members | **AWS Cost:** ~$0.25 | **Depends On:** Issue #25 | **Blocks:** Issues #27–#33

**Description:** peak_users=1000, 5 reps, seeds 132–136, 6 algorithms. Record ASG scaling event timestamps for E7.

---

### Issue #27 — E3: 50× Traffic Spike (Stress Test)

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** All 4 members | **AWS Cost:** ~$0.30 | **Depends On:** Issue #26 | **Blocks:** Issues #28, #32

---

### Issue #28 — E4: 100× Traffic Spike (Conditional)

**Labels:** `type:experiment` `phase:6-experiments` `priority:normal` `costs:aws-billing`  
**Owner:** Devkanti Sarkar | **AWS Cost:** ~$0.15 | **Depends On:** Issue #27 (run only if E3 shows PPO P95 < 2000ms)

---

### Issue #29 — E5: Repeated Flash-Sale Patterns

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** Mohar Gorai | **AWS Cost:** ~$0.75 | **Depends On:** Issue #26

---

### Issue #30 — E6: Noisy/Transient Spikes

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** Agrima Gupta | **AWS Cost:** ~$0.25 | **Depends On:** Issue #26

---

### Issue #31 — E7: Cold-Start / Scaling Response Time

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** All 4 members | **AWS Cost:** ~$0.25 | **Depends On:** Issue #26

---

### Issue #32 — E8: Cost/Performance Trade-off

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** Devkanti Sarkar | **AWS Cost:** ~$0.50 | **Depends On:** Issue #26

---

### Issue #33 — E9: DRL Algorithm Comparison (PPO vs DQN)

**Labels:** `type:analysis` `phase:6-experiments` `priority:high` `costs:free`  
**Owner:** All 4 members | **AWS Cost:** $0 (reuses E2/E3 data) | **Depends On:** Issues #26, #27

---

### Issue #34 — E10: Reward Ablation Study

**Labels:** `type:experiment` `phase:6-experiments` `priority:critical-path` `costs:aws-billing`  
**Owner:** All 4 members | **AWS Cost:** ~$0.50 | **Depends On:** Issue #26

---

## Phase 7 — Results & Statistical Validation

**Milestone:** M7 | **Duration:** 4–5 days | **Local only**

---

### Issue #35 — Raw Data Collection and Verification

**Labels:** `type:analysis` `phase:7-results` `priority:critical-path` `costs:free`  
**Owner:** Agrima Gupta | **Depends On:** All Phase 6 issues | **Blocks:** Issues #36–#37

---

### Issue #36 — Statistical Analysis and Results Table

**Labels:** `type:analysis` `phase:7-results` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar | **Depends On:** Issue #35 | **Blocks:** Issue #41

---

### Issue #37 — Generate Paper-Quality Figures

**Labels:** `type:analysis` `phase:7-results` `priority:high` `costs:free`  
**Owner:** Mohar Gorai | **Depends On:** Issue #35 | **Blocks:** Issue #41

**8 required figures:** (1) System architecture, (2) Traffic profile, (3) Reward convergence, (4) Latency CDF, (5) P95 bar chart, (6) CPU timeline, (7) Scaling reaction time, (8) Ablation study. All PDF vector format.

---

### Issue #38 — Reproducibility Verification

**Labels:** `type:reproducibility` `phase:7-results` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** Any member who was NOT the primary E2 experimenter | **Depends On:** Issue #36 | **Blocks:** Issue #43

---

## Phase 8 — Conference Paper Preparation

**Milestone:** M8 | **Duration:** 7–10 days | **Local only**

---

### Issue #39 — Paper Outline, Section Assignment & Template Setup

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `priority:critical-path` `costs:free`  
**Owner:** All (Devkanti leads) | **Depends On:** Issue #36 | **Blocks:** Issues #40–#43

---

### Issue #40 — Write Algorithm Pseudocode

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `priority:high` `costs:free`  
**Owner:** Devkanti Sarkar | **Depends On:** Issue #11

---

### Issue #41 — Write Results Section and Tables

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar | **Depends On:** Issues #36, #37 | **Blocks:** Issue #43

---

### Issue #42 — Write Related Work Section

**Labels:** `type:research` `phase:8-paper` `component:paper` `priority:high` `costs:free`  
**Owner:** Agrima Gupta | **Depends On:** Issue #39 | **Blocks:** Issue #43

---

### Issue #43 — Full Paper Internal Review and Revision Round

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `priority:critical-path` `costs:free`  
**Owner:** All 4 members | **Depends On:** Issues #39–#42 | **Blocks:** Issue #45

---

## Phase 9 — Demo, Reproducibility & Teardown

**Milestone:** M9 | **Duration:** 2–3 days

---

### Issue #44 — Build Live Monitoring Dashboard

**Labels:** `type:implementation` `phase:9-demo` `component:metrics` `priority:normal` `costs:free`  
**Owner:** Agrima Gupta | **Depends On:** Issue #23

**Description:** `src/metrics/visualiser.py` — Python Dash app, runs **locally** on experimenter's laptop. Reads `state/current.json` from S3 every 5s. No AWS CloudWatch Dashboard provisioned (not required for research).

---

### Issue #45 — Tag v1.0 GitHub Release

**Labels:** `type:documentation` `phase:9-demo` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar | **Depends On:** Issue #43, Issue #38 | **Blocks:** Issue #46

**Release assets:** `ppo_flash_v1.zip`, `dqn_flash_v1.zip`, `environment.yml`, `requirements.txt`, S3 results link. Release notes include: seeds, Python version, SB3 version, reproducibility command, AWS cost actuals.

---

### Issue #46 — Execute AWS Teardown

**Labels:** `type:infrastructure` `phase:9-demo` `component:aws-infra` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** Devkanti Sarkar | **Depends On:** Issue #45

**Description:** Execute `scripts/teardown.sh` — delete ALB, target group, ASG (desired=0), Lambda functions, inference EC2, DynamoDB table, S3 bucket (after local backup). Verify $0 ongoing charges in Cost Explorer.

> **Do NOT delete S3 until all experiment results are downloaded locally and committed to the release.**

---

## 13. Critical Path

```
Issue #1 (ADR-001 + ADR-002)
    ↓
Issue #2 (Repo scaffold)
    ↓
Issue #3 (Python environment)
    ↓
Issue #8 (Traffic generator) ─────────────────────┐
    ↓                                               │
Issue #5 (FlashSaleEnv) ←───────────────────────────┘
    ↓
Issue #11 (PPO training)
    ↓
Issue #13 (Local eval GATE CHECK)
    ↓
Issue #14 (AWS billing alarm GATE CHECK)
    ↓
Issues #15 → #16 → #17 → #18 (AWS stack)
    ↓
Issues #20 → #21 → #22 → #23 (Lambda + inference server)
    ↓
Issue #24 (End-to-end integration GATE CHECK)
    ↓
Issue #25 (E1) → Issue #26 (E2) → Issue #27 (E3)
    ↓
Issue #35 (Data verification) → Issue #36 (Stats)
    ↓
Issues #39 → #41 (Paper outline → Results section)
    ↓
Issue #43 (Paper review) → Issue #45 (v1.0 release) → Issue #46 (Teardown)
```

---

## 14. Parallelisable Work

| After | Can run in parallel |
|-------|-------------------|
| Issue #3 (env) | Issues #4, #6, #7, #10 |
| Issue #11 started | Issue #12 (DQN, frozen env) |
| Issue #26 (E2) | Issues #29, #30, #31, #32, #33 |
| Issue #35 (data) | Issues #36, #37 |
| Issue #39 (paper outline) | Issues #40, #41, #42 |

---

## 15. Release Checkpoints

| Tag | Trigger | Gate |
|-----|---------|------|
| `v0.1-env-ready` | Issue #13 closes | PPO converges; local eval passes |
| `v0.2-aws-live` | Issue #24 closes | Integration test all 6 checks pass |
| `v0.3-experiments-done` | Issue #35 closes | All E1–E10 raw data verified |
| `v0.4-analysis-done` | Issue #37 closes | Stats table + 8 PDF figures |
| `v1.0` | Issue #43 closes | Paper reviewed; reproducibility passed; AWS teardown complete |

---

## 16. Contribution Tracking

| Category | Devkanti | Agrima | Mohar | [4th] |
|----------|---------|--------|-------|-------|
| Research | PRD lead, ADR-001 | Research gap, related work | H-MAS insight, Alibaba trace | TBD |
| Development | FlashSaleEnv, PPO, local eval | Repo scaffold, baselines, DQN | Flask backend, JMeter, EC2/ASG/ALB | TBD |
| Cloud Integration | AWS setup, IAM, inference server, teardown | State collector Lambda, S3/DynamoDB | Scaling Lambda, custom CW metrics, EC2 | TBD |
| AI/ML | PPO training | DQN training, E9 comparison | E5 repeated bursts | TBD |
| Testing | Local gate check, env tests | Baseline/metric tests, E1/E6 | Integration test, E7 | TBD |
| Documentation | ADR-001/002, paper intro/methods | PRD updates, paper setup/results, refs | Paper conclusion, architecture diagram | TBD |
| GitHub | Release tagging, PR coordination | ≥2 PRs, ≥2 reviews | ≥2 PRs, ≥2 reviews | ≥2 PRs, ≥2 reviews |

**Minimum per BCSE355L guidelines:** ≥ 2 PRs opened · ≥ 2 code reviews · weekly commits · continuous documentation updates

**Branch naming:** `feature/phase{N}-{description}` · `experiment/e{N}-{algo}` · `docs/phase{N}-{doc}`  
**Commit convention:** `[PHASE-{N}] {verb} {what}`

---

*Document version 1.1 — reflects cost-optimisation review (ADR-002). Core research methodology unchanged.*
