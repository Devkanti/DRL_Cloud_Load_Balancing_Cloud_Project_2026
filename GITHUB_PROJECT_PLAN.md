# GitHub Project Plan — FlashBalanceAI
## BCSE355L Cloud Architecture Design | Deep Reinforcement Learning-Based Adaptive Cloud Load Balancing

**Project:** FlashBalanceAI: PPO Framework for Adaptive Cloud Load Balancing under Flash-Sale Burst Traffic
**Repository:** `DRL_Cloud_Load_Balancing_Cloud_Project_2026`
**Team:** Devkanti Sarkar (24BIT0162) · Agrima Gupta (24BIT0253) · Mohar Gorai · [4th member]
**Branch Model:** `main` ← `develop` ← `feature/*` (already implemented — not redesigned here)
**Version:** 1.2 | **Date:** 2026-09-23

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

All GitHub Issues must carry at least one **type** label, one **phase** label, and one **component** label.

### Type Labels

| Label | Colour | Meaning |
|-------|--------|---------|
| `type:research` | `#0075ca` | Literature review, gap analysis, hypothesis formation |
| `type:implementation` | `#e4e669` | Source-code tasks (src/, tests/, configs/) |
| `type:experiment` | `#d93f0b` | Requires running code against AWS or local env |
| `type:analysis` | `#f9d0c4` | Statistical analysis, figure generation, notebook work |
| `type:documentation` | `#0e8a16` | README, ADR, PRD updates, paper sections |
| `type:infrastructure` | `#5319e7` | AWS setup, IAM, CloudFormation, deploy scripts |
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
| `priority:critical` | Blocks ≥2 downstream issues |
| `priority:high` | Blocks ≥1 downstream issue |
| `priority:normal` | Standard priority |
| `status:blocked` | Cannot start — dependency not met |
| `status:in-progress` | Currently being worked on |
| `cost:aws-billing` | This task triggers AWS charges — see ADR-002 |
| `cost:free-tier` | Confirmed within AWS Free Tier |
| `good-first-issue` | Suitable for a team member new to the codebase |

---

## 2. Milestone Overview

| Milestone | Phase | Issues | Key Deliverable | Exit Gate | Est. Duration |
|-----------|-------|--------|-----------------|-----------|---------------|
| **M0** Design Lock | 0 | #1–#2 | ADR-001.md locked, repo scaffolded | All 4 members sign ADR | 2 days |
| **M1** Local Foundation | 1 | #3–#7 | FlashSaleEnv + baselines working locally | `pytest` green, reward sanity check passes | 7 days |
| **M2** Dataset & Traffic | 2 | #8–#10 | Traffic generator + JMeter plans verified | 5 scenario plots generated | 5 days |
| **M3** DRL Implementation | 3 | #11–#13 | PPO + DQN trained locally, eval gate passed | PPO reward > RR on val seeds | 10 days |
| **M4** AWS Infrastructure | 4 | #14–#19 | All AWS resources provisioned | Health checks green, $5 billing alarm active | 5 days |
| **M5** AWS Integration | 5 | #20–#24 | End-to-end pipeline: JMeter → ALB → Lambda → DynamoDB | Integration test passes, ALB deleted post-test | 7 days |
| **M6** Experiments | 6 | #25–#34 | E1–E10 raw results collected | 5 reps × 10 experiments, results in S3 | 14 days |
| **M7** Results & Stats | 7 | #35–#38 | Statistical validation + publication figures | p < 0.05, 95% CI, 5 figures ready | 7 days |
| **M8** Paper | 8 | #39–#43 | Conference paper draft complete | All sections drafted, ready for review | 14 days |
| **M9** Demo & Release | 9 | #44–#46 | Live demo, v1.0 tag, AWS teardown | All AWS resources deleted, cost = $0 ongoing | 3 days |

**Milestone dependencies:** M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9  
*(M2 and parts of M1 can run in parallel after M0 is complete)*

---

## Phase 0 — Design Lock & Repository Setup

**Objective:** Lock all architecture decisions and scaffold the repository so all four team members start from an identical, agreed foundation.
**Duration:** 2 days
**AWS Cost:** $0

---

### Issue #1 — Architecture Decision Record (ADR-001)

**Phase label:** `phase:0-design`
**Type labels:** `type:documentation`, `type:research`
**Component:** `component:environment`
**Priority:** `priority:critical`
**Owner/Role:** Devkanti Sarkar (lead) + all members (sign-off)

**Description:**
Create and lock `decisions/ADR-001.md` containing all 24 architecture decisions (D1–D24) as defined in IMPLEMENTATION_PLAN_PART1.md T0.1. Every team member must commit a written sign-off.

**Tasks:**
- [ ] Hold synchronous team meeting (≥ 1 hour) to resolve any remaining design disagreements
- [ ] Document all decisions in `decisions/ADR-001.md` covering: primary DRL (PPO), primary baseline (DQN), action space (discrete N=4), state dimensions (23-dim), reward weights (w1=0.4, w2=0.2, w3=0.2, w4=0.2), AWS region (us-east-1), training location (local first), traffic generator approach
- [ ] Create `decisions/ADR-002.md` — Cost Optimisation Record (ALB delete-per-session, CloudFront removed, API Gateway not in primary flow, SageMaker last resort, 8 custom CloudWatch metrics)
- [ ] Each team member adds their sign-off line to ADR-001.md and commits

**Dependencies:** None
**Deliverables:** `decisions/ADR-001.md`, `decisions/ADR-002.md`
**Acceptance Criteria:**
- `decisions/ADR-001.md` exists with all D1–D24 decisions documented
- All 4 team members have a commit touching the sign-off section
- No open architecture questions remain (OI-3 resolved)
- `decisions/ADR-002.md` documents the 14-service cost classification

**Estimated Effort:** 4 hours (1h meeting + 3h write-up)

---

### Issue #2 — Repository Scaffold

**Phase label:** `phase:0-design`
**Type labels:** `type:implementation`, `type:documentation`
**Component:** `component:aws-infra`
**Priority:** `priority:critical`
**Labels:** `good-first-issue`
**Owner/Role:** Agrima Gupta (24BIT0253)

**Description:**
Create the complete repository directory structure as specified in IMPLEMENTATION_PLAN_PART1.md T0.2. Create placeholder `__init__.py` files, YAML configs, notebook stubs, `.gitignore`, and infrastructure keepfiles.

**Tasks:**
- [ ] Create all `src/*/__ init__.py` placeholder files with module docstrings
- [ ] Create `configs/ppo_config.yaml` with locked hyperparameters from T3.1
- [ ] Create `configs/dqn_config.yaml` with locked hyperparameters from T3.2
- [ ] Create `configs/traffic_config.yaml` with scenario definitions from T2.1
- [ ] Create `configs/aws_config.yaml` with resource name templates (no secrets)
- [ ] Create notebook stubs: `notebooks/01_traffic_analysis.ipynb`, `02_ppo_training.ipynb`, `03_results_analysis.ipynb`, `04_ablation_study.ipynb`
- [ ] Create `src/aws/inference_server.py` scaffold (EC2 PPO inference server — Issue #22 implements)
- [ ] Create `src/aws/inference_coordinator.py` scaffold (Lambda coordinator — Issue #22 implements)
- [ ] Create `.gitignore` covering Python, venvs, Jupyter checkpoints, model zips, AWS credentials, JMeter `.jtl` files
- [ ] Create `experiments/results/.gitkeep`, `experiments/analysis/.gitkeep`, `infra/cloudformation/.gitkeep`
- [ ] Verify all YAML files are valid (`python -c "import yaml; yaml.safe_load(open(f))"`)
- [ ] Verify all `__init__.py` files have valid Python syntax

**Dependencies:** Issue #1
**Deliverables:** Full directory tree, configs, notebook stubs, `.gitignore`
**Acceptance Criteria:**
- `git ls-tree -r --name-only HEAD` shows all required directories
- All YAML files parse without error
- All Python files pass `python -m py_compile`
- `.gitignore` present and covers model binaries, secrets, JMeter outputs

**Estimated Effort:** 4 hours

---

## Phase 1 — Local Development Foundation

**Objective:** Build all core Python components (environment, backend, baselines, metrics) that run entirely locally — no AWS required.
**Duration:** 7 days
**AWS Cost:** $0

---

### Issue #3 — Python Environment Setup

**Phase label:** `phase:1-local-dev`
**Type labels:** `type:implementation`, `type:reproducibility`
**Component:** `component:environment`
**Priority:** `priority:critical`
**Labels:** `good-first-issue`
**Owner/Role:** [4th member]

**Description:**
Create a reproducible Python 3.11 environment that all team members can replicate identically. Pin all dependency versions.

**Tasks:**
- [ ] Create `requirements.txt` with pinned versions: `stable-baselines3==2.3.0`, `gymnasium==0.29.1`, `torch==2.2.0`, `flask==3.0.3`, `boto3==1.34.0`, `numpy==1.26.4`, `pandas==2.2.2`, `matplotlib==3.8.4`, `scipy==1.13.0`, `pyyaml==6.0.1`, `pytest==8.2.0`
- [ ] Create `environment.yml` for conda: `name: flashbalance`, `python=3.11`
- [ ] Document setup steps in `README.md` under "Getting Started"
- [ ] Verify `pip install -r requirements.txt` succeeds on a clean environment
- [ ] Verify `conda env create -f environment.yml` succeeds

**Dependencies:** Issue #2
**Deliverables:** `requirements.txt`, `environment.yml`, updated `README.md`
**Acceptance Criteria:**
- `pip install -r requirements.txt && python -c "from stable_baselines3 import PPO; print('OK')"` exits 0
- All 4 members confirm successful environment creation on their machines

**Estimated Effort:** 3 hours

---

### Issue #4 — Flask Backend Server (EC2 Simulator)

**Phase label:** `phase:1-local-dev`
**Type labels:** `type:implementation`
**Component:** `component:backend`
**Priority:** `priority:high`
**Owner/Role:** Mohar Gorai

**Description:**
Implement `src/backend/app.py` — a Flask server that simulates one EC2 backend instance. Each instance tracks its own CPU utilisation, active connections, and request queue, and responds to HTTP requests with configurable artificial latency.

**Tasks:**
- [ ] Implement `GET /health` endpoint returning `{"status": "healthy", "instance_id": str}`
- [ ] Implement `POST /request` endpoint that simulates processing: `latency = base_latency * (1 + cpu_util)`, returns `{"latency_ms": float, "status": 200}`
- [ ] Simulate CPU utilisation: `cpu = active_connections / max_connections` (configurable `max_connections`, default 100)
- [ ] Implement connection semaphore so `active_connections` tracks in-flight requests correctly
- [ ] Expose `GET /metrics` returning JSON: `{"cpu_util": float, "active_connections": int, "queue_depth": int, "response_time_ema": float}`
- [ ] Write unit tests in `src/tests/test_backend.py`
- [ ] Document API in docstrings

**Dependencies:** Issue #3
**Deliverables:** `src/backend/app.py`, `src/tests/test_backend.py`
**Acceptance Criteria:**
- `pytest src/tests/test_backend.py` passes
- `/metrics` returns all 4 required fields
- Latency scales linearly with simulated load

**Estimated Effort:** 6 hours

---

### Issue #5 — FlashSaleEnv (Gymnasium Environment)

**Phase label:** `phase:1-local-dev`
**Type labels:** `type:implementation`
**Component:** `component:environment`
**Priority:** `priority:critical`
**Owner/Role:** Devkanti Sarkar

**Description:**
Implement `src/environment/flash_sale_env.py` — the custom Gymnasium environment that represents the load-balancing MDP. This is the core research artifact.

**Tasks:**
- [ ] Inherit from `gymnasium.Env`; register as `FlashSaleEnv-v1`
- [ ] **Observation space:** `Box(low=0, high=∞, shape=(23,))` — 5N+3 dimensions: CPU×4, active_conn×4, queue_depth×4, response_time_ema×4, health_status×4, arrival_rate (norm), burst_indicator, time_since_last_spike
- [ ] **Action space:** `Discrete(4)` — index of backend instance to route next request to
- [ ] **Reward function:** `R = 0.4·R_lat + 0.2·R_util + 0.2·R_tput + 0.2·R_sla` where each component is normalised to [0, 1]; load weights from `configs/ppo_config.yaml`
- [ ] **Step logic:** receive action → route request to selected backend → collect metrics → compute reward → check episode termination (7900 steps or SLA violation threshold)
- [ ] **Reset:** initialise all backends to idle state; accept `seed` parameter for reproducibility
- [ ] Pass `gymnasium.utils.env_checker.check_env()` without warnings
- [ ] Write unit tests in `src/tests/test_environment.py` and `src/tests/test_reward.py`

**Dependencies:** Issue #4 (backend), Issue #3 (env)
**Deliverables:** `src/environment/flash_sale_env.py`, `src/tests/test_environment.py`, `src/tests/test_reward.py`
**Acceptance Criteria:**
- `check_env(FlashSaleEnv())` passes with 0 warnings
- Observation shape is always `(23,)`
- Reward is in `[-1, 1]` range
- `pytest src/tests/test_environment.py src/tests/test_reward.py` green

**Estimated Effort:** 10 hours

---

### Issue #6 — Baseline Algorithms

**Phase label:** `phase:1-local-dev`
**Type labels:** `type:implementation`
**Component:** `component:baselines`
**Priority:** `priority:high`
**Owner/Role:** Mohar Gorai

**Description:**
Implement the four baseline load-balancing algorithms that PPO and DQN will be compared against: Round Robin (RR), Weighted Round Robin (WRR), Least Connections (LC), and Threshold-based Autoscaler.

**Tasks:**
- [ ] Implement `src/baselines/round_robin.py`: stateless cyclic routing, `select_backend(state) -> int`
- [ ] Implement `src/baselines/weighted_round_robin.py`: weights proportional to `1/cpu_util`
- [ ] Implement `src/baselines/least_connections.py`: always route to backend with fewest active connections
- [ ] Implement `src/baselines/threshold_autoscaler.py`: add/remove backends when CPU > 70% / < 30%
- [ ] All baselines share the same `select_backend(obs: np.ndarray) -> int` interface (compatible with FlashSaleEnv)
- [ ] Write tests in `src/tests/test_baselines.py`

**Dependencies:** Issue #5
**Deliverables:** `src/baselines/*.py`, `src/tests/test_baselines.py`
**Acceptance Criteria:**
- All 4 baselines implement `select_backend(obs)` returning `int ∈ {0,1,2,3}`
- `pytest src/tests/test_baselines.py` green
- RR distributes requests evenly over 100 steps (verified in test)

**Estimated Effort:** 5 hours

---

### Issue #7 — Metrics Collector

**Phase label:** `phase:1-local-dev`
**Type labels:** `type:implementation`
**Component:** `component:metrics`
**Priority:** `priority:normal`
**Owner/Role:** Agrima Gupta

**Description:**
Implement `src/metrics/collector.py` to aggregate per-step metrics during experiment runs, and `src/metrics/visualiser.py` for a simple Dash dashboard.

**Tasks:**
- [ ] Implement `MetricsCollector` class: tracks `p95_latency`, `p99_latency`, `throughput_rps`, `cpu_utilisation_mean`, `sla_violation_rate`, `reward_episode_mean` per episode
- [ ] Implement `export_csv(path)` method — output format compatible with `notebooks/03_results_analysis.ipynb`
- [ ] Implement `src/metrics/visualiser.py`: Dash app with 4 live-updating charts (reward, latency, CPU, SLA rate)
- [ ] Ensure `MetricsCollector` is serialisable to JSON for Lambda/S3 upload
- [ ] Write tests in `src/tests/test_metrics.py`

**Dependencies:** Issue #5
**Deliverables:** `src/metrics/collector.py`, `src/metrics/visualiser.py`, `src/tests/test_metrics.py`
**Acceptance Criteria:**
- `export_csv()` produces valid CSV with all 6 metric columns
- `pytest src/tests/test_metrics.py` green
- Dash app starts without import errors

**Estimated Effort:** 5 hours

---

## Phase 2 — Dataset & Traffic Generation

**Objective:** Build and validate the traffic workload: synthetic flash-sale generator, JMeter HTTP test plans, and Alibaba trace preprocessing.
**Duration:** 5 days
**AWS Cost:** $0

---

### Issue #8 — Synthetic Traffic Generator

**Phase label:** `phase:2-dataset`
**Type labels:** `type:implementation`
**Component:** `component:traffic-gen`
**Priority:** `priority:critical`
**Owner/Role:** Agrima Gupta

**Description:**
Implement `src/traffic/traffic_generator.py` — the synthetic flash-sale traffic generator that produces request-rate time series for all experiment scenarios defined in `configs/traffic_config.yaml`.

**Tasks:**
- [ ] Implement `TrafficGenerator` class with constructor accepting `burst_multiplier`, `noise_std_fraction`, `seed`, and episode config (warmup, pre_burst, spike_onset, peak, cooldown steps from config)
- [ ] Implement `generate() -> np.ndarray` returning arrival rate at each 100ms step; total length = 7 900 steps
- [ ] Implement phases: warmup (100 rps) → pre-burst (300 rps) → spike onset (linear ramp to `100 * burst_multiplier`) → peak → cooldown (exponential decay)
- [ ] Add Gaussian noise: `rate_t += np.random.normal(0, noise_std_fraction * rate_t)`
- [ ] Support all scenarios from `configs/traffic_config.yaml`: `e1_baseline`, `e2_10x`, `e3_50x`, `e4_100x`, `e6_noisy`, `train_episodes`, `val_episodes`
- [ ] Write `src/tests/test_traffic_generator.py`
- [ ] Run `notebooks/01_traffic_analysis.ipynb` to generate and save all scenario plots

**Dependencies:** Issue #3
**Deliverables:** `src/traffic/traffic_generator.py`, `src/tests/test_traffic_generator.py`, scenario plots in `experiments/results/traffic_profiles/`
**Acceptance Criteria:**
- `e2_10x` scenario peaks at `100 * 10 = 1000 rps` ± 5%
- `generate()` returns array of length exactly 7900
- `pytest src/tests/test_traffic_generator.py` green
- All 5 scenario plots saved

**Estimated Effort:** 6 hours

---

### Issue #9 — JMeter HTTP Test Plans

**Phase label:** `phase:2-dataset`
**Type labels:** `type:implementation`, `type:experiment`
**Component:** `component:traffic-gen`
**Priority:** `priority:high`
**Labels:** `cost:aws-billing`
**Owner/Role:** Devkanti Sarkar

**Description:**
Create Apache JMeter test plans (`.jmx` files) for injecting HTTP load directly against the ALB DNS name during Phase 5–6 experiments. JMeter replaces API Gateway in the primary data flow (ADR-002).

**Tasks:**
- [ ] Create `src/traffic/jmeter_configs/flashsale_e2_10x.jmx` — 10× burst scenario, 5 min peak, 100 threads
- [ ] Create `src/traffic/jmeter_configs/flashsale_e3_50x.jmx` — 50× burst, 200 threads
- [ ] Create `src/traffic/jmeter_configs/flashsale_e4_100x.jmx` — 100× burst, 400 threads
- [ ] Create `src/traffic/jmeter_configs/flashsale_e6_noisy.jmx` — 2× burst with 40% noise
- [ ] Configure each plan: ramp-up matches `spike_onset_steps` (100 × 100ms = 10s), peak duration 300s, record P95/P99 latency and throughput
- [ ] Parameterise `${ALB_DNS}` as a JMeter variable so plans work without hardcoded endpoints
- [ ] Verify plans load in JMeter GUI without errors

**Dependencies:** Issue #2
**Deliverables:** `src/traffic/jmeter_configs/*.jmx` (4 plans)
**Acceptance Criteria:**
- All 4 `.jmx` files open cleanly in JMeter 5.6+
- `${ALB_DNS}` variable is used consistently — no hardcoded IP/DNS
- Ramp-up time = 10 s, peak duration ≥ 300 s in all plans

**Estimated Effort:** 5 hours

---

### Issue #10 — Alibaba Cloud Cluster Trace Preprocessing

**Phase label:** `phase:2-dataset`
**Type labels:** `type:research`, `type:implementation`
**Component:** `component:dataset`
**Priority:** `priority:normal`
**Owner/Role:** Mohar Gorai

**Description:**
Download, preprocess, and document the Alibaba Cloud Cluster Trace dataset for use as a real-world traffic reference in the paper. This dataset motivates the traffic generator parameters.

**Tasks:**
- [ ] Download Alibaba Cluster Trace v2018 from the official repository
- [ ] Save raw files to `dataset/alibaba_cluster_trace/` (excluded from git via `.gitignore`)
- [ ] Write `dataset/preprocess_alibaba.py`: extract request arrival rates, resample to 100ms bins, identify flash-event intervals (CPU spikes > 5× baseline)
- [ ] Document dataset statistics in `dataset/README.md`: source URL, licence, size, preprocessing steps
- [ ] Verify burst multiplier distribution from real data supports 10×–100× synthetic generator parameters
- [ ] Upload processed summary CSV to S3 (`dataset/alibaba_summary.csv`)

**Dependencies:** Issue #3
**Deliverables:** `dataset/preprocess_alibaba.py`, `dataset/README.md`, processed summary CSV
**Acceptance Criteria:**
- `dataset/README.md` includes source citation, download URL, licence
- Preprocessing script runs without errors on the raw trace
- Summary statistics table shows at least one event with ≥ 10× burst

**Estimated Effort:** 6 hours

---

## Phase 3 — DRL Implementation

**Objective:** Train PPO and DQN agents on FlashSaleEnv locally, validate convergence, and pass the local evaluation gate before any AWS spending.
**Duration:** 10 days
**AWS Cost:** $0 (local training)

---

### Issue #11 — PPO Agent Training

**Phase label:** `phase:3-drl`
**Type labels:** `type:implementation`, `type:experiment`
**Component:** `component:ppo-agent`
**Priority:** `priority:critical`
**Owner/Role:** Devkanti Sarkar

**Description:**
Train the PPO agent on FlashSaleEnv using Stable-Baselines3 v2.3.0 with the hyperparameters locked in `configs/ppo_config.yaml`. Save the best checkpoint for Phase 5 deployment.

**Tasks:**
- [ ] Implement `src/agents/ppo_agent.py`: `train()`, `evaluate()`, `save()`, `load()` using `stable_baselines3.PPO`
- [ ] Load all hyperparameters from `configs/ppo_config.yaml` — do NOT hardcode values
- [ ] Use `n_envs=4` vectorised environments, `MlpPolicy`, `net_arch=[64,64]`
- [ ] Attach `EvalCallback` using val seeds 112–126 with `eval_freq=50_000`
- [ ] Attach `CheckpointCallback` saving every 100k steps to `models/ppo/`
- [ ] Train for `total_timesteps=2_000_000`
- [ ] Monitor reward curve — confirm steady improvement by step 500k
- [ ] Save best model to `models/ppo/best_model.zip`
- [ ] Run `notebooks/02_ppo_training.ipynb` end-to-end and save output
- [ ] Write `src/tests/test_ppo_agent.py`

**Dependencies:** Issue #5 (FlashSaleEnv), Issue #3 (env)
**Deliverables:** `src/agents/ppo_agent.py`, `models/ppo/best_model.zip`, trained notebook output, `src/tests/test_ppo_agent.py`
**Acceptance Criteria:**
- Training completes 2M timesteps without crash
- Episode reward on val seeds is higher than Round Robin baseline (local gate check)
- Best model file exists and loads without error: `PPO.load("models/ppo/best_model.zip")`
- `pytest src/tests/test_ppo_agent.py` green

**Estimated Effort:** 12 hours (training wall time ≈ 2–4 h depending on hardware)

---

### Issue #12 — DQN Agent Training

**Phase label:** `phase:3-drl`
**Type labels:** `type:implementation`, `type:experiment`
**Component:** `component:dqn-agent`
**Priority:** `priority:high`
**Owner/Role:** Mohar Gorai

**Description:**
Train the DQN baseline agent using the same FlashSaleEnv, same seed, and identical network architecture as PPO for a fair comparison. See `configs/dqn_config.yaml`.

**Tasks:**
- [ ] Implement `src/agents/dqn_agent.py`: `train()`, `evaluate()`, `save()`, `load()` using `stable_baselines3.DQN`
- [ ] Load all hyperparameters from `configs/dqn_config.yaml`
- [ ] Network architecture **must** match PPO: `net_arch=[64,64]`
- [ ] Use `seed_train=42` — same as PPO (ensures identical environment sequence for fair comparison)
- [ ] Train for `total_timesteps=2_000_000`
- [ ] Save best model to `models/dqn/best_model.zip`
- [ ] Write `src/tests/test_dqn_agent.py`

**Dependencies:** Issue #5, Issue #3
**Deliverables:** `src/agents/dqn_agent.py`, `models/dqn/best_model.zip`, `src/tests/test_dqn_agent.py`
**Acceptance Criteria:**
- Training completes 2M timesteps without crash
- Both PPO and DQN trained with same seed; their val-set reward curves are recorded in the same plot
- `pytest src/tests/test_dqn_agent.py` green

**Estimated Effort:** 8 hours

---

### Issue #13 — Local Evaluation Gate Check

**Phase label:** `phase:3-drl`
**Type labels:** `type:testing`, `type:experiment`
**Component:** `component:ppo-agent`
**Priority:** `priority:critical`
**Owner/Role:** Agrima Gupta

**Description:**
Run the local pre-AWS evaluation gate: compare PPO, DQN, and all 4 baselines on val seeds 112–126. PPO must outperform all baselines on P95 latency before any AWS money is spent.

**Tasks:**
- [ ] Write `src/tests/test_local_eval_gate.py` that evaluates all agents/baselines on val seeds 112–126 across all 5 scenarios
- [ ] Record: P95 latency, throughput, SLA violation rate, mean episode reward for each agent
- [ ] Assert: PPO mean P95 latency < Round Robin mean P95 latency
- [ ] Assert: PPO mean episode reward > DQN mean episode reward
- [ ] Export results to `experiments/results/local_gate_results.csv`
- [ ] Create `experiments/results/local_gate_results.csv` summary — include in PR for team review

**Dependencies:** Issue #11, Issue #12, Issue #6
**Deliverables:** `src/tests/test_local_eval_gate.py`, `experiments/results/local_gate_results.csv`
**Acceptance Criteria:**
- Gate test file runs end-to-end in < 30 minutes on a laptop
- PPO P95 latency < RR P95 latency on val seeds (gate PASS)
- Results CSV has columns: `agent`, `scenario`, `seed`, `p95_latency_ms`, `throughput_rps`, `sla_violation_rate`, `mean_reward`

**Estimated Effort:** 4 hours

---

## Phase 4 — AWS Infrastructure Deployment

**Objective:** Provision all required AWS resources in us-east-1 at minimum cost. ALB must be deleted (not just stopped) between experiment sessions.
**Duration:** 5 days
**AWS Cost:** ~$3–5 (estimated; ALB delete-per-session per ADR-002)

---

### Issue #14 — Billing Alarm Setup

**Phase label:** `phase:4-aws-deploy`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:critical`
**Labels:** `cost:aws-billing`
**Owner/Role:** Devkanti Sarkar

**Description:**
Set up AWS billing alarms before any other AWS resource is created. Two alarms: $5 warning and $15 hard ceiling (ADR-001 D22). This is the first AWS task.

**Tasks:**
- [ ] Enable billing alerts in AWS account settings (one-time)
- [ ] Create CloudWatch billing alarm at $5 (SNS → email notification)
- [ ] Create CloudWatch billing alarm at $15 (SNS → email notification to all 4 team members)
- [ ] Verify alarm triggers correctly (test with $0.01 threshold, then restore)
- [ ] Document alarm ARNs in `configs/aws_config.yaml` comments

**Dependencies:** Issue #2 (aws_config.yaml exists)
**Deliverables:** Two active billing alarms in AWS console
**Acceptance Criteria:**
- Both alarms visible in CloudWatch console
- All 4 team email addresses subscribed to SNS topic
- No other AWS resources created before this issue is closed

**Estimated Effort:** 1 hour

---

### Issue #15 — IAM Roles & Policies

**Phase label:** `phase:4-aws-deploy`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:critical`
**Labels:** `cost:free-tier`
**Owner/Role:** Agrima Gupta

**Description:**
Create the minimum-privilege IAM roles required by Lambda functions and EC2 instances per ADR-001 D20.

**Tasks:**
- [ ] Implement `src/aws/iam_setup.py` using boto3
- [ ] Create `FlashBalanceAI-Lambda-Role` with policies: `AWSLambdaBasicExecutionRole`, `CloudWatchReadOnlyAccess`, `AmazonDynamoDBFullAccess`, `AmazonS3FullAccess` (scoped to project bucket only)
- [ ] Create `FlashBalanceAI-EC2-Role` and instance profile: `AmazonS3ReadOnlyAccess` (model download only), `CloudWatchAgentServerPolicy`
- [ ] Use least-privilege: restrict S3 access to `flashbalanceai-{ACCOUNT_ID}` bucket only
- [ ] Document role ARNs — store in AWS SSM Parameter Store (not in git)

**Dependencies:** Issue #14
**Deliverables:** `src/aws/iam_setup.py`, IAM roles created in AWS
**Acceptance Criteria:**
- Both roles visible in IAM console
- Lambda can write to DynamoDB and read from S3 (verified by test invocation)
- No wildcard `*` permissions in any policy

**Estimated Effort:** 3 hours

---

### Issue #16 — S3 Bucket Setup

**Phase label:** `phase:4-aws-deploy`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:high`
**Labels:** `cost:free-tier`
**Owner/Role:** [4th member]

**Description:**
Create the S3 bucket `flashbalanceai-{ACCOUNT_ID}` in us-east-1 and configure lifecycle rules to avoid unexpected storage costs.

**Tasks:**
- [ ] Create bucket with naming convention from `configs/aws_config.yaml`
- [ ] Block all public access
- [ ] Create folder structure: `models/`, `state/`, `results/`, `dataset/`
- [ ] Set lifecycle rule: delete objects in `state/` after 7 days
- [ ] Upload trained PPO model after Issue #11: `aws s3 cp models/ppo/best_model.zip s3://flashbalanceai-{ACCOUNT_ID}/models/ppo_flash_v1.zip`
- [ ] Upload trained DQN model after Issue #12

**Dependencies:** Issue #15
**Deliverables:** S3 bucket with correct structure and lifecycle policy
**Acceptance Criteria:**
- Bucket exists; `aws s3 ls s3://flashbalanceai-{ACCOUNT_ID}/` shows correct prefixes
- Lifecycle rule on `state/` prefix confirmed in console
- Public access is blocked

**Estimated Effort:** 2 hours

---

### Issue #17 — EC2 Auto Scaling Group (Backend Instances)

**Phase label:** `phase:4-aws-deploy`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:critical`
**Labels:** `cost:aws-billing`
**Owner/Role:** Devkanti Sarkar

**Description:**
Launch 4 t2.micro EC2 backend instances in an Auto Scaling Group. Each instance runs the Flask backend server (`src/backend/app.py`). **Stop all instances when not actively running experiments.**

**Tasks:**
- [ ] Create launch template: AMI = Amazon Linux 2023, instance type = `t2.micro`, security group allowing port 5000 from ALB only, user-data script that installs Flask and starts `app.py`
- [ ] Create ASG `FlashBalanceAI-ASG`: min=2, desired=4, max=8, cooldown=180s
- [ ] Tag all instances: `Project=FlashBalanceAI`, `Phase=4`, `Owner={team}`
- [ ] Implement `src/aws/deploy.py` functions: `create_asg()`, `stop_asg()`, `delete_asg()`
- [ ] Verify health checks: `curl http://{instance_ip}:5000/health` returns `{"status": "healthy"}`
- [ ] **IMPORTANT:** Add `stop_all_instances()` call to teardown script (Issue #46)

**Dependencies:** Issue #15, Issue #16, Issue #4 (backend app)
**Deliverables:** Running ASG with 4 healthy backend instances, `src/aws/deploy.py`
**Acceptance Criteria:**
- 4 instances show "InService" in ASG console
- `/health` endpoint returns 200 on all 4 instances
- All instances tagged correctly

**Estimated Effort:** 6 hours

---

### Issue #18 — ALB Setup (Flash-Session Only)

**Phase label:** `phase:4-aws-deploy`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:critical`
**Labels:** `cost:aws-billing`
**Owner/Role:** Agrima Gupta

**Description:**
Create an Application Load Balancer targeting the backend ASG. **Per ADR-002: the ALB must be DELETED (not stopped) at the end of each experiment session** — ALB charges accumulate even when idle.

**Tasks:**
- [ ] Create ALB `FlashBalanceAI-ALB` in us-east-1 with HTTP listener on port 80
- [ ] Create target group pointing to backend instances on port 5000; health check: `GET /health`, interval 30s
- [ ] Register all 4 EC2 instances in the target group
- [ ] Record `alb_dns` and `target_group_arn` in `configs/aws_config.yaml` (as comments — not secrets)
- [ ] Implement `create_alb()` and `delete_alb()` in `src/aws/deploy.py`
- [ ] **TEST delete_alb():** create, verify DNS resolves, delete, verify DNS no longer resolves
- [ ] Add `delete_alb()` to teardown script (Issue #46)
- [ ] Document "ALB lifecycle" in `decisions/ADR-002.md` comments section

**Dependencies:** Issue #17
**Deliverables:** ALB DNS name, target group ARN, `delete_alb()` function verified
**Acceptance Criteria:**
- ALB health check shows all 4 targets healthy
- `create_alb()` / `delete_alb()` cycle works without console intervention
- Teardown script confirmed to call `delete_alb()`

**Estimated Effort:** 4 hours

---

### Issue #19 — DynamoDB Table Setup

**Phase label:** `phase:4-aws-deploy`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:high`
**Labels:** `cost:free-tier`
**Owner/Role:** [4th member]

**Description:**
Create the `routing_decisions` DynamoDB table used by the inference pipeline to store state vectors and routing decisions.

**Tasks:**
- [ ] Create table `routing_decisions`: partition key `pk` (String), billing mode = PAY_PER_REQUEST (on-demand — free tier covers student workload)
- [ ] Add TTL attribute `expires_at` (epoch seconds) — set TTL = now + 24h on all writes to auto-purge old records
- [ ] Implement `src/aws/dynamo_utils.py`: `write_state()`, `read_latest_state()`, `write_action()`, `read_latest_action()`
- [ ] Write integration test: write a mock 23-dim state, read it back, verify

**Dependencies:** Issue #15
**Deliverables:** DynamoDB table, `src/aws/dynamo_utils.py`
**Acceptance Criteria:**
- Table exists with correct key schema
- TTL enabled (prevents unbounded storage growth)
- `write_state()` + `read_latest_state()` round-trip test passes

**Estimated Effort:** 3 hours

---

## Phase 5 — AWS Integration & End-to-End Pipeline

**Objective:** Connect all AWS components into a working end-to-end pipeline: JMeter → ALB → Backend EC2 × 4 ← Lambda StateCollector ← DynamoDB ← Lambda InferenceCoordinator ← EC2 Inference Server.
**Duration:** 7 days
**AWS Cost:** ~$5–8

---

### Issue #20 — Lambda State Collector

**Phase label:** `phase:5-integration`
**Type labels:** `type:implementation`, `type:infrastructure`
**Component:** `component:aws-lambda`
**Priority:** `priority:critical`
**Labels:** `cost:free-tier`
**Owner/Role:** Devkanti Sarkar

**Description:**
Implement and deploy `src/aws/cloudwatch_collector.py` — Lambda function that polls CloudWatch every 30 seconds, builds the 23-dimensional state vector, and writes it to DynamoDB.

**Tasks:**
- [ ] Implement `build_state_vector()`: fetch CPU×4, active_conn×4, queue_depth×4, response_time_ema×4, health×4, arrival_rate, burst_indicator, time_since_spike from CloudWatch and ALB metrics
- [ ] Custom CloudWatch metrics constraint (ADR-002): **exactly 8 custom metrics** — `QueueDepth×4` + `ResponseTimeEMA×4` — all others use standard AWS metrics
- [ ] Implement `lambda_handler(event, context)` writing state JSON to DynamoDB with TTL
- [ ] Package and deploy to Lambda with 30-second EventBridge trigger
- [ ] Write integration test: invoke Lambda manually, verify DynamoDB record appears within 5s

**Dependencies:** Issue #17, Issue #18, Issue #19
**Deliverables:** Deployed Lambda `FlashBalanceAI-StateCollector`, DynamoDB records
**Acceptance Criteria:**
- Lambda invocation logs show no errors
- DynamoDB shows state record updated within 35 s of deployment
- State vector length is exactly 23
- Custom CloudWatch metric count ≤ 8 (free tier)

**Estimated Effort:** 8 hours

---

### Issue #21 — CloudWatch Agent on EC2 Backends

**Phase label:** `phase:5-integration`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:high`
**Labels:** `cost:free-tier`
**Owner/Role:** Mohar Gorai

**Description:**
Install and configure the CloudWatch agent on all 4 backend EC2 instances to publish the custom `QueueDepth` and `ResponseTimeEMA` metrics to the `FlashBalanceAI/Instances` namespace.

**Tasks:**
- [ ] Install CloudWatch agent via SSM on all 4 instances
- [ ] Configure agent to publish `QueueDepth` and `ResponseTimeEMA` metrics from `src/backend/app.py` `/metrics` endpoint every 10 seconds
- [ ] Verify metrics appear in CloudWatch console under `FlashBalanceAI/Instances` namespace
- [ ] Confirm exactly 8 custom metrics active (4 instances × 2 metrics) — within free tier 10 metric limit

**Dependencies:** Issue #17, Issue #7 (MetricsCollector)
**Deliverables:** CloudWatch agent configured on all EC2 instances, 8 custom metrics visible
**Acceptance Criteria:**
- CloudWatch console shows 8 metrics in `FlashBalanceAI/Instances` namespace
- Data points appearing every ≤ 30 s on all 4 instances
- No agent errors in CloudWatch Logs

**Estimated Effort:** 4 hours

---

### Issue #22 — EC2 Inference Server + Lambda Coordinator

**Phase label:** `phase:5-integration`
**Type labels:** `type:implementation`, `type:infrastructure`
**Component:** `component:inference-server`, `component:aws-lambda`
**Priority:** `priority:critical`
**Labels:** `cost:aws-billing`
**Owner/Role:** Agrima Gupta

**Description:**
Implement the full inference pipeline: (A) EC2 inference server loading the trained PPO model and serving actions on port 6000, and (B) Lambda InferenceCoordinator reading state from DynamoDB, calling the server, and writing the routing decision back.

**Tasks (Component A — EC2 Inference Server):**
- [ ] Complete `src/aws/inference_server.py` (scaffold created in Issue #2): implement S3 model download, `PPO.load()`, Flask HTTP endpoint `POST /action`
- [ ] Deploy on a dedicated `t2.micro` instance (separate from backend instances)
- [ ] Add startup script to user-data: `python inference_server.py --bucket flashbalanceai-{ACCOUNT_ID} --port 6000`
- [ ] Open security group port 6000 to Lambda VPC only

**Tasks (Component B — Lambda Coordinator):**
- [ ] Complete `src/aws/inference_coordinator.py` (scaffold created in Issue #2): implement DynamoDB state read, HTTP POST to inference server, DynamoDB action write
- [ ] Set `INFERENCE_SERVER_URL` environment variable to EC2 private IP:6000
- [ ] Deploy Lambda with 2s timeout (inference must be < 2s)

**Tasks (Integration):**
- [ ] End-to-end test: trigger coordinator → state read → inference server POST → action written to DynamoDB → confirmed by backend routing

**Dependencies:** Issue #11 (trained PPO model), Issue #20 (state in DynamoDB), Issue #19 (DynamoDB)
**Deliverables:** Running inference server on EC2, deployed Lambda coordinator, end-to-end test passing
**Acceptance Criteria:**
- `curl http://{inference_server_ip}:6000/health` returns `{"status": "ok", "model_loaded": true}`
- Lambda coordinator returns `{"action": 0|1|2|3}` within 2 s
- DynamoDB shows routing decision written after each Lambda invocation
- `pytest src/tests/test_inference_pipeline.py` green

**Estimated Effort:** 10 hours

---

### Issue #23 — Scaling Trigger Lambda

**Phase label:** `phase:5-integration`
**Type labels:** `type:implementation`, `type:infrastructure`
**Component:** `component:aws-lambda`
**Priority:** `priority:high`
**Labels:** `cost:free-tier`
**Owner/Role:** Mohar Gorai

**Description:**
Implement `src/aws/scaling_trigger.py` — Lambda function that triggers ASG scale-out/in based on the routing action and burst indicator from the inference pipeline.

**Tasks:**
- [ ] Implement `lambda_handler`: read current action and burst_indicator from DynamoDB; if `burst_indicator=True` and all 4 backends at > 70% CPU, call `asg.set_desired_capacity(count+1)`; if `burst_indicator=False` and avg CPU < 30%, scale in
- [ ] Add cooldown logic: do not trigger scale-out if last scale-out was within `asg_cooldown_seconds=180`
- [ ] Deploy Lambda; trigger via InferenceCoordinator at end of action selection

**Dependencies:** Issue #22
**Deliverables:** Deployed Lambda `FlashBalanceAI-ScalingTrigger`
**Acceptance Criteria:**
- Scale-out triggered when CPU > 70% sustained for 2 periods
- 3-minute cooldown respected (no thrashing)
- Lambda executes in < 5s

**Estimated Effort:** 5 hours

---

### Issue #24 — End-to-End Integration Test

**Phase label:** `phase:5-integration`
**Type labels:** `type:testing`, `type:experiment`
**Component:** `component:aws-infra`
**Priority:** `priority:critical`
**Labels:** `cost:aws-billing`
**Owner/Role:** Agrima Gupta

**Description:**
Run the full end-to-end integration test: JMeter → ALB → Backend EC2 × 4, with PPO inference pipeline active. Validate that all components work together before the main experimental campaign. Delete ALB immediately after test.

**Tasks:**
- [ ] Start all AWS resources (recreate ALB with `create_alb()`)
- [ ] Run JMeter `flashsale_e2_10x.jmx` for 5 minutes against `${ALB_DNS}`
- [ ] Verify in parallel: DynamoDB shows routing decisions updating, CloudWatch shows CPU across instances, inference server logs show predictions
- [ ] Collect metrics: P95 latency, throughput, SLA violation rate
- [ ] Save raw JMeter `.jtl` to `experiments/results/integration_test/`
- [ ] **IMMEDIATELY after test: run `delete_alb()`** — confirm ALB deleted in console
- [ ] Document integration test results in PR description

**Dependencies:** Issue #20, Issue #21, Issue #22, Issue #23, Issue #9
**Deliverables:** Integration test results, ALB confirmed deleted
**Acceptance Criteria:**
- JMeter run completes without errors
- PPO inference pipeline responds to every 30s polling cycle during test
- ALB deleted within 15 minutes of test end
- P95 latency logged (baseline for Phase 6)

**Estimated Effort:** 4 hours (plus AWS run time)

---

## Phase 6 — Experimental Campaign

**Objective:** Run all 10 experiments (E1–E10) defined in PRD §24, collecting 5 repetitions each with fixed test seeds 127–141.
**Duration:** 14 days
**AWS Cost:** ~$5–8 (ALB recreated and deleted per session)

> **Seed protocol:** All Phase 6 experiments use test seeds 127–141. Train and val seeds (42–126) are not used here. Each experiment session: `create_alb()` → run experiments → `delete_alb()`.

---

### Issue #25 — Experiment E1: Baseline Traffic (1× Burst)

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:ppo-agent` | **Labels:** `cost:aws-billing`
**Owner/Role:** Devkanti Sarkar

Evaluate PPO, DQN, RR, WRR, LC, Threshold on no-burst scenario (1× baseline traffic). 5 reps, seeds 127–131.
Save results to `experiments/results/e1_baseline/`.

**Acceptance Criteria:** 5 result files per agent, P95 latency < 200ms for all agents (no burst = easy condition).

---

### Issue #26 — Experiment E2: 10× Flash-Sale Burst

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:ppo-agent` | **Labels:** `cost:aws-billing`
**Owner/Role:** Agrima Gupta

Core experiment. Evaluate all 6 agents on 10× burst. 5 reps. Save to `experiments/results/e2_10x/`.
**Primary hypothesis test:** PPO P95 latency < RR P95 latency (p < 0.05).

---

### Issue #27 — Experiment E3: 50× Flash-Sale Burst

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:ppo-agent` | **Labels:** `cost:aws-billing`
**Owner/Role:** Mohar Gorai

Extreme burst. Evaluate all 6 agents. 5 reps. Save to `experiments/results/e3_50x/`.

---

### Issue #28 — Experiment E4: 100× Flash-Sale Burst

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:ppo-agent` | **Labels:** `cost:aws-billing`
**Owner/Role:** [4th member]

Most extreme burst. Evaluate all 6 agents. 5 reps. Save to `experiments/results/e4_100x/`.

---

### Issue #29 — Experiment E5: AWS Auto Scaling Comparison

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:aws-infra` | **Labels:** `cost:aws-billing`
**Owner/Role:** Devkanti Sarkar

Compare PPO routing with and without ASG scale-out active. Quantify DRL value beyond standard AWS autoscaling. 5 reps on E2 scenario.

---

### Issue #30 — Experiment E6: Noisy Traffic (40% Noise)

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:ppo-agent` | **Labels:** `cost:aws-billing`
**Owner/Role:** Agrima Gupta

Evaluate robustness to high-noise traffic (2× burst + 40% noise). 5 reps. Save to `experiments/results/e6_noisy/`.

---

### Issue #31 — Experiment E7: Cold-Start Latency

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:inference-server` | **Labels:** `cost:aws-billing`
**Owner/Role:** Mohar Gorai

Measure inference server startup latency (time from Lambda invocation to first action) and action selection latency per call. 5 reps. Compare to SLA budget (< 500ms).

---

### Issue #32 — Experiment E8: Multi-Region Simulation

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:aws-infra` | **Labels:** `cost:aws-billing`
**Owner/Role:** [4th member]

Simulate geographic traffic distribution by varying backend instance availability (randomly disable 1–2 instances mid-experiment). Evaluate PPO adaptation. 5 reps.

---

### Issue #33 — Experiment E9: DRL Transfer — Unseen Burst Profile

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment` | **Component:** `component:ppo-agent` | **Labels:** `cost:aws-billing`
**Owner/Role:** Devkanti Sarkar

Test PPO on a burst profile not seen during training (triangular burst instead of exponential). Evaluate zero-shot generalisation. 5 reps.

---

### Issue #34 — Experiment E10: Reward Weight Ablation

**Phase label:** `phase:6-experiments` | **Type:** `type:experiment`, `type:analysis` | **Component:** `component:ppo-agent`
**Owner/Role:** Agrima Gupta

Systematically vary reward weights (w1, w2, w3, w4) and retrain PPO (500k steps each). Evaluate impact on P95 latency. Run entirely **locally** (no AWS). Execute `notebooks/04_ablation_study.ipynb`.

**Acceptance Criteria:** Heatmap figure generated, baseline weights (0.4, 0.2, 0.2, 0.2) confirmed optimal or updated in ADR-001.

---

## Phase 7 — Results & Statistical Validation

**Objective:** Aggregate all E1–E10 results, compute statistics, generate publication-quality figures.
**Duration:** 7 days | **AWS Cost:** $0

---

### Issue #35 — Statistical Analysis

**Phase label:** `phase:7-results`
**Type labels:** `type:analysis`
**Component:** `component:paper`
**Priority:** `priority:critical`
**Owner/Role:** Agrima Gupta

**Tasks:**
- [ ] Load all experiment CSVs from `experiments/results/`
- [ ] Compute mean ± 95% CI for each metric × agent × experiment
- [ ] Run paired Student's t-test (PPO vs DQN) and paired Wilcoxon test for non-normal distributions
- [ ] Record p-values in `experiments/analysis/statistical_tests.csv`
- [ ] Verify: ≥ 5 reps per cell, 95% CI computed correctly (`±t_{0.975, n-1} × SE`)

**Acceptance Criteria:** `statistical_tests.csv` shows p < 0.05 for PPO vs RR on E2 (primary claim), or paper claims revised if not.

**Estimated Effort:** 8 hours

---

### Issue #36 — Publication Figures

**Phase label:** `phase:7-results`
**Type labels:** `type:analysis`
**Component:** `component:paper`
**Priority:** `priority:high`
**Owner/Role:** Devkanti Sarkar

**Tasks:**
- [ ] Fig 1: P95 latency comparison bar chart (E1–E4) with error bars — all 6 agents
- [ ] Fig 2: Throughput vs burst multiplier line chart
- [ ] Fig 3: SLA violation rate grouped bar chart
- [ ] Fig 4: Reward convergence curves (PPO vs DQN training)
- [ ] Fig 5: Reward weight ablation heatmap (E10)
- [ ] Save all figures to `experiments/results/figures/` as 300 dpi PNG and PDF

**Acceptance Criteria:** All 5 figures generated without errors, PDF format suitable for paper submission.

**Estimated Effort:** 6 hours

---

### Issue #37 — Reproducibility Package

**Phase label:** `phase:7-results`
**Type labels:** `type:reproducibility`
**Component:** `component:paper`
**Priority:** `priority:high`
**Owner/Role:** Mohar Gorai

**Tasks:**
- [ ] Document all random seeds used (train: 42–111, val: 112–126, test: 127–141) in `experiments/analysis/seed_manifest.csv`
- [ ] Write `experiments/analysis/reproduce_all.sh`: end-to-end script that recreates all figures from saved model checkpoints and test seeds
- [ ] Verify `reproduce_all.sh` produces figures that match originals (within ±2% tolerance)
- [ ] Pin software versions in `requirements.txt` and `environment.yml`

**Acceptance Criteria:** `reproduce_all.sh` runs end-to-end and all figures match originals.

**Estimated Effort:** 5 hours

---

### Issue #38 — Notebooks Finalisation

**Phase label:** `phase:7-results`
**Type labels:** `type:documentation`, `type:analysis`
**Component:** `component:paper`
**Priority:** `priority:normal`
**Owner/Role:** [4th member]

**Tasks:**
- [ ] Run all 4 notebooks with final data and clear all previous outputs
- [ ] Re-execute `01_traffic_analysis.ipynb`, `02_ppo_training.ipynb`, `03_results_analysis.ipynb`, `04_ablation_study.ipynb` from top to bottom
- [ ] Commit final executed notebooks with outputs to `feature/agrimagupta`

**Acceptance Criteria:** All notebooks execute with 0 errors and committed outputs match figures in `experiments/results/figures/`.

**Estimated Effort:** 3 hours

---

## Phase 8 — Conference Paper Preparation

**Objective:** Write a complete conference paper suitable for IEEE/ACM submission.
**Duration:** 14 days | **AWS Cost:** $0

---

### Issue #39 — Paper: Abstract, Introduction, Related Work

**Phase label:** `phase:8-paper`
**Type labels:** `type:documentation`, `type:research`
**Component:** `component:paper`
**Priority:** `priority:high`
**Owner/Role:** Devkanti Sarkar

**Tasks:**
- [ ] Write Abstract (≤ 200 words): problem, method (PPO), key results (P95 improvement %, SLA violation reduction %)
- [ ] Write Introduction: flash-sale problem, motivation, contributions (3–5 numbered), paper structure
- [ ] Write Related Work: table of 10+ cited papers vs this work, covering DRL-based LB, AWS cloud autoscaling, flash-sale traffic management

**Acceptance Criteria:** Related work table covers all 10 papers surveyed in Phase-I documents. Abstract states quantitative result.

**Estimated Effort:** 12 hours

---

### Issue #40 — Paper: System Design & Methodology

**Phase label:** `phase:8-paper`
**Type labels:** `type:documentation`
**Component:** `component:paper`
**Priority:** `priority:critical`
**Owner/Role:** Agrima Gupta

**Tasks:**
- [ ] Write System Design section: MDP formulation, state space (23-dim), action space (N=4), reward function (4-component), FlashSaleEnv description
- [ ] Write Methodology section: PPO algorithm description, SB3 implementation, network architecture ([64,64] MLP), training procedure, DQN baseline setup
- [ ] Include AWS architecture diagram (from `feature/AgrimaGupta/architecture/`)
- [ ] Write Traffic Generator section: synthetic generator, scenario definitions, Alibaba trace alignment

**Acceptance Criteria:** Section contains all MDP equations, architecture diagram, and references all locked config values from YAML files.

**Estimated Effort:** 10 hours

---

### Issue #41 — Paper: Experimental Setup & Results

**Phase label:** `phase:8-paper`
**Type labels:** `type:documentation`, `type:analysis`
**Component:** `component:paper`
**Priority:** `priority:critical`
**Owner/Role:** Mohar Gorai

**Tasks:**
- [ ] Write Experimental Setup section: AWS configuration, instance types, experiment scenarios E1–E10, evaluation metrics, statistical methodology (5 reps, 95% CI, paired t-test)
- [ ] Write Results section: include all 5 figures from Issue #36, describe PPO vs baseline findings, state significance levels for all comparisons

**Acceptance Criteria:** Results section references all 5 figures. Every quantitative claim has a p-value or CI range cited from `statistical_tests.csv`.

**Estimated Effort:** 10 hours

---

### Issue #42 — Paper: Discussion & Conclusion

**Phase label:** `phase:8-paper`
**Type labels:** `type:documentation`, `type:research`
**Component:** `component:paper`
**Priority:** `priority:high`
**Owner/Role:** [4th member]

**Tasks:**
- [ ] Write Discussion: ablation insights (E10), limitations (single region, synthetic workload), practical deployment considerations
- [ ] Write Conclusion: restate contributions, key findings, future work (multi-region, continuous action space, real e-commerce dataset)
- [ ] Write Acknowledgements section

**Estimated Effort:** 6 hours

---

### Issue #43 — Paper: Final Review & Formatting

**Phase label:** `phase:8-paper`
**Type labels:** `type:documentation`
**Component:** `component:paper`
**Priority:** `priority:high`
**Owner/Role:** Agrima Gupta (lead review) + all members

**Tasks:**
- [ ] Assemble full paper draft in IEEE/ACM LaTeX template
- [ ] Internal review: all 4 members read and comment
- [ ] Address all review comments
- [ ] Confirm all figures are high-resolution (≥ 300 dpi), all references in BibTeX
- [ ] Run grammar check and ensure all claims are backed by data

**Acceptance Criteria:** All 4 team members sign off on final draft. Paper length within conference page limit.

**Estimated Effort:** 8 hours

---

## Phase 9 — Demo, Reproducibility & Teardown

**Objective:** Deliver a live demo, tag v1.0 release, and fully tear down all AWS resources.
**Duration:** 3 days | **AWS Cost:** $1–2 (demo session ALB)

---

### Issue #44 — Demo Dashboard

**Phase label:** `phase:9-demo`
**Type labels:** `type:implementation`
**Component:** `component:metrics`
**Priority:** `priority:normal`
**Owner/Role:** Mohar Gorai

**Tasks:**
- [ ] Complete `src/metrics/visualiser.py` Dash app with live-updating charts for demo
- [ ] Add side-by-side PPO vs RR routing animation during live JMeter run
- [ ] Test on a 2-minute JMeter run (E2 scenario, reduced load) before demo day

**Acceptance Criteria:** Dashboard shows live metrics updating every 30s during demo JMeter run.

**Estimated Effort:** 6 hours

---

### Issue #45 — v1.0 Release Tag

**Phase label:** `phase:9-demo`
**Type labels:** `type:documentation`, `type:reproducibility`
**Component:** `component:paper`
**Priority:** `priority:high`
**Owner/Role:** Devkanti Sarkar

**Tasks:**
- [ ] Ensure `main` branch contains all merged work from `develop`
- [ ] Update `README.md`: project description, architecture diagram, quick-start, team, citation
- [ ] Create git tag: `git tag -a v1.0 -m "FlashBalanceAI v1.0 — BCSE355L final submission"` and push
- [ ] Create GitHub Release: attach paper PDF, reproducibility package, final figures
- [ ] Verify: tag visible on GitHub, release notes complete

**Acceptance Criteria:** `git tag v1.0` pushed to remote; GitHub Release created with all deliverables attached.

**Estimated Effort:** 3 hours

---

### Issue #46 — AWS Teardown

**Phase label:** `phase:9-demo`
**Type labels:** `type:infrastructure`
**Component:** `component:aws-infra`
**Priority:** `priority:critical`
**Labels:** `cost:aws-billing`
**Owner/Role:** All members

**Description:**
Delete ALL AWS resources to eliminate ongoing charges. Verify in billing console that no billable resources remain.

**Tasks:**
- [ ] Run `src/aws/deploy.py teardown_all()`: delete ALB → delete target group → terminate all EC2 instances → delete ASG → delete launch template → delete Lambda functions × 3 → delete DynamoDB table → delete CloudWatch alarms × 2 → delete IAM roles × 2 → delete CloudWatch log groups
- [ ] Verify S3 bucket objects: download final results archive, then delete bucket or set lifecycle to delete all objects after 30 days
- [ ] Check AWS Cost Explorer: confirm zero billable resources in all categories
- [ ] Check billing alarm: should show no projected charges
- [ ] Document teardown completion in `decisions/ADR-002.md`

**Acceptance Criteria:**
- AWS console shows 0 running EC2 instances, 0 ALBs, 0 Lambda functions, 0 DynamoDB tables in us-east-1
- Final AWS bill ≤ $15 total (target $15 budget per ADR-002)
- All 4 team members confirm teardown complete

**Estimated Effort:** 2 hours

---

## 13. Critical Path

The following sequence of issues cannot be parallelised — each one blocks the next:

```
#1 ADR
→ #2 Scaffold
  → #3 Python env
    → #5 FlashSaleEnv ──────────────────────────┐
    → #8 Traffic Generator                       │
      → #11 PPO Training ──────────────────────→ #13 Local Gate
        → #14 Billing Alarm                      │
          → #15 IAM                              │
            → #17 EC2/ASG ──────────────────────→ #18 ALB
              → #20 State Collector              │
                → #22 Inference Server+Lambda ──→ #24 E2E Integration Test
                  → #26 E2 Core Experiment
                    → #35 Statistical Analysis
                      → #36 Figures
                        → #40 Paper Methods
                          → #43 Paper Review
                            → #45 v1.0 Release
                              → #46 Teardown
```

**Total critical path:** ~55 days (parallel work reduces wall time to ~35 days)

---

## 14. Parallelisable Work

These groups of issues can proceed simultaneously once their shared dependency is met:

| Can start after | Parallel work |
|-----------------|---------------|
| Issue #2 (scaffold) | #3 (env setup), #9 (JMeter plans), #10 (Alibaba trace) |
| Issue #5 (FlashSaleEnv) | #6 (baselines), #7 (metrics), #8 (traffic gen), #12 (DQN training) |
| Issue #13 (local gate) | #14–#19 (all AWS infra issues) in sequence per member |
| Issue #24 (e2e test) | #25–#34 (E1–E10 experiments) — assign 2–3 experiments per member |
| Issue #35 (stats) | #36 (figures), #37 (repro package), #38 (notebooks) |
| Issue #39 (intro) | #40 (methodology), #41 (results), #42 (discussion) |

---

## 15. Release Checkpoints

| Tag | Milestone | Trigger | Contents |
|-----|-----------|---------|----------|
| `v0.1-scaffold` | M0 complete | Issue #2 merged to develop | Directory structure, configs, ADR, .gitignore |
| `v0.2-local` | M1–M2 complete | Issues #3–#10 merged | FlashSaleEnv, baselines, traffic generator, JMeter plans |
| `v0.3-drl` | M3 complete | Issues #11–#13 merged | Trained PPO + DQN models, local gate results |
| `v0.4-aws` | M4–M5 complete | Issues #14–#24 merged | Full AWS pipeline working, integration test passed |
| `v0.5-results` | M6–M7 complete | Issues #25–#38 merged | All experiments done, figures generated |
| `v0.9-paper` | M8 complete | Issues #39–#43 merged | Paper draft complete |
| **`v1.0`** | M9 complete | Issues #44–#46 merged | Final release: paper + code + teardown confirmed |

---

## 16. Contribution Tracking

Per BCSE355L guideline: each team member must have ≥ 2 PRs merged, ≥ 2 code reviews, and regular weekly commits.

| Team Member | Primary Issues | Secondary Issues | PR Requirement | Code Review Requirement |
|-------------|---------------|------------------|----------------|------------------------|
| **Devkanti Sarkar** (24BIT0162) | #1, #9, #11, #14, #17, #20, #25, #29, #33, #36, #39, #45 | Review: #5, #22 | ≥ 2 PRs | ≥ 2 reviews |
| **Agrima Gupta** (24BIT0253) | #2, #7, #8, #13, #15, #18, #22, #24, #26, #30, #35, #40, #43 | Review: #11, #35 | ≥ 2 PRs | ≥ 2 reviews |
| **Mohar Gorai** | #4, #6, #10, #12, #21, #23, #27, #31, #37, #41, #44 | Review: #6, #24 | ≥ 2 PRs | ≥ 2 reviews |
| **[4th member]** | #3, #16, #19, #28, #32, #38, #42 | Review: #13, #46 | ≥ 2 PRs | ≥ 2 reviews |

### Contribution Area Mapping

| Contribution Area | Issues | Primary Member |
|-------------------|--------|---------------|
| Literature / Research | #1, #39 | Devkanti |
| Research Gap Analysis | #1, #10 | Devkanti, Mohar |
| Development (local) | #3–#8, #11–#13 | All |
| Cloud Integration | #14–#24 | Agrima, Devkanti |
| AI/ML (DRL) | #11, #12, #13, #34 | Devkanti, Mohar, Agrima |
| Experimentation | #25–#34 | All |
| Testing | #5, #6, #7, #13, #24 | All |
| Documentation | #1, #2, #39–#43 | All |
| Presentation | #44, #45 | Mohar, Devkanti |
| GitHub Commits | All issues | All (weekly minimum) |

---

*Generated from PRD.md (v1.0) + IMPLEMENTATION_PLAN_PART1.md + IMPLEMENTATION_PLAN_PART2.md. Do not duplicate content from those documents — this file tracks execution, not specification.*
