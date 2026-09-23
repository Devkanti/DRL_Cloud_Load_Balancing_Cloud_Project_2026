# GitHub Project Plan — FlashBalanceAI
## BCSE355L Cloud Architecture Design | Deep Reinforcement Learning-Based Adaptive Cloud Load Balancing

**Project:** FlashBalanceAI: PPO Framework for Adaptive Cloud Load Balancing under Flash-Sale Burst Traffic  
**Repository:** `DRL_Cloud_Load_Balancing_Cloud_Project_2026`  
**Team:** Devkanti Sarkar (24BIT0162) · Agrima Gupta (24BIT0253) · Mohar Gorai · [4th member]  
**Branch Model:** `main` ← `develop` ← `feature/*` (already implemented — not redesigned here)  
**Document Version:** 1.0 | **Date:** 2026-08-20

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
| `type:infrastructure` | `#5319e7` | AWS setup, IAM, CloudFormation, CI/CD |
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
| `phase:5-integration` | Lambda integration, end-to-end pipeline |
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
| `component:aws-lambda` | Lambda functions (collector, inference, scaling) |
| `component:aws-infra` | EC2, ALB, ASG, CloudWatch, S3, DynamoDB |
| `component:metrics` | MetricsCollector and visualiser |
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
| `gate:required` | This issue is a gate check — work downstream cannot start until it is closed |
| `blocked` | Cannot proceed; blocking issue must be linked |
| `good-first-issue` | Suitable as a first task for a new contributor |

---

## 2. Milestone Overview

| # | Milestone | Phase(s) | Calendar Target | Exit Gate |
|---|-----------|----------|-----------------|-----------|
| M0 | Design Lock & Repo Scaffold | 0 | Week 1 Day 1–2 | ADR-001.md signed by all 4 members; repo structure verified |
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

---

## Phase 0 — Design Lock & Repository Setup

**Milestone:** M0 | **Duration:** 1–2 days | **All costs: Free**

**Objective:** Lock all architecture decisions and create the canonical repository structure so every team member builds to the same specification.

**Deliverables:** `decisions/ADR-001.md`, fully scaffolded repository directories, `.gitignore`, `README.md` stub, branch structure.

**Exit Criteria:** All 4 members have committed a sign-off to `ADR-001.md`. `git ls-tree -r --name-only HEAD` shows all required directories. `pip install -r requirements.txt` succeeds on all 4 machines.

---

### Issue #1 — Write Architecture Decision Record (ADR-001)

**Labels:** `type:documentation` `phase:0-design` `component:environment` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar (lead) — all 4 members must sign off  
**Estimated Effort:** 3 hours (1 meeting + write-up)  
**Depends On:** Nothing (first issue)  
**Blocks:** All other issues

**Description:**  
Hold a synchronous meeting to resolve any design disagreements from the three Phase-I PDFs. Document all locked decisions in `decisions/ADR-001.md`.

**Decisions to lock:**
- Primary DRL algorithm: **PPO** via Stable-Baselines3
- Primary baseline: **DQN** (identical architecture)
- Action space: **Discrete, N=4** backend instances
- State dimensions: **23** (5N+3, N=4)
- Reward weights: `w1=0.4, w2=0.2, w3=0.2, w4=0.2` (subject to E10 ablation)
- AWS region: **us-east-1**
- Max EC2 instances: 4 backend + 1 inference = 5 total t2.micro
- Training location: **local laptop first**; SageMaker only as fallback
- Traffic generator: `src/traffic/traffic_generator.py` + JMeter

**Acceptance Criteria:**
- [ ] `decisions/ADR-001.md` exists in the repository
- [ ] All 4 members have committed their acknowledgement (Git author name visible in log)
- [ ] No unresolved design questions remain open
- [ ] Document references PRD.md Sections 7–12 for full context

---

### Issue #2 — Scaffold Repository Directory Structure

**Labels:** `type:implementation` `phase:0-design` `priority:critical-path` `costs:free` `good-first-issue`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #1 (ADR must be signed before structure is locked)  
**Blocks:** Issues #3–#7

**Description:**  
Create the full canonical directory tree as specified in `IMPLEMENTATION_PLAN_PART1.md` T0.2. Include all `__init__.py` placeholders, `configs/`, `experiments/`, `notebooks/`, `infra/`, `decisions/`, and `src/` subdirectories.

**Files to create:**
- `src/environment/__init__.py`, `src/agents/__init__.py`, `src/baselines/__init__.py`
- `src/traffic/__init__.py`, `src/backend/__init__.py`, `src/aws/__init__.py`, `src/metrics/__init__.py`
- `src/tests/` (empty directory with `.gitkeep`)
- `experiments/results/.gitkeep`, `experiments/analysis/.gitkeep`
- `notebooks/01_traffic_analysis.ipynb` (empty), `02_ppo_training.ipynb`, `03_results_analysis.ipynb`, `04_ablation_study.ipynb`
- `configs/ppo_config.yaml`, `configs/dqn_config.yaml`, `configs/traffic_config.yaml`, `configs/aws_config.yaml`
- Updated `.gitignore` entries: `*.pyc`, `__pycache__/`, `.env`, `*.jtl`, `models/*.zip`, `data/alibaba_trace/`, `experiments/results/raw/`

**Acceptance Criteria:**
- [ ] `git ls-tree -r --name-only HEAD` shows all directories listed above
- [ ] No import errors from any `__init__.py`
- [ ] `.gitignore` does not accidentally exclude source files
- [ ] `configs/` YAML files contain valid (even if placeholder) content
- [ ] PR merged into `develop` with at least one other member as reviewer

---

## Phase 1 — Local Development Foundation

**Milestone:** M1 | **Duration:** 5–7 days | **All costs: Free (local only)**

**Objective:** Build all core software components locally — Flask backend, Gymnasium environment, baseline load balancers, and metrics collection — before any AWS or DRL work begins.

**Deliverables:** `src/backend/app.py`, `src/environment/flash_sale_env.py`, `src/baselines/` (4 classes), `src/metrics/collector.py`, `environment.yml`, `requirements.txt`, all passing unit tests.

**Exit Criteria:** `pytest src/tests/ -v` passes with 0 failures. `check_env(FlashSaleEnv())` raises no errors. All 4 members can activate the conda env and run the verify command.

---

### Issue #3 — Create Reproducible Python Environment

**Labels:** `type:implementation` `phase:1-local-dev` `type:reproducibility` `priority:critical-path` `costs:free` `good-first-issue`  
**Owner:** All 4 members (each must verify on their own machine)  
**Estimated Effort:** 1 hour  
**Depends On:** Issue #2  
**Blocks:** Issues #4–#8

**Description:**  
Create a reproducible conda environment that all 4 team members can activate identically. Pin all package versions as specified in `IMPLEMENTATION_PLAN_PART1.md` T1.1.

**Key packages (pinned):**
- `stable-baselines3==2.3.0`, `gymnasium==0.29.1`, `torch==2.2.2`
- `numpy==1.26.4`, `pandas==2.2.1`, `boto3==1.34.84`
- `flask==3.0.3`, `dash==2.17.0`, `plotly==5.22.0`
- `pytest==8.1.1`, `pytest-cov==5.0.0`, `pyyaml==6.0.1`
- `requests==2.31.0`, `locust==2.24.1`

**Acceptance Criteria:**
- [ ] `environment.yml` and `requirements.txt` committed to repo root
- [ ] `python -c "import stable_baselines3; import gymnasium; print('OK')"` succeeds on all 4 machines
- [ ] Python version is exactly 3.11 (not 3.12 — SB3 compatibility confirmed on 3.11)
- [ ] No version conflicts reported by `pip check`

---

### Issue #4 — Implement Flask Backend Server (EC2 Simulator)

**Labels:** `type:implementation` `phase:1-local-dev` `component:backend` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #9 (JMeter plans), Issue #14 (EC2 deployment)

**Description:**  
Create `src/backend/app.py` — a Flask HTTP server that simulates an EC2 product-catalogue backend with configurable artificial latency, three endpoints, and background CloudWatch metric publishing.

**Endpoints required:**
- `GET /health` → `{"status": "healthy", "instance": INSTANCE_ID}`
- `GET /product/<pid>` → JSON with configurable latency (BASE_LATENCY_MS env var)
- `GET /metrics` → `{"queue_depth": int, "instance": INSTANCE_ID}`

**Configuration via environment variables:** `BASE_LATENCY_MS`, `INSTANCE_ID`, `PORT`

**Acceptance Criteria:**
- [ ] All 3 endpoints return HTTP 200 with valid JSON
- [ ] Latency measurably increases when `BASE_LATENCY_MS` is raised
- [ ] 4 instances start on ports 5001–5004 without port conflict
- [ ] `src/backend/requirements_backend.txt` contains `flask==3.0.3 gunicorn==21.2.0`
- [ ] PR includes a brief manual test log (curl output) as a comment

---

### Issue #5 — Implement Custom Gymnasium Environment (FlashSaleEnv)

**Labels:** `type:implementation` `phase:1-local-dev` `component:environment` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 6 hours  
**Depends On:** Issue #3, Issue #8 (traffic generator must exist first — can stub `generate_profile`)  
**Blocks:** Issues #11–#12 (PPO/DQN training)

**Description:**  
Create `src/environment/flash_sale_env.py` implementing `gymnasium.Env` with a 23-dimensional state space (5N+3, N=4), discrete action space (N=4), four-component burst-aware reward function, and full episode lifecycle.

**State vector:** `[cpu_util×4, active_conn×4, queue_depth×4, resp_time_ema×4, health_status×4, arrival_rate_norm, burst_indicator, time_since_spike]`

**Reward function:** `R = 0.4·R_lat + 0.2·R_util + 0.2·R_tput + 0.2·R_sla` (weights per ADR-001)

**Unit tests to write in `src/tests/test_environment.py`:**
- `test_obs_shape()` — assert `obs.shape == (23,)` for N=4
- `test_action_space()` — assert action space size == 4
- `test_episode_completes()` — run 7800 steps, verify `done == True`
- `test_reward_range()` — reward in `[-3, 2]` for all states
- `test_reward_components()` — verify each R_i component in expected range

**Acceptance Criteria:**
- [ ] `pytest src/tests/test_environment.py -v` passes (0 failures)
- [ ] `stable_baselines3.common.env_checker.check_env(FlashSaleEnv())` raises no warnings or errors
- [ ] One full episode (7800 steps) completes in < 10 seconds on a laptop
- [ ] `reset(seed=42)` twice produces identical initial observations

---

### Issue #6 — Implement Baseline Load Balancers

**Labels:** `type:implementation` `phase:1-local-dev` `component:baselines` `priority:high` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 3 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #12 (local evaluation)

**Description:**  
Create software implementations of all four non-DRL baselines in `src/baselines/`. All must expose a `.select(state)` method returning an integer in `[0, N-1]`.

**Classes to implement:**
1. `src/baselines/round_robin.py` — `RoundRobinLB`
2. `src/baselines/weighted_round_robin.py` — `WeightedRoundRobinLB` (weight ∝ 1/cpu_util)
3. `src/baselines/least_connections.py` — `LeastConnectionsLB`
4. `src/baselines/threshold_autoscaler.py` — `ThresholdAutoscaler` (mimics CloudWatch CPU>70% alarm logic with consecutive-period counting)

**Unit tests in `src/tests/test_baselines.py`:**
- Each baseline's `select()` returns an integer in `[0, N-1]`
- `ThresholdAutoscaler.check_scaling()` returns `"scale_out"` after 2 consecutive high-CPU states
- All baselines run 7800 steps in `FlashSaleEnv` without error or exception

**Acceptance Criteria:**
- [ ] `pytest src/tests/test_baselines.py -v` passes (0 failures)
- [ ] All baselines complete a full 7800-step episode in `FlashSaleEnv` without error
- [ ] No hardcoded instance counts (N must be constructor-configurable)

---

### Issue #7 — Implement Metrics Collection Module

**Labels:** `type:implementation` `phase:1-local-dev` `component:metrics` `priority:high` `costs:free`  
**Owner:** All (any available member — ~2 hours)  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #12 (local evaluation)

**Description:**  
Create `src/metrics/collector.py` with `MetricsCollector` class. Must record per-step data and compute standardised summary statistics for both local and AWS experiment runs.

**Summary statistics required:** `mean_response_ms`, `p95_response_ms`, `p99_response_ms`, `throughput_served`, `failure_rate`, `mean_cpu`, `total_reward`

**Unit tests in `src/tests/test_metrics.py`:**
- `compute_summary()` returns all 7 required keys
- Collector records 7800 steps without memory issues (< 50 MB RAM)
- `save(path)` produces a valid CSV readable by `pandas.read_csv()`

**Acceptance Criteria:**
- [ ] `pytest src/tests/test_metrics.py -v` passes
- [ ] `compute_summary()` returns all required keys with numerically plausible values
- [ ] CSV output is tab/comma separated and includes a header row

---

## Phase 2 — Dataset & Traffic Generation

**Milestone:** M2 | **Duration:** 3–4 days | **Parallel with Phase 1 after Issue #3**

**Objective:** Build the synthetic flash-sale traffic generator, JMeter test plans, and acquire/process the Alibaba supplementary dataset.

**Deliverables:** `src/traffic/traffic_generator.py`, `configs/traffic_config.yaml`, JMeter `.jmx` files, `data/alibaba_processed.csv`, traffic profile plots in `notebooks/01_traffic_analysis.ipynb`.

**Exit Criteria:** All traffic generator unit tests pass. Visual inspection of profile plots confirms correct phase transitions (warm-up → pre-burst → spike → peak → cooldown). JMeter completes a 13-minute local run against Flask backends.

---

### Issue #8 — Implement Synthetic Flash-Sale Traffic Generator

**Labels:** `type:implementation` `phase:2-dataset` `component:traffic-gen` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 4 hours  
**Depends On:** Issue #3  
**Blocks:** Issue #5 (Gymnasium env), Issues #11–#12 (training)

**Description:**  
Create `src/traffic/traffic_generator.py` with `TrafficConfig` dataclass and `generate_profile()` / `generate_repeated_bursts()` functions producing normalised arrival-rate time series with correct flash-sale phase transitions and reproducible seeding.

**Traffic phases:** warm-up (2 min) → pre-burst (1 min, 3× baseline) → spike onset (10s, linear ramp) → peak (5 min, burst_multiplier×) → cooldown (exponential decay back to 1×)

**Scenarios in `configs/traffic_config.yaml`:** `e1_baseline` (1×), `e2_10x` (10×), `e3_50x` (50×), `e4_100x` (100×), `e6_noisy` (2×, noise_std=0.40)

**Unit tests in `src/tests/test_traffic_generator.py`:**
- `test_profile_length()` — `len(profile) == n_steps`
- `test_baseline_range()` — warm-up steps have mean ≈ 1.0 ± 0.2
- `test_burst_peak()` — peak steps have mean ≈ burst_multiplier ± 10%
- `test_reproducibility()` — same seed → bit-identical profile
- `test_different_seeds()` — different seeds → different profiles

**Acceptance Criteria:**
- [ ] All 5 unit tests pass
- [ ] Visual plot in `notebooks/01_traffic_analysis.ipynb` shows correct phase transitions for 10× and 50× scenarios
- [ ] `generate_repeated_bursts(n_bursts=3)` produces concatenated profile of correct length

---

### Issue #9 — Create JMeter Load-Injection Test Plans

**Labels:** `type:implementation` `phase:2-dataset` `component:traffic-gen` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 3 hours  
**Depends On:** Issue #4 (Flask backend must be runnable)  
**Blocks:** Issues #19–#28 (experiment runs)

**Description:**  
Create parameterised Apache JMeter 5.6.3 test plans for all experiment scenarios. Plans must be runnable locally against Flask and against AWS ALB with a single property change.

**Files to create in `src/traffic/jmeter_configs/`:**
- `flash_sale_10x.jmx` — peak_users=1000, 13-minute total
- `flash_sale_50x.jmx` — peak_users=5000, 13-minute total
- `flash_sale_repeated_bursts.jmx` — 3 consecutive 10× bursts, ~45 minutes
- `flash_sale_noisy.jmx` — 2× bursts with noise_std=0.40

**Each plan must include:** Thread Group with ramp-up, HTTP Sampler (`GET /product/${__Random(1,1000)}`), Response Assertion (status 200), CSV listener saving to `experiments/results/`.

**Parameterisation via JMeter properties:** `-Jhost=ALB_DNS -Jpeak_users=1000 -Jresultsfile=...`

**Acceptance Criteria:**
- [ ] Each `.jmx` file completes a 13-minute run against localhost (4 Flask servers on ports 5001–5004)
- [ ] Output CSV has columns: `timeStamp`, `elapsed`, `label`, `responseCode`, `bytes`
- [ ] No hardcoded host/port in plan XML

---

### Issue #10 — Acquire and Preprocess Alibaba Cluster Trace (Supplementary)

**Labels:** `type:research` `phase:2-dataset` `component:dataset` `priority:normal` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #3  
**Blocks:** Nothing critical (supplementary only)

**Description:**  
Download Alibaba Cloud Cluster Trace v2018 from https://github.com/alibaba/clusterdata. Process into a normalised arrival-rate time series for supplementary comparison against the synthetic dataset.

> **⚠️ Research Integrity:** The Alibaba trace is batch job scheduling data, NOT e-commerce request routing data. It must NOT be used as the primary evaluation dataset. Use only for: (a) arrival-rate distribution comparison in paper supplementary, (b) optional agent pre-training warm-start.

**Steps:**
1. Download `batch_task.tar.gz` (~2 GB compressed)
2. Add `data/alibaba_trace/` to `.gitignore`
3. Process in `notebooks/01_traffic_analysis.ipynb` — compute per-minute arrival counts, normalise to FlashSale-Synthetic format
4. Save `data/alibaba_processed.csv`
5. Add plot comparing Alibaba arrival-rate distribution vs synthetic flash-sale profile

**Acceptance Criteria:**
- [ ] `data/alibaba_processed.csv` loads without error via `pandas.read_csv()`
- [ ] Per-minute arrival-rate distribution plot committed to notebook
- [ ] Raw trace directory is gitignored (NOT committed to repo)
- [ ] Notebook cell contains explicit comment: "Alibaba trace = batch jobs, supplementary use only"

---

## Phase 3 — DRL Implementation

**Milestone:** M3 | **Duration:** 7–10 days | **Local only**

**Objective:** Train PPO and DQN agents to convergence on `FlashSaleEnv`, verify reward convergence, and run a local evaluation gate check before touching AWS.

**Deliverables:** `models/ppo/ppo_flash_v1.zip`, `models/dqn/dqn_flash_v1.zip`, TensorBoard reward convergence curves, preliminary local results table in `notebooks/03_results_analysis.ipynb`.

**Exit Criteria (gate check — mandatory before Phase 4):** PPO reward curve shows monotonic improvement. `model.predict(obs)` returns action in `[0, 3]` within 5ms. PPO outperforms Round Robin on P95 latency by ≥ 10% in local evaluation. No NaN/Inf values in any metric.

---

### Issue #11 — Train PPO Agent

**Labels:** `type:implementation` `phase:3-drl` `component:ppo-agent` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 9 hours (5h implementation + 4h training)  
**Depends On:** Issue #5 (environment), Issue #8 (traffic generator)  
**Blocks:** Issue #13 (local evaluation), Issue #17 (PPO inference Lambda)

**Description:**  
Create `src/agents/ppo_agent.py` with `train_ppo()` and `load_ppo()` functions using Stable-Baselines3 PPO. Train for 2M timesteps with 4 vectorised environments, EvalCallback for early stopping, and CheckpointCallback every 100k steps.

**Hyperparameters (from `configs/ppo_config.yaml`):**
- `learning_rate: 0.0003`, `n_steps: 2048`, `batch_size: 64`, `n_epochs: 10`
- `gamma: 0.99`, `gae_lambda: 0.95`, `clip_range: 0.2`, `ent_coef: 0.01`
- `net_arch: [64, 64]`, `total_timesteps: 2000000`

**Training seeds:** SEED_TRAIN=42, SEED_VAL=43. Test seeds (127–141) must NOT be used during training or hyperparameter tuning.

**Convergence criterion:** Mean episode reward change < 1% over last 500k steps.

**Acceptance Criteria:**
- [ ] Training completes without error
- [ ] TensorBoard shows reward monotonically increasing and plateauing (screenshot committed to `decisions/`)
- [ ] `model.predict(obs)` returns integer in `[0, 3]` within 5ms
- [ ] Final model saved as `models/ppo/ppo_flash_v1.zip`
- [ ] `pytest src/tests/test_ppo_agent.py -v` passes

---

### Issue #12 — Train DQN Agent

**Labels:** `type:implementation` `phase:3-drl` `component:dqn-agent` `priority:high` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 9 hours (4h implementation + 5h training)  
**Depends On:** Issue #5 (environment must be frozen before DQN training starts), Issue #11 started  
**Blocks:** Issue #13 (local evaluation)

**Description:**  
Create `src/agents/dqn_agent.py` using Stable-Baselines3 DQN. Train on identical `FlashSaleEnv` environment version as PPO for a fair comparison. Architecture must match PPO: 2-layer MLP [64, 64], same action/observation space.

**Hyperparameters (from `configs/dqn_config.yaml`):**
- `learning_rate: 0.0001`, `buffer_size: 100000`, `learning_starts: 10000`
- `batch_size: 64`, `gamma: 0.99`, `target_update_interval: 1000`
- `exploration_fraction: 0.15`, `exploration_final_eps: 0.05`

> **⚠️ Important:** Use the same test seeds (127–141) as PPO for comparison. The environment version must be frozen at the same commit as T3.1 — do not update `FlashSaleEnv` after DQN training begins.

**Acceptance Criteria:**
- [ ] DQN training completes without error
- [ ] TensorBoard shows reward improvement over training steps
- [ ] Final model saved as `models/dqn/dqn_flash_v1.zip`
- [ ] `pytest src/tests/test_dqn_agent.py -v` passes

---

### Issue #13 — Local Evaluation Gate Check (All 6 Algorithms)

**Labels:** `type:experiment` `phase:3-drl` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** All 4 members (collaborative notebook session)  
**Estimated Effort:** 4 hours  
**Depends On:** Issues #11, #12, #6, #7  
**Blocks:** Issue #14 (AWS deployment — must not start until this gate passes)

**Description:**  
Run all 6 algorithms (PPO, DQN, RR, WRR, LC, Threshold) on test seeds 127–131 in `FlashSaleEnv`. Generate a preliminary comparison table. This is a mandatory gate check — AWS deployment cannot begin until PPO passes the quality threshold.

**Notebook:** `notebooks/03_results_analysis.ipynb`

**Gate pass conditions:**
1. PPO reward curve shows monotonic improvement (no flat-line from step 0)
2. PPO outperforms Round Robin on P95 latency by ≥ 10% on at least 3 of 5 test seeds
3. No NaN or Inf values in any metric for any algorithm
4. DQN produces results meaningfully different from random action selection

**If gate fails:** Debug `FlashSaleEnv` reward function and retrain before proceeding. Do not advance to Phase 4.

**Deliverables:**
- Preliminary results table (P95 latency, CPU utilisation, SLA violation rate) committed to `experiments/analysis/`
- Reward convergence plots for PPO and DQN

**Acceptance Criteria:**
- [ ] All 6 algorithms complete 5 full episodes without error
- [ ] Gate pass conditions 1–4 all satisfied
- [ ] Preliminary table committed and linked in this issue's closing comment
- [ ] Team explicitly agrees in a comment that Phase 4 may begin

---

## Phase 4 — AWS Infrastructure Deployment

**Milestone:** M4 | **Duration:** 3–4 days | **⚠️ Some tasks incur AWS charges**

**Objective:** Provision all required AWS resources safely, with billing alarms set before any chargeable resource is launched.

**Deliverables:** Live AWS stack (EC2 ASG, ALB, Lambda×3, CloudWatch, S3, DynamoDB), IAM roles, billing alarm at $5.

**Exit Criteria:** All AWS resources exist and are reachable. `aws sts get-caller-identity` returns correct account. EC2 health checks pass. Billing alarm confirmed in CloudWatch console.

> **⚠️ Safety Rule:** Issue #14 (billing alarm) must be merged and confirmed before any other Phase 4 issue is opened.

---

### Issue #14 — AWS Account Safety Setup & Billing Alarm

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #13 (gate check must pass)  
**Blocks:** All other Phase 4 issues

**Description:**  
Configure the AWS student account with billing protection before any chargeable resources are launched.

**Steps:**
1. Enable Cost Explorer and Billing Dashboard
2. Create CloudWatch billing alarm: threshold $5 → SNS email notification
3. Create hard budget ceiling: $20 total via AWS Budgets
4. Set default region to `us-east-1` in AWS CLI config
5. Document account ID and region in `configs/aws_config.yaml` (no secrets — use env vars for credentials)
6. Verify EC2 t2.micro instance quota ≥ 5 in us-east-1 (request increase if needed)

**Acceptance Criteria:**
- [ ] Billing alarm confirmed active in CloudWatch console (screenshot in PR)
- [ ] `aws sts get-caller-identity` returns correct account ID
- [ ] EC2 quota ≥ 5 t2.micro instances confirmed
- [ ] `configs/aws_config.yaml` committed (no credentials — region and resource name templates only)

---

### Issue #15 — Create IAM Roles and Policies

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #14  
**Blocks:** Issues #16–#18 (all resource setup)

**Description:**  
Create IAM roles following least-privilege principle. Implement in `src/aws/iam_setup.py` using boto3 so setup is reproducible.

**Roles to create:**
1. `FlashBalanceAI-Lambda-Role` — permissions: `cloudwatch:GetMetricStatistics`, `cloudwatch:PutMetricData`, `s3:GetObject`, `s3:PutObject`, `dynamodb:PutItem`, `elasticloadbalancing:ModifyTargetGroup`, `autoscaling:SetDesiredCapacity`
2. `FlashBalanceAI-EC2-Role` — permissions: `cloudwatch:PutMetricData`, `s3:GetObject`

**Acceptance Criteria:**
- [ ] Both IAM roles exist in the account (verify via `aws iam get-role`)
- [ ] `src/aws/iam_setup.py` script creates roles idempotently (safe to re-run)
- [ ] No wildcard `*` permissions granted
- [ ] IAM policy JSON documents committed to `decisions/iam_policies/`

---

### Issue #16 — Provision S3 Bucket and Upload Trained Models

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:aws-billing`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 1 hour  
**Depends On:** Issue #15, Issue #11 (trained PPO model)  
**Blocks:** Issues #17–#18 (Lambda functions need models in S3)

**Description:**  
Create the S3 bucket `flashbalanceai-{ACCOUNT_ID}` in us-east-1 with versioning enabled and public access blocked. Upload trained model artifacts.

**Bucket structure:**
```
flashbalanceai-{ACCOUNT_ID}/
├── models/ppo_flash_v1.zip
├── models/dqn_flash_v1.zip
├── state/current.json          (written by Lambda at runtime)
├── results/                    (experiment raw data)
└── configs/                    (runtime configuration copies)
```

**Acceptance Criteria:**
- [ ] Bucket exists with versioning enabled and all public access blocked
- [ ] `aws s3 ls s3://flashbalanceai-{ACCOUNT_ID}/models/` shows both model zips
- [ ] `src/aws/deploy.py` contains bucket creation logic (idempotent)

---

### Issue #17 — Launch EC2 Backend Instances and Auto Scaling Group

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:aws-billing`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 4 hours  
**Depends On:** Issue #15, Issue #4 (Flask backend must be ready for AMI/UserData)  
**Blocks:** Issue #18 (ALB needs target instances)

**Description:**  
Launch 4 t2.micro EC2 instances running the Flask backend via ASG. Configure UserData to install dependencies and start the Flask app on boot.

**Configuration:**
- Instance type: t2.micro (Free Tier eligible, ≤ 750 hrs/month)
- AMI: Amazon Linux 2023 (us-east-1 latest)
- ASG name: `FlashBalanceAI-ASG`, min=2, desired=4, max=8
- Health check: HTTP `/health` on port 5000
- UserData: installs Python 3.11, copies `src/backend/app.py` from S3, starts gunicorn

**Acceptance Criteria:**
- [ ] ASG shows 4 healthy instances in EC2 console
- [ ] `curl http://{INSTANCE_IP}:5000/health` returns `{"status": "healthy"}`
- [ ] CloudWatch agent publishing `CPUUtilization` for each instance
- [ ] ASG scale-out policy: CPU > 70% for 2 periods → +1 instance; scale-in: CPU < 30% for 5 periods → -1

---

### Issue #18 — Deploy Application Load Balancer

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:critical-path` `costs:aws-billing`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #17  
**Blocks:** Issue #20 (CloudWatch state collector Lambda)

**Description:**  
Create an Application Load Balancer (`FlashBalanceAI-ALB`) with a target group (`FlashBalanceAI-TG`) routing to the 4 EC2 instances. Enable access logs to S3.

**Configuration:**
- Scheme: internet-facing, IPv4
- Listener: HTTP:80 → forward to `FlashBalanceAI-TG`
- Target group: HTTP, port 5000, health check path `/health`
- Stickiness: disabled (required for load balancing experiments)

**Acceptance Criteria:**
- [ ] ALB DNS name resolves and returns HTTP 200 from `/health`
- [ ] All 4 targets show `healthy` in target group
- [ ] ALB access logs enabled to `s3://flashbalanceai-{ACCOUNT_ID}/logs/alb/`
- [ ] `aws elbv2 describe-load-balancers` output saved to `decisions/aws_resource_ids.md`

---

### Issue #19 — Create DynamoDB Routing Decisions Table

**Labels:** `type:infrastructure` `phase:4-aws-deploy` `component:aws-infra` `priority:normal` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 0.5 hours  
**Depends On:** Issue #14  
**Blocks:** Issue #21 (PPO inference Lambda writes to DynamoDB)

**Description:**  
Create DynamoDB table `routing_decisions` for logging inference decisions during experiments.

**Schema:** partition key `timestamp` (String), attributes: `experiment_id`, `action`, `state` (JSON string), `arrival_rate`

**Acceptance Criteria:**
- [ ] Table exists: `aws dynamodb describe-table --table-name routing_decisions`
- [ ] Table is in `PAY_PER_REQUEST` billing mode (Free Tier: 25 GB + 200M requests/month)

---

## Phase 5 — AWS Integration & End-to-End Pipeline

**Milestone:** M5 | **Duration:** 5–7 days | **⚠️ Minimal AWS charges during testing**

**Objective:** Deploy Lambda functions, connect CloudWatch state collection to PPO inference, and verify the complete data flow end-to-end before running experiments.

**Deliverables:** 3 Lambda functions deployed, custom CloudWatch namespace `FlashBalanceAI/Instances` publishing, S3 state file updating every 30s, DynamoDB accumulating routing rows.

**Exit Criteria (gate check — mandatory before Phase 6):** T5.5 integration test passes all 6 checks — traffic distributes to all 4 instances, no Lambda errors, DynamoDB has entries, no single instance receives > 80% of requests.

---

### Issue #20 — Deploy CloudWatch State Collector Lambda

**Labels:** `type:implementation` `phase:5-integration` `component:aws-lambda` `priority:critical-path` `costs:aws-billing`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 5 hours  
**Depends On:** Issues #16, #17, #18  
**Blocks:** Issue #21 (inference Lambda reads from S3 state written by this Lambda)

**Description:**  
Create and deploy `src/aws/cloudwatch_collector.py` as Lambda function `FlashBalanceAI-StateCollector`. Polls CloudWatch every 30s, builds the 23-dimensional state vector, writes `state/current.json` to S3.

**State vector construction:** `[cpu×4, connections×4, queue_depth×4, resp_ema×4, health×4, arrival_rate_norm, burst_indicator, time_since_spike]`

**Trigger:** EventBridge rule (30s polling — minimum resolution achievable without Step Functions)

> **⚠️ Note:** AWS EventBridge minimum rate is 1 minute. For 30s polling use a CloudWatch Alarm with a 30s evaluation period as trigger, or accept 1-minute polling and document this as a methodological limitation.

**Acceptance Criteria:**
- [ ] Lambda deploys without error
- [ ] `aws s3 cp s3://flashbalanceai-{ACCOUNT_ID}/state/current.json -` shows a valid 23-element array with CPU values in `[0, 1]`
- [ ] No Lambda errors in CloudWatch Logs for 5 consecutive invocations
- [ ] Lambda execution time < 5s (well within 15s timeout)

---

### Issue #21 — Deploy PPO Inference Server

**Labels:** `type:implementation` `phase:5-integration` `component:ppo-agent` `component:aws-lambda` `priority:critical-path` `costs:aws-billing`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 6 hours  
**Depends On:** Issues #16, #20  
**Blocks:** Issue #24 (end-to-end integration test)

**Description:**  
Deploy `src/aws/inference_server.py` as a Flask application on a dedicated EC2 t2.micro inference server. A lightweight Lambda (`FlashBalanceAI-InferenceCoordinator`) calls the inference server every 30s and updates ALB target weights based on the PPO action.

> **Architecture decision:** Use EC2 inference server rather than Lambda packaging to avoid the 250MB unzipped Lambda size limit imposed by Stable-Baselines3. SB3 model is loaded at EC2 startup from S3.

**Inference loop:**
1. EC2 inference server loads PPO model from S3 at startup
2. Lightweight Lambda reads `state/current.json` from S3
3. Lambda calls `POST /infer` on inference EC2
4. Inference EC2 returns `{"action": int}`
5. Lambda logs action to DynamoDB and updates ALB target group attributes

**Acceptance Criteria:**
- [ ] Inference server starts and loads PPO model from S3 within 60s
- [ ] `/infer` endpoint returns an integer in `[0, 3]` for valid 23-dim input
- [ ] DynamoDB `routing_decisions` table accumulates rows during operation
- [ ] CloudWatch Logs show no errors for 5 consecutive inference cycles

---

### Issue #22 — Deploy Scaling Trigger Lambda

**Labels:** `type:implementation` `phase:5-integration` `component:aws-lambda` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 3 hours  
**Depends On:** Issues #20, #17  
**Blocks:** Issue #24 (integration test)

**Description:**  
Create and deploy `src/aws/scaling_trigger.py` as Lambda function `FlashBalanceAI-ScalingTrigger`. Handles both reactive scaling (CPU > 70% CloudWatch alarm) and proactive pre-scaling (when `burst_indicator=1` in state vector).

**Scaling logic:**
- Proactive: `burst_indicator=1` → immediately add 2 instances (bypass cooldown)
- Reactive: CPU alarm ALARM state → add 1 instance (with cooldown)
- Scale-in: CPU < 30% for 5 consecutive periods → remove 1 instance

**CloudWatch alarms to create:**
- `FlashBalanceAI-HighCPU` — CPU > 70% for 2 evaluation periods (each 30s)
- `FlashBalanceAI-LowCPU` — CPU < 30% for 5 evaluation periods

**Acceptance Criteria:**
- [ ] Manually raising a test instance's CPU above 70% causes ASG desired capacity to increase within 120s
- [ ] Lambda free tier usage confirmed (< 1M invocations/month)
- [ ] Scale-in tested without disrupting active connections

---

### Issue #23 — Add Custom CloudWatch Metrics Publishing to Flask Backend

**Labels:** `type:implementation` `phase:5-integration` `component:backend` `component:aws-lambda` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 2 hours  
**Depends On:** Issues #20, #4  
**Blocks:** Issue #24 (integration test — state collector needs these metrics)

**Description:**  
Add a background metrics-publishing thread to `src/backend/app.py` that pushes `QueueDepth` and `ResponseTimeEMA` to CloudWatch namespace `FlashBalanceAI/Instances` every 30s per instance.

**Metrics published per instance:**
- `QueueDepth` (requests currently being processed)
- `ResponseTimeEMA` (normalised to [0, 1] where 1 = 500ms)

**Acceptance Criteria:**
- [ ] CloudWatch console shows `FlashBalanceAI/Instances` namespace with 4 instance dimensions
- [ ] Metrics update every 30s (confirmed via `aws cloudwatch get-metric-statistics`)
- [ ] 8 metrics total (2 × 4 instances) — within Free Tier (10 custom metrics free)

---

### Issue #24 — End-to-End Integration Test (Gate Check)

**Labels:** `type:testing` `phase:5-integration` `priority:critical-path` `gate:required` `costs:aws-billing`  
**Owner:** All 4 members  
**Estimated Effort:** 3 hours  
**Depends On:** Issues #20–#23  
**Blocks:** All Phase 6 experiment issues (none may start until this gate passes)

**Description:**  
Run a 5-minute JMeter test at baseline load (100 req/s) against the ALB DNS and verify all 6 pipeline components simultaneously.

**Verification checklist:**
1. CloudWatch: `FlashBalanceAI/Instances` metrics updating every 30s ✓
2. S3 `state/current.json`: valid 23-element vector with CPU in [0, 1] ✓
3. DynamoDB `routing_decisions`: rows accumulating with valid `action` values ✓
4. ALB access logs: requests distributed across all 4 instances ✓
5. No Lambda errors in CloudWatch Logs ✓
6. No single instance receiving > 80% of requests ✓

**Acceptance Criteria:**
- [ ] All 6 checklist items confirmed and documented with evidence (CloudWatch screenshots, S3 JSON snippet, DynamoDB row count)
- [ ] Test cost recorded: estimated ~$0.05 for 1 hour
- [ ] Gate sign-off committed to `decisions/integration_test_report.md` by all 4 members

---

## Phase 6 — Experimental Campaign

**Milestone:** M6 | **Duration:** 7–10 days | **⚠️ Primary cost phase (~$8–18 total)**

**Objective:** Execute experiments E1–E10 with the specified repetitions and collect all raw data. No analysis in this phase — data collection only.

**Deliverables:** Raw `.jtl` files and CloudWatch metric exports for every run, uploaded to S3 and committed (or linked) in `experiments/results/`.

**Exit Criteria:** Every experiment (E1–E10) has raw data files for the specified minimum repetitions. No file is empty or corrupted. All raw data is backed up to S3.

> **⚠️ Research Integrity:** No results from these experiments may be reported in the paper until (a) the experiment has completed the minimum specified repetitions, (b) all raw data has been saved and committed, and (c) statistical analysis has been completed per Phase 7.

---

### Issue #25 — E1: Baseline Traffic Calibration (1× Load)

**Labels:** `type:experiment` `phase:6-experiments` `priority:critical-path` `costs:aws-billing`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 4 hours | **AWS Cost:** ~$0.50  
**Depends On:** Issue #24 (integration gate must pass)  
**Blocks:** Issues #26–#28 (all spike experiments)

**Description:**  
Run all 6 algorithms at constant 100 req/s for 13 minutes each. Purpose: confirm systems work at normal load and calibrate baselines. If PPO shows large divergence from RR at baseline, the reward function needs debugging.

**Parameters:** 100 req/s constant, 5 repetitions per algorithm, seeds 127–131, 6 algorithms (PPO, DQN, RR, WRR, LC, Threshold).

**Metrics collected per run:** mean latency (ms), P95/P99 latency (ms), throughput (req/s), failure rate (%), CPU utilisation per instance (%), load imbalance CV.

**Expected outcome:** All algorithms perform similarly (difference < 10%). Any large PPO divergence at baseline indicates a bug.

**Acceptance Criteria:**
- [ ] 30 `.jtl` files (6 algorithms × 5 seeds) committed to `experiments/results/e1/`
- [ ] No run file is empty or < 100 rows
- [ ] Raw data uploaded to `s3://flashbalanceai-{ACCOUNT_ID}/results/e1/`
- [ ] PPO and RR P95 difference at baseline < 20% (sanity check — not a result claim)

---

### Issue #26 — E2: 10× Traffic Spike (Primary Experiment)

**Labels:** `type:experiment` `phase:6-experiments` `priority:critical-path` `costs:aws-billing`  
**Owner:** All 4 members  
**Estimated Effort:** 6 hours | **AWS Cost:** ~$0.50  
**Depends On:** Issue #25 (E1 must complete with no anomalies)  
**Blocks:** Issues #27–#33 (E3, E5, E6, E7, E9 all depend on E2 data)

**Description:**  
Primary experiment answering RQ1 (PPO vs Threshold) and RQ2 (PPO vs DQN). Traffic profile: warm-up (2 min, 100 rps) → pre-burst (1 min, 300 rps) → spike onset (10s, 0→1000 rps) → peak (5 min, 1000 rps) → cooldown (5 min).

**Parameters:** peak_users=1000, 5 repetitions per algorithm, seeds 132–136, 6 algorithms.

**Additional measurement:** Record ASG scaling event timestamp for E7 cold-start analysis.

**Expected (unvalidated) outcomes:**
- H1: PPO P95 < 500ms during peak
- H2: Threshold P95 800ms–2000ms during spike onset

**Acceptance Criteria:**
- [ ] 30 `.jtl` files committed to `experiments/results/e2/`
- [ ] CloudWatch metric exports for all 5 seeds committed to `experiments/results/e2/cloudwatch/`
- [ ] ASG scaling event timestamps recorded for each run
- [ ] Raw data uploaded to S3

---

### Issue #27 — E3: 50× Traffic Spike (Stress Test)

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** All 4 members  
**Estimated Effort:** 5 hours | **AWS Cost:** ~$0.60  
**Depends On:** Issue #26 (E2 must show no infrastructure ceiling issues)  
**Blocks:** Issues #28, #32 (E4 conditional, E9 comparison)

**Description:**  
Stress test at 50× baseline. Same protocol as E2 with peak_users=5000. Infrastructure ceiling (max 8 t2.micro instances) likely reached.

**Gate condition before running:** E2 must complete successfully. If E2 shows infrastructure ceiling issues (all instances at 100% CPU), redesign before running E3.

**Acceptance criteria (to proceed to E4):** EC2 instances must not crash. If all instances hit 100% CPU, reduce to 4 repetitions and document the ceiling as a methodological limitation.

**Acceptance Criteria:**
- [ ] 30 `.jtl` files committed to `experiments/results/e3/` (or 24 with documented ceiling reduction)
- [ ] Any infrastructure ceiling events documented in `experiments/results/e3/notes.md`
- [ ] Raw data uploaded to S3

---

### Issue #28 — E4: 100× Traffic Spike (Conditional — Infrastructure Ceiling)

**Labels:** `type:experiment` `phase:6-experiments` `priority:normal` `costs:aws-billing`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 3 hours | **AWS Cost:** ~$0.30  
**Depends On:** Issue #27 (run only if E3 shows PPO maintaining < 2000ms P95)  
**Blocks:** Nothing (independent analysis)

**Description:**  
Upper bound experiment — run only if E3 succeeded. peak_users=10000, shortened 8-minute duration, 3 repetitions. Results reported as infrastructure ceiling analysis, not algorithm comparison.

> **Condition:** Run ONLY if E3 showed PPO maintaining P95 < 2000ms at 50×. Otherwise document as out-of-scope with justification.

**Acceptance Criteria:**
- [ ] If run: 18 `.jtl` files committed to `experiments/results/e4/`
- [ ] If not run: `experiments/results/e4/skipped.md` explaining the decision with E3 evidence

---

### Issue #29 — E5: Repeated Flash-Sale Patterns

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 5 hours | **AWS Cost:** ~$1.50  
**Depends On:** Issue #26 (E2 complete)  
**Blocks:** Nothing critical (independent)

**Description:**  
Test recovery between bursts. 3 consecutive 10× bursts with 2-minute gaps. Purpose: validate that PPO adapts across multiple events within one session.

**Parameters:** 3 algorithms (PPO, DQN, Threshold — most relevant for repeated events), 5 repetitions, ~45-minute runs.

**Key measurement:** PPO reward per burst window — does reward improve from burst 1 to burst 3?

**Acceptance Criteria:**
- [ ] 15 `.jtl` files committed to `experiments/results/e5/`
- [ ] Per-burst reward windows extracted and committed to `experiments/results/e5/burst_windows.csv`

---

### Issue #30 — E6: Noisy/Transient Spikes

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 4 hours | **AWS Cost:** ~$0.50  
**Depends On:** Issue #26 (E2 complete)  
**Blocks:** Nothing critical (independent)

**Description:**  
Test whether PPO over-reacts to short transient noise. Baseline 100 rps with random 2× spikes lasting 10–30 seconds, noise_std=0.40. This tests the noise-filtering insight from Mohar's H-MAS research (simplified via `burst_indicator`).

**Expected outcome:** PPO should NOT trigger scale-out for transient 2× spikes. Threshold may oscillate.

**Acceptance Criteria:**
- [ ] 15 `.jtl` files committed to `experiments/results/e6/`
- [ ] Scale-out event counts per algorithm per run recorded

---

### Issue #31 — E7: Cold-Start / Scaling Response Time

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** All 4 members  
**Estimated Effort:** 4 hours | **AWS Cost:** ~$0.50  
**Depends On:** Issue #26 (E2 complete)  
**Blocks:** Nothing critical (independent)

**Description:**  
Measure time from burst onset to N+1 instances being healthy in the target group. Start with desired-capacity=2, trigger 10× burst, measure time until 3rd instance is healthy.

**Measurement protocol (mandatory):**
- t₀ = burst onset timestamp (JMeter ramp-up start)
- t₁ = ASG desired-capacity change timestamp (CloudWatch event)
- t₂ = new instance `healthy` in ALB target group
- Scaling reaction time = t₂ − t₀

**Hypothesis H8 (unvalidated):** PPO proactive pre-scaling reduces reaction time by 50–70% vs Threshold reactive.

**Acceptance Criteria:**
- [ ] 10 timing records committed (5 PPO proactive + 5 Threshold reactive) to `experiments/results/e7/scaling_times.csv`
- [ ] t₀, t₁, t₂ explicitly recorded for each run with sub-second precision

---

### Issue #32 — E8: Cost/Performance Trade-off

**Labels:** `type:experiment` `phase:6-experiments` `priority:high` `costs:aws-billing`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 8 hours | **AWS Cost:** ~$1.00  
**Depends On:** Issue #26 (E2 complete — uses E2 traffic profile)  
**Blocks:** Nothing critical

**Description:**  
Vary the utilisation reward weight w₂ to understand cost vs P95 latency trade-off. Three configurations: (a) w₂=0.10 performance focus, (b) w₂=0.20 default, (c) w₂=0.40 cost focus.

**For each config:** Retrain PPO with modified `ppo_config.yaml` reward weights (use Google Colab to save SageMaker cost — ~4 hours training each), then run E2-equivalent experiment.

**Metrics:** P95 latency, average CPU utilisation, estimated AWS cost (instances × hours).

**Acceptance Criteria:**
- [ ] 3 retrained models committed to `models/ppo/` with config-specific names
- [ ] 9 `.jtl` files committed to `experiments/results/e8/` (3 configs × 3 seeds)
- [ ] Cost estimates per config recorded in `experiments/results/e8/cost_comparison.csv`

---

### Issue #33 — E9: DRL Algorithm Comparison (PPO vs DQN)

**Labels:** `type:analysis` `phase:6-experiments` `priority:high` `costs:free`  
**Owner:** All 4 members  
**Estimated Effort:** 2 hours | **AWS Cost:** $0 (uses E2/E3 data)  
**Depends On:** Issues #26, #27 (E2 and E3 data already collected)  
**Blocks:** Issue #36 (statistical analysis)

**Description:**  
Extract and compare PPO vs DQN results from E2 and E3 using the same test seeds. Answers RQ2: does PPO provide measurable advantage over DQN?

**Analysis:** Paired t-test on P95 latency (PPO vs DQN), α=0.05. Report mean P95, 95% CI, Cohen's d effect size, p-value.

> **⚠️ Integrity Note:** Do not adjust test seeds or filter runs after seeing results. Report all 5 repetitions regardless of outcome.

**Acceptance Criteria:**
- [ ] `notebooks/04_ablation_study.ipynb` contains PPO vs DQN statistical comparison cell
- [ ] Paired t-test result committed (t-statistic, p-value, Cohen's d) even if result is not significant
- [ ] No post-hoc seed filtering

---

### Issue #34 — E10: Reward Ablation Study

**Labels:** `type:experiment` `phase:6-experiments` `priority:critical-path` `costs:aws-billing`  
**Owner:** All 4 members  
**Estimated Effort:** 10 hours | **AWS Cost:** ~$1.00  
**Depends On:** Issue #26 (E2 complete — uses E2 traffic profile)  
**Blocks:** Issue #37 (ablation figures)

**Description:**  
Validate each reward component's contribution. Five variants: (a) full reward default, (b) w₁=0 no R_lat, (c) w₂=0 no R_util, (d) w₃=0 no R_tput, (e) w₄=0 no R_sla.

**For each variant:** Modify `ppo_config.yaml` → retrain PPO (local/Colab) → evaluate on same 3 test seeds as E2.

**Critical for paper:** Without E10, reviewers will correctly challenge the reward weights as arbitrary.

**Hypothesis H7 (unvalidated):** Removing R_sla (w₄=0) should increase P99 latency most.

**Acceptance Criteria:**
- [ ] 5 retrained models committed to `models/ppo/ablation/`
- [ ] 15 `.jtl` files committed to `experiments/results/e10/` (5 variants × 3 seeds)
- [ ] Ablation comparison table (P95 latency, SLA violation rate, CPU%) committed to `experiments/analysis/`

---

## Phase 7 — Results & Statistical Validation

**Milestone:** M7 | **Duration:** 4–5 days | **Local only**

**Objective:** Compute all reported statistics with correct methodology, generate paper-quality figures, and run reproducibility verification.

**Deliverables:** Completed `notebooks/03_results_analysis.ipynb` with full stats table, 8 paper-quality PDF figures in `experiments/analysis/figures/`, reproducibility report.

**Exit Criteria:** All ??? cells in the results table filled with actual experimental data. All 8 figures exported as PDF. Reproducibility verification run within ±5%.

---

### Issue #35 — Raw Data Collection and Verification

**Labels:** `type:analysis` `phase:7-results` `priority:critical-path` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 3 hours  
**Depends On:** All Phase 6 issues  
**Blocks:** Issues #36–#37 (statistical analysis and figures)

**Description:**  
Download all experiment results from S3, verify completeness, and check for data gaps. Every experiment run must have a corresponding raw data file before analysis begins.

**Verification checks per experiment:**
- Each `.jtl` file has approximately 7800 rows (13 min × 10 req/s average)
- `pandas.read_csv(jtl_file).isnull().sum()` == 0
- All specified repetitions are present

**Acceptance Criteria:**
- [ ] Completeness matrix committed to `experiments/analysis/data_completeness.csv` (rows=experiments, columns=algorithms/seeds, values=row counts)
- [ ] Any missing runs documented with reason in `experiments/analysis/missing_data.md`
- [ ] All raw data committed or S3 links recorded in `experiments/results/README.md`

---

### Issue #36 — Statistical Analysis and Results Table

**Labels:** `type:analysis` `phase:7-results` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 6 hours  
**Depends On:** Issue #35  
**Blocks:** Issue #41 (paper results section)

**Description:**  
Execute the full statistical analysis pipeline in `notebooks/03_results_analysis.ipynb`. Compute all required statistics following the pre-registered methodology.

**Required statistics per algorithm per experiment:**
- Mean ± std P95 latency across 5 seeds
- 95% confidence interval (t-distribution, n=5)
- Paired t-test vs PPO: t-statistic, p-value (α=0.05)
- Cohen's d effect size

**Results table to complete (all `???` replaced with actual values):**

| Algorithm | Mean P95 (ms) | Std | 95% CI | t vs PPO | p-value | Cohen's d |
|-----------|--------------|-----|--------|----------|---------|-----------|
| PPO | ??? | ??? | ??? | — | — | — |
| DQN | ??? | ??? | ??? | ??? | ??? | ??? |
| RoundRobin | ??? | ??? | ??? | ??? | ??? | ??? |
| LeastConn | ??? | ??? | ??? | ??? | ??? | ??? |
| Threshold | ??? | ??? | ??? | ??? | ??? | ??? |

> **⚠️ Integrity:** Fill only with actual experimental results. Never fill with hypothetical or expected values.

**Acceptance Criteria:**
- [ ] All `???` cells in both results table and ablation table filled with actual data
- [ ] Statistical analysis notebook cells produce reproducible output when re-run with `Restart & Run All`
- [ ] No data from test seeds 127–141 used anywhere in training or hyperparameter tuning (confirmed)

---

### Issue #37 — Generate Paper-Quality Figures

**Labels:** `type:analysis` `phase:7-results` `priority:high` `costs:free`  
**Owner:** Mohar Gorai  
**Estimated Effort:** 6 hours  
**Depends On:** Issue #35  
**Blocks:** Issue #41 (paper results section)

**Description:**  
Generate all 8 required figures using matplotlib 3.8 + seaborn 0.13. All figures must be PDF vector format, font size ≥ 8pt (IEEE minimum), correct column widths.

**Required figures:**
1. **Figure 1** — System architecture diagram (draw.io or matplotlib, vector)
2. **Figure 2** — Traffic profile showing all phases for 10× and 50× scenarios
3. **Figure 3** — Reward convergence curves (PPO and DQN, episode reward vs steps, shaded 95% CI)
4. **Figure 4** — Latency CDF (P5–P99 for all 6 algorithms at 10× burst)
5. **Figure 5** — P95 latency comparison bar chart (E1, E2, E3 grouped by algorithm)
6. **Figure 6** — CPU utilisation timeline (time-series during 10× burst, one line per instance)
7. **Figure 7** — Scaling reaction time bar chart (PPO proactive vs Threshold reactive, E7)
8. **Figure 8** — Ablation study bar chart (E10 results, each reward variant)

**Format requirements:**
- All figures: PDF vector format (not PNG)
- Font ≥ 8pt, `plt.rcParams.update({'font.size': 9})`
- Single-column: 3.5 in wide; double-column: 7.16 in wide
- Save to `experiments/analysis/figures/fig{N}_{name}.pdf`

**Acceptance Criteria:**
- [ ] All 8 figures exist as PDF files in `experiments/analysis/figures/`
- [ ] No figure uses PNG/raster format
- [ ] Figure 3 shows actual training curves (not placeholder data)
- [ ] Figure 5 contains all three spike levels (E1, E2, E3) for all 6 algorithms

---

### Issue #38 — Reproducibility Verification

**Labels:** `type:reproducibility` `phase:7-results` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** Any team member who was NOT the primary experimenter for E2  
**Estimated Effort:** 3 hours  
**Depends On:** Issue #36  
**Blocks:** Issue #42 (paper submission — must pass before submission)

**Description:**  
Verify that a fresh run with documented seeds produces results within ±5% of reported values. Must be performed on a different team member's machine.

**Steps:**
1. Clone repo fresh on a different machine
2. Recreate conda environment from `environment.yml`
3. Run E2 with seeds 132–133 (2 repetitions only)
4. Compare P95 latency to recorded values — must be within ±5%
5. Document any deviations in `decisions/reproducibility_report.md`

**Acceptance Criteria:**
- [ ] Both verification runs produce P95 latency within ±5% of original recorded values
- [ ] `decisions/reproducibility_report.md` committed with machine specs, exact commands run, and comparison table
- [ ] If deviation > 5%: root cause identified and documented before paper submission

---

## Phase 8 — Conference Paper Preparation

**Milestone:** M8 | **Duration:** 7–10 days | **Local only**

**Objective:** Write a complete, submission-ready conference paper draft (IEEE 2-column format, 6–8 pages) targeting IEEE Cloud / ICDCS / CCGRID 2026–2027.

**Deliverables:** Complete paper draft with all sections, algorithm pseudocode, related work with 15–20 verified citations, results tables with actual data.

**Exit Criteria:** All paper sections drafted. Results section contains only experimentally validated values (no `???` or hypothetical values). Reproducibility check (Issue #38) passed.

---

### Issue #39 — Paper Outline, Section Assignment & Template Setup

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `priority:critical-path` `costs:free`  
**Owner:** All 4 members (Devkanti leads)  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #36 (stats must be ready to write Results section)  
**Blocks:** Issues #40–#43

**Description:**  
Assign paper sections to team members, set internal deadlines, download IEEE template, create LaTeX project in `docs/paper/`.

**Section assignments:**
| Section | Owner | Target Length |
|---------|-------|--------------|
| Abstract | Devkanti (lead) | 250 words |
| 1. Introduction | Devkanti | 1 page |
| 2. Related Work | Agrima | 1.5 pages |
| 3. System Design | Agrima (all contribute) | 1.5 pages |
| 4. Methodology | Devkanti + Mohar | 2 pages |
| 5. Experimental Setup | Agrima | 1 page |
| 6. Results | Devkanti (all contribute) | 2 pages |
| 7. Discussion | All | 0.5 page |
| 8. Conclusion | Mohar | 0.3 page |
| References | Agrima (ongoing) | 15–20 refs |

**Acceptance Criteria:**
- [ ] `docs/paper/` directory with IEEEtran template committed
- [ ] Section assignments documented in `decisions/paper_assignments.md` with internal deadlines
- [ ] LaTeX compiles to PDF without errors (even with placeholder text)

---

### Issue #40 — Write Algorithm Pseudocode

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `component:ppo-agent` `priority:high` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 2 hours  
**Depends On:** Issue #11 (PPO implementation complete)  
**Blocks:** Nothing critical (can proceed in parallel with other paper sections)

**Description:**  
Write paper-ready pseudocode for Algorithm 1 (PPO Training for FlashBalanceAI) and Algorithm 2 (inference loop) in LaTeX `algorithm2e` format for inclusion in the Methodology section.

**Pseudocode must cover:**
- Algorithm 1: PPO training with FlashSaleEnv, GAE computation, clipped surrogate objective, evaluation callback
- Algorithm 2: AWS inference loop (CloudWatch → state vector → PPO predict → ALB update → DynamoDB log)

**Acceptance Criteria:**
- [ ] Both algorithms typeset in `algorithm2e` format and compile without error
- [ ] Notation is consistent with the MDP formulation in PRD.md Section 9
- [ ] No fabricated or placeholder algorithm steps

---

### Issue #41 — Write Results Section and Tables

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar (all contribute figures)  
**Estimated Effort:** 4 hours  
**Depends On:** Issues #36, #37 (statistical analysis and figures complete)  
**Blocks:** Issue #43 (paper review)

**Description:**  
Write the Results section (Section 6) of the paper. All tables must be filled with actual experimental results from Phase 7. All figure references must point to the generated PDF figures.

**Required paper elements in this section:**
- Table 1: Performance comparison under 10× burst (all algorithms × all metrics, mean ± std, 5 seeds)
- Table 2: Ablation study results (E10 — 5 reward variants × 3 metrics)
- Figure references: Figures 4–8 with captions
- Statistical significance annotations (†) per PRD Section 14.2

> **⚠️ Integrity:** Table 1 must not contain any `[fill]` cells or hypothetical values. Every cell must come from Issue #36 analysis.

**Acceptance Criteria:**
- [ ] Table 1 is complete with all actual experimental values and standard deviations
- [ ] Table 2 (ablation) is complete with E10 data
- [ ] All 5 research questions (RQ1–RQ5) are explicitly answered in the text

---

### Issue #42 — Write Related Work Section

**Labels:** `type:research` `phase:8-paper` `component:paper` `priority:high` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 5 hours  
**Depends On:** Issue #39 (section assignment)  
**Blocks:** Issue #43 (paper review)

**Description:**  
Write Related Work (Section 2) citing all 10 surveyed papers plus 5–8 additional papers. Organise into three groups: (a) DRL for cloud scheduling, (b) DRL for load balancing, (c) AWS/cloud autoscaling.

**Mandatory citations (verified against Google Scholar):**
1. Zhou et al. (2024) — DRL survey, Springer AI Review
2. Hu et al. (2025) — D4PG network routing, IEEE Trans. Ind. Informatics
3. Chen et al. (2026) — MDP DRL for e-commerce, Elsevier ESWA
4. Femminella & Reali (2024) — PPO for serverless HPA, MDPI Computers
5. Funika et al. (2023) — PPO-LSTM heterogeneous cloud, Springer J. Supercomputing
6. Jian et al. (2024) — DRS K8s scheduler, Wiley SPE
7. Yamsani & Chenna Reddy (2026) — SLA-DRL, Nature Scientific Reports
8. Zhou et al. (2024) — TS-SDTRA PPO+DQN, Elsevier Computer Communications
9. Li et al. (2024) — Distributional RL batch scheduling, IEEE TPDS
10. MAS-H2 paper — Hierarchical Multi-Agent autoscaling
11. Mnih et al. (2015) — DQN, Nature (foundational)
12. Schulman et al. (2017) — PPO, arXiv (foundational)
13. Sutton & Barto (2018) — RL textbook
14–15. Flash-sale / thundering-herd papers (2022+)

> **⚠️ Integrity:** Only cite papers actually read. Do not fabricate DOIs, volume numbers, or page ranges. Verify all citations against Google Scholar before submission.

**Acceptance Criteria:**
- [ ] 15–20 citations, all with complete and verified bibliographic information
- [ ] Explicit research gap paragraph identifying what no surveyed paper does (per PRD Section 16)
- [ ] No fabricated citations

---

### Issue #43 — Full Paper Internal Review and Revision Round

**Labels:** `type:documentation` `phase:8-paper` `component:paper` `priority:critical-path` `costs:free`  
**Owner:** All 4 members  
**Estimated Effort:** 8 hours (combined across team)  
**Depends On:** Issues #39–#42 (all sections drafted)  
**Blocks:** Issue #45 (final release)

**Description:**  
Internal peer-review round where each team member reviews at least one other member's sections. All comments must be addressed before the paper is considered submission-ready.

**Review checklist:**
- [ ] No result claims unsupported by experimental data
- [ ] All hypotheses are labelled as expected/unvalidated until confirmed
- [ ] All limitations from PRD Section 20 are acknowledged in the paper
- [ ] References list complete with no `???` entries
- [ ] LaTeX compiles to ≤ 8 pages in IEEEtran 2-column format
- [ ] All figure captions are self-contained
- [ ] Abstract matches actual results (written last, after experiments)

**Acceptance Criteria:**
- [ ] Paper compiles to PDF ≤ 8 pages with no LaTeX errors or warnings
- [ ] At least 2 review passes completed (evidenced by PR review comments)
- [ ] All review comments marked as resolved or explicitly deferred with justification

---

## Phase 9 — Demo, Reproducibility & Teardown

**Milestone:** M9 | **Duration:** 2–3 days**

**Objective:** Create a live demo (optional), build the monitoring dashboard, tag the v1.0 release, and execute complete AWS teardown to prevent billing runoff.

**Deliverables:** v1.0 GitHub release with model artifacts, `scripts/teardown.sh` executed and verified, `decisions/reproducibility_report.md` finalised.

**Exit Criteria:** `v1.0` tag exists on `main`. AWS Cost Explorer confirms all chargeable resources deleted. Teardown script verified.

---

### Issue #44 — Build Live Monitoring Dashboard

**Labels:** `type:implementation` `phase:9-demo` `component:metrics` `priority:normal` `costs:free`  
**Owner:** Agrima Gupta  
**Estimated Effort:** 4 hours  
**Depends On:** Issue #23 (CloudWatch metrics must be publishing)  
**Blocks:** Nothing critical

**Description:**  
Create `src/metrics/visualiser.py` as a Python Dash application showing live experiment metrics: response time EMA per instance, CPU utilisation per instance, and routing distribution histogram. Auto-refreshes every 5 seconds from S3 state.

**Key components:**
- `dcc.Graph` for response time EMA bar chart
- `dcc.Graph` for CPU utilisation per instance
- `dcc.Graph` for routing decision histogram (last 100 decisions)
- `dcc.Interval` with 5s refresh pulling from `state/current.json` in S3

**Acceptance Criteria:**
- [ ] `python src/metrics/visualiser.py` starts without error at `http://localhost:8050`
- [ ] All 3 graphs update when S3 state is refreshed
- [ ] Dashboard works in read-only mode (no writes to AWS)

---

### Issue #45 — Tag v1.0 GitHub Release

**Labels:** `type:documentation` `phase:9-demo` `priority:critical-path` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 1 hour  
**Depends On:** Issue #43 (paper review complete), Issue #38 (reproducibility verified)  
**Blocks:** Issue #46 (teardown — teardown only after release is tagged)

**Description:**  
Create the `v1.0` annotated Git tag on `main` and publish a GitHub Release with all required artifacts.

**Release assets to attach:**
- `ppo_flash_v1.zip` — final trained PPO model
- `dqn_flash_v1.zip` — final trained DQN model
- `environment.yml` — pinned conda environment
- `requirements.txt` — pinned pip dependencies
- Link to `experiments/results/` S3 bucket (or compressed subset if < 100MB)
- Release notes summarising: implemented features, experiment results (brief), known limitations, reproducibility instructions

**Release notes must include:**
- Exact seeds: `SEED_TRAIN=42, SEED_VAL=43, SEED_TEST=44`
- AWS region: `us-east-1`
- Python version: 3.11
- Stable-Baselines3 version: 2.3.0
- Quick-start reproducibility command

**Acceptance Criteria:**
- [ ] `git tag -l` shows `v1.0` on `main`
- [ ] GitHub Releases page shows release with all listed assets attached
- [ ] Release notes contain reproducibility instructions runnable by a third party
- [ ] All PRD research objectives (O1–O8) are addressed in release notes (even if results are negative)

---

### Issue #46 — Execute AWS Teardown

**Labels:** `type:infrastructure` `phase:9-demo` `component:aws-infra` `priority:critical-path` `gate:required` `costs:free`  
**Owner:** Devkanti Sarkar  
**Estimated Effort:** 1 hour  
**Depends On:** Issue #45 (release tagged — all experimental data must be backed up locally AND in release before teardown)  
**Blocks:** Nothing (final issue)

**Description:**  
Execute the complete teardown procedure to delete all chargeable AWS resources and prevent unexpected billing.

**⚠️ Do NOT run teardown until:** (a) v1.0 GitHub release is tagged, (b) all `.jtl` files are downloaded locally, (c) all CloudWatch exports are committed, (d) S3 experiment data is backed up.

**Teardown order (from `scripts/teardown.sh`):**
1. Set ASG desired-capacity=0, min=0 → wait 120s for instance termination
2. Delete ASG
3. Delete ALB
4. Delete Target Group
5. Delete Lambda functions (StateCollector, ScalingTrigger, InferenceCoordinator)
6. Stop SageMaker notebook instance (if running)
7. Delete DynamoDB table `routing_decisions`
8. Empty and delete S3 bucket (requires explicit `yes` confirmation)
9. Delete EC2 Launch Template
10. Delete CloudWatch alarms and EventBridge rules

**Acceptance Criteria:**
- [ ] AWS Console: EC2 instances list empty in us-east-1
- [ ] AWS Console: No active ALB or Target Groups
- [ ] AWS Console: Lambda functions list empty
- [ ] AWS Console: S3 bucket deleted (or confirmed empty)
- [ ] AWS Cost Explorer shows no ongoing hourly charges
- [ ] Teardown completion screenshot committed to `decisions/teardown_confirmation.md`

---

## 13. Critical Path

The critical path represents the minimum-duration sequence of tasks that determine the earliest possible completion date. No task on the critical path can slip without delaying the entire project.

```
Issue #1 (ADR-001)
    ↓
Issue #2 (Repo scaffold)
    ↓
Issue #3 (Python environment)
    ↓
Issue #8 (Traffic generator) ──────────────────────────────────────┐
    ↓                                                               │
Issue #5 (FlashSaleEnv)                                            │
    ↓                                                               │
Issue #11 (PPO training) ←─────────────────────────────────────────┘
    ↓
Issue #13 (Local evaluation GATE CHECK)
    ↓
Issue #14 (AWS billing alarm GATE CHECK)
    ↓
Issues #15 → #16 → #17 → #18 (AWS stack)
    ↓
Issues #20 → #21 → #22 → #23 (Lambda functions)
    ↓
Issue #24 (End-to-end integration GATE CHECK)
    ↓
Issue #25 (E1: Baseline calibration)
    ↓
Issue #26 (E2: 10× Primary experiment)
    ↓
Issue #27 (E3: 50× Stress test)
    ↓
Issue #35 (Raw data verification)
    ↓
Issue #36 (Statistical analysis)
    ↓
Issues #39 → #41 (Paper outline → Results section)
    ↓
Issue #43 (Paper internal review)
    ↓
Issue #45 (v1.0 Release tag)
    ↓
Issue #46 (AWS Teardown)
```

**Total calendar time on critical path: ~35 days with 4 students working in parallel**

---

## 14. Parallelisable Work

The following issue groups can be worked simultaneously once their shared dependency is met:

| After | Can run in parallel |
|-------|-------------------|
| Issue #3 (env ready) | Issues #4, #6, #7, #10 — Flask backend, baselines, metrics, Alibaba dataset |
| Issue #3 (env ready) | Issue #8 (traffic generator, parallel with Issue #4) |
| Issue #11 started | Issue #12 (DQN training — must use frozen env version) |
| Issue #24 (integration gate) | Issues #29, #30, #31 (E5, E6, E7 — after E2 complete) |
| Issue #26 (E2 complete) | Issues #28, #29, #30, #31, #32, #33 (E4–E9) |
| Issue #35 (data verified) | Issues #36, #37 (statistical analysis and figures) |
| Issue #39 (paper outline) | Issues #40, #41, #42 (pseudocode, results, related work) |
| Issue #43 (paper review) | Issue #44 (monitoring dashboard) |

---

## 15. Release Checkpoints

| Tag | Trigger | Contents | Status Gate |
|-----|---------|----------|-------------|
| `v0.1-env-ready` | Issue #13 closes (local eval gate passes) | Trained PPO+DQN models, all unit tests passing, preliminary local results table | Issue #13 acceptance criteria met |
| `v0.2-aws-live` | Issue #24 closes (integration gate passes) | AWS stack operational, Lambda functions deployed, integration test report | Issue #24 acceptance criteria met |
| `v0.3-experiments-done` | Issue #35 closes (all raw data verified) | All E1–E10 raw data in `experiments/results/`, completeness matrix | Issue #35 acceptance criteria met |
| `v0.4-analysis-done` | Issue #37 closes (all 8 figures generated) | Complete stats table, all 8 PDF figures, reproducibility report | Issues #36, #37, #38 all closed |
| `v1.0` | Issue #43 closes (paper review complete) | Final paper draft, model artifacts, environment files, release notes with reproducibility instructions | Issues #38, #43, #45 all closed |

---

## 16. Contribution Tracking

The guideline requires individual contribution tracking. Every team member must maintain evidence across the following categories in their commit history and PR participation.

| Contribution Category | Devkanti Sarkar | Agrima Gupta | Mohar Gorai | [4th Member] |
|----------------------|-----------------|--------------|-------------|--------------|
| **Research / Literature** | PRD authorship, ADR-001 lead, reward function design | Research gap analysis, DQN/PPO comparison rationale, related work section | H-MAS noise-filtering literature, Alibaba trace analysis | TBD |
| **Development** | `FlashSaleEnv`, PPO agent, local evaluation, results section | Repo scaffold, baselines, DQN agent, S3/DynamoDB infra, dashboard | Flask backend, JMeter plans, EC2/ASG setup, ALB, scaling Lambda, figures | TBD |
| **Cloud Integration** | AWS account safety, IAM, PPO inference server, teardown | CloudWatch state collector Lambda, custom metrics | Scaling trigger Lambda, custom CloudWatch publishing, EC2/ALB | TBD |
| **AI/ML** | PPO training lead, hyperparameter configuration | DQN training, E9 comparison analysis | E5 repeated-burst experiment, reward convergence curves | TBD |
| **Testing** | Local evaluation gate check, unit tests (env, PPO) | Unit tests (baselines, metrics), E1/E6 experiments, data verification | Integration test participation, E5/E7 experiments | TBD |
| **Documentation** | ADR-001, paper Introduction/Methodology/Abstract | Repo structure, paper Related Work/System Design/Experimental Setup, references | Paper Conclusion/Discussion, architecture diagram | TBD |
| **Presentation / GitHub** | Release tagging, paper review coordination | PR reviews (≥2 per guideline requirement), commit history continuity | PR reviews (≥2), commit history | TBD — ≥2 PRs |

### Minimum GitHub Participation Requirements (per BCSE355L Guidelines)

Each team member must satisfy all of the following before the final submission:

- [ ] **≥ 2 Pull Requests opened** (any branch → `develop` or `develop` → `main`)
- [ ] **≥ 2 PR code reviews** participated in (as reviewer, with substantive comments)
- [ ] **Weekly commits** — at least 1 meaningful commit per calendar week during the project
- [ ] **Continuous documentation updates** — README, ADR, or paper section updated alongside code commits
- [ ] **Commit messages** follow the project convention: `[PHASE-N] <verb> <what>` (e.g., `[PHASE-3] implement PPO training with SB3 callback`)
- [ ] **Individual contribution** traceable to named GitHub issues assigned to the member

### Branch Naming Convention

```
feature/phase{N}-{short-description}    # e.g. feature/phase1-flash-sale-env
fix/issue{N}-{short-description}        # e.g. fix/issue5-reward-nan
experiment/e{N}-{algorithm}             # e.g. experiment/e2-ppo-10x-spike
docs/phase{N}-{document}               # e.g. docs/phase8-related-work-section
```

### Commit Message Convention

```
[PHASE-{N}] {verb} {subject}

Examples:
[PHASE-1] implement FlashSaleEnv with 23-dim state space
[PHASE-3] train PPO agent to convergence (2M steps, reward plateau confirmed)
[PHASE-6] run E2 10x spike experiment seeds 132-136 (30 jtl files)
[PHASE-8] draft results section with actual E2/E3 statistical analysis
```

---

*This plan is derived exclusively from `PRD.md` and `IMPLEMENTATION_PLAN_PART1.md` / `IMPLEMENTATION_PLAN_PART2.md`. It does not redesign the technical architecture, duplicate methodology details, or invent work beyond what is specified in those documents. For full MDP formulation, reward function derivation, AWS service justification, and experimental methodology, refer to `PRD.md` Sections 9–14 and `IMPLEMENTATION_PLAN_PART2.md` Phases 6–7.*
