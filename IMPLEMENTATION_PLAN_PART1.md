# IMPLEMENTATION PLAN — FlashBalanceAI (Part 1 of 2)
## Phases 0–4: Design → Local Dev → Dataset → DRL → AWS Deployment

**Document Version:** 1.0 | **Date:** 2026-08-20  
**Team:** Devkanti Sarkar · Agrima Gupta · Mohar Gorai · [4th member]  
**Read alongside:** `PRD.md` for full methodology, MDP formulation, and research context.  

> **Part 1 covers:** Phase 0 (Design Lock), Phase 1 (Local Dev), Phase 2 (Dataset/Traffic), Phase 3 (DRL Implementation), Phase 4 (AWS Deployment).  
> **Part 2 covers:** Phase 5 (AWS Integration), Phase 6 (Experiments), Phase 7 (Stats), Phase 8 (Paper), Phase 9 (Demo/Teardown), full Timeline table.

---

## CONVENTIONS

```
[CRITICAL PATH]  — cannot be skipped or parallelised; blocks downstream work
[PARALLEL]       — safe to run alongside other tasks
[COSTS MONEY]    — triggers AWS billing; read carefully
[FREE TIER]      — confirmed within AWS Free Tier or student credit allowance
[LOCAL ONLY]     — no AWS needed; runs on team laptops
```

---

## PHASE 0 — FINAL ARCHITECTURE / DESIGN LOCK

**Duration:** 1–2 days | **Prerequisites:** PRD.md reviewed by all 4 members

---

### T0.1 — Architecture Decision Record [CRITICAL PATH]

**Objective:** Lock all design decisions so no one builds contradictory components.

**Steps:**
1. All 4 members read PRD.md Sections 7–12 in full.
2. Hold a 1-hour synchronous meeting (video or in-person) to resolve any disagreements.
3. Confirm the following decisions in writing (add to a `decisions/ADR-001.md` file in the repo):
   - Primary DRL algorithm: **PPO** (not DDPG, not H-MAS)
   - Primary DRL baseline: **DQN** (identical architecture)
   - Action space: **Discrete, N=4 backend instances**
   - State dimensions: **23-dimensional** (5N+3, N=4)
   - Reward weights: **w1=0.4, w2=0.2, w3=0.2, w4=0.2** (subject to E10 ablation)
   - AWS region: **us-east-1** (lowest cost, widest service availability)
   - Max EC2 instances: **4 backend + 1 inference/control** = 5 total t2.micro
   - Training location: **local laptop first** (SageMaker only if local fails)
   - Traffic generator: **Python custom (`traffic_generator.py`)** + JMeter for load injection
4. Get written acknowledgement (Git commit) from each team member.

**Dependencies:** None  
**Tool:** Google Docs / GitHub wiki / `decisions/ADR-001.md`  
**Input:** PRD.md  
**Output:** `decisions/ADR-001.md` — locked architecture record  
**Estimated Time:** 3 hours (1 meeting + doc write-up)  
**Acceptance Criteria:** All 4 members have committed a sign-off to the ADR file. No open questions remain.  
**Costs Money?** No.

---

### T0.2 — Repository Structure Setup [CRITICAL PATH]

**Objective:** Create the full directory structure so every team member knows where to put files.

**Steps:**
1. On the `main` branch, create the following structure:
```
DRL_Cloud_Load_Balancing_Cloud_Project_2026/
├── decisions/
│   └── ADR-001.md
├── src/
│   ├── environment/
│   │   ├── flash_sale_env.py        # Custom Gymnasium environment
│   │   └── __init__.py
│   ├── agents/
│   │   ├── ppo_agent.py             # PPO training + inference
│   │   ├── dqn_agent.py             # DQN training + inference
│   │   └── __init__.py
│   ├── baselines/
│   │   ├── round_robin.py
│   │   ├── weighted_round_robin.py
│   │   ├── least_connections.py
│   │   └── threshold_autoscaler.py
│   ├── traffic/
│   │   ├── traffic_generator.py     # Synthetic flash-sale generator
│   │   ├── jmeter_configs/          # JMeter .jmx test plans
│   │   └── __init__.py
│   ├── backend/
│   │   ├── app.py                   # Flask backend simulating EC2 server
│   │   ├── requirements_backend.txt
│   │   └── __init__.py
│   ├── aws/
│   │   ├── cloudwatch_collector.py  # Lambda: state from CloudWatch
│   │   ├── ppo_inference.py         # Lambda: PPO inference + ALB update
│   │   ├── scaling_trigger.py       # Lambda: ASG scale-out/in
│   │   ├── deploy.py                # boto3 infrastructure setup
│   │   ├── iam_setup.py             # IAM roles and policies
│   │   └── __init__.py
│   ├── metrics/
│   │   ├── collector.py             # Local metrics aggregation
│   │   ├── visualiser.py            # Dash dashboard
│   │   └── __init__.py
│   └── tests/
│       ├── test_environment.py
│       ├── test_reward.py
│       ├── test_traffic_generator.py
│       ├── test_baselines.py
│       ├── test_ppo_agent.py
│       └── test_dqn_agent.py
├── notebooks/
│   ├── 01_traffic_analysis.ipynb
│   ├── 02_ppo_training.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_ablation_study.ipynb
├── configs/
│   ├── ppo_config.yaml              # PPO hyperparameters
│   ├── dqn_config.yaml              # DQN hyperparameters
│   ├── traffic_config.yaml          # Traffic generator parameters
│   └── aws_config.yaml              # AWS resource identifiers
├── experiments/
│   ├── results/                     # Raw JMeter .jtl files, CloudWatch exports
│   └── analysis/                    # Processed CSVs, plots
├── infra/
│   └── cloudformation/              # Optional: CloudFormation templates
├── PRD.md
├── IMPLEMENTATION_PLAN_PART1.md
├── IMPLEMENTATION_PLAN_PART2.md
├── requirements.txt                 # Pinned Python dependencies
├── environment.yml                  # Conda environment
├── .gitignore
└── README.md
```
2. Create placeholder `__init__.py` files. Create `requirements.txt` (see T1.2).
3. Add `.gitignore` entries: `*.pyc`, `__pycache__/`, `.env`, `*.jtl`, `models/*.zip`, `experiments/results/raw/`.
4. Push to `main`. Create branches: `dev/environment`, `dev/agents`, `dev/aws`, `dev/experiments`.

**Dependencies:** T0.1  
**Tool:** Git, local terminal  
**Input:** ADR-001.md  
**Output:** Populated repository with correct structure  
**Estimated Time:** 2 hours  
**Acceptance Criteria:** `git ls-tree -r --name-only HEAD` shows all directories. `pip install -r requirements.txt` succeeds.  
**Costs Money?** No.

---

## PHASE 1 — LOCAL DEVELOPMENT

**Duration:** 5–7 days | **All tasks [LOCAL ONLY] unless noted**

---

### T1.1 — Python Environment Setup [CRITICAL PATH][LOCAL ONLY]

**Objective:** Reproducible Python environment across all 4 team members.

**Steps:**
1. Install Python 3.11 (not 3.12 — Stable-Baselines3 compatibility confirmed on 3.11).
2. Create conda environment:
```bash
conda create -n flashbalance python=3.11 -y
conda activate flashbalance
```
3. Install dependencies:
```bash
pip install stable-baselines3==2.3.0 \
            gymnasium==0.29.1 \
            torch==2.2.2 \
            numpy==1.26.4 \
            pandas==2.2.1 \
            boto3==1.34.84 \
            flask==3.0.3 \
            dash==2.17.0 \
            plotly==5.22.0 \
            pytest==8.1.1 \
            pytest-cov==5.0.0 \
            pyyaml==6.0.1 \
            requests==2.31.0 \
            locust==2.24.1
```
4. Export: `conda env export > environment.yml` and `pip freeze > requirements.txt`.
5. Verify: `python -c "import stable_baselines3; import gymnasium; print('OK')"`.

**Dependencies:** T0.2  
**Tool:** conda, pip  
**Input:** None  
**Output:** `environment.yml`, `requirements.txt`, working conda env  
**Estimated Time:** 1 hour  
**Acceptance Criteria:** All 4 team members can activate the env and run the verify command without errors.  
**Costs Money?** No.

---

### T1.2 — Flask Backend Server [LOCAL ONLY][PARALLEL]

**Objective:** Simulated EC2 backend server that responds to HTTP requests with configurable artificial latency (simulates CPU-bound product-page rendering).

**Steps:**
1. Create `src/backend/app.py`:
```python
import time, random, os
from flask import Flask, jsonify, request

app = Flask(__name__)
BASE_LATENCY_MS = float(os.environ.get("BASE_LATENCY_MS", "20"))
INSTANCE_ID = os.environ.get("INSTANCE_ID", "server-0")

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "instance": INSTANCE_ID}), 200

@app.route("/product/<int:pid>")
def product(pid):
    # Simulate CPU-bound work; latency increases with queue depth
    latency = BASE_LATENCY_MS + random.gauss(0, 5)  # ±5ms jitter
    time.sleep(max(0, latency / 1000.0))
    return jsonify({"product_id": pid, "instance": INSTANCE_ID, "price": 99.99}), 200

@app.route("/metrics")
def metrics():
    # Returns queue depth metric for CloudWatch custom publishing
    return jsonify({"queue_depth": 0, "instance": INSTANCE_ID}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
```
2. Test locally: start 4 instances on ports 5001–5004 using `INSTANCE_ID=server-0 PORT=5001 python app.py`.
3. Verify health check: `curl http://localhost:5001/health` returns `{"status": "healthy"}`.
4. Create `src/backend/requirements_backend.txt`: `flask==3.0.3 gunicorn==21.2.0`.

**Dependencies:** T1.1  
**Tool:** Python, Flask  
**Input:** None  
**Output:** `src/backend/app.py` — working HTTP server with `/health`, `/product/<id>`, `/metrics` endpoints  
**Estimated Time:** 2 hours  
**Acceptance Criteria:** All 3 endpoints return correct HTTP 200 with valid JSON. Latency measurably increases when BASE_LATENCY_MS is raised.  
**Costs Money?** No.

---

### T1.3 — Custom Gymnasium Environment [CRITICAL PATH][LOCAL ONLY]

**Objective:** Discrete-time simulation environment for PPO/DQN training that mimics the AWS backend without requiring live AWS.

**Steps:**
1. Create `src/environment/flash_sale_env.py` implementing `gymnasium.Env`:
```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlashSaleEnv(gym.Env):
    """
    State: [cpu_util x N, active_conn x N, queue_depth x N,
            resp_time_ema x N, health_status x N,
            arrival_rate_norm, burst_indicator, time_since_spike]
    Action: discrete integer in {0, ..., N-1} — target backend instance
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_instances=4, episode_steps=7800, seed=42):
        super().__init__()
        self.n = n_instances
        self.episode_steps = episode_steps  # 13 min at 100ms steps
        self.observation_space = spaces.Box(
            low=0.0, high=10.0,
            shape=(5 * self.n + 3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n)
        self._seed = seed
        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cpu = np.full(self.n, 0.1, dtype=np.float32)
        self.conn = np.zeros(self.n, dtype=np.float32)
        self.queue = np.zeros(self.n, dtype=np.float32)
        self.resp_ema = np.full(self.n, 0.1, dtype=np.float32)
        self.health = np.ones(self.n, dtype=np.float32)
        self.arrival_rate = 1.0  # normalised: 1.0 = baseline
        self.last_spike_step = -1000
        self.traffic_profile = self._generate_traffic_profile()
        obs = self._get_obs()
        return obs, {}

    def _generate_traffic_profile(self):
        """Generates arrival rate normalised to baseline for each step."""
        from src.traffic.traffic_generator import generate_profile
        return generate_profile(self.episode_steps, seed=self._seed)

    def _get_obs(self):
        burst_ind = float(self.arrival_rate > 2.0)
        steps_since = min((self.step_count - self.last_spike_step) / 600.0, 1.0)
        return np.concatenate([
            self.cpu, self.conn, self.queue,
            self.resp_ema, self.health,
            [self.arrival_rate / 10.0, burst_ind, steps_since]
        ]).astype(np.float32)

    def step(self, action):
        # Update arrival rate from traffic profile
        self.arrival_rate = self.traffic_profile[self.step_count]
        if self.arrival_rate > 2.0 and self.step_count - self.last_spike_step > 50:
            self.last_spike_step = self.step_count

        # Simulate server state update based on routing action
        load = self.arrival_rate / self.n  # simplified: load if evenly distributed
        chosen_load = self.arrival_rate * 0.9  # 90% goes to chosen server

        self.cpu[action] = min(1.0, self.cpu[action] * 0.85 + chosen_load * 0.15)
        for i in range(self.n):
            if i != action:
                self.cpu[i] = max(0.05, self.cpu[i] * 0.9 - 0.01)

        # Response time EMA
        base_resp = 0.1 + self.cpu[action] * 0.8  # normalised 0–1 (0=20ms, 1=200ms)
        self.resp_ema[action] = 0.7 * self.resp_ema[action] + 0.3 * base_resp

        # Reward computation
        reward = self._compute_reward(action)

        self.step_count += 1
        done = self.step_count >= self.episode_steps
        return self._get_obs(), reward, done, False, {}

    def _compute_reward(self, action):
        avg_resp = float(np.mean(self.resp_ema))
        avg_cpu  = float(np.mean(self.cpu))
        tput     = min(1.0, 1.0 / (1.0 + self.cpu[action]))
        sla_viol = float(self.resp_ema[action] > 0.7)  # >0.7 ~ >500ms
        r_lat  = -(avg_resp / 0.2) + 1.0
        r_util = avg_cpu - abs(avg_cpu - 0.70)
        r_tput = tput
        r_sla  = -2.0 * sla_viol
        return 0.4 * r_lat + 0.2 * r_util + 0.2 * r_tput + 0.2 * r_sla
```
2. Write unit tests in `src/tests/test_environment.py`:
   - `test_obs_shape()`: assert `obs.shape == (23,)` for N=4
   - `test_action_space()`: assert action space size == N
   - `test_episode_completes()`: run 7800 steps, verify `done == True`
   - `test_reward_range()`: reward in [-3, 2] for all states
3. Run: `pytest src/tests/test_environment.py -v`.

**Dependencies:** T1.1, T1.2 (traffic generator referenced)  
**Tool:** Python, Gymnasium, pytest  
**Input:** `traffic_generator.py` (T2.1)  
**Output:** `src/environment/flash_sale_env.py`, passing unit tests  
**Estimated Time:** 6 hours  
**Acceptance Criteria:** All unit tests pass. `check_env(FlashSaleEnv())` (SB3 utility) raises no errors. One full episode (7800 steps) completes in < 10 seconds on a laptop.  
**Costs Money?** No.

---

### T1.4 — Baseline Load Balancers [LOCAL ONLY][PARALLEL]

**Objective:** Software implementations of all non-DRL baselines for fair comparison.

**Steps:**
1. Create `src/baselines/round_robin.py`:
```python
class RoundRobinLB:
    def __init__(self, n): self.n = n; self.idx = 0
    def select(self, state=None):
        chosen = self.idx % self.n
        self.idx += 1
        return chosen
```
2. Create `src/baselines/weighted_round_robin.py`:
```python
import numpy as np
class WeightedRoundRobinLB:
    def __init__(self, n): self.n = n; self.counts = np.zeros(n)
    def select(self, state):
        # weight inversely proportional to cpu_util
        cpu = state[:self.n]
        weights = 1.0 / (cpu + 0.01)
        weights /= weights.sum()
        chosen = np.random.choice(self.n, p=weights)
        return int(chosen)
```
3. Create `src/baselines/least_connections.py`:
```python
import numpy as np
class LeastConnectionsLB:
    def __init__(self, n): self.n = n
    def select(self, state):
        conn = state[self.n:2*self.n]  # active_conn slice of state
        return int(np.argmin(conn))
```
4. Create `src/baselines/threshold_autoscaler.py` — simulates CloudWatch threshold logic:
```python
class ThresholdAutoscaler:
    """Mimics CloudWatch alarm: CPU > 0.70 for 2 consecutive periods → scale out."""
    def __init__(self, n, low=0.30, high=0.70, consecutive=2):
        self.n = n; self.low = low; self.high = high
        self.consecutive = consecutive; self.counter_high = 0; self.counter_low = 0
    def check_scaling(self, state):
        avg_cpu = state[:self.n].mean()
        if avg_cpu > self.high:
            self.counter_high += 1; self.counter_low = 0
            if self.counter_high >= self.consecutive:
                return "scale_out"
        elif avg_cpu < self.low:
            self.counter_low += 1; self.counter_high = 0
            if self.counter_low >= 5:
                return "scale_in"
        return "no_action"
    def select(self, state):
        # Round-robin routing (threshold autoscaler only controls count)
        return RoundRobinLB(self.n).select()
```
5. Write tests in `src/tests/test_baselines.py` — verify all `select()` returns integer in `[0, N-1]`.

**Dependencies:** T1.1  
**Tool:** Python  
**Input:** None  
**Output:** 4 baseline classes, passing unit tests  
**Estimated Time:** 3 hours  
**Acceptance Criteria:** `pytest src/tests/test_baselines.py -v` passes. All baselines run 7800 steps in FlashSaleEnv without error.  
**Costs Money?** No.

---

### T1.5 — Metrics Collection Module [LOCAL ONLY][PARALLEL]

**Objective:** Collect, aggregate, and store all experiment metrics in a standardised format for statistical analysis.

**Steps:**
1. Create `src/metrics/collector.py`:
```python
import numpy as np, pandas as pd, time

class MetricsCollector:
    def __init__(self, algorithm_name, experiment_id, seed):
        self.algo = algorithm_name; self.exp_id = experiment_id; self.seed = seed
        self.records = []

    def record_step(self, step, action, reward, state, response_time_ms,
                    cpu_utils, is_sla_violation):
        self.records.append({
            "step": step, "action": action, "reward": reward,
            "response_time_ms": response_time_ms,
            "avg_cpu": float(np.mean(cpu_utils)),
            "max_cpu": float(np.max(cpu_utils)),
            "is_sla_violation": int(is_sla_violation),
            "timestamp": time.time()
        })

    def compute_summary(self):
        df = pd.DataFrame(self.records)
        return {
            "algorithm": self.algo,
            "experiment_id": self.exp_id,
            "seed": self.seed,
            "mean_response_ms": df["response_time_ms"].mean(),
            "p95_response_ms": df["response_time_ms"].quantile(0.95),
            "p99_response_ms": df["response_time_ms"].quantile(0.99),
            "throughput_served": len(df[df["response_time_ms"] < 2000]),
            "failure_rate": df["is_sla_violation"].mean(),
            "mean_cpu": df["avg_cpu"].mean(),
            "total_reward": df["reward"].sum(),
        }

    def save(self, path):
        pd.DataFrame(self.records).to_csv(path, index=False)
```
2. Write `src/tests/test_metrics.py` — verify compute_summary keys and value ranges.

**Dependencies:** T1.1  
**Tool:** Python, pandas, numpy  
**Output:** `src/metrics/collector.py`, passing tests  
**Estimated Time:** 2 hours  
**Acceptance Criteria:** Collector records 7800 steps without memory issues. `compute_summary()` returns all required keys with plausible values.  
**Costs Money?** No.

---

## PHASE 2 — DATASET + TRAFFIC GENERATION

**Duration:** 3–4 days | **[LOCAL ONLY]** | **[PARALLEL with Phase 1 after T1.1]**

---

### T2.1 — Synthetic Flash-Sale Traffic Generator [CRITICAL PATH][LOCAL ONLY]

**Objective:** Parameterisable traffic generator producing arrival rate time-series with configurable burst multipliers and reproducible seeds.

**Steps:**
1. Create `src/traffic/traffic_generator.py`:
```python
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class TrafficConfig:
    baseline_rps: float = 100.0
    burst_multiplier: float = 10.0   # 10x, 50x, or 100x
    warmup_steps: int = 1200         # 2 min at 100ms
    pre_burst_steps: int = 600       # 1 min
    spike_onset_steps: int = 100     # 10 sec (ramp-up)
    peak_steps: int = 3000           # 5 min
    cooldown_steps: int = 3000       # 5 min
    noise_std_fraction: float = 0.10 # 10% Gaussian noise
    seed: int = 42

def generate_profile(n_steps: int, config: TrafficConfig = None) -> np.ndarray:
    """Returns normalised arrival rate (1.0 = baseline) for each 100ms step."""
    if config is None:
        config = TrafficConfig()
    rng = np.random.default_rng(config.seed)

    profile = np.ones(n_steps, dtype=np.float32)
    warmup_end = config.warmup_steps
    pre_burst_end = warmup_end + config.pre_burst_steps
    onset_end = pre_burst_end + config.spike_onset_steps
    peak_end = onset_end + config.peak_steps

    # Pre-burst: 3x baseline
    profile[warmup_end:pre_burst_end] = 3.0
    # Ramp up (linear) to burst_multiplier
    profile[pre_burst_end:onset_end] = np.linspace(3.0, config.burst_multiplier,
                                                     config.spike_onset_steps)
    # Peak
    profile[onset_end:min(peak_end, n_steps)] = config.burst_multiplier
    # Cooldown (exponential decay)
    if peak_end < n_steps:
        tail = n_steps - peak_end
        profile[peak_end:] = config.burst_multiplier * np.exp(
            -np.linspace(0, 3, tail))
        profile[peak_end:] = np.maximum(profile[peak_end:], 1.0)

    # Add Gaussian noise
    noise = rng.normal(0, config.noise_std_fraction, n_steps)
    profile = np.maximum(0.5, profile * (1 + noise))
    return profile

def generate_repeated_bursts(n_bursts=3, config: TrafficConfig = None) -> np.ndarray:
    """Generate episode with multiple flash-sale events (Experiment E5)."""
    if config is None: config = TrafficConfig()
    single = generate_profile(7800, config)
    gap = np.ones(1200, dtype=np.float32)  # 2 min gap between bursts
    result = single
    for i in range(1, n_bursts):
        cfg = TrafficConfig(**{**config.__dict__, "seed": config.seed + i})
        result = np.concatenate([result, gap, generate_profile(7800, cfg)])
    return result
```
2. Create `configs/traffic_config.yaml`:
```yaml
baseline_rps: 100
scenarios:
  e1_baseline:   { burst_multiplier: 1.0,  noise_std_fraction: 0.05, seed: 44 }
  e2_10x:        { burst_multiplier: 10.0, noise_std_fraction: 0.10, seed: 44 }
  e3_50x:        { burst_multiplier: 50.0, noise_std_fraction: 0.10, seed: 44 }
  e4_100x:       { burst_multiplier: 100.0, noise_std_fraction: 0.10, seed: 44 }
  e6_noisy:      { burst_multiplier: 2.0,  noise_std_fraction: 0.40, seed: 44 }
  train_episodes: { burst_multiplier: 10.0, noise_std_fraction: 0.10, seed: 42 }
  val_episodes:   { burst_multiplier: 10.0, noise_std_fraction: 0.10, seed: 43 }
```
3. Plot each traffic profile to `notebooks/01_traffic_analysis.ipynb` — verify visually.
4. Write `src/tests/test_traffic_generator.py`:
   - `test_profile_length()`: assert `len(profile) == n_steps`
   - `test_baseline_range()`: baseline steps have mean ≈ 1.0 ± 0.2
   - `test_burst_peak()`: peak steps have mean ≈ burst_multiplier ± 10%
   - `test_reproducibility()`: same seed → identical profile
   - `test_different_seeds()`: different seeds → different profiles

**Dependencies:** T1.1  
**Tool:** Python, numpy  
**Input:** `configs/traffic_config.yaml`  
**Output:** `src/traffic/traffic_generator.py`, passing tests, traffic profile plots  
**Estimated Time:** 4 hours  
**Acceptance Criteria:** All tests pass. Visual inspection confirms correct phase transitions. Two profiles with seed=42 are bit-identical.  
**Costs Money?** No.

---

### T2.2 — JMeter Test Plans [PARALLEL]

**Objective:** JMeter `.jmx` test plans for injecting synthetic traffic into the deployed AWS backend during Phases 5–6.

**Steps:**
1. Download Apache JMeter 5.6.3 (free, open source).
2. Create `src/traffic/jmeter_configs/flash_sale_10x.jmx` with:
   - Thread Group: Ramp-up 100 users → 1000 users in 10 seconds
   - HTTP Sampler: `GET http://{ALB_DNS}/product/${__Random(1,1000)}`
   - Duration Controller: 13 minutes total
   - Response Assertion: status code 200
   - Listeners: Summary Report, Response Time Graph, Aggregate Report (save as CSV)
3. Duplicate for 50× scenario: 5000 users peak.
4. Add a CSV Data Set Config pointing to `configs/traffic_config.yaml` arrival rates.
5. Parameterise with JMeter properties: `-Jhost=ALB_DNS -Jpeak_users=1000`.
6. Test locally against 4 Flask backends: `jmeter -n -t flash_sale_10x.jmx -Jhost=localhost -Jport=5001`.

**Dependencies:** T1.2  
**Tool:** Apache JMeter 5.6.3  
**Input:** Flask backend running locally  
**Output:** `.jmx` test plans + sample local JMeter CSV  
**Estimated Time:** 3 hours  
**Acceptance Criteria:** JMeter test completes 13-minute run against localhost, produces CSV with latency/throughput columns.  
**Costs Money?** No (local only at this stage).

---

### T2.3 — Alibaba Cluster Trace Acquisition [PARALLEL]

**Objective:** Download Alibaba Cloud Cluster Trace for supplementary pre-training analysis (NOT primary dataset).

**Steps:**
1. Visit https://github.com/alibaba/clusterdata — choose `cluster-trace-v2018`.
2. Download `batch_task.tar.gz` (≈ 2 GB compressed) from the release page.
3. Extract to `data/alibaba_trace/` (add to `.gitignore`).
4. Pre-process in `notebooks/01_traffic_analysis.ipynb`:
   - Load `batch_task.csv` (columns: `task_name`, `start_time`, `end_time`, `plan_cpu`, `plan_mem`)
   - Compute per-minute request arrival counts (proxy for arrival rate)
   - Normalise to match FlashSale-Synthetic format
   - Save processed file to `data/alibaba_processed.csv`

**⚠️ INTEGRITY NOTE:** The Alibaba trace is batch job scheduling data, NOT e-commerce request routing data. Do not claim it validates flash-sale routing. Use only for: (a) comparing arrival rate distributions in the paper's supplementary material, (b) optional agent pre-training warm-start.

**Dependencies:** T1.1, internet access  
**Tool:** Python, pandas  
**Input:** Alibaba GitHub release  
**Output:** `data/alibaba_processed.csv`, distribution comparison plots  
**Estimated Time:** 2 hours  
**Acceptance Criteria:** `alibaba_processed.csv` loads without error. Per-minute arrival rate distribution plotted.  
**Costs Money?** No (public dataset).

---

## PHASE 3 — DRL IMPLEMENTATION

**Duration:** 7–10 days | **[LOCAL ONLY]**

---

### T3.1 — PPO Agent: Training [CRITICAL PATH][LOCAL ONLY]

**Objective:** Train PPO agent on `FlashSaleEnv`, checkpoint to disk, achieve convergence.

**Steps:**
1. Create `src/agents/ppo_agent.py`:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
)
from src.environment.flash_sale_env import FlashSaleEnv
import yaml, os

def train_ppo(config_path="configs/ppo_config.yaml", save_dir="models/ppo"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env = make_vec_env(FlashSaleEnv, n_envs=4,
                       env_kwargs={"n_instances": 4, "seed": 42})
    eval_env = FlashSaleEnv(n_instances=4, seed=43)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        policy_kwargs={"net_arch": cfg["net_arch"]},
        tensorboard_log=f"{save_dir}/tb_logs/",
        verbose=1
    )

    callbacks = [
        CheckpointCallback(save_freq=100_000, save_path=f"{save_dir}/checkpoints/",
                           name_prefix="ppo_flash"),
        EvalCallback(eval_env, best_model_save_path=f"{save_dir}/best/",
                     eval_freq=50_000, n_eval_episodes=3, verbose=1)
    ]

    model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks,
                progress_bar=True)
    model.save(f"{save_dir}/ppo_flash_final")
    return model
```
2. Create `configs/ppo_config.yaml`:
```yaml
learning_rate: 0.0003
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
net_arch: [64, 64]
total_timesteps: 2000000
```
3. Run training: `python -m src.agents.ppo_agent` (~2–4 hours on laptop CPU with 4 envs).
4. Monitor TensorBoard: `tensorboard --logdir models/ppo/tb_logs/`.
5. Convergence criterion: mean episode reward stops increasing (< 1% change over 500k steps).
6. Save final model as `models/ppo/ppo_flash_v1.zip`.

**Expected training time (CPU only):**
- Laptop (4 cores, no GPU): ~3–4 hours for 2M steps with n_envs=4
- Google Colab (free, CPU): ~5–6 hours
- SageMaker ml.t2.medium: ~4–5 hours ($0.046/hr → ≈ $0.20 total) [COSTS MONEY]

**Dependencies:** T1.3 (environment), T1.1 (environment)  
**Tool:** Stable-Baselines3, PyTorch, TensorBoard  
**Input:** `FlashSaleEnv`, `configs/ppo_config.yaml`  
**Output:** `models/ppo/ppo_flash_v1.zip`, TensorBoard reward curves  
**Estimated Time:** 5 hours implementation + 4 hours training = 9 hours total  
**Acceptance Criteria:** Training completes without error. TensorBoard shows reward increasing and plateauing. `model.predict(obs)` returns action in `[0, 3]` within 5ms.  
**Costs Money?** No (local). If using SageMaker: ~$0.20–$0.50.

---

### T3.2 — DQN Agent: Training [PARALLEL with T3.1]

**Objective:** Train DQN baseline on identical `FlashSaleEnv` for direct PPO comparison.

**Steps:**
1. Create `src/agents/dqn_agent.py`:
```python
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.environment.flash_sale_env import FlashSaleEnv
import yaml

def train_dqn(config_path="configs/dqn_config.yaml", save_dir="models/dqn"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env = FlashSaleEnv(n_instances=4, seed=42)
    eval_env = FlashSaleEnv(n_instances=4, seed=43)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        target_update_interval=cfg["target_update_interval"],
        train_freq=cfg["train_freq"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_final_eps=cfg["exploration_final_eps"],
        policy_kwargs={"net_arch": cfg["net_arch"]},
        tensorboard_log=f"{save_dir}/tb_logs/",
        verbose=1
    )
    callbacks = [
        CheckpointCallback(save_freq=100_000, save_path=f"{save_dir}/checkpoints/",
                           name_prefix="dqn_flash"),
        EvalCallback(eval_env, best_model_save_path=f"{save_dir}/best/",
                     eval_freq=50_000, n_eval_episodes=3, verbose=1)
    ]
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks,
                progress_bar=True)
    model.save(f"{save_dir}/dqn_flash_final")
    return model
```
2. Create `configs/dqn_config.yaml`:
```yaml
learning_rate: 0.0001
buffer_size: 100000
learning_starts: 10000
batch_size: 64
gamma: 0.99
target_update_interval: 1000
train_freq: 4
exploration_fraction: 0.15
exploration_final_eps: 0.05
net_arch: [64, 64]
total_timesteps: 2000000
```
3. Train identically to PPO: `python -m src.agents.dqn_agent`.
4. **IMPORTANT:** Use the same test seeds (44–58) as PPO for the comparison. Never use validation seeds for final evaluation.

**Dependencies:** T1.3, T3.1 (environment must be finalised before training starts)  
**Tool:** Stable-Baselines3  
**Input:** `FlashSaleEnv`, `configs/dqn_config.yaml`  
**Output:** `models/dqn/dqn_flash_v1.zip`, TensorBoard reward curves  
**Estimated Time:** 4 hours implementation + 5 hours training = 9 hours  
**Acceptance Criteria:** Same criteria as T3.1. `model.predict(obs)` returns action in `[0, 3]`.  
**Costs Money?** No (local).

---

### T3.3 — Local Evaluation: All Algorithms [CRITICAL PATH]

**Objective:** Run all 6 algorithms (PPO, DQN, RR, WRR, LC, Threshold) on the test set in `FlashSaleEnv`. Confirm PPO and DQN behave plausibly before AWS deployment.

**Steps:**
1. Create `notebooks/03_results_analysis.ipynb` with the evaluation loop:
```python
import numpy as np
from src.environment.flash_sale_env import FlashSaleEnv
from src.agents.ppo_agent import load_ppo
from src.agents.dqn_agent import load_dqn
from src.baselines.round_robin import RoundRobinLB
from src.baselines.least_connections import LeastConnectionsLB
from src.metrics.collector import MetricsCollector

TEST_SEEDS = list(range(127, 142))  # 15 test episodes
N_REPS = 5

algorithms = {
    "PPO": load_ppo("models/ppo/ppo_flash_v1.zip"),
    "DQN": load_dqn("models/dqn/dqn_flash_v1.zip"),
    "RoundRobin": RoundRobinLB(4),
    "LeastConnections": LeastConnectionsLB(4),
}
results = {}
for name, agent in algorithms.items():
    run_results = []
    for seed in TEST_SEEDS[:N_REPS]:
        env = FlashSaleEnv(n_instances=4, seed=seed)
        collector = MetricsCollector(name, f"local_eval_{seed}", seed)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent.select(obs)
            obs, reward, done, _, _ = env.step(action)
            collector.record_step(env.step_count, action, reward, obs,
                                  obs[3*4 + int(action)] * 500,  # resp_ema → ms
                                  obs[:4], obs[3*4 + int(action)] > 0.7)
        run_results.append(collector.compute_summary())
    results[name] = run_results
```
2. Compute mean ± std for each metric across 5 runs.
3. Plot reward convergence curves for PPO and DQN.
4. Generate preliminary comparison table (P95 latency, CPU utilisation, SLA violation rate).
5. **GATE CHECK:** If PPO does NOT outperform Round Robin on P95 latency by at least 10%, debug the environment and reward function before proceeding to AWS deployment.

**Dependencies:** T3.1, T3.2, T1.4, T1.5  
**Tool:** Python, Jupyter  
**Input:** Trained models, test seeds  
**Output:** Preliminary results table, reward convergence plots  
**Estimated Time:** 4 hours  
**Acceptance Criteria:** PPO and DQN produce results meaningfully different from random. PPO reward curve shows monotonic improvement. No NaN/Inf values in any metric.  
**Costs Money?** No.

---

## PHASE 4 — AWS DEPLOYMENT

**Duration:** 5–7 days | **[COSTS MONEY — read all cost notes carefully]**

> **Cost optimisation revision (2026-08-20):** This phase reflects the revised low-cost architecture from PRD §12. Key changes: (1) API Gateway removed from primary data path — JMeter targets ALB DNS directly; (2) CloudFront removed entirely; (3) SageMaker demoted to optional fallback (T4.7); (4) custom CloudWatch metrics reduced to 8 (within free tier); (5) ALB must be deleted between experiment phases to avoid idle charges. Research methodology unchanged.

---

### AWS COST ANALYSIS AND FREE TIER VERIFICATION

**⚠️ IMPORTANT:** Based on AWS pricing as of mid-2026 for us-east-1. Always verify at https://aws.amazon.com/pricing/ before spending.

| Service | Free Tier | Beyond Free Tier | Notes |
|---------|-----------|------------------|-------|
| EC2 t2.micro | 750 hrs/month (new accounts) | $0.0116/hr | 5 instances stopped between sessions; active ~4 hrs/day × 14 exp days = 280 hrs ≈ **within free tier if new account** |
| ALB | **NOT free tier** | $0.0225/hr + $0.008/LCU-hr | Active ~4 hrs/day × 14 days = 56 hrs → **~$1.30 + LCU** — **delete between experiment phases** |
| Lambda | 1M requests/month free | $0.20/1M req beyond | < 100k invocations → **free** |
| CloudWatch custom metrics | 10 metrics free | $0.30/metric/month beyond | **8 custom metrics → free tier** (reduced from ~20) |
| CloudWatch alarms | 10 alarms free | $0.10/alarm/month beyond | 5 alarms used → **free** |
| S3 | 5 GB free | $0.023/GB/month | ~2–3 GB total → **free** |
| SageMaker ml.t2.medium | **NOT free tier** | $0.046/hr | Optional fallback only; ~$0.20–$0.50 if used once |
| API Gateway | *(removed from primary path)* | — | JMeter → ALB DNS directly; API GW not provisioned |
| ASG | Free (EC2 cost only) | — | No additional cost |
| DynamoDB | 25 GB + 200M req/month free | — | **Free** |
| SNS | 1M notifications free | — | **Free** |
| IAM | Always free | — | **Free** |
| CloudFront | *(removed from architecture)* | — | Not provisioned |

**REVISED ESTIMATED TOTAL AWS COST FOR FULL EXPERIMENT CAMPAIGN:**

| Scenario | Original Estimate | Revised Estimate |
|----------|------------------|-----------------|
| Conservative (4 hrs active/day, 14 days) | $12–18 | **$4–8** |
| Realistic (6 hrs active/day, 14 days) | $20–30 | **$7–12** |
| Maximum (8 hrs active/day, 14 days) | $35–45 | **$10–15** |

> **This design aims to minimise AWS expenditure. It does not guarantee zero cost. ALB and EC2 will incur charges during active experiment windows. Do not leave resources running overnight.**

**BILLING SAFEGUARDS (mandatory before starting Phase 4):**
1. Set a CloudWatch billing alarm at **$5** (warning email) and **$20** (action email).
2. Set AWS Budgets alert: Monthly cost > $15 → email all 4 team members immediately.
3. **ALWAYS stop EC2 instances (desired=0, min=0) and delete ALB after each experiment session.**
4. Enable Cost Explorer to monitor daily spend; check each morning during experiment phase.

---

### T4.1 — AWS Account Setup and Safety [CRITICAL PATH][COSTS MONEY]

**Objective:** Prepare AWS account with billing safeguards, IAM roles, and resource limits before any chargeable resources are created.

**Steps:**
1. Sign in to AWS Console as root user → go to Billing → Billing Preferences → enable billing alerts.
2. Create CloudWatch billing alarm:
   - Metric: `EstimatedCharges`, threshold: $5 (warning)
   - Create second alarm at $25 (action: email team)
3. Create AWS Budget: $20/month, 80% threshold alert.
4. Switch to IAM admin user (never use root for resources):
```bash
aws iam create-user --user-name flashbalance-admin
aws iam attach-user-policy --user-name flashbalance-admin \
    --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
aws iam create-access-key --user-name flashbalance-admin
```
5. Configure AWS CLI: `aws configure` → enter access key, secret, region=us-east-1, output=json.
6. Verify: `aws sts get-caller-identity` returns your account ID.
7. Check EC2 t2.micro quota: `aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A` → must be ≥ 5.
8. If quota < 5: request increase via Service Quotas console (takes 1–2 business days).

**Dependencies:** None (first AWS task)  
**Tool:** AWS Console, AWS CLI  
**Input:** AWS student account credentials  
**Output:** Billing alarms set, IAM admin user created, CLI configured  
**Estimated Time:** 2 hours  
**Acceptance Criteria:** `aws cloudwatch describe-alarms` shows 2 billing alarms. `aws budgets describe-budgets --account-id <ID>` shows 1 budget.  
**Costs Money?** IAM, CloudWatch alarms, Budgets: **Free**. The alarm creation itself is free.

---

### T4.2 — IAM Roles and Policies [CRITICAL PATH]

**Objective:** Create least-privilege IAM roles for Lambda, EC2, and SageMaker.

**Steps:**
1. Create `src/aws/iam_setup.py` and run it:
```python
import boto3, json

iam = boto3.client('iam')

LAMBDA_TRUST = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                   "Action": "sts:AssumeRole"}]
})

LAMBDA_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {"Effect": "Allow",
         "Action": ["cloudwatch:GetMetricStatistics", "cloudwatch:PutMetricData",
                    "cloudwatch:DescribeAlarms"],
         "Resource": "*"},
        {"Effect": "Allow",
         "Action": ["elasticloadbalancing:ModifyTargetGroupAttributes",
                    "elasticloadbalancing:DescribeTargetGroups",
                    "elasticloadbalancing:DescribeTargetHealth"],
         "Resource": "*"},
        {"Effect": "Allow",
         "Action": ["autoscaling:SetDesiredCapacity",
                    "autoscaling:DescribeAutoScalingGroups"],
         "Resource": "*"},
        {"Effect": "Allow",
         "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": "arn:aws:s3:::flashbalanceai-*/*"},
        {"Effect": "Allow",
         "Action": ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem"],
         "Resource": "arn:aws:dynamodb:us-east-1:*:table/routing_decisions"},
        {"Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream",
                    "logs:PutLogEvents"],
         "Resource": "*"}
    ]
})

# Create role
role = iam.create_role(RoleName="FlashBalanceAI-Lambda-Role",
                        AssumeRolePolicyDocument=LAMBDA_TRUST)
iam.put_role_policy(RoleName="FlashBalanceAI-Lambda-Role",
                    PolicyName="FlashBalanceLambdaPolicy",
                    PolicyDocument=LAMBDA_POLICY)
print("Lambda role ARN:", role['Role']['Arn'])
```
2. Run: `python src/aws/iam_setup.py` → note the role ARN, add to `configs/aws_config.yaml`.
3. Create SageMaker execution role similarly (trust: `sagemaker.amazonaws.com`) with `AmazonSageMakerFullAccess` + `S3ReadWrite` scoped to the project bucket.

**How to delete:** `aws iam delete-role-policy --role-name FlashBalanceAI-Lambda-Role --policy-name FlashBalanceLambdaPolicy` then `aws iam delete-role --role-name FlashBalanceAI-Lambda-Role`.

**Dependencies:** T4.1  
**Tool:** AWS IAM, boto3  
**Output:** IAM roles created, ARNs in `configs/aws_config.yaml`  
**Estimated Time:** 2 hours  
**Acceptance Criteria:** `aws iam get-role --role-name FlashBalanceAI-Lambda-Role` returns HTTP 200. Policy document matches intended permissions.  
**Costs Money?** IAM is **free**.

---

### T4.3 — S3 Bucket and Model Upload [COSTS MONEY — minimal]

**Objective:** Create S3 bucket, upload trained models and traffic config.

**Steps:**
1. Create bucket (replace `{ACCOUNT_ID}` with your 12-digit account ID):
```bash
aws s3 mb s3://flashbalanceai-{ACCOUNT_ID} --region us-east-1
aws s3api put-bucket-versioning \
    --bucket flashbalanceai-{ACCOUNT_ID} \
    --versioning-configuration Status=Enabled
aws s3api put-public-access-block \
    --bucket flashbalanceai-{ACCOUNT_ID} \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```
2. Upload trained models:
```bash
aws s3 cp models/ppo/ppo_flash_v1.zip s3://flashbalanceai-{ACCOUNT_ID}/models/ppo_flash_v1.zip
aws s3 cp models/dqn/dqn_flash_v1.zip s3://flashbalanceai-{ACCOUNT_ID}/models/dqn_flash_v1.zip
aws s3 cp configs/traffic_config.yaml s3://flashbalanceai-{ACCOUNT_ID}/configs/traffic_config.yaml
```
3. Add bucket name to `configs/aws_config.yaml`.

**How to delete:** `aws s3 rb s3://flashbalanceai-{ACCOUNT_ID} --force`.

**Dependencies:** T4.2, T3.1 (trained model)  
**Tool:** AWS S3, AWS CLI  
**Output:** S3 bucket with uploaded models  
**Estimated Time:** 1 hour  
**Acceptance Criteria:** `aws s3 ls s3://flashbalanceai-{ACCOUNT_ID}/models/` shows both model files.  
**Costs Money?** Storage: First 5 GB free → **Free** for our use.

---

### T4.4 — EC2 Backend Instances and Auto Scaling Group [COSTS MONEY]

**Objective:** Launch 4 t2.micro EC2 instances running the Flask backend, inside an Auto Scaling Group.

**Steps:**
1. Create a launch template using the AWS Console or CLI:
```bash
# User data script (base64-encode before use):
# #!/bin/bash
# yum update -y && yum install python3-pip -y
# pip3 install flask gunicorn
# aws s3 cp s3://flashbalanceai-{ACCOUNT_ID}/backend/app.py /home/ec2-user/app.py
# INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
# export INSTANCE_ID BASE_LATENCY_MS=20
# gunicorn -w 1 -b 0.0.0.0:5000 app:app &

aws ec2 create-launch-template \
    --launch-template-name FlashBalanceAI-Backend \
    --version-description "v1" \
    --launch-template-data '{
      "InstanceType": "t2.micro",
      "ImageId": "ami-0c02fb55956c7d316",
      "IamInstanceProfile": {"Name": "FlashBalanceAI-EC2-Profile"},
      "SecurityGroupIds": ["sg-XXXX"],
      "UserData": "BASE64_USER_DATA_HERE"
    }'
```
2. Create security group (allow port 5000 inbound from ALB SG only, port 22 from your IP):
```bash
aws ec2 create-security-group \
    --group-name FlashBalanceAI-Backend-SG \
    --description "Backend EC2 instances"
aws ec2 authorize-security-group-ingress \
    --group-name FlashBalanceAI-Backend-SG \
    --protocol tcp --port 5000 --source-group FlashBalanceAI-ALB-SG
```
3. Create Auto Scaling Group:
```bash
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name FlashBalanceAI-ASG \
    --launch-template "LaunchTemplateName=FlashBalanceAI-Backend,Version=1" \
    --min-size 2 --max-size 8 --desired-capacity 4 \
    --availability-zones us-east-1a us-east-1b \
    --default-cooldown 180
```
4. **IMPORTANT — shut down after each session:**
```bash
aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name FlashBalanceAI-ASG --desired-capacity 0 --min-size 0
```

**How to delete everything:**
```bash
aws autoscaling delete-auto-scaling-group --auto-scaling-group-name FlashBalanceAI-ASG --force-delete
aws ec2 delete-launch-template --launch-template-name FlashBalanceAI-Backend
aws ec2 delete-security-group --group-name FlashBalanceAI-Backend-SG
```

**Dependencies:** T4.2, T1.2  
**Tool:** AWS EC2, Auto Scaling, AWS CLI  
**Output:** 4 running t2.micro instances, ASG created  
**Estimated Time:** 4 hours  
**Acceptance Criteria:** `aws ec2 describe-instances --filters Name=tag:aws:autoscaling:groupName,Values=FlashBalanceAI-ASG` shows 4 running instances. `curl http://{INSTANCE_IP}:5000/health` returns `{"status":"healthy"}`.  
**Costs Money?** **YES** — t2.micro × 4 × hours running. Stay within free tier 750 hrs/month. Set desired-capacity=0 when not experimenting.

---

### T4.5 — Application Load Balancer [COSTS MONEY]

**Objective:** Create ALB with target group pointing to backend EC2 instances.

> **Cost discipline (revised 2026-08-20):** The ALB must be **deleted** (not just stopped) between experiment phases because AWS charges $0.0225/hr even when idle. Recreating it before each experiment session adds ~20 minutes of setup time but eliminates idle charges. Save the ALB ARN and Target Group ARN to `configs/aws_config.yaml` for quick recreation. The `src/aws/deploy.py` script must support idempotent ALB creation.

**Steps:**
1. Create ALB:
```bash
aws elbv2 create-load-balancer \
    --name FlashBalanceAI-ALB \
    --subnets subnet-XXXX subnet-YYYY \
    --security-groups sg-ALB \
    --scheme internet-facing \
    --type application
# Save ALB ARN and DNS name to configs/aws_config.yaml
```
2. Create target group:
```bash
aws elbv2 create-target-group \
    --name FlashBalanceAI-TG \
    --protocol HTTP --port 5000 \
    --vpc-id vpc-XXXX \
    --health-check-path /health \
    --health-check-interval-seconds 10 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 2
```
3. Register all 4 EC2 instances as targets.
4. Create listener: HTTP:80 → forward to target group.
5. Enable ALB access logs: `aws elbv2 modify-load-balancer-attributes --load-balancer-arn <ARN> --attributes Key=access_logs.s3.enabled,Value=true Key=access_logs.s3.bucket,Value=flashbalanceai-{ACCOUNT_ID} Key=access_logs.s3.prefix,Value=logs/alb`
6. Save ALB DNS name to `configs/aws_config.yaml` as `alb_dns`.
7. **Test:** JMeter targeting `http://{ALB_DNS}/product/1` directly (no API Gateway required).

**How to delete (run after every experiment session):**
```bash
aws elbv2 delete-load-balancer --load-balancer-arn <ARN>
aws elbv2 delete-target-group --target-group-arn <ARN>
# Also set ASG desired=0 to stop backend EC2 instances
aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name FlashBalanceAI-ASG \
    --desired-capacity 0 --min-size 0
```

**⚠️ COST NOTE:** ALB is NOT free tier. $0.0225/hr fixed + LCU charges.
- Active during experiments only: ~4 hrs/day × 14 experiment days = 56 hrs → **~$1.30 total fixed** (+ LCU ~$0.50 estimated).
- If left running 24/7 for 30 days: ~$16.20 — **do not leave running overnight**.

**Dependencies:** T4.4
**Tool:** AWS ALB, AWS CLI
**Output:** ALB DNS name, target group ARN, access logs enabled
**Estimated Time:** 2 hours
**Acceptance Criteria:** `aws elbv2 describe-target-health --target-group-arn <ARN>` shows all 4 targets healthy. `curl http://{ALB_DNS}/product/1` returns 200. ALB access logs appearing in S3 within 5 minutes of first request.
**Costs Money?** **YES — $0.0225/hr fixed charge.** Delete after every experiment session.

---

### T4.6 — DynamoDB Table [FREE TIER]

**Objective:** Create routing_decisions table for audit logging.

**Steps:**
```bash
aws dynamodb create-table \
    --table-name routing_decisions \
    --attribute-definitions \
        AttributeName=timestamp,AttributeType=S \
        AttributeName=experiment_id,AttributeType=S \
    --key-schema \
        AttributeName=timestamp,KeyType=HASH \
        AttributeName=experiment_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1
```

**How to delete:** `aws dynamodb delete-table --table-name routing_decisions`.

**Dependencies:** T4.1  
**Estimated Time:** 30 minutes  
**Costs Money?** On-demand DynamoDB: first 25 GB and 200M requests free → **Free** for our use.

---

### T4.7 — SageMaker Training (Optional Fallback) [COSTS MONEY — USE ONLY IF LOCAL FAILS]

**Objective:** Use SageMaker only if local laptop training AND Google Colab both fail to produce a converged PPO model within the available time.

**Decision rule:** SageMaker is a last resort. Use it if and only if:
- Local laptop training (T3.1) does not complete 2M steps within 8 hours, AND
- Google Colab (free) is unavailable or produces numerical instability

> **Preferred alternatives in order:** (1) Laptop CPU (~3–4 hrs for 2M steps), (2) Google Colab free GPU/TPU (~1–2 hrs), (3) SageMaker ml.t2.medium (~$0.20–$0.50). Do not provision SageMaker unless the team has explicitly confirmed both (1) and (2) are insufficient.

**Steps (if needed):**
1. Open SageMaker Console → Notebook Instances → Create notebook instance.
2. Instance type: **ml.t2.medium** (2 vCPU, 4 GB RAM) — cheapest available.
3. EBS volume: 10 GB gp2.
4. IAM role: `FlashBalanceAI-SageMaker-Role` (created in T4.2).
5. Clone project repo into the notebook.
6. Run `notebooks/02_ppo_training.ipynb` (copy of local training notebook).
7. **IMMEDIATELY stop the notebook instance after training completes:**
```bash
aws sagemaker stop-notebook-instance --notebook-instance-name FlashBalanceAI-Training
```
8. Upload trained model to S3 (training script should do this automatically).

**How to delete:** `aws sagemaker delete-notebook-instance --notebook-instance-name FlashBalanceAI-Training` (must be stopped first).

**Dependencies:** T4.3, T3.1 (training script ready)  
**Estimated Time:** 1 hour setup + 4–5 hours training  
**Costs Money?** **YES — $0.046/hr × 5 hrs = $0.23.** Stop immediately after training.  
**Safer alternative:** Use Google Colab (free, Python 3.10 compatible) or team member's laptop with 16GB RAM.

---

**End of Part 1**

*Continue reading `IMPLEMENTATION_PLAN_PART2.md` for:*
*Phase 5 (AWS Integration) · Phase 6 (Experiments E1–E10) · Phase 7 (Statistical Validation) · Phase 8 (Paper Preparation) · Phase 9 (Demo/Teardown) · Full Timeline Table*
