# PRD — FlashBalanceAI
## Deep Reinforcement Learning-Based Adaptive Cloud Load Balancing for Flash-Sale Traffic

**Document Version:** 1.0  
**Date:** 2026-08-20  
**Authors:** Devkanti Sarkar (24BIT0162) · Agrima Gupta (24BIT0253) · Mohar Gorai ·   
**Branch:** `prd`

---

## SOURCE MATERIAL RECONCILIATION NOTE

This PRD was synthesised from three individual Phase-I research documents:

| Source | Student | Primary Approach | Key Contribution |
|--------|---------|------------------|------------------|
| `devkanti.pdf` (Cloud_Project.docx) | Devkanti Sarkar | PPO + burst-aware multi-objective reward | Literature survey (5 papers), PPO selection rationale, 4-component reward function, AWS-native architecture, expected 30% p95 improvement |
| `agrima.pdf` (Research_Gap_Student2_UPDATED_v2.docx) | Agrima Gupta | DQN vs PPO request-level routing (FlashBalanceAI) | Deep 5-paper analysis, DQN/PPO comparison rationale, synthetic traffic generator, DynamoDB/Cognito/CloudFront services, SageMaker-trained agents |
| `mohar.pdf` (Project_Phase_1.docx) | Mohar Gorai | Hierarchical Multi-Agent System + DDPG | H-MAS noise-filtering layer, DDPG for continuous routing, Alibaba Cloud Cluster Trace dataset |

**Architecture Decision:** After reconciling all three documents, **PPO is selected as the primary DRL algorithm**. DQN is retained as a primary baseline. DDPG and H-MAS components are repurposed as ablation variants and theoretical context. The rationale is detailed in Section 10.

---

## 1. Project Title

**FlashBalanceAI: A Proximal Policy Optimization Framework for Adaptive Cloud Load Balancing under Flash-Sale Burst Traffic**

*Subtitle:* Real-Time, Multi-Objective DRL-Driven Traffic Management on AWS for E-Commerce Platforms

---

## 2. Executive Summary

Flash-sale events in e-commerce platforms create extreme traffic spikes — often 10× to 100× within seconds — that overwhelm traditional static load balancers and reactive autoscalers. Existing Deep Reinforcement Learning (DRL) solutions address general cloud workloads, network-level routing, or serverless edge computing but none specifically target the *thundering herd* pattern of flash-sale bursts in a production AWS deployment.

FlashBalanceAI proposes a **PPO-based adaptive load balancing framework** that integrates directly with AWS-native services (ALB, EC2 Auto Scaling Groups, CloudWatch, Lambda, S3, API Gateway) and is trained on a synthetic flash-sale traffic generator calibrated to real-world e-commerce burst patterns. The system introduces a **four-component burst-aware reward function** that simultaneously optimises response time, resource utilisation, throughput, and SLA violation penalties.

Primary baselines are Round Robin, Least Connections, threshold-based autoscaling, and DQN. The system is designed for reproducible experimental validation on a student AWS account with minimum cost, targeting conference publication (IEEE/ACM Cloud/ICDCS/CCGRID 2026–2027).

---

## 3. Problem Statement

E-commerce flash-sale events exhibit two pathological traffic characteristics that defeat existing infrastructure:

1. **Thundering herd onset:** Traffic escalates from baseline (~100 req/s) to peak (~10,000–100,000 req/s) within 2–10 seconds — faster than any reactive autoscaler's cold-start latency (AWS ASG scale-up: 2–5 minutes).
2. **Transient noise within the burst:** Millisecond-level traffic spikes within the overall burst cause standard DRL agents to oscillate — overreacting to noise, thrashing auto-scaling, and degrading performance.

**Consequence:** Server CPU saturation → latency spikes → request queue overflow → HTTP 500/503 errors → revenue loss. Industry reports (cited in Devkanti's survey) estimate that every second of additional latency during a flash sale reduces conversions by 7%.

**Formal problem:** Given a stream of incoming HTTP requests at time *t* with arrival rate λ(t) that follows a flash-sale burst profile, distribute requests across *N* heterogeneous backend EC2 instances such that P95 latency remains ≤ 500 ms, throughput is maximised, SLO violations are minimised, and AWS operational cost is minimised — all simultaneously, in real time, without manual reconfiguration.

---

## 4. Motivation

**Why DRL specifically?**

- Traditional algorithms (Round Robin, Least Connections) are stateless and cannot learn temporal traffic patterns.
- Threshold-based autoscaling (CPU > 70% → add instance) reacts after the spike, not before it.
- Rule-based predictive scaling requires manual schedule configuration and breaks for unannounced sales.
- DRL agents learn from experience, improve with each flash-sale event, and can achieve proactive pre-scaling by detecting early-warning signals in CloudWatch metrics.

**Why this gap specifically?**

Five surveyed papers (Devkanti) and five additional papers (Agrima) confirm that no existing work:
- Tests DRL under flash-sale burst traffic (10×–100× spikes within seconds)
- Integrates DRL directly with AWS production services (ELB, ASG, Lambda, CloudWatch)
- Compares DQN and PPO on identical flash-sale workloads with P95/P99 latency metrics
- Uses a multi-objective reward balancing cost AND performance simultaneously

**Why PPO (and not DQN or DDPG)?** → See Section 10.

---

## 5. Research Objectives

| # | Objective | Measurable Target |
|---|-----------|-------------------|
| O1 | Develop a PPO-based DRL agent for per-request load balancing under flash-sale burst traffic | Agent trains to convergence on synthetic flash-sale workload |
| O2 | Design a four-component burst-aware reward function | All four components validated individually (ablation study) |
| O3 | Integrate DRL agent with AWS: ALB, ASG, CloudWatch, Lambda, S3, API Gateway | End-to-end pipeline operational in AWS student account |
| O4 | Reduce P95 response time by ≥ 30% vs threshold-based autoscaling during 10× spike | Measured across ≥ 5 independent experimental runs |
| O5 | Maintain ≥ 85% CPU utilisation during peak load (avoid over-provisioning) | Measured from CloudWatch during peak window |
| O6 | Reduce AWS operational cost by 20–25% vs static over-provisioned baseline | Measured from AWS Cost Explorer during equivalent experiment runs |
| O7 | Compare PPO vs DQN on identical flash-sale workloads | Statistical comparison with confidence intervals |
| O8 | Achieve full reproducibility with open-source code, fixed seeds, and documented environment | Third party can reproduce within ± 5% of reported metrics |

---

## 6. Scope and Non-Scope

### In Scope
- PPO agent for request-level routing across N EC2 backend instances (N configurable: 2–8)
- DQN baseline on identical environment
- Threshold-based autoscaling baseline (AWS CloudWatch alarms + ASG)
- Round Robin and Least Connections baselines (software-implemented)
- Synthetic flash-sale traffic generator (normal → spike → peak → cooldown)
- AWS deployment: ALB, EC2 (t2.micro/t3.micro), Lambda, CloudWatch, S3, API Gateway
- Metrics: avg latency, P95/P99 latency, throughput, failure rate, CPU utilisation, cost
- Single AWS region (us-east-1)
- HTTP/1.1 request routing (not gRPC, not WebSocket)
- Offline training (SageMaker notebook or local), online inference

### Out of Scope
- Multi-region or multi-cloud deployment
- Database-layer load balancing (RDS, DynamoDB routing)
- Container orchestration (ECS, EKS) — EC2-based only
- Real production traffic data (legal/privacy constraints)
- GPU training (cost constraints)
- DDPG or H-MAS as primary agents (used only as theoretical baselines in related work discussion)
- Payment processing, user authentication, product catalogue services

---

## 7. Proposed System Explanation

FlashBalanceAI is a two-stage system:

**Stage 1 — Request Routing (PPO Agent):** Every incoming HTTP request is intercepted at the API Gateway / ALB layer. A lightweight PPO inference agent (running on a dedicated EC2 t2.micro or as a Lambda function) observes the current system state (from CloudWatch) and selects which backend EC2 instance should handle the request. The routing decision is executed in < 5 ms.

**Stage 2 — Instance Scaling (CloudWatch + Lambda + ASG):** A parallel control loop monitors aggregate CloudWatch metrics every 30 seconds. When the DRL agent's state signals high utilisation, a Lambda function triggers ASG scale-out *before* CPU saturation occurs. This is the proactive pre-scaling mechanism that differentiates FlashBalanceAI from reactive threshold autoscaling.

**Training:** Offline, using a simulated environment that replays synthetic flash-sale traffic traces. The PPO agent is trained using Stable-Baselines3 on a local machine or SageMaker notebook instance. The trained policy (neural network weights) is stored in S3 and loaded into the inference endpoint.

**Monitoring:** A CloudWatch dashboard and a lightweight Python/Dash monitoring dashboard visualise real-time routing decisions, per-instance CPU, latency histograms, and SLA violation counts.

---

## 8. Detailed End-to-End Architecture

```
                         ┌─────────────────────────────────────────────────┐
                         │              AWS Cloud (us-east-1)              │
                         │                                                 │
 Users (JMeter)          │  ┌─────────────┐    ┌──────────────────────┐  │
 flash-sale traffic ────▶│  │ API Gateway │───▶│  ALB (HTTP Listener) │  │
                         │  └─────────────┘    └──────────┬───────────┘  │
                         │                               │               │
                         │  ┌────────────────────────────▼────────────┐  │
                         │  │         PPO Inference Agent             │  │
                         │  │   (Lambda / EC2 t2.micro endpoint)      │  │
                         │  │  - Reads state from CloudWatch          │  │
                         │  │  - Selects target backend instance      │  │
                         │  │  - Executes ALB target group change     │  │
                         │  └────────────┬────────────────────────────┘  │
                         │               │ routes to                      │
                         │  ┌────────────▼──────────────────────────┐    │
                         │  │     EC2 Backend Pool (Auto Scaling)   │    │
                         │  │  ┌──────────┐ ┌──────────┐ ┌──────┐  │    │
                         │  │  │ Server-1 │ │ Server-2 │ │  ... │  │    │
                         │  │  │ t2.micro │ │ t2.micro │ │      │  │    │
                         │  │  └──────────┘ └──────────┘ └──────┘  │    │
                         │  └───────────────────────────────────────┘    │
                         │         │ metrics (CPU,latency,queue)         │
                         │  ┌──────▼──────────────────────────────────┐  │
                         │  │          Amazon CloudWatch              │  │
                         │  │  - Custom metrics namespace             │  │
                         │  │  - Alarms → Lambda → ASG scale-out     │  │
                         │  │  - Dashboard                           │  │
                         │  └─────────────────────────────────────────┘  │
                         │                                                │
                         │  ┌──────────────┐   ┌─────────────────────┐  │
                         │  │ Amazon S3    │   │  SageMaker Notebook │  │
                         │  │ model weights│   │  (Offline Training) │  │
                         │  │ traffic logs │   │  ml.t2.medium       │  │
                         │  └──────────────┘   └─────────────────────┘  │
                         │                                                │
                         │  ┌───────────────────┐                        │
                         │  │    IAM Roles      │                        │
                         │  │  least-privilege  │                        │
                         │  └───────────────────┘                        │
                         └────────────────────────────────────────────────┘
```

**Data Flow:**
1. JMeter (local) generates HTTP traffic following flash-sale burst profile → API Gateway → ALB
2. ALB forwards to PPO inference endpoint (Lambda or EC2)
3. PPO endpoint queries CloudWatch for current state (last 10s sliding window)
4. PPO selects backend instance index → ALB routes request to target instance
5. Backend instance processes request → returns response → ALB → JMeter (records latency)
6. CloudWatch agent on each EC2 instance publishes CPU, memory, active connection count every 10s
7. Lambda function triggered by CloudWatch alarm (CPU > 70% for 2 consecutive periods) → triggers ASG scale-out
8. Every 60s: reward is computed from CloudWatch metrics → stored in DynamoDB → used for offline policy update

---

## 9. Detailed Methodology

### 9.1 MDP Formulation

The load balancing problem is formulated as a Markov Decision Process (S, A, P, R, γ):

- **State space S:** System state observed at decision epoch *t*
- **Action space A:** Routing decision (which backend instance receives the next request)
- **Transition P:** Environment dynamics (traffic arrival, server processing, auto-scaling)
- **Reward R:** Four-component burst-aware reward (see Section 9.4)
- **Discount factor γ:** 0.99 (long-horizon optimisation)

**Decision epoch:** Every incoming request (event-driven) or every Δt = 100 ms (time-driven). For this project: **time-driven at 100 ms intervals** to avoid per-request Lambda cost explosion.

### 9.2 State Space

The state vector **s_t ∈ ℝ^(5N+3)** where N = number of active backend instances:

```
s_t = [
  cpu_util[1..N],         # CPU utilisation for each instance (0–1), shape: (N,)
  active_conn[1..N],      # Active HTTP connections per instance, normalised (0–1), shape: (N,)
  queue_depth[1..N],      # Request queue depth per instance, normalised (0–1), shape: (N,)
  resp_time_ema[1..N],    # Exponential moving avg response time per instance (0–1), shape: (N,)
  health_status[1..N],    # Binary: 1=healthy, 0=unhealthy, shape: (N,)
  arrival_rate_norm,      # Current request arrival rate λ(t) normalised against baseline (0–10+)
  burst_indicator,        # Derived: λ(t)/λ_baseline > 2 → 1, else 0
  time_since_last_spike   # Seconds since last burst onset, normalised (0–1)
]
```

**State dimensions:** For N=4 backend instances → 4×5 + 3 = **23-dimensional state**

**Source of state observations:**
- `cpu_util`: CloudWatch `CPUUtilization` metric per EC2 instance (60s default → custom 10s metric)
- `active_conn`: CloudWatch `ActiveConnectionCount` from ALB target group
- `queue_depth`: Custom CloudWatch metric published by backend Flask app
- `resp_time_ema`: CloudWatch `TargetResponseTime` from ALB
- `health_status`: ALB target group health check
- `arrival_rate_norm`: CloudWatch `RequestCount` per 10s window / baseline request count
- `burst_indicator`: Derived in Lambda state collector
- `time_since_last_spike`: Tracked in Lambda state collector using last burst timestamp

### 9.3 Action Space

**Discrete action space A = {0, 1, ..., N-1}** where action *a* = select backend instance *a* as the routing target for the current time window.

**Justification for discrete vs continuous:**
- PPO works with both. Discrete is simpler, lower variance, and directly maps to ALB target group selection.
- DDPG requires continuous actions (e.g., traffic split percentages). While this enables finer-grained control, it adds training instability that is not warranted for our baseline comparison. DDPG is left as a future extension.
- For N=4 instances: |A| = 4 actions.
- At inference: agent selects action *a*, ALB weights are adjusted to route 100% traffic to instance *a* for the next 100 ms window. In production, a weighted split would be more realistic — this is noted as a limitation.

### 9.4 Reward Function

The four-component burst-aware reward at time step *t*:

```
R(t) = w₁·R_lat(t) + w₂·R_util(t) + w₃·R_tput(t) + w₄·R_sla(t)
```

**Components:**

**R_lat(t) — Response time penalty:**
```
R_lat(t) = −(avg_response_time(t) / latency_target) + 1
         = −(L_t / 200ms) + 1
```
Range: large positive when L_t << 200ms, negative when L_t > 200ms.

**R_util(t) — Resource utilisation reward:**
```
R_util(t) = avg_cpu_util(t) − |avg_cpu_util(t) − util_target|
          where util_target = 0.70 (70%)
```
Rewards being near 70% utilisation. Penalises both under-utilisation (waste) and over-utilisation (saturation). Inspired by Chen et al. (2026, Elsevier).

**R_tput(t) — Throughput reward:**
```
R_tput(t) = requests_served(t) / requests_arrived(t)
```
Fraction of requests successfully served (not timed out or failed). Range: [0, 1].

**R_sla(t) — SLA violation penalty:**
```
R_sla(t) = −k · I(p95_latency(t) > SLA_threshold)
          where SLA_threshold = 500ms, k = 2.0
```
Binary penalty of magnitude *k* whenever P95 latency breaches the SLA. This directly implements the SLA-first philosophy from Yamsani & Chenna Reddy (2026, Nature Scientific Reports, as cited in Agrima's survey).

**Weights (initial values, tuned via ablation study):**

| Weight | Value | Justification |
|--------|-------|---------------|
| w₁ (latency) | 0.40 | Primary user-facing metric during flash sales |
| w₂ (utilisation) | 0.20 | Cost control — avoid idle over-provisioning |
| w₃ (throughput) | 0.20 | Ensure requests are served, not just distributed |
| w₄ (SLA) | 0.20 | Hard guardrail on tail latency SLA |

**Note:** These weights are hyperparameters subject to ablation study E10. Claims about optimal weights must be experimentally validated, not assumed.

### 9.5 Agent/Environment Interaction

```
for each time step t = 0, 1, 2, ...:
    1. Collect state s_t from CloudWatch (10s sliding window metrics)
    2. PPO policy π_θ(a | s_t) → sample action a_t
    3. Execute a_t: update ALB target weights via boto3 API
    4. Environment steps: requests arrive, servers process, metrics update
    5. Collect next state s_{t+1} and compute R(t)
    6. Store (s_t, a_t, R(t), s_{t+1}, done) in rollout buffer
    7. Every K=2048 steps: run PPO update (see Section 9.6)
```

### 9.6 Training Methodology

**Algorithm:** Proximal Policy Optimization (PPO-Clip) [Schulman et al., 2017]

**Neural network architecture:**
- Actor network (policy): 2-layer MLP, [64, 64] hidden units, ReLU activations → softmax output over N actions
- Critic network (value function): 2-layer MLP, [64, 64] hidden units, ReLU → scalar value
- Shared trunk (optional): single 64-unit layer → separate actor/critic heads

**Training hyperparameters (initial values):**

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 3×10⁻⁴ (Adam) | Schulman et al. standard |
| Clip ratio ε | 0.2 | PPO paper default |
| Value loss coefficient c₁ | 0.5 | Standard |
| Entropy coefficient c₂ | 0.01 | Encourages exploration |
| GAE λ | 0.95 | Standard |
| Discount γ | 0.99 | Long horizon |
| Rollout steps K | 2048 | Stable-Baselines3 default |
| Mini-batch size | 64 | Standard |
| Update epochs | 10 | PPO standard |
| Max training steps | 2×10⁶ | Monitor convergence |

**Training environment:** Discrete-time simulation of AWS backend (Gymnasium-compatible custom environment). NOT trained directly against live AWS during initial development.

**Training procedure:**
1. Initialise custom Gymnasium environment (`FlashSaleEnv`)
2. Generate synthetic flash-sale episode (30 min simulated → ~18,000 decision steps at 100ms)
3. Run PPO training with Stable-Baselines3 for 2M steps
4. Checkpoint every 100k steps to S3
5. Evaluate on held-out episodes every 100k steps
6. Early stopping if reward plateaus for 500k steps

### 9.7 Inference Methodology

1. Load trained PPO policy weights from S3 into memory
2. Deploy as AWS Lambda function (128MB → 512MB if needed) or EC2 t2.micro
3. On each 100ms tick: call CloudWatch `get_metric_statistics` for all instances
4. Assemble state vector, run `policy.predict(state)` → action (< 2ms inference)
5. Update ALB listener rule weights via `elbv2.modify_rule()` boto3 call
6. Log action + state to DynamoDB for offline analysis

### 9.8 Traffic Generation Methodology

**Tool:** Apache JMeter (local) + custom Python traffic generator (`traffic_generator.py`)

**Flash-sale traffic profile (per episode):**

| Phase | Duration | Request Rate | Description |
|-------|----------|--------------|-------------|
| Warm-up | 2 min | 100 req/s | Normal baseline traffic |
| Pre-announcement | 1 min | 300 req/s | Anticipatory browsing increase |
| Spike onset | 10 sec | 1,000–10,000 req/s | Flash sale opens |
| Peak | 5 min | 2,000–5,000 req/s | Sustained peak load |
| Cooldown | 5 min | 500 → 100 req/s | Gradual return to baseline |
| Total episode | ~13 min | — | One complete flash-sale event |

**Burst multipliers tested:** 10×, 50× (primary), 100× (if infrastructure permits)

**Request types:** 80% product page GET, 15% cart PUT, 5% checkout POST (realistic e-commerce mix)

**Reproducibility:** All traffic generators use fixed random seeds (42 for training, 43 for validation, 44 for test).

### 9.9 Flash-Sale Burst Modelling

Arrival rate λ(t) is modelled as a **superposition of baseline Poisson traffic and a log-normal burst**:

```
λ(t) = λ_baseline + λ_burst · LogNormal(μ=log(peak_rate), σ=0.3) · Gaussian_envelope(t; t_onset, σ_burst)
```

Where:
- `λ_baseline = 100 req/s`
- `λ_burst` = multiplier (10×, 50×, 100× λ_baseline)
- `t_onset` = flash-sale start time (randomised ± 30s for test robustness)
- `σ_burst = 60s` (flash sale lasts approximately 5 minutes at peak)

**Noise injection:** 10% Gaussian noise added to arrival rate to simulate transient spikes within the burst (tests the H-MAS noise-filtering hypothesis from Mohar's document).

### 9.10 Scaling Methodology

**Two-loop control:**
- **Fast loop (PPO agent, 100ms):** Per-request routing decisions, no instance changes
- **Slow loop (CloudWatch + Lambda + ASG, 30s):** Instance count adjustment

**Proactive pre-scaling trigger:** If `burst_indicator = 1` (arrival rate > 2× baseline) AND `time_since_last_spike < 600s` → Lambda pre-scales by +2 instances before CPU hits threshold.

**Reactive scaling:** Standard CloudWatch alarm: CPU > 70% for 2 consecutive 30s periods → Lambda calls `autoscaling:SetDesiredCapacity` (+1 instance). CPU < 30% for 5 consecutive periods → scale in (-1 instance).

**Scale-in protection:** Minimum 2 instances always running. Maximum 8 instances (student account quota).

### 9.11 Request Routing Methodology

1. Request arrives at ALB (HTTP listener on port 80)
2. ALB evaluates listener rules (default: weighted target group)
3. PPO inference endpoint called every 100ms to update weights
4. ALB routes using current weight vector [w₁, w₂, ..., wN] where wᵢ ∝ capacity of instance i
5. Sticky sessions DISABLED (stateless routing for clean measurement)

### 9.12 Monitoring/Feedback Loop

```
CloudWatch (10s metrics) → Lambda state collector → S3 state buffer
                                                        ↓
                                         PPO inference endpoint (100ms)
                                                        ↓
                                         ALB weight update (boto3)
                                                        ↓
                             DynamoDB (log: state, action, reward, timestamp)
                                                        ↓
                                         Python dashboard (real-time viz)
```

---

## 10. DRL Techniques Used and Algorithm Selection

### 10.1 PRIMARY ALGORITHM: Proximal Policy Optimization (PPO)

**Why PPO was selected:**

| Criterion | DQN | PPO | DDPG |
|-----------|-----|-----|------|
| Action space | Discrete ✓ | Both ✓ | Continuous only |
| Sample efficiency | Moderate | Moderate-High | High |
| Training stability | Moderate | High (clip ratio) | Low (sensitive to hyperparams) |
| Policy type | Value-based | Policy gradient | Actor-critic |
| Convergence | Often unstable for large N | Reliable | Unreliable without careful tuning |
| Parallel workers | No | Yes (VecEnv) | No |
| Literature validation for cloud | P5 (Funika), P4 (Femminella) | Both confirm PPO > threshold | P4 (Mohar) uses it but no comparison |

**Mathematical foundation:**

PPO-Clip maximises:
```
L^CLIP(θ) = E_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t)]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` — probability ratio between new and old policy
- `Â_t` — Generalised Advantage Estimate (GAE): `Â_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}`, `δ_t = r_t + γV(s_{t+1}) − V(s_t)`
- `ε = 0.2` — clip ratio preventing large policy updates

The clip constraint solves the **policy collapse** problem that makes vanilla policy gradient methods unstable for bursty environments with high reward variance — a critical property for flash-sale traffic where reward signals swing dramatically between phases.

**Full PPO objective:**
```
L(θ) = L^CLIP(θ) − c₁·L^VF(θ) + c₂·S[π_θ](s_t)
```

Where L^VF is the value function loss (MSE), S is entropy bonus.

### 10.2 PRIMARY BASELINE: DQN (Deep Q-Network)

**Why DQN as baseline (not discarded):**

DQN (Mnih et al., 2015) is the most widely used DRL algorithm for discrete routing problems. Agrima's document explicitly proposes a DQN vs PPO comparison as its core novelty, and Zhou et al. (2024, Elsevier, cited in Agrima's survey) is the only surveyed paper that compares both on the same platform. Retaining DQN as the primary DRL baseline preserves this comparison and enables the key research question: *Does the policy-gradient stability of PPO provide measurable benefit over value-based DQN for flash-sale routing?*

**DQN implementation:** Standard DQN with experience replay (buffer size 100k) and target network (update every 1000 steps). Identical network architecture to PPO (2-layer MLP, [64, 64]).

### 10.3 WHY DDPG WAS NOT SELECTED AS PRIMARY

Mohar's document proposes DDPG for continuous routing. **DDPG is rejected as primary** for the following reasons:

1. **Continuous action space mismatch:** Our ALB routing is inherently discrete (select one of N target groups). DDPG forces a continuous → discrete mapping that adds unnecessary complexity.
2. **Training instability:** DDPG is notoriously sensitive to hyperparameter choices and learning rate schedules. For a 4-student project with limited experiment budget, PPO's reliability is preferable.
3. **Literature evidence:** No paper in our survey demonstrates DDPG outperforming PPO for request-level routing under burst traffic. DDPG's advantage is in continuous control (robotics, continuous resource allocation).
4. **Student account constraints:** DDPG training requires more wall-clock time to converge, increasing SageMaker/EC2 training costs.

**DDPG retained as:** A theoretical discussion point in related work, and optionally as an E9 algorithm comparison experiment if time permits.

### 10.4 WHY H-MAS WAS NOT SELECTED AS PRIMARY

Mohar's document proposes a Hierarchical Multi-Agent System (H-MAS). **H-MAS is rejected as primary** for the following reasons:

1. **Complexity vs benefit ratio:** H-MAS requires designing, training, and coordinating multiple agents with a hierarchy — doubling implementation complexity for uncertain benefit.
2. **Noise filtering can be achieved differently:** The key benefit of H-MAS (filtering transient noise within bursts) can be achieved by the `burst_indicator` feature in our state space (a low-pass filter on arrival rate) and by PPO's natural variance reduction through clipped updates.
3. **No strong empirical evidence:** The MAS-H2 paper (cited in Mohar's survey) focuses on autoscaling conflicts, not continuous routing. The benefit for flash-sale routing specifically is unproven.
4. **Conference paper scope:** A clean PPO vs DQN comparison with a novel reward function is a tighter, more defensible contribution than an unvalidated multi-agent hierarchy.

**H-MAS retained as:** Related work discussion. The `burst_indicator` state feature is a simplified proxy for the noise-filtering role of the H-MAS supervisor layer.

---

## 11. Data/Dataset Strategy

### 11.1 Existing Datasets from PDFs

| Dataset | Source | Cited By | Description | Suitability |
|---------|--------|----------|-------------|-------------|
| Alibaba Cloud Cluster Trace | github.com/alibaba/clusterdata | Mohar | ~5 GB, 1M+ records, 12 features, CPU/memory/latency/request size, CSV/JSON | Useful for pre-training; NOT flash-sale traffic — batch job cluster trace |
| Google Cluster Trace | Google Research | Devkanti (P1) | Large-scale cluster scheduling trace | No burst profile, not e-commerce |
| NASA HTTP logs | NASA Kennedy Space Center | Devkanti (P1, P5) | 1995 HTTP server logs, ~1.8M requests | Too old (1995), not representative |
| ClarkNet HTTP logs | ClarkNet ISP | Devkanti (P5) | 1995 HTTP server logs | Too old, not representative |
| Azure Functions traces | Microsoft Azure | Agrima (P2) | Real serverless traffic traces | Serverless pattern, not e-commerce burst |

**⚠️ CRITICAL FLAG:** NASA HTTP logs and ClarkNet logs are from 1995. Funika et al. (2023) use them as a limitation (cited in Devkanti's survey). These datasets are explicitly NOT suitable for flash-sale validation. Do not use them as primary datasets.

**⚠️ FLAG:** Alibaba Cluster Trace contains batch job scheduling data, not real-time request routing data. It can be used to pre-train the agent on general load patterns but NOT as the primary evaluation dataset.

### 11.2 Primary Dataset: Synthetic Flash-Sale Traffic

**Name:** FlashSale-Traffic-Synthetic v1.0  
**Generator:** `src/backend/traffic_generator.py` (custom Python, MIT license)  
**Basis:** Calibrated against published e-commerce traffic reports (Amazon Prime Day 2023, Alibaba 11.11 2023 traffic pattern papers) — specific citations to be added in paper.

**Schema (12 features):**

| Feature | Type | Description |
|---------|------|-------------|
| `timestamp` | float | Unix timestamp (ms resolution) |
| `request_id` | str | UUID |
| `request_type` | categorical | GET/PUT/POST |
| `payload_size_bytes` | int | Request body size |
| `arrival_rate_per_sec` | float | λ(t) at this timestamp |
| `target_instance` | int | Ground truth (optimal oracle routing) |
| `instance_cpu_util` | float[N] | CPU utilisation at arrival time |
| `instance_active_conn` | float[N] | Active connections at arrival time |
| `response_time_ms` | float | Measured response time (set during replay) |
| `phase` | categorical | warm-up / pre-burst / spike / peak / cooldown |
| `burst_multiplier` | float | Current λ(t)/λ_baseline |
| `seed` | int | Episode random seed |

### 11.3 Preprocessing

1. **Normalisation:** Min-max normalise all continuous features to [0, 1] using training set statistics (prevent data leakage: fit scaler on training split only)
2. **Sliding window aggregation:** Aggregate request-level features into 10s windows for state construction
3. **Reward computation:** Compute R(t) components offline from logged metrics
4. **Missing value handling:** Forward-fill missing CloudWatch metrics (< 0.1% expected)
5. **Outlier clipping:** Clip response times at 99th percentile of training set to reduce reward variance

### 11.4 Train/Validation/Test Split

| Split | Proportion | Description | Seeds |
|-------|-----------|-------------|-------|
| Training | 70% | Agent training episodes | 42–111 (70 episodes) |
| Validation | 15% | Hyperparameter tuning, early stopping | 112–126 (15 episodes) |
| Test | 15% | Final evaluation, reported results | 127–141 (15 episodes) |

**No data leakage:** Scaler is fit on training episodes only. Test episodes are not touched until final evaluation. Validation episodes are used only for checkpoint selection.

### 11.5 Data Leakage Prevention

- Scaler statistics derived from training set only
- Test set seeds not used during any training or hyperparameter tuning
- Baseline algorithms (Round Robin, Least Connections) see the same test episodes as PPO/DQN
- No future traffic information in state vector (no lookahead features)

---

## 12. AWS Architecture

### 12.1 Complete AWS Service Inventory

| # | Service | Resource/Type | Role | Estimated Cost |
|---|---------|--------------|------|----------------|
| 1 | EC2 | t2.micro (up to 8 instances) | Backend servers (Flask apps simulating product catalogue) | Free tier: 750 hrs/mo; beyond: ~$0.0116/hr per instance |
| 2 | ALB (Application Load Balancer) | 1 ALB, 1 HTTPS listener | Route traffic to backend EC2 instances; target group weights updated by PPO | ~$0.008/LCU-hr + $0.0225/hr fixed; not free tier |
| 3 | AWS Lambda | Python 3.12, 512MB, 30s timeout | (a) State collector: polls CloudWatch → builds state vector; (b) Scaling trigger: calls ASG; (c) PPO inference (optional) | First 1M requests/mo free; well within student budget |
| 4 | Amazon CloudWatch | Custom namespace, 10s resolution | Collect CPU, connections, latency; trigger alarms; dashboard | 10 custom metrics free; beyond: $0.30/metric/mo |
| 5 | Amazon S3 | Standard, 1 bucket | Model checkpoints, traffic logs, experiment results | First 5 GB free; beyond: $0.023/GB/mo |
| 6 | SageMaker | ml.t2.medium notebook instance | Offline PPO/DQN training; only on during training | $0.046/hr; stop immediately after training |
| 7 | Auto Scaling Groups | Min=2, Max=8, t2.micro | Horizontal scaling of backend pool | No additional cost beyond EC2 |
| 8 | API Gateway | REST API, no custom domain | Public endpoint for JMeter traffic injection | First 1M calls/mo free |
| 9 | IAM | Roles + Policies | Least-privilege access for Lambda, EC2, SageMaker | Free |
| 10 | DynamoDB | On-demand, 1 table | Routing decision log, experiment metadata | First 25 GB free |
| 11 | Amazon SNS | 1 topic, email | Alarm notifications | First 1M notifications free |
| 12 | CloudFront | Optional | Dashboard CDN | First 1 TB data transfer free |

### 12.2 Exact Role of Each Service

**EC2 (Backend servers):**
- Run a lightweight Flask application that simulates an e-commerce product page
- CPU-intensive endpoint: `GET /product/<id>` performs a 10–50ms computation to simulate DB query
- CloudWatch agent installed: publishes CPU, memory, active connections every 10s to custom namespace `FlashBalanceAI/Instances`

**ALB:**
- Single listener on port 80 (HTTP for student account; HTTPS requires ACM certificate)
- One target group per backend instance OR one weighted target group with configurable weights
- Target group health check: `GET /health` every 10s, unhealthy threshold = 2
- PPO agent modifies `Weights` via `elbv2.modify_target_group_attributes` every 100ms

**Lambda — State Collector (runs every 10s):**
```python
# Pseudo-code
def collect_state():
    metrics = cloudwatch.get_metric_statistics(...)  # for each instance
    state = build_state_vector(metrics)
    s3.put_object(Key='state/current.json', Body=json.dumps(state))
```

**Lambda — PPO Inference (runs every 100ms):**
```python
# Pseudo-code
def inference():
    state = json.loads(s3.get_object(Key='state/current.json'))
    action = policy.predict(np.array(state))  # loaded from S3 at cold start
    elbv2.modify_listener_rule(weights=one_hot(action, N_instances))
```

**Lambda — Scaling Trigger (triggered by CloudWatch alarm):**
```python
def scale_out(event, context):
    current = autoscaling.describe_auto_scaling_groups(...)['desired']
    autoscaling.set_desired_capacity(DesiredCapacity=min(current + 1, MAX_INSTANCES))
```

**CloudWatch:**
- Metrics: `CPUUtilization` (built-in), `ActiveConnectionCount` (ALB), `TargetResponseTime` (ALB), `QueueDepth` (custom from Flask)
- Alarms: CPU > 70% for 2×30s → SNS → Lambda scale-out; CPU < 30% for 5×30s → scale-in
- Dashboard: real-time view of all backend instances + routing decision histogram

**S3:**
- Bucket: `flashbalanceai-{account_id}`
- Folders: `models/`, `traffic_logs/`, `experiment_results/`, `state/`
- Model loading: Lambda reads model at cold start; cached in `/tmp` for warm invocations

**SageMaker:**
- Used ONLY for offline training (Phase 3)
- Notebook instance: ml.t2.medium (2 vCPU, 4 GB RAM — sufficient for PPO MLP training)
- Training script: `src/backend/train_ppo.py` using Stable-Baselines3
- Training duration: ~2–4 hours for 2M steps with N=4 (local training preferred if SageMaker costs exceed $5)
- Stop instance immediately after training: `sagemaker.stop_notebook_instance()`

**ASG (Auto Scaling Groups):**
- Launch template: t2.micro, Amazon Linux 2, user-data installs Flask + CloudWatch agent
- Scaling policies: triggered by Lambda (not standard CloudWatch-native scaling) to allow proactive pre-scaling
- Cooldown: 180s (prevents thrashing)

**API Gateway:**
- REST API: `POST /request` → forwards to ALB
- Stage: `dev` (no custom domain for student account)
- Throttling: 10,000 req/s burst (default)

**IAM:**
- Role: `FlashBalanceAI-Lambda-Role` → policies: `CloudWatchReadOnly`, `EC2ReadOnly`, `elasticloadbalancing:ModifyRule`, `autoscaling:SetDesiredCapacity`, `s3:GetObject/PutObject`, `dynamodb:PutItem/GetItem`
- Role: `FlashBalanceAI-SageMaker-Role` → policies: `SageMakerFullAccess`, `S3ReadWrite` (scoped to `flashbalanceai-*` bucket)
- No root account keys. All access via assumed roles.

**DynamoDB:**
- Table: `routing_decisions` (partition key: `timestamp`, sort key: `request_id`)
- Stores: state vector, action taken, reward computed, episode ID
- Used for offline analysis and reward verification

### 12.3 Data Flow Between Services

```
JMeter → API Gateway → ALB
                         ↓ (every request)
               EC2 Backend (Flask)
                         ↓ (every 10s)
               CloudWatch Custom Metrics
                         ↓ (every 10s)
               Lambda State Collector → S3 (state/current.json)
                         ↓ (every 100ms)
               Lambda PPO Inference → reads S3 state → ALB modify weights
                         ↓ (every step)
               DynamoDB (log action + state + reward)
                         ↓ (alarm triggers)
               CloudWatch Alarm → SNS → Lambda Scaling → ASG
```

### 12.4 Security / IAM

- All Lambda functions use least-privilege IAM roles scoped to specific resources
- No public S3 bucket. All S3 access via pre-signed URLs or Lambda internal access
- ALB not publicly exposed beyond what JMeter needs (security group: allow port 80 from JMeter EC2 only)
- No SSH keys committed to git. EC2 accessed via AWS Systems Manager Session Manager
- DynamoDB table: IAM-controlled, no public access
- CloudWatch logs: retained 30 days (cost control)

### 12.5 Logging / Monitoring

- CloudWatch Logs: Lambda function logs (function errors, state collection, inference latency)
- CloudWatch Metrics: custom namespace `FlashBalanceAI/*` for all application metrics
- CloudWatch Dashboard: `FlashBalanceAI-Monitor` — real-time view
- DynamoDB: full action/state/reward audit trail for post-experiment analysis
- Local Python dashboard (Dash/Plotly): real-time visualisation during experiments

### 12.6 Model Storage / Deployment

- Training output: `models/ppo_flash_v{version}.zip` → S3
- Deployment: Lambda `init()` downloads from S3 to `/tmp/model.zip` at cold start, unzips, loads via `PPO.load()`
- Versioning: S3 versioning enabled on `models/` prefix; every experiment writes a new version
- Rollback: `boto3.client('s3').copy_object()` to restore previous checkpoint

---

## 13. Baselines

| Baseline | Implementation | Justification |
|----------|---------------|---------------|
| Round Robin (RR) | Software-implemented, cyclic allocation | Universal standard; used in 4/10 surveyed papers |
| Weighted Round Robin (WRR) | Weight ∝ instance capacity | Slightly smarter RR; Agrima's comparison |
| Least Connections (LC) | Route to instance with fewest active connections | Common production heuristic |
| Threshold-Based Autoscaling | AWS CloudWatch alarm: CPU > 70% → scale out, CPU < 30% → scale in | Current production standard; Devkanti's primary comparison baseline |
| DQN | Deep Q-Network (Mnih et al., 2015), identical architecture to PPO | Primary DRL baseline; enables Agrima's DQN vs PPO comparison |
| Random | Uniform random instance selection | Lower bound; confirms that learning matters |

**Note:** DDPG and H-MAS are NOT included as implementation baselines due to scope constraints. They are discussed as related work.

---

## 14. Evaluation Methodology

### 14.1 Primary Metrics

| Metric | Definition | Target (PPO) | Measurement |
|--------|-----------|--------------|-------------|
| Average Latency | Mean response time across all requests in experiment window (ms) | < 200ms at 10× burst | CloudWatch `TargetResponseTime` + JMeter |
| P95 Latency | 95th percentile response time (ms) | < 500ms at 10× burst | JMeter aggregate report |
| P99 Latency | 99th percentile response time (ms) | < 1000ms at 10× burst | JMeter aggregate report |
| Throughput | Successfully served requests per second (req/s) | Maximised | JMeter throughput counter |
| Request Failure Rate | HTTP 5xx errors / total requests (%) | < 1% at 10× burst | JMeter error summary |
| SLA Violation Rate | % of requests with latency > SLA threshold (500ms) | < 5% at 10× burst | Computed from JMeter results |
| CPU Utilisation (avg) | Mean CPU across all active instances during peak (%) | 70–85% | CloudWatch `CPUUtilization` |
| CPU Utilisation (max) | Peak CPU across any single instance (%) | < 95% | CloudWatch |
| Load Imbalance Index | Coefficient of variation of CPU across instances | < 0.2 | Computed from CloudWatch |
| Scaling Reaction Time | Time from burst onset to N+1 instances active (seconds) | < 120s (PPO proactive) vs 180s+ (threshold) | CloudWatch EC2 instance start timestamps |
| Cost per 1M Requests | Total AWS cost during experiment / requests served × 10⁶ | 20–25% lower than static baseline | AWS Cost Explorer |
| Total AWS Cost | Absolute cost during experiment window | Minimised | AWS Cost Explorer |

### 14.2 Statistical Analysis Requirements

- **Repetitions:** Each experiment repeated N_rep ≥ 5 times with different seeds
- **Reported values:** Mean ± standard deviation for all metrics
- **Confidence intervals:** 95% CI using t-distribution (small sample)
- **Significance testing:** Paired t-test (PPO vs each baseline) on P95 latency; reject H₀ at α = 0.05
- **Effect size:** Cohen's d reported alongside p-values
- **No cherry-picking:** All N_rep runs reported, not just best run

---

## 15. Previous Approach vs Our Approach

*Based exclusively on evidence from the surveyed papers.*

| Aspect | DRS (Jian et al., 2024, Wiley) | PPO-HPA (Femminella & Reali, 2024, MDPI) | SLA-DRL (Yamsani et al., 2026, Nature) | TS-SDTRA (Zhou et al., 2024, Elsevier) | DistRL (Li et al., 2024, IEEE TPDS) | FlashBalanceAI (This Project) |
|--------|-------------------------------|------------------------------------------|----------------------------------------|----------------------------------------|------------------------------------|-------------------------------|
| Algorithm | DRL (custom) | PPO | Custom DRL + SVRS | PPO, PPO-LSTM, DQN, D3QN | Distributional RL | PPO (+ DQN baseline) |
| Traffic type | Generic microservices | General serverless | General IoT tasks | General edge computing | Batch jobs | Flash-sale burst (10×–100×) |
| Action | Pod scheduling | HPA threshold config | Task assignment | Per-task routing | Job-to-VM assignment | Per-request routing to EC2 |
| Platform | On-premises Kubernetes | OpenFaaS on hardware | Simulation | SDN simulation | Cloud simulation | AWS (EC2, ALB, Lambda) |
| Flash-sale traffic | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| P95/P99 latency metric | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Cost optimisation | ✗ | Partial (over-provisioning penalty) | ✗ | ✗ | ✗ | ✓ (multi-objective reward) |
| DQN vs PPO comparison | ✗ | ✗ | ✗ | ✓ (SDN sim only) | ✗ | ✓ (on AWS, flash-sale) |
| AWS-native deployment | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Proactive pre-scaling | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Reproducibility | Partial | ✓ (code shared) | ✓ (GitHub) | Partial | Partial | ✓ (open-source, fixed seeds) |

---

## 16. Research Gap

Based on the 10 surveyed papers (5 from Devkanti, 5 from Agrima), the following research gaps are **verified as real** (not claimed without evidence):

**Verified Gap G1:** No existing DRL-based load balancing work specifically targets flash-sale burst traffic characterised by 10×–100× request rate escalation within seconds. All surveyed papers use general workloads (batch jobs, generic microservices, serverless, SDN edge).

**Verified Gap G2:** No surveyed paper deploys a DRL load balancer on AWS production services (ALB, ASG, Lambda, CloudWatch). Existing work uses simulations, generic Kubernetes, or on-premises hardware.

**Verified Gap G3:** Only one surveyed paper (Zhou et al., 2024, Elsevier) compares DQN and PPO on the same platform, and this comparison is in an SDN edge simulation — not cloud, not flash-sale traffic.

**Verified Gap G4:** No surveyed paper uses a multi-objective reward function that simultaneously optimises latency, cost, throughput, and SLA violations for e-commerce burst traffic.

**Verified Gap G5:** P95/P99 latency metrics are absent from all surveyed papers except Yamsani et al. (2026) — and that work does not test under flash-sale conditions.

**Unverified / Requires Validation:**
- "First DRL load balancer on AWS" — **do not claim without a comprehensive literature search beyond these 10 papers**.
- Exact percentage improvements (30% P95 reduction) — **these are EXPECTED results, not validated**.

---

## 17. Novelty

### 17.1 Novelty Claimed in the PDFs

| Claim | Source | Defensible? |
|-------|--------|-------------|
| First DRL load balancer specifically for flash-sale burst traffic | Devkanti, Agrima | Partially — within the 10 surveyed papers. Cannot claim globally without broader search. |
| Novel 4-component burst-aware reward function | Devkanti | Defensible — specific combination of latency + cost + throughput + SLA penalty for flash-sale context is novel |
| DQN vs PPO comparison on AWS flash-sale setup | Agrima | Defensible — not done in surveyed literature |
| DDPG + H-MAS integration | Mohar | NOT RETAINED — too speculative |
| AWS-native DRL deployment (ELB, ASG, Lambda, CloudWatch) | Devkanti, Agrima | Defensible within surveyed scope |
| Proactive pre-scaling before burst onset | Devkanti | Defensible as combination — not seen in surveyed papers |
| 30% P95 reduction, 85% utilisation, 20–25% cost saving | Devkanti | EXPECTED RESULTS ONLY — not yet experimentally validated |

### 17.2 Defensible Novelty After Verification

**N1. Flash-Sale Burst-Aware DRL Load Balancing on AWS**
Within the surveyed literature (10 papers, top-tier venues, 2023–2026), no work combines: (a) DRL-based per-request routing, (b) AWS-native deployment (ALB + ASG + Lambda + CloudWatch), and (c) evaluation under flash-sale burst traffic profiles. This combination is the primary novelty claim.

**N2. Four-Component Multi-Objective Burst-Aware Reward Function**
The specific combination of response time penalty, utilisation reward, throughput fraction, and SLA violation penalty — tuned and validated for flash-sale burst conditions — constitutes a novel reward engineering contribution.

**N3. DQN vs PPO Empirical Comparison on AWS Flash-Sale Workloads with P95/P99 Metrics**
This is the only comparison in our surveyed scope that uses both algorithms on real AWS infrastructure with e-commerce burst traffic and tail latency metrics.

**N4. Proactive Pre-Scaling via Burst Indicator Signal**
The `burst_indicator` state feature enables the agent to trigger pre-scaling before CPU saturation — a behaviour not achievable with standard threshold-based CloudWatch alarms.

### 17.3 Claims to NOT Make

- "No existing work addresses DRL for cloud load balancing" — false; the field has extensive literature
- "First to use PPO for cloud autoscaling" — false; Femminella & Reali (2024) precede this work
- "First to use DDPG for continuous resource allocation" — false; well-established
- Any specific numerical results (30%, 85%, 20–25%) until experimentally validated

---

## 18. Expected Contributions

1. **Algorithmic:** Four-component burst-aware reward function for DRL load balancing in flash-sale contexts
2. **Empirical:** First (within surveyed scope) DQN vs PPO comparison on AWS with flash-sale workloads and P95/P99 latency metrics
3. **Systems:** Open-source end-to-end AWS deployment pipeline (boto3-integrated PPO inference, Lambda state collector, CloudWatch feedback loop)
4. **Practical:** Flash-sale traffic generator calibrated to e-commerce burst profiles, with reproducible seeds
5. **Negative results:** Empirical evidence on where threshold-based autoscaling fails specifically, and by how much

---

## 19. Expected Results / Hypotheses

**⚠️ ALL RESULTS BELOW ARE EXPECTED (UNVALIDATED HYPOTHESES). NONE ARE EXPERIMENTALLY CONFIRMED.**

| Hypothesis | Expected Direction | Basis |
|------------|-------------------|-------|
| H1: PPO outperforms Round Robin on P95 latency at 10× burst | PPO ≥ 30% lower P95 | MDP formulation should capture burst dynamics; RR ignores server state |
| H2: PPO outperforms Threshold-based autoscaling on P95 latency | PPO ≥ 30% lower P95 | Proactive pre-scaling avoids cold-start delay |
| H3: PPO achieves higher CPU utilisation than static baselines | 70–85% for PPO vs ~40–50% for static | Multi-objective reward penalises under-utilisation |
| H4: PPO reduces AWS cost vs static over-provisioned baseline | 20–25% cost reduction | Fewer idle instances needed |
| H5: PPO outperforms DQN on P95 latency at 50× burst | PPO marginally better, within 10% | PPO's policy gradient stability should benefit under high reward variance |
| H6: All DRL methods degrade gracefully at 100× burst | All methods exceed SLA; PPO degrades least | Infrastructure ceiling limits all methods |
| H7: Ablation: removing SLA penalty from reward increases P99 latency | +20–40% P99 increase | SLA term is responsible for tail latency control |
| H8: Proactive pre-scaling reduces scaling reaction time | 50–70% reduction in time-to-scale | Burst indicator triggers early ASG call |

---

## 20. Risks, Limitations, and Threats to Validity

### Technical Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Lambda cold-start latency > 100ms PPO inference budget | High | Pre-warm Lambda with scheduled events; or migrate inference to dedicated EC2 |
| ALB weight update latency > routing window | Medium | Use weighted target group (not listener rules) for faster updates |
| PPO fails to converge on synthetic environment | Medium | Monitor reward curve; reduce learning rate; simplify state space |
| CloudWatch metric delay (60s default) makes state stale | Medium | Use custom 10s metric resolution; accept 10s lag |
| Student account EC2 quota insufficient (> 5 instances) | Medium | Request quota increase in advance; design for N=4 maximum |
| SageMaker cost overrun | Low-Medium | Use local training (laptop/Colab) as primary; SageMaker only as validation |

### Methodological Limitations

1. **Simulated environment gap:** PPO is trained in a software simulation, not against live AWS. The sim-to-real gap may reduce performance. Real-world AWS latency (network jitter, EC2 variability) is not captured in training.
2. **Synthetic traffic:** FlashSale-Traffic-Synthetic is not real production data. Patterns may not match actual e-commerce flash sales.
3. **Single-region evaluation:** Results may not generalise to multi-region or CDN-fronted deployments.
4. **Discrete routing oversimplification:** In production, ALB uses weighted routing, not pure one-hot instance selection. Our 100ms time-sliced routing approximates weighted routing.
5. **N=4 scale limit:** Student account limits evaluation to 4–8 instances. Results for hundreds of instances cannot be extrapolated without further validation.
6. **No database layer:** Backend is stateless Flask; no DB under load. Real e-commerce has DB bottlenecks.

### Threats to Validity

| Threat | Type | Mitigation |
|--------|------|------------|
| Overfitting to specific traffic seed | Internal | Use held-out test seeds; report variance |
| Baseline configuration bias | Internal | Use optimal hyperparameters for all baselines |
| AWS billing surprises | External | Set billing alarm at $5; budget ceiling $20 total |
| Confirmation bias in metric selection | Internal | Pre-register metrics before running experiments |
| Software bugs in reward computation | Internal | Unit-test reward function; verify against manual calculation |

---

## 21. Reproducibility Requirements

1. **Code:** All code committed to `main` branch, MIT license, `requirements.txt` with pinned versions
2. **Seeds:** All random seeds documented (`SEED_TRAIN=42`, `SEED_VAL=43`, `SEED_TEST=44`)
3. **Model:** Final model weights stored in S3 and GitHub release (zipped)
4. **Environment:** `environment.yml` (conda) and `requirements.txt` (pip) with exact package versions
5. **Infrastructure:** AWS CDK or Terraform script to recreate AWS infrastructure
6. **Experiments:** Each experiment run has a unique `experiment_id` stored in DynamoDB with full config
7. **Data:** Traffic generator code + seeds allow re-generation of all training/test episodes
8. **Results:** Raw JMeter `.jtl` files and CloudWatch metric exports stored in S3 and GitHub release
9. **Paper:** Methodology section must be self-sufficient to recreate the experiment from scratch

---

## 22. Conference-Paper-Ready Research Questions / Hypotheses

**RQ1 (Primary):** Does a PPO-based DRL load balancer achieve significantly lower P95 latency than threshold-based autoscaling during e-commerce flash-sale burst traffic on AWS?

**RQ2 (Comparison):** Does PPO provide measurable performance advantage over DQN for real-time request routing under flash-sale burst conditions?

**RQ3 (Reward):** Which components of the multi-objective burst-aware reward function contribute most to P95 latency improvement? (Ablation)

**RQ4 (Scaling):** Does the burst-indicator-driven proactive pre-scaling mechanism reduce scaling reaction time compared to threshold-based autoscaling?

**RQ5 (Cost):** What is the trade-off between P95 latency improvement and AWS operational cost across different burst multipliers?

**H1 (Formal):** H₀: μ(P95_PPO) ≥ μ(P95_Threshold). H₁: μ(P95_PPO) < μ(P95_Threshold). Test: paired t-test, α = 0.05, N_rep ≥ 5.

---

## 23. Suggested Paper Structure

1. **Abstract** (250 words): Problem, approach, key results (to be filled after experiments)
2. **Introduction**: Flash-sale traffic challenge; DRL motivation; contributions list (5 bullets)
3. **Related Work**: 8–10 papers in 3 groups: (a) DRL for cloud scheduling, (b) DRL for load balancing, (c) AWS/cloud autoscaling; explicit gap identification
4. **System Design**: Architecture diagram; data flow; AWS service integration
5. **Methodology**: MDP formulation; state/action/reward; PPO algorithm; training procedure
6. **Experimental Setup**: Traffic generator; baselines; metrics; statistical analysis plan; AWS configuration
7. **Results**: Table: all baselines × all metrics (E1–E9); graphs: latency CDF, reward convergence, cost comparison, scaling timeline; ablation table (E10)
8. **Discussion**: Answer RQ1–RQ5; explain unexpected results; H-MAS insight from Mohar's work as future direction
9. **Limitations and Threats to Validity**
10. **Conclusion**: Summary of contributions; future work (DDPG, multi-region, real production data)
11. **References**: 15–20 papers; include all 10 surveyed papers

---

## 24. Required Experiments Before Claiming Publication Results

| Exp | Name | Purpose | Minimum Runs |
|-----|------|---------|--------------|
| E1 | Baseline Traffic (1× load) | Confirm all systems work; calibrate baselines | 5 |
| E2 | 10× Traffic Spike | Primary comparison: all baselines | 5 |
| E3 | 50× Traffic Spike | Stress test; expected degradation in simple baselines | 5 |
| E4 | 100× Traffic Spike | Upper bound; expected infrastructure ceiling | 3 |
| E5 | Repeated Flash Sales | Multiple spikes within one episode; recovery behaviour | 5 |
| E6 | Noisy/Transient Spikes | Short-lived 2× spikes within baseline; test over-reaction | 5 |
| E7 | Cold-Start / Scaling Response | Measure time-to-scale for PPO vs threshold autoscaling | 5 |
| E8 | Cost/Performance Trade-off | Vary w₂ (utilisation weight) and measure cost vs P95 | 3 per config |
| E9 | DRL Algorithm Comparison (PPO vs DQN) | Same environment, both algorithms, identical test episodes | 5 each |
| E10 | Reward Ablation Study | Remove each reward component one at a time | 3 per ablation |

**Minimum total experiment budget:** E1+E2+E3+E4+E5+E6+E7+(E8×3)+E9+E10 ≈ **60–80 runs**

**No result from any of these experiments may be reported in the paper until the experiment has been executed with the specified repetitions and statistical analysis has been completed.**

---

## FINAL RECOMMENDED ARCHITECTURE

### DRL Algorithm
**PPO (Proximal Policy Optimization)** with 2-layer MLP actor-critic [64, 64], trained using Stable-Baselines3. Discrete action space (N=4 backend instances). Clip ratio 0.2, GAE λ=0.95, 2M training steps.

### Baseline DRL
**DQN** with experience replay and target network, identical architecture, for mandatory PPO vs DQN comparison.

### AWS Services (Minimum Required)
EC2 (t2.micro × 4), ALB (1), Lambda (3 functions), CloudWatch (custom metrics), S3 (1 bucket), SageMaker (notebook ml.t2.medium, stopped after training), API Gateway (REST), ASG (1 group), IAM (2 roles), DynamoDB (1 table).

### Dataset / Traffic
Synthetic FlashSale-Traffic-Synthetic v1.0 generated by `traffic_generator.py`. Calibrated to 10× and 50× burst multipliers. No real production data required. Alibaba Cluster Trace used only for supplementary pre-training analysis.

### Reward Function
Four-component: R = 0.4·R_lat + 0.2·R_util + 0.2·R_tput + 0.2·R_sla. Weights subject to ablation study (E10).

### Baselines
Round Robin, Weighted Round Robin, Least Connections, Threshold-based autoscaling (CloudWatch + ASG), DQN.

### Experiments
E1–E10 as specified in Section 24, minimum 5 repetitions each, 95% CI, paired t-test for RQ1/RQ2.

### Why This Is The Right Architecture
- PPO's stability under high-variance flash-sale rewards is theoretically and empirically supported
- AWS-native integration is novel within the surveyed literature
- DQN baseline enables the PPO vs DQN comparison that Agrima's document identifies as a core contribution
- Mohar's H-MAS insight is preserved through the `burst_indicator` state feature (simplified noise filter)
- Cost is minimised by: using t2.micro instances, stopping SageMaker immediately after training, using Lambda for inference (free tier), and limiting experiment duration

**Estimated total AWS cost for full experimental campaign: $15–35 USD** (within student credit budget)
