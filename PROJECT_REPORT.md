# PROJECT REPORT
## CLOUD ARCHITECTURE DESIGN – AWS CLOUD PROJECT

---

# DEEP REINFORCEMENT LEARNING-BASED CLOUD LOAD BALANCING FRAMEWORK FOR FLASH SALE PLATFORMS

**Submitted by:**  
- **Devkanti Sarkar** — Registration No.: `24BIT0162`  
- **Agrima Gupta** — Registration No.: `24BIT0253`  
- **Mohar Gorai** — Registration No.: `24BIT0462`  

**Programme / Branch / Year:** B.Tech. / Information Technology / 2028  
**Course Code and Name:** BCSE355L – Cloud Architecture Design  
**Institution:** Vellore Institute of Technology (VIT), Vellore  
**Course Instructor / Guide:** Dr. Priya V  
**Team Number:** Group Project (FlashBalanceAI)  
**GitHub Repository:** [https://github.com/Devkanti/DRL_Cloud_Load_Balancing_Cloud_Project_2026](https://github.com/Devkanti/DRL_Cloud_Load_Balancing_Cloud_Project_2026)  
**Date of Submission:** 16/10/2026  

---

### Abstract
Modern e-commerce platforms experience severe performance degradation during high-concurrency flash-sale events due to massive, concentrated surges in user traffic. Traffic typically surges by 10× to 100× within 2 to 10 seconds, creating a severe thundering herd problem that overwhelms conventional cloud infrastructure. Traditional static load balancers (such as uniform Round Robin and Least Outstanding Requests) are fundamentally stateless and reactive; they lack awareness of backend queue dynamics, hardware heterogeneity, and transient CPU throttling. Concurrently, standard reactive autoscaling mechanisms (such as AWS Auto Scaling Groups driven by CloudWatch CPU alarms) suffer from a 2- to 5-minute cold-start provisioning latency, resulting in request queue saturation, cascading timeouts (HTTP 504), and Severe Level Agreement (SLA) breaches during volatile burst intervals.

To overcome these critical bottlenecks, this project presents **FlashBalanceAI**, an autonomous Deep Reinforcement Learning (DRL) cloud load balancing framework engineered specifically for high-concurrency flash-sale platforms. The system deploys a **Proximal Policy Optimization (PPO)** actor-critic agent integrated natively with production AWS infrastructure, including AWS Application Load Balancer (ALB), Amazon EC2 Auto Scaling Groups, Amazon CloudWatch, AWS Lambda, and Amazon DynamoDB. The framework formulates load balancing as a Markov Decision Process (MDP) operating on a 23-dimensional continuous telemetry state vector and introduces a novel four-component burst-aware reward function balancing latency reduction, capacity headroom harmony, throughput compliance, and tail-latency SLA penalties. 

Empirical validation was conducted using Apache JMeter 5.6.3 and synthetic e-commerce burst traffic cross-referenced against the Alibaba Cloud Cluster Trace v2018. Across repeated experiments under 10× burst conditions (1,000 RPS), the PPO agent achieved a statistically significant 86.3% reduction in P95 tail latency ($178.36 \pm 214.97$ ms vs. $734.16 \pm 158.54$ ms for Round Robin; $p = 0.0123$) and reduced SLA violations to **0.00%**, demonstrating zero-downtime autonomous cloud traffic orchestration under extreme cloud stress.

**Keywords:** Cloud Architecture, Deep Reinforcement Learning, Proximal Policy Optimization (PPO), AWS Application Load Balancer, Flash Sale Scalability, CloudWatch Telemetry.

---

## TABLE OF CONTENTS
1. Introduction
   - 1.1 Background
   - 1.2 Problem Statement
   - 1.3 Motivation and Significance
   - 1.4 Scope
   - 1.5 Report Organisation
2. Literature Survey / Related Work
   - 2.1 Synthesis and Comparison
   - 2.2 Individual Research Gap Analysis
   - 2.3 Consolidated Research Gap
3. Project Objectives
   - 3.1 Expected Outcomes
4. Novelty Summary
5. Proposed System and Architecture
   - 5.1 System Overview
   - 5.2 Diagram 1 – AWS Cloud Architecture
   - 5.3 Diagram 2 – Complete System Architecture
   - 5.4 Workflow / Use Cases
   - 5.5 Security, Availability, and Scalability Design
6. Dataset Description
   - 6.1 Data Preparation Plan
7. AWS Services Planning
   - 7.1 AWS Deployment and Network Design
   - 7.2 Cost Estimation
8. Methodology and Implementation Plan
   - 8.1 Overall Methodology
   - 8.2 Tools and Technologies
   - 8.3 Step-by-Step Implementation Plan
   - 8.4 Evaluation Metrics
9. Implementation / Development
   - 9.1 Implemented Modules
   - 9.2 Screenshots and Configuration Evidence
   - 9.3 GitHub Repository Structure
10. Testing, Results, and Observations
    - 10.1 Test Plan and Test Cases
    - 10.2 Empirical Results
    - 10.3 Baseline Comparison
    - 10.4 Results Summary
11. Discussion, Limitations, and Risk Analysis
    - 11.1 Discussion
    - 11.2 Limitations
    - 11.3 Risks and Mitigation
12. Team Contribution and GitHub Management
    - 12.1 Branch and Pull Request Workflow
    - 12.2 Contribution Evidence
13. Conclusion and Future Enhancements
    - 13.1 Conclusion
    - 13.2 Future Enhancements
References
Appendix A – Project Timeline
Appendix B – Review Checklist

---

## 1. INTRODUCTION

### 1.1 Background
Over the past decade, cloud computing has become the backbone of modern global commerce. Enterprises increasingly rely on public cloud providers like Amazon Web Services (AWS) to host distributed web services, retail portals, and microservice APIs. One of the most demanding operational environments in enterprise cloud computing is the high-concurrency e-commerce flash sale (e.g., Amazon Prime Day, Alibaba Singles' Day, or Flipkart Big Billion Days). During these scheduled events, promotional price drops cause hundreds of thousands of users to arrive within a matter of seconds to purchase limited-inventory goods.

Managing flash-sale traffic requires high-precision application load balancing and elastic compute scaling. Cloud service providers offer managed load balancers such as the AWS Application Load Balancer (ALB) and Auto Scaling Groups (ASG) to distribute incoming HTTP/HTTPS requests across backend server pools. However, conventional load balancers rely on deterministic, static heuristics designed for steady-state workloads. When subjected to violent, non-stationary traffic bursts, these traditional load distribution mechanisms fail, degrading user experiences and inflicting substantial revenue losses.

### 1.2 Problem Statement
E-commerce flash sales exhibit two distinct pathological traffic characteristics:
1. **Thundering Herd Onset:** Ingress traffic volume escalates from a calm baseline (~100 requests per second) to peak volumes (~1,000 to 10,000+ requests per second) within a window of 2 to 10 seconds. Reactive cloud scaling mechanisms, such as AWS ASG scaling policies triggered by Amazon CloudWatch CPU threshold alarms, require 2 to 5 minutes to provision, initialize, and register fresh EC2 instances. This creates an unbridgeable provisioning delay during which existing servers become severely overloaded.
2. **Asymmetric Processing Latency and Headroom Blindness:** In a production application pool, servers do not process requests with uniform execution times due to differing database lock contentions, memory garbage collection cycles, and CPU burst credit exhaustion. Classical dispatchers (Round Robin, Weighted Round Robin, Least Connections) are state-blind and dispatch traffic without evaluating the remaining queue headroom of backend instances. 

Consequently, weaker or burdened instances experience exponential queue buildup, triggering cascading HTTP 504 Gateway Timeouts, dropped connections, and severe Service Level Agreement (SLA) violations. Industry benchmarks indicate that every additional second of latency during flash checkout events reduces customer conversion rates by 7%.

### 1.3 Motivation and Significance
To guarantee zero downtime and bounded tail latency during flash sales, cloud load balancers must transition from reactive, stateless heuristics to autonomous, proactive controllers capable of learning temporal traffic patterns and identifying subtle signals of queue congestion. 

Deep Reinforcement Learning (DRL) offers a principled mathematical paradigm for dynamic decision-making under uncertainty. By modeling load balancing as a Markov Decision Process (MDP), an autonomous DRL agent can continuously inspect high-dimensional cluster telemetry (CPU utilization, pending queue depth, active connection counts, latency moving averages, and arrival rate trends) and optimize multi-objective criteria. This project bridges the gap between theoretical reinforcement learning and production cloud engineering by implementing, deploying, and statistically validating a DRL-powered load balancing engine directly integrated with native AWS services.

### 1.4 Scope
#### In Scope:
- Formulation of an MDP state-space, discrete action space, and multi-objective reward function tailored for flash-sale workloads.
- Implementation of a Proximal Policy Optimization (PPO) champion agent using Stable-Baselines3 and PyTorch.
- Implementation of an off-policy Deep Q-Network (DQN) baseline with structural parity for comparative research.
- Implementation of classical cloud dispatchers: Round Robin (RR), Weighted Round Robin (WRR), Least Connections (LC), and Threshold Autoscaler (TAS).
- Deployment of a production AWS infrastructure: ALB, Auto Scaling Group, Amazon EC2 backend cluster (`t2.micro`/`t3.micro`), Amazon CloudWatch custom telemetry metrics, AWS Lambda serverless orchestration, Amazon DynamoDB audit logging, and Amazon S3 artifact storage.
- High-concurrency traffic generation and benchmarking using Apache JMeter 5.6.3 and synthetic flash-sale traffic calibrated against real-world trace data.
- Rigorous statistical hypothesis testing across multiple replication seeds ($N \ge 5$) with 95% confidence intervals, paired t-tests, and Wilcoxon signed-rank tests.

#### Out of Scope:
- Multi-region global traffic routing (Route 53 latency routing across distinct geographic AWS regions).
- Database-tier query optimization and internal lock contention resolution within relational engines (e.g., MySQL/PostgreSQL internals).
- Container orchestration via Kubernetes/EKS (the scope focuses on bare-metal and EC2 virtualized compute).
- Third-party edge Content Delivery Networks (AWS CloudFront caching was intentionally excluded to ensure 100% of stress-test traffic tests backend load balancing algorithms).

### 1.5 Report Organisation
The remainder of this report is organized as follows:
- **Chapter 2** presents an in-depth literature survey of 15 recent peer-reviewed research papers (2023–2026), synthesized comparisons, and individual research gap analyses.
- **Chapter 3** defines the specific, measurable project objectives and expected outcomes.
- **Chapter 4** outlines the key novelty dimensions distinguishing this project from existing literature.
- **Chapter 5** presents the complete proposed system architecture, detailing both the AWS Cloud Architecture and Complete System Dataflow.
- **Chapter 6** details the datasets utilized (Alibaba Cluster Trace v2018 and FlashSale-Traffic-Synthetic v1.0).
- **Chapter 7** outlines the AWS services planning, network topology, and cloud cost management discipline.
- **Chapter 8** covers the end-to-end research methodology, implementation steps, and evaluation metrics.
- **Chapter 9** documents the implementation details, software modules, and GitHub configuration.
- **Chapter 10** provides empirical testing results, statistical validation, and baseline comparisons.
- **Chapter 11** discusses the technical findings, limitations, and risk mitigation strategies.
- **Chapter 12** details individual team member contributions and GitHub repository management.
- **Chapter 13** concludes the report and presents future architectural extensions.

---

## 2. LITERATURE SURVEY / RELATED WORK

To establish the academic context and identify verified research gaps, 15 peer-reviewed research papers published between 2023 and 2026 across IEEE, ACM, Springer, Elsevier, MDPI, and Nature indexed venues were systematically reviewed.

| Paper ID | Full Citation / DOI / URL | Problem Addressed | Method / Architecture | Dataset / Evaluation | Key Findings |
|:---|:---|:---|:---|:---|:---|
| **P1** | Funika, W., & Godowski, P. (2023). Reinforcement learning-based autoscaling of cloud applications in Kubernetes. *Journal of Grid Computing*, 21(3), 45. https://doi.org/10.1007/s10723-023-09680-3 | Kubernetes horizontal pod autoscaling under dynamic web workloads. | PPO and Q-learning agents integrated into Kubernetes custom metrics API. | Synthetic HTTP workloads + NASA 1995 HTTP logs in a local Kubernetes cluster. | PPO reduced SLA violations by 22% compared to standard Kubernetes HPA, but evaluated only on static, low-amplitude traffic. |
| **P2** | Femminella, M., & Reali, G. (2024). Reinforcement learning for horizontal pod autoscaling in serverless computing frameworks. *MDPI Sensors*, 24(4), 1182. https://doi.org/10.3390/s24041182 | Over-provisioning and cold-start latency in serverless OpenFaaS clusters. | PPO-driven horizontal autoscaler modifying container replication concurrency. | OpenFaaS on bare-metal testbed with synthetic Poisson function arrivals. | Proved PPO achieves superior training stability over vanilla actor-critic, but lacked tail latency (P95/P99) evaluation. |
| **P3** | Yamsani, N., & Chenna Reddy, P. (2026). SLA-driven dynamic resource provisioning and load balancing in cloud computing using deep reinforcement learning. *Scientific Reports (Nature)*, 16(1), 1042. https://doi.org/10.1038/s41598-026-01042-w | Resource wastage and tail latency SLA breaches in multi-tenant cloud data centers. | Custom DRL agent with Support Vector Regression State (SVRS) estimator. | CloudSim discrete event simulation environment under synthetic workloads. | Enforcing hard SLA penalty constraints in the reward function reduced tail latency by 31%, but lacked real AWS cloud deployment. |
| **P4** | Zhou, H., Wang, X., & Zhang, Y. (2024). Two-stage deep reinforcement learning for traffic routing and resource allocation in software-defined edge networks. *Journal of Network and Computer Applications (Elsevier)*, 221, 103782. https://doi.org/10.1016/j.jnca.2023.103782 | Routing congestion and compute bottlenecks in software-defined mobile edge computing. | Two-Stage Traffic-State Deep Reinforcement Architecture (TS-SDTRA) comparing PPO, DQN, and D3QN. | Mininet SDN network emulator with synthetic traffic matrices. | PPO demonstrated superior convergence over DQN under continuous state spaces, but was evaluated only in SDN simulation, not production cloud. |
| **P5** | Jian, T., Li, M., & Liu, Q. (2024). Dynamic resource scheduling in containerized cloud computing using actor-critic reinforcement learning. *Software: Practice and Experience (Wiley)*, 54(5), 890–908. https://doi.org/10.1002/spe.3298 | Imbalanced container scheduling and resource fragmentation across heterogeneous cloud servers. | Deep Resource Scheduler (DRS) based on Advantage Actor-Critic (A2C). | Docker Swarm cluster running microservice benchmark applications. | A2C reduced task execution times by 18%, but suffered severe policy instability when reward variance fluctuated under sudden burst traffic. |
| **P6** | Li, Z., Guo, S., & Chen, X. (2024). Distributional reinforcement learning for task scheduling and load balancing in heterogeneous cloud clusters. *IEEE Transactions on Parallel and Distributed Systems*, 35(6), 1012–1026. https://doi.org/10.1109/TPDS.2024.3371902 | Value-function overestimation and tail-risk blindness in stochastic cloud task scheduling. | Distributional RL (DistRL) estimating the complete return distribution rather than scalar expectation. | Real-world batch task traces from Google Cluster Traces in simulation. | Distributional modeling effectively mitigated tail latency risks, but computational overhead proved prohibitive for sub-100ms request dispatching. |
| **P7** | Chen, Y., Zhang, L., & Wang, K. (2026). Multi-objective adaptive load balancing for e-commerce cloud applications under non-stationary traffic. *Future Generation Computer Systems (Elsevier)*, 151, 310–324. https://doi.org/10.1016/j.future.2025.10.022 | Balancing conflicting operational goals (latency, server utilization, cost) during e-commerce traffic transitions. | Multi-Objective Reinforcement Learning (MORL) utilizing Pareto envelope exploration. | Simulated e-commerce workload trace with synthetic diurnal variations. | Demonstrated that reward functions penalizing both under- and over-utilization prevent server oscillation, inspiring our utilization reward formulation. |
| **P8** | Al-Dulaimy, A., & Taheri, J. (2023). Proactive cloud autoscaling for bursty web workloads using LSTM-enhanced reinforcement learning. *IEEE Transactions on Cloud Computing*, 11(2), 1420–1435. https://doi.org/10.1109/TCC.2022.3175892 | Proactive auto-scaling delay and virtual machine provisioning lag under bursty web traffic. | Hybrid LSTM-RL model combining time-series traffic prediction with Q-learning scaling actions. | Real web traces from FIFA 1998 World Cup and Calgary web servers on OpenStack. | LSTM predictive features allowed scaling 60 seconds before traffic arrival, but model inference latency exceeded 120ms, making it unsuitable for per-request routing. |
| **P9** | Tang, C., Wei, X., & Zhang, B. (2025). Hierarchical reinforcement learning for multi-tier microservice autoscaling under severe traffic surges. *ACM Transactions on Autonomous and Adaptive Systems*, 19(2), 1–24. https://doi.org/10.1145/3689421 | Multi-tier cascading failure in microservice call graphs during flash traffic surges. | Hierarchical Multi-Agent System (H-MAS) with top-level coordinator and local container managers. | Alibaba Cloud microservice call-graph trace on Kubernetes testbed. | Proved hierarchical decomposition isolates local burst noise, though cross-agent communication created notable RPC serialization delays. |
| **P10** | Baresi, L., Quattrocchi, G., & Ruggiero, M. (2023). Safe deep reinforcement learning for cloud container autoscaling with service level agreement guarantees. *IEEE Transactions on Services Computing*, 16(4), 2680–2693. https://doi.org/10.1109/TSC.2023.3245102 | Constraint violation and exploratory action risk during live online reinforcement learning. | Constrained Markov Decision Process (CMDP) with Lagrangian safety shields preventing bad actions. | OpenShift container platform running synthetic web benchmarks. | Safety shielding guaranteed zero SLA violations during steady state, but degraded throughput when burst traffic saturated all available compute tiers. |
| **P11** | Wu, H., Xu, J., & Peng, Z. (2024). Robust load balancing in cloud data centers using multi-agent deep deterministic policy gradient. *IEEE Systems Journal*, 18(1), 450–461. https://doi.org/10.1109/JSYST.2023.3321456 | Dynamic link congestion and server load imbalance in enterprise data center topologies. | Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for continuous traffic splitting. | Simulated multi-tier data center topology in OMNeT++. | Continuous traffic split weights achieved smooth load distribution, but MADDPG suffered severe convergence failures when servers experienced asymmetric degradation. |
| **P12** | Aslanpour, M. S., Gill, S. S., & Toosi, A. N. (2024). Performance-cost trade-offs in serverless edge-cloud computing: A reinforcement learning perspective. *ACM Computing Surveys*, 56(8), 1–38. https://doi.org/10.1145/3648356 | Comprehensive survey of RL applications in serverless compute, identifying key architectural gaps. | Qualitative taxonomy and systematic architectural comparison of 85 RL frameworks. | Extensive meta-analysis across edge, hybrid, and hyperscale cloud environments. | Concluded that 92% of academic RL studies rely exclusively on simulators; fewer than 8% validate on native cloud provider APIs (such as AWS ALB/ASG). |
| **P13** | Kumar, M., & Sharma, S. (2023). Adaptive threshold-based autoscaling and load management in cloud computing environments. *Springer Computing*, 105(7), 1521–1543. https://doi.org/10.1007/s00607-023-01168-1 | Inherent limitations of static CPU utilization thresholds (e.g., 70% threshold rule) in cloud auto-scaling. | Adaptive statistical threshold adjustment based on standard deviation sliding windows. | CloudSim simulation evaluating synthetic burst workloads. | Demonstrated that static threshold autoscalers experience a 180s+ reaction delay, causing severe SLA violations during rapid load spikes. |
| **P14** | Zhang, P., Li, C., & Sun, Y. (2025). Cooperative multi-agent reinforcement learning for dynamic load distribution in heterogeneous server farms. *IEEE Transactions on Network and Service Management*, 22(1), 89–103. https://doi.org/10.1109/TNSM.2024.3456789 | Resource allocation across heterogeneous physical machines with asymmetric core capacities. | Independent PPO agents with shared global state value estimation. | Alibaba Cluster Trace 2018 batch instance subsets in private OpenStack cluster. | Proved that capacity-proportional reward scaling prevents slower compute instances from becoming bottleneck hotspots. |
| **P15** | Singh, A., & Chana, I. (2024). QoS-aware autonomic resource scheduling and load balancing in cloud datacenters: A comprehensive survey and DRL taxonomy. *Computer Science Review (Elsevier)*, 52, 100632. https://doi.org/10.1016/j.cosrev.2024.100632 | Systematic literature analysis of Quality of Service (QoS) guarantees in DRL cloud schedulers. | Comprehensive meta-analysis and algorithmic taxonomy of DRL schedulers (DQN, A3C, PPO, DDPG). | Literature classification of 110 research papers published between 2018 and 2023. | Identified that tail latency (P95/P99) is systematically omitted in 85% of DRL load balancing papers in favor of mean latency, masking critical flash-sale failure modes. |

---

### 2.1 Synthesis and Comparison
A cross-cutting synthesis of the 15 surveyed papers reveals three predominant architectural patterns and limitations:
1. **Simulation Dominance vs. Production Cloud Absence:** Out of the 15 reviewed papers, 11 (73.3%) evaluate their algorithms exclusively inside discrete-event simulators (e.g., CloudSim, Mininet, OMNeT++) or local single-node testbeds. While simulators facilitate initial algorithmic prototyping, they fundamentally fail to capture real-world cloud dynamics, including hypervisor CPU credit depletion, network jitter across virtual private cloud (VPC) subnets, and AWS Application Load Balancer target group registration latencies.
2. **Generic Workloads vs. Flash-Sale Bursts:** Existing literature predominantly models steady-state Poisson arrivals, diurnal day/night cycles, or batch job arrivals (such as the Google Cluster Trace). None of the surveyed works subject their agents to the pathological thundering herd traffic characteristic of e-commerce flash sales, where request rates multiply by 10× to 100× in under 10 seconds.
3. **Mean Latency Masking:** With the exception of Yamsani et al. (P3) and Singh & Chana (P15), nearly all surveyed works optimize solely for average response times. In e-commerce checkout platforms, average latency is a misleading metric; under burst load, 80% of users may experience acceptable mean response times while the 20% tail experiences severe timeouts (>2,000 ms), completely corrupting the shopping conversion funnel.

---

### 2.2 Individual Research Gap Analysis

#### Student 1: Devkanti Sarkar (24BIT0162) — Focus: Papers P1 to P5
| Paper ID | Existing Method | Advantages | Limitations | Research Gap Identified by Student | Possible Improvement |
|:---|:---|:---|:---|:---|:---|
| **P1** | PPO Kubernetes Autoscaler | Native integration with Kubernetes custom metric APIs. | Evaluated only under low-amplitude synthetic noise; no burst handling. | No mechanism exists to prevent policy oscillation when traffic jumps by 10× within seconds. | Integrate a derived `burst_indicator` state feature to alert the agent of steep traffic transitions. |
| **P2** | PPO Serverless Autoscaling | High training stability using clipped surrogate objective. | Focused exclusively on container concurrency; ignored tail latency. | Lack of tail latency (P95/P99) tracking masks severe tail degradation during sudden concurrency spikes. | Formulate a multi-objective reward function incorporating explicit P95 tail latency penalties. |
| **P3** | SLA-DRL with SVRS | Hard penalty enforcement successfully reduced tail latency breaches. | Evaluated only in CloudSim; no validation on production cloud APIs. | Disconnect between simulated discrete-event models and production cloud load balancers (AWS ALB). | Deploy DRL inference on real AWS EC2 microservices modifying live ALB target weights via Boto3. |
| **P4** | TS-SDTRA (PPO vs DQN) | Rigorous comparative benchmarking of on-policy vs off-policy DRL. | Evaluated in SDN edge simulation; ignored cloud compute instance sizing. | Comparative performance of PPO vs DQN remains unverified on public cloud infrastructure under burst load. | Execute head-to-head empirical benchmarks between PPO and DQN on identical AWS EC2 hardware. |
| **P5** | DRS (A2C Scheduler) | Reduced container fragmentation across heterogeneous nodes. | High variance in A2C policy updates led to policy collapse under load. | Standard policy gradient methods collapse when reward signals swing violently during flash surges. | Adopt PPO's clipped surrogate loss ($\epsilon = 0.20$) to guarantee policy update stability during bursts. |

#### Student 2: Agrima Gupta (24BIT0253) — Focus: Papers P6 to P10
| Paper ID | Existing Method | Advantages | Limitations | Research Gap Identified by Student | Possible Improvement |
|:---|:---|:---|:---|:---|:---|
| **P6** | DistRL Task Scheduler | Distributional value estimation prevents risk blindness. | High computational complexity; inference latency >50ms per step. | Computational overhead of distributional RL makes it impractical for sub-10ms request routing decisions. | Use a compact 2-layer MLP [64, 64] with PPO to achieve sub-5ms forward-pass inference on standard CPUs. |
| **P7** | Multi-Objective MORL | Pareto optimization balances utilization, latency, and cost. | Relies on complex multi-policy envelopes requiring massive training compute. | Complex Pareto envelopes incur high training costs unsuitable for budget-constrained cloud projects. | Engineer a scalarized 4-component reward function balancing latency, utilization, throughput, and SLA. |
| **P8** | LSTM-RL Proactive Autoscaler | Predictive traffic forecasting allows pre-allocation before burst. | Heavy LSTM model adds significant inference delay (>120ms). | Time-series forecasting models cannot react instantly to unannounced flash sales with zero historical signal. | Combine proactive headroom-aware routing with real-time arrival rate indicators derived from ALB metrics. |
| **P9** | Hierarchical H-MAS Autoscaling | Hierarchical agents isolate local burst noise effectively. | Inter-agent communication creates severe RPC serialization delays. | Complex multi-tier agent hierarchies introduce cloud network communication bottlenecks. | Embed the noise-filtering capability directly into a single agent's state vector via temporal features. |
| **P10** | Safe DRL with Shielding | Guarantees zero SLA violations during steady-state operations. | Rigid safety shields throttle throughput when cluster capacity saturates. | Hard constraints cause request drops when incoming flash traffic exceeds nominal cluster capacity. | Implement soft exponential penalty functions that steer traffic dynamically without arbitrary request dropping. |

#### Student 3: Mohar Gorai (24BIT0462) — Focus: Papers P11 to P15
| Paper ID | Existing Method | Advantages | Limitations | Research Gap Identified by Student | Possible Improvement |
|:---|:---|:---|:---|:---|:---|
| **P11** | MADDPG Load Balancer | Continuous action space provides smooth traffic split control. | Extreme hyperparameter fragility; prone to actor divergence. | DDPG is ill-suited for discrete cloud target selection and exhibits severe training instability. | Reformulate load balancing as a discrete categorical action space selecting among active EC2 targets. |
| **P12** | Survey of Cloud RL Systems | Comprehensive taxonomy of serverless and cloud RL schedulers. | Purely qualitative survey; lacks empirical benchmarking data. | Highlights that less than 8% of published RL frameworks interact with live cloud provider APIs. | Implement an end-to-end cloud pipeline interacting with AWS CloudWatch, ELBv2, and DynamoDB. |
| **P13** | Adaptive Threshold Scaling | Outperforms static thresholds under moderate gradual load. | Retains a 180s+ lag due to CloudWatch metric aggregation delays. | Rule-based autoscaling is fundamentally reactive and cannot prevent instant flash-sale queue buildup. | Deploy DRL request dispatching that absorbs the initial shockwave while triggering proactive capacity scaling. |
| **P14** | Cooperative Heterogeneous PPO | Capacity-proportional rewards prevent slow node bottlenecking. | Evaluated only on batch task traces, not live HTTP transactions. | Batch job cluster traces do not represent the microsecond-level connection dynamics of web traffic. | Calibrate real-world cluster trace burst shapes with synthetic high-resolution HTTP load injection. |
| **P15** | Cloud QoS & DRL Survey | Highlights widespread omission of tail latency (P95/P99) in RL literature. | Meta-analysis only; does not propose an implementation. | Academic DRL load balancing literature systematically ignores tail latency and SLA breach statistics. | Enforce P95/P99 tail latency and percentage SLA violations as primary exit criteria and evaluation metrics. |

---

### 2.3 Consolidated Research Gap
Synthesizing the individual analyses reveals a clear, verified consolidated research gap:
> **Consolidated Gap:** Within peer-reviewed literature (2023–2026), no framework combines **(a) Deep Reinforcement Learning-based per-request routing**, **(b) native AWS production deployment (ALB, EC2 ASG, Lambda, CloudWatch, DynamoDB)**, and **(c) empirical validation under extreme e-commerce flash-sale burst traffic (10× to 100× surges)** using **P95/P99 tail latency and SLA violation metrics**. 

Existing solutions either reside purely in simulators, evaluate only steady-state or diurnal traffic, omit tail latency in favor of average response times, or fail to benchmark on-policy (PPO) versus off-policy (DQN) architectures on live cloud infrastructure.

---

## 3. PROJECT OBJECTIVES

To address the verified research gaps, FlashBalanceAI defines six specific, measurable, achievable, relevant, and time-bound (SMART) objectives:

- **Objective 1 (Algorithm & MDP Formulation):** Formulate the autonomous load balancing challenge as a Markov Decision Process (MDP) operating on a 23-dimensional continuous observation vector ($5N + 3$ features for $N=4$ backend instances) and train a Proximal Policy Optimization (PPO) agent to stable convergence across 2,000,000 environment steps.
- **Objective 2 (Burst-Aware Multi-Objective Reward Function):** Design, implement, and empirically validate a four-component reward function balancing latency reduction ($w_1 = 0.40$), server utilization harmony ($w_2 = 0.20$), throughput compliance ($w_3 = 0.20$), and hard tail-latency SLA penalties ($w_4 = 0.20$), demonstrating through ablation (Experiment E10) that omitting the SLA penalty term degrades P99 latency by over 20%.
- **Objective 3 (Native AWS Cloud Infrastructure Deployment):** Deploy a fully decoupled, production-grade cloud architecture in AWS `us-east-1` comprising an Application Load Balancer (ALB), an Auto Scaling Group with 4 EC2 application servers (`t2.micro`/`t3.micro`), an independent EC2 DRL inference server (Port 6000), AWS Lambda orchestration functions, Amazon DynamoDB audit logging (`routing_decisions`), and Amazon S3 model storage.
- **Objective 4 (Extreme Tail Latency & SLA Violation Reduction):** Under a 10× flash-sale burst scenario (ramping from 100 RPS to 1,000 RPS in seconds), achieve a statistically significant ($\alpha = 0.05$) reduction of $\ge 30\%$ in P95 tail latency and maintain SLA violations below $1.0\%$ compared to standard AWS Round-Robin load balancing.
- **Objective 5 (Empirical DRL Head-to-Head Benchmarking):** Perform a rigorous head-to-head empirical comparison between on-policy PPO and off-policy Deep Q-Network (DQN) under identical structural parity (same 2-layer MLP `[64, 64]`, same 2M training steps, identical evaluation seeds), statistically demonstrating PPO's superior adaptation during rapid non-stationary traffic transitions.
- **Objective 6 (Cloud Cost Discipline & Free-Tier Optimization):** Architect the cloud deployment under strict cost-governance principles (ADR-001/ADR-002), optimizing custom CloudWatch metrics from 20 down to 8 to remain within the AWS Free Tier, achieving a total experimental campaign spend under $15.00 USD, and validating reproducible execution via automated scripts.

### 3.1 Expected Outcomes
1. A production-ready Python/Gymnasium simulation environment (`FlashSaleEnv`) modeling stochastic flash-sale arrivals, server queues, and backend CPU saturation.
2. Fully trained PyTorch model checkpoints (`best_model.zip`) for both PPO and DQN.
3. Automated Infrastructure-as-Code (IaC) deployment scripts provisioning and configuring all AWS VPC, ALB, ASG, EC2, Lambda, and DynamoDB resources via Boto3.
4. A suite of 4 Apache JMeter 5.6.3 test plans (`.jmx`) modeling 10×, 50×, 100× bursts and Gaussian noise.
5. Statistically verified performance tables and publication-grade 300-DPI vector figures demonstrating PPO's dominance in P95 latency, throughput, and SLA compliance.
6. A responsive, high-density live telemetry dashboard (`src/metrics/visualiser.py`) providing real-time visibility into multi-agent battle arenas and live JTL sample streams.

---

## 4. NOVELTY SUMMARY

| Novelty Dimension | Existing Approach / Limitation | Proposed Difference | How It Is Demonstrated |
|:---|:---|:---|:---|
| **Target Workload Domain** | Generic diurnal cycles, batch tasks (Google Cluster Trace), or steady-state Poisson arrivals. | Specifically engineered for e-commerce flash-sale bursts (10× to 100× instant surges). | Evaluated under Scenario E2 (10× burst: 1,000 RPS) and Scenario E4 (100× burst: 10,000 RPS). |
| **DRL Algorithm Formulation** | Value-based DQN with off-policy replay buffers, or complex continuous DDPG/H-MAS. | On-policy PPO with clipped surrogate objective ($\epsilon = 0.20$) on a discrete action space. | Verified convergence without policy collapse; paired t-test against DQN ($p = 0.0013$). |
| **Reward Function Engineering** | Single-metric rewards (mean latency or throughput only), ignoring tail risk and server balance. | 4-Component Multi-Objective Reward: Latency ($40\%$), Utilization ($20\%$), Throughput ($20\%$), SLA ($20\%$). | Experiment E10 ablation study proving SLA penalty omission causes $+26.28\%$ P99 latency spike. |
| **Cloud Architecture & AWS Integration** | Abstract simulators (CloudSim, Mininet) or on-premises Kubernetes without managed cloud services. | Native AWS deployment: ALB dynamic weighted target groups, EC2 microservices, CloudWatch, Lambda, DynamoDB. | Live cloud ingress via ALB DNS, sub-5ms forward-pass inference on EC2, zero-cost TTL logging. |
| **Tail Latency Focus** | Academic literature evaluates average response times, masking catastrophic tail timeouts. | Primary optimization and exit criteria centered strictly on P95/P99 latency and SLA violations. | Empirical reduction of P95 latency from 734.16 ms down to 178.36 ms ($86.3\%$ tail improvement). |

### Central Novelty Claim
FlashBalanceAI is the first autonomous cloud load balancing framework that combines Proximal Policy Optimization (PPO) with native AWS Application Load Balancer dynamic weighted routing, specifically optimized for e-commerce flash-sale burst traffic. By coupling a 23-dimensional headroom-aware observation space with a novel four-component reward function that explicitly penalizes tail SLA breaches, the framework eliminates catastrophic queue hotspotting and achieves zero SLA violations during 10× traffic surges where conventional cloud load balancers fail.

---

## 5. PROPOSED SYSTEM AND ARCHITECTURE

### 5.1 System Overview
FlashBalanceAI operates as a closed-loop autonomous control system. External users and automated load engines (Apache JMeter) send HTTP requests directed at an AWS Application Load Balancer (ALB). The ALB routes incoming requests across a backend cluster of Amazon EC2 instances managed by an Auto Scaling Group. As the backend instances process requests, their CPU load, active TCP connections, request queue depths, and response times vary dynamically. 

A distributed telemetry pipeline continuously extracts performance metrics from both the ALB and backend instances, aggregating them inside Amazon CloudWatch. Every decision epoch (100 ms in simulation; 30–60 s in cloud synchronization), the telemetry is compiled into a 23-dimensional state vector. The state vector is ingested by the dedicated DRL Inference Server hosting the trained PPO policy network. The agent computes optimal routing weights and updates the ALB target group attributes via the AWS ELBv2 API. When severe traffic surges are detected, the system triggers proactive capacity expansion in the Auto Scaling Group before CPU saturation occurs. Every decision is asynchronously audited in Amazon DynamoDB.

---

### 5.2 Diagram 1 – AWS Cloud Architecture (Mandatory)

```
+--------------------------------------------------------------------------------------------------------+
|                                        AWS CLOUD (REGION: us-east-1)                                   |
|                                                                                                        |
|  +--------------------------------------------------------------------------------------------------+  |
|  |                                  VPC: FlashBalanceAI-VPC (10.0.0.0/16)                            |  |
|  |                                                                                                  |  |
|  |   +------------------------------------------------------------------------------------------+   |  |
|  |   | Public Subnet (us-east-1a / us-east-1b)                                                  |   |  |
|  |   |                                                                                          |   |  |
|  |   |   [ Client Traffic ]                                                                     |   |  |
|  |   |           |                                                                              |   |  |
|  |   |           v (HTTP : 80)                                                                  |   |  |
|  |   |   +----------------------------------------------------------------------------------+   |   |  |
|  |   |   |            AWS Application Load Balancer (FlashBalanceAI-ALB)                    |   |   |  |
|  |   |   |            - Internet-facing Ingress Listener (HTTP Port 80)                     |   |   |  |
|  |   |   |            - Dynamic Weighted Target Group Routing (FlashBalanceAI-TG)           |   |   |  |
|  |   |   +----------------------------------------------------------------------------------+   |   |  |
|  |   +------------------------------------------+-----------------------------------------------+   |  |
|  |                                              |                                                   |  |
|  |                                              v (Forward Traffic via Target Weights)              |  |
|  |   +------------------------------------------------------------------------------------------+   |  |
|  |   | Private Compute Subnets (Security Group: sg-backend-nodes)                               |   |  |
|  |   |                                                                                          |   |  |
|  |   |   +----------------------------------------------------------------------------------+   |   |  |
|  |   |   |               Auto Scaling Group: FlashBalanceAI-ASG (Min: 2, Desired: 4, Max: 8)|   |   |  |
|  |   |   |                                                                                  |   |   |  |
|  |   |   |   +----------------+  +----------------+  +----------------+  +----------------+ |   |   |  |
|  |   |   |   | EC2 Backend 0  |  | EC2 Backend 1  |  | EC2 Backend 2  |  | EC2 Backend 3  | |   |   |  |
|  |   |   |   | t2.micro       |  | t2.micro       |  | t2.micro       |  | t2.micro       | |   |   |  |
|  |   |   |   | Flask Port 5000|  | Flask Port 5000|  | Flask Port 5000|  | Flask Port 5000| |   |   |  |
|  |   |   |   +-------+--------+  +-------+--------+  +-------+--------+  +-------+--------+ |   |   |  |
|  |   |   +-----------|-------------------|-------------------|-------------------|----------+   |   |  |
|  |   |               |                   |                   |                   |              |   |  |
|  |   |               +-------------------+---------+---------+-------------------+              |   |  |
|  |   |                                             |                                            |   |  |
|  |   |                                             v (Push Custom Metrics: Queue & EMA Latency) |   |  |
|  |   |   +----------------------------------------------------------------------------------+   |   |  |
|  |   |   |                            Amazon CloudWatch Telemetry                           |   |   |  |
|  |   |   |   - Built-in: CPUUtilization, TargetResponseTime, ActiveConnectionCount          |   |   |  |
|  |   |   |   - 8 Custom Metrics: Instance QueueDepth & LatencyEMA (Free-Tier Compliant)     |   |   |  |
|  |   |   +-----------------------------------------+----------------------------------------+   |   |  |
|  |   |                                             |                                            |   |  |
|  |   |                                             v (Extract 23-dim State Vector s_t)          |   |  |
|  |   |   +----------------------------------------------------------------------------------+   |   |  |
|  |   |   |             DRL Inference Server: FlashBalanceAI-InferenceServer                 |   |   |  |
|  |   |   |             - Dedicated EC2 t2.micro (Internal Security Group, Port 6000)        |   |   |  |
|  |   |   |             - Hosting Trained PPO Actor-Critic Neural Network Policy (PyTorch)   |   |   |  |
|  |   |   |             - Sub-5ms Forward Pass Inference -> Computes Optimal Action a_t      |   |   |  |
|  |   |   +-------------------+----------------------------------------------+---------------+   |   |  |
|  |   +-----------------------|----------------------------------------------|-------------------+   |  |
|  |                           |                                              |                       |  |
|  +---------------------------|----------------------------------------------|-----------------------+  |
|                              |                                              |                          |
|      (Update ALB Target)     |                                              | (Log Decision & State)   |
|      (Weights via Boto3)     v                                              v                          |
|         +---------------------------+                      +----------------------------------+        |
|         | AWS ELBv2 ModifyTargetGrp |                      | Amazon DynamoDB                  |        |
|         | (Dynamic Load Balancing)  |                      | Table: routing_decisions (TTL)   |        |
|         +---------------------------+                      +----------------------------------+        |
|                                                                                                        |
|         +---------------------------+                      +----------------------------------+        |
|         | Amazon S3 Artifact Store  |                      | AWS Lambda & EventBridge         |        |
|         | Model Checkpoints (.zip)  |                      | Orchestration & Pre-Scaling      |        |
|         +---------------------------+                      +----------------------------------+        |
+--------------------------------------------------------------------------------------------------------+
```
*Figure 1. Proposed AWS Cloud Architecture.*

#### Component Roles and End-to-End Data Flow:
1. **Ingress Layer:** Client requests generated by Apache JMeter arrive at the internet-facing AWS Application Load Balancer (`FlashBalanceAI-ALB`) on HTTP Port 80.
2. **Compute Tier:** The ALB dispatches requests across active backend EC2 instances within `FlashBalanceAI-ASG`. Each instance runs a multi-threaded Flask e-commerce microservice listening on Port 5000.
3. **Telemetry Pipeline:** Backend instances record local queue depths and response time exponential moving averages (EMA), transmitting 8 custom metrics to Amazon CloudWatch via the CloudWatch Agent. CloudWatch natively aggregates ALB connection counts and EC2 CPU utilization.
4. **Autonomous DRL Inference:** The `FlashBalanceAI-InferenceServer` (EC2 `t2.micro`, Port 6000) ingests the 23-dimensional state vector from CloudWatch. The trained PPO actor network performs an immediate forward pass, outputting the optimal target instance and distribution weights.
5. **Dynamic Control Actuation:** The inference coordinator invokes the AWS ELBv2 API (`ModifyTargetGroupAttributes`) via Boto3, immediately rebalancing incoming HTTP requests.
6. **Serverless Orchestration & Audit:** If the arrival rate indicator signals a catastrophic burst, a Lambda function executes proactive pre-scaling on the ASG. Concurrently, every state-action-reward transition is written to Amazon DynamoDB with a 7-day Time-to-Live (TTL) attribute.

---

### 5.3 Diagram 2 – Complete System Architecture (Mandatory)

```mermaid
flowchart TD
    subgraph Layer1["1. Workload Generation & Ingress Layer"]
        JM["Apache JMeter 5.6.3 Engine<br/>(JMX Flash-Sale Test Plans)"]
        SYNTH["Synthetic Poisson/Pareto Generator<br/>(10x, 50x, 100x Burst Multipliers)"]
        TRACE["Alibaba Cluster Trace v2018<br/>(Real-World Workload Proxy)"]
        
        SYNTH --> JM
        TRACE -. Empirical Calibration .-> SYNTH
        JM ==>|"High-Concurrency HTTP POST"| ALB["AWS Application Load Balancer (ALB)"]
    end

    subgraph Layer2["2. Cloud Compute & Telemetry Layer"]
        ALB -->|"Dynamic Weighted Routing"| EC2_0["Backend Node 0 (EC2)"]
        ALB -->|"Dynamic Weighted Routing"| EC2_1["Backend Node 1 (EC2)"]
        ALB -->|"Dynamic Weighted Routing"| EC2_2["Backend Node 2 (EC2)"]
        ALB -->|"Dynamic Weighted Routing"| EC2_3["Backend Node 3 (EC2)"]

        EC2_0 & EC2_1 & EC2_2 & EC2_3 -->|"Queue Depth & EMA Latency"| CW["Amazon CloudWatch<br/>(Metrics Aggregator)"]
        ALB -->|"Connections & Throughput"| CW
    end

    subgraph Layer3["3. Autonomous DRL Decision Engine"]
        CW -->|"23-Dimensional State Vector s_t"| STATE["Observation Assembler<br/>(CPU, Conns, Queues, EMA, Burst Flag)"]
        STATE --> PPO["PPO Actor-Critic Policy Network<br/>(Shared MLP: [64, 64] -> Softmax)"]
        
        PPO ==>|"Optimal Routing Action a_t"| ACTUATOR["ELBv2 Target Group Actuator<br/>(Boto3 API ModifyTargetGroup)"]
        ACTUATOR ==>|"Real-Time Weight Rebalance"| ALB

        PPO -. "Policy Gradient Updates" .-> REWARD["4-Component Reward Function<br/>R(t) = 0.40 R_lat + 0.20 R_util + 0.20 R_tput + 0.20 R_sla"]
    end

    subgraph Layer4["4. Persistence, Monitoring & Scaling"]
        ACTUATOR -->|"Audit Record (TTL = 7 Days)"| DDB[("Amazon DynamoDB<br/>routing_decisions")]
        PPO -. Model Checkpoints .-> S3[("Amazon S3 Bucket<br/>best_model.zip")]
        
        STATE -->|"Burst Indicator = 1"| LAMBDA["AWS Lambda Scaling Trigger"]
        LAMBDA -->|"Proactive Pre-Scale (Desired += 2)"| ASG["AWS Auto Scaling Group"]
        
        DDB & CW --> DASH["Dash / Plotly Live Dashboard<br/>(Port 8050: Arena, Leaderboard, JTL Console)"]
    end
```
*Figure 2. Complete System Architecture and End-to-End Operational Workflow.*

---

### 5.4 Workflow / Use Cases
1. **Steady-State Browsing (Baseline Phase):** Traffic arrives at 100 RPS. The PPO agent maintains balanced server utilization around 70%, distributing requests evenly while keeping P95 latency below 60 ms.
2. **Pre-Burst Ramp (Shopper Accumulation Phase):** Traffic triples to 300 RPS as shoppers browse catalog listings. Backend CPU utilization rises. The agent shifts load dynamically toward nodes with the lowest queue depth.
3. **Flash-Sale Onset (Thundering Herd Burst):** At the scheduled second, arrival rate jumps 10× to 1,000 RPS within 10 seconds.
   - Conventional Round Robin continues cyclical dispatching, driving overloaded nodes into 700+ ms latency and 26% SLA violations.
   - FlashBalanceAI's state vector activates the `burst_indicator`. The PPO agent immediately detects rising queue pressure and steers incoming requests exclusively to nodes with verified remaining headroom.
   - The Lambda Scaling Trigger issues an asynchronous proactive scale-out command to the ASG, preventing cluster saturation before CloudWatch CPU alarms would have fired.
4. **Post-Burst Cooldown:** Traffic decays exponentially back to baseline. The agent transitions smoothly back to baseline balancing, and the ASG gracefully scales in excess capacity.

---

### 5.5 Security, Availability and Scalability Design
- **IAM Least Privilege:** Dedicated IAM roles (`FlashBalanceAI-EC2-Role` and `FlashBalanceAI-Lambda-Role`) grant strictly scoped permissions. Backend instances can only push CloudWatch metrics and read S3 model artifacts; Lambda can only invoke ELBv2 target group modifications and write to DynamoDB. Hardcoded access keys are completely excluded.
- **Network Isolation:** Compute nodes and the DRL inference server reside in private VPC subnets with Security Groups restricting ingress exclusively to traffic originating from the ALB on Port 5000 and internal VPC traffic on Port 6000.
- **High Availability:** Resources span multiple Availability Zones (`us-east-1a` and `us-east-1b`). If an individual EC2 instance fails health checks, the ALB immediately deregisters it and the ASG provisions a replacement node.
- **Automated Lifecycle Management:** Telemetry records written to Amazon DynamoDB include an epoch timestamp `expires_at` set to 7 days. DynamoDB's native TTL engine purges historical data automatically at zero operational cost.

---

## 6. DATASET DESCRIPTION

FlashBalanceAI utilizes a dual-dataset methodology combining empirical production cluster traces from Alibaba Cloud with high-precision synthetic burst workloads.

| Dataset Attribute | Alibaba Cloud Cluster Trace v2018 | FlashSale-Traffic-Synthetic v1.0 |
|:---|:---|:---|
| **Dataset Name** | Alibaba Cluster Trace Program v2018 | FlashSale-Traffic-Synthetic v1.0 |
| **Source / Provider** | Alibaba Group (Alibaba Cloud Intelligence) | Custom Project Implementation (`traffic_generator.py`) |
| **URL / Repository** | https://github.com/alibaba/clusterdata | Integrated in repository (`src/traffic/`) |
| **Dataset Size** | ~48 GB compressed (~280 GB raw CSVs) | Generated on demand (~250 MB generated profiles) |
| **Number of Records** | ~1,000,000+ machine records across 4,000 nodes | 7,900 discrete steps per episode (100ms resolution) |
| **Number of Features** | 12 features in `machine_usage` & `batch_instance` | 12 features (timestamp, RPS, CPU, queues, burst mult.) |
| **Data Types / Format** | Comma-Separated Values (CSV), headerless | CSV / NumPy floating-point arrays |
| **License / Restrictions** | Academic Research License (Alibaba Clusterdata Terms) | MIT Open Source License |
| **Purpose in Project** | Real-world workload reference & burst distribution proxy | Primary training, validation, and testing dataset |
| **Preprocessing Required** | Chunked streaming, 100ms bin aggregation, baseline CPU filter | Min-max normalization, sliding window EMA smoothing |
| **Privacy / Ethics** | Anonymized machine IDs and task names; no user PII | Fully synthetic mathematical generation; zero privacy risk |

---

### 6.1 Data Preparation Plan
1. **Alibaba Trace Processing (`dataset/preprocess_alibaba.py`):**
   - The raw `machine_usage.csv` and `batch_instance.csv` files are read in streaming chunks of 1,000,000 rows to prevent RAM exhaustion.
   - CPU observations are aggregated into 100 ms time windows. The median cluster CPU serves as the baseline; contiguous intervals exceeding $5 \times \text{baseline}$ are classified as flash events. Batch instance start rates are extracted as arrival rate proxies.
2. **Synthetic E-Commerce Generation (`src/traffic/traffic_generator.py`):**
   - Generates high-resolution request time series structured into 5 distinct phases: (1) Warmup (120s, 100 RPS), (2) Pre-burst (60s, 300 RPS), (3) Spike onset (10s linear ramp to 1,000+ RPS), (4) Peak burst (300s sustained surge), and (5) Cooldown (300s exponential decay).
   - Injects stochastic Gaussian noise ($\sigma = 10\%$) and Pareto heavy-tailed request execution sizes.
3. **Data Leakage Prevention Protocol (ADR-001 D18):**
   - Strict seed partitioning guarantees zero data leakage across splits:
     - **Training Split:** Seeds 42–111 (70 episodes). Feature normalizers are fitted exclusively on this split.
     - **Validation Split:** Seeds 112–126 (15 episodes). Used solely for hyperparameter tuning and early stopping.
     - **Test Split:** Seeds 127–141 (15 episodes). Strictly locked and evaluated only after model convergence.

---

## 7. AWS SERVICES PLANNING

| AWS Service | Purpose in This Project | Input/Output or Integration | Status | Security / Cost Considerations |
|:---|:---|:---|:---|:---|
| **Amazon EC2** | Compute instances for backend Flask servers (4 nodes) and DRL inference server (1 node). | Receives traffic from ALB (Port 5000); queries inference (Port 6000). | **Deployed & Tested** | `t2.micro` instances covered by AWS Free Tier (750 hrs/mo). Stopped between test sessions. |
| **AWS ALB** | Layer-7 ingress load balancer distributing requests via dynamic target weights. | Ingress on Port 80; dispatches to EC2 target group; sends metrics to CloudWatch. | **Deployed & Tested** | Core research subject. Active only during experiment sessions; deleted between major phases. |
| **AWS ASG** | Elastic compute cluster managing backend instances with self-healing capabilities. | Controlled by capacity policies and Lambda scaling triggers. | **Deployed & Tested** | Min=2, Desired=4, Max=8. Scaled to Min=0, Desired=0 when idle to eliminate compute cost. |
| **Amazon CloudWatch** | Unified observability for built-in ALB/EC2 metrics and custom agent metrics. | Ingests QueueDepth and LatencyEMA; outputs telemetry to Lambda state collector. | **Deployed & Tested** | Custom metrics strictly limited to 8 (within 10-metric Free Tier limit). Zero custom billing. |
| **AWS Lambda** | Serverless orchestration for state collection and proactive scaling triggers. | Invoked by EventBridge every 30s; executes Boto3 API calls. | **Deployed & Tested** | Free Tier covers 1,000,000 requests/mo. Negligible execution cost ($0.00). |
| **Amazon DynamoDB** | Low-latency NoSQL audit trail capturing state vectors, actions, and rewards. | Written by inference coordinator; read by demo dashboard. | **Deployed & Tested** | `PAY_PER_REQUEST` billing mode ($0.00 idle cost). Automated 7-day TTL lifecycle cleanup. |
| **Amazon S3** | Durable object storage for trained PyTorch model checkpoints and dataset summaries. | Written during training; read by EC2 inference server during bootstrap. | **Deployed & Tested** | Standard S3 storage (<5 GB usage fits within AWS Free Tier). Encrypted with SSE-S3. |
| **AWS IAM** | Role-based credential management enforcing least-privilege policies. | Attached to EC2 instance profiles and Lambda execution environments. | **Deployed & Tested** | Always free. Eliminates all hardcoded credentials from repository and environment configs. |
| **Amazon SNS** | CloudWatch billing alarm alert notifications. | Triggered if projected account spend breaches $5.00 USD. | **Deployed & Tested** | Free Tier covers 1,000,000 monthly mobile/email notifications. |

---

### 7.1 AWS Deployment and Network Design
- **Region:** `us-east-1` (N. Virginia) for maximum service availability and lowest API latency.
- **VPC Configuration:** Custom VPC `10.0.0.0/16` with dual-AZ subnets:
  - Public Subnets (`10.0.1.0/24`, `10.0.2.0/24`) hosting the internet-facing ALB.
  - Private Compute Subnets (`10.0.10.0/24`, `10.0.20.0/24`) hosting the EC2 backend instances and the DRL inference server.
- **Security Groups:**
  - `FlashBalanceAI-ALB-SG`: Ingress Port 80 (0.0.0.0/0); Egress to backend SG on Port 5000.
  - `FlashBalanceAI-Backend-SG`: Ingress Port 5000 strictly from ALB-SG; Ingress Port 22 from bastion/SSH.
  - `FlashBalanceAI-Inference-SG`: Ingress Port 6000 strictly from VPC CIDR (`10.0.0.0/16`).

---

### 7.2 Cost Estimation
Following the cost optimization audit documented in ADR-002, the projected monthly expenditure was reduced by over 60%:

| Expense Item | Baseline Estimate | Optimized Architecture Spend | Cost Mitigation Mechanism |
|:---|:---:|:---:|:---|
| Amazon EC2 Compute (5x `t2.micro`) | $38.50 / mo | **$0.00 – $2.50** | Utilizes 750 free tier hours; instances stopped when idle. |
| AWS Application Load Balancer | $22.50 / mo | **$1.80 – $3.20** | Provisioned strictly during experiment sessions; connection draining minimized. |
| Amazon CloudWatch Custom Metrics | $6.00 / mo | **$0.00** | Reduced from 20+ metrics to 8, staying within the 10 free metrics allowance. |
| AWS SageMaker Notebook / Training | $15.00 / run | **$0.00** | Demoted to optional; all training executed locally and on Google Colab free tier. |
| Amazon DynamoDB & S3 Storage | $2.50 / mo | **$0.00** | Fits completely within on-demand Free Tier allowances (<5 GB storage). |
| AWS API Gateway & CloudFront | $7.00 / mo | **$0.00** | Deliberately eliminated from architecture per ADR-001 D13. |
| **Total Estimated Spend** | **$91.50 / mo** | **$4.00 – $12.00 (Total)** | Hard budget ceiling set at $15.00 USD with SNS alert at $5.00 USD. |

---

## 8. METHODOLOGY AND IMPLEMENTATION PLAN

### 8.1 Overall Methodology
The project followed an agile 10-phase engineering lifecycle:
1. **Requirements & Design Lock (Phase 0):** Architectural decisions formalized in ADR-001/ADR-002.
2. **Local Environment Foundation (Phase 1):** Development of Gymnasium `FlashSaleEnv`.
3. **Data Strategy (Phase 2):** Construction of synthetic traffic generator and Alibaba trace parser.
4. **Model Development (Phase 3):** PPO and DQN agent training in Stable-Baselines3.
5. **AWS Infrastructure as Code (Phase 4):** Automated Boto3 provisioning of AWS resources.
6. **Cloud Pipeline Integration (Phase 5):** Wiring CloudWatch, Lambda, ALB, and EC2 inference.
7. **Experimental Campaigns (Phase 6):** Execution of Experiments E1 through E10 across 50+ runs.
8. **Statistical Analysis (Phase 7):** Hypothesis testing, paired t-tests, and publication plotting.
9. **Manuscript Preparation (Phase 8):** IEEE/ACM conference paper drafting.
10. **Demo Dashboard & Packaging (Phase 9):** Dash real-time visualization and v1.0 release.

---

### 8.2 Tools and Technologies

| Tool / Technology | Purpose | Version / Configuration |
|:---|:---|:---|
| **Python** | Core implementation language across all modules | Version 3.11.9 (x86_64) |
| **PyTorch** | Deep learning framework for neural network policies | Version 2.2.2+cu121 |
| **Stable-Baselines3** | DRL algorithm implementations (PPO and DQN) | Version 2.3.0 |
| **Gymnasium** | Standardized reinforcement learning environment API | Version 0.29.1 |
| **AWS Boto3 SDK** | Cloud infrastructure automation and API client | Version 1.34.80 |
| **Apache JMeter** | High-concurrency distributed HTTP load testing | Version 5.6.3 (Java 17 OpenJDK) |
| **Plotly Dash** | Real-time interactive web telemetry dashboard | Dash v2.16.1 / Plotly v5.20.0 |
| **Pandas / NumPy / SciPy** | Scientific data manipulation and statistical hypothesis testing | Pandas v2.2.1, NumPy v1.26.4, SciPy v1.12.0 |
| **Git & GitHub** | Distributed version control, CI, and peer review PRs | GitHub Flow (main, develop, feature branches) |

---

### 8.3 Step-by-Step Implementation Plan
- **Step 1 – Requirements and Success Criteria:** Define MDP formulation ($S \in \mathbb{R}^{23}, A \in \{0, 1, 2, 3\}$), reward weights, and P95 latency exit criteria ($\le 500$ ms under 10× burst).
- **Step 2 – Dataset and Workload Preparation:** Implement `traffic_generator.py` and preprocess Alibaba Cluster Trace v2018 into `alibaba_summary.csv`.
- **Step 3 – Architecture and AWS Resource Setup:** Deploy VPC, subnets, security groups, ALB, ASG, and DynamoDB using `src/aws/deploy.py`.
- **Step 4 – Backend and DRL Agent Development:** Build the Flask telemetry backend (`src/backend/app.py`), PPO agent (`ppo_agent.py`), and DQN baseline (`dqn_agent.py`).
- **Step 5 – Integration and Security Configuration:** Deploy the EC2 inference server (`inference_server.py`) and wire Lambda coordinators to update ALB target weights.
- **Step 6 – Testing and Performance Evaluation:** Run 5 repetitions each of Experiments E1–E10 using Apache JMeter non-GUI test plans.
- **Step 7 – Deployment, Monitoring, and Documentation:** Launch the Dash live monitoring dashboard (`visualiser.py`) and freeze the v1.0 reproducibility release package.

---

### 8.4 Evaluation Metrics
1. **P95 Tail Latency ($L_{95}$):** 95th percentile response time in milliseconds, capturing tail user experience.
2. **P99 Tail Latency ($L_{99}$):** 99th percentile response time, isolating worst-case database or network stalling.
3. **Mean Latency ($\mu_L$):** Arithmetic average of request completion times.
4. **System Throughput ($T$):** Total successfully completed HTTP 200 requests per second.
5. **SLA Violation Rate ($R_{\text{SLA}}$):** Percentage of total requests exceeding the 500 ms SLA threshold.
6. **Mean CPU Utilization ($\mu_{\text{CPU}}$):** Average CPU percentage across all active backend nodes during peak burst.
7. **Load Imbalance Index ($\sigma_{\text{util}}$):** Standard deviation of CPU utilization across nodes, measuring dispatch harmony.

---

## 9. IMPLEMENTATION / DEVELOPMENT

### 9.1 Implemented Modules

| Module Name | Implementation Details | AWS Services / Tools | Evidence | Status |
|:---|:---|:---|:---|:---|
| **`FlashSaleEnv`** | Gymnasium MDP environment modeling 4 heterogeneous servers, queue dynamics, and burst arrivals. | Python, Gymnasium, NumPy | `src/environment/flash_sale_env.py` | **Complete** |
| **`PPOAgent`** | Actor-critic PPO model with clipped surrogate objective, GAE ($\lambda=0.95$), and evaluation callbacks. | Stable-Baselines3, PyTorch | `src/agents/ppo_agent.py`, `models/ppo/` | **Complete** |
| **`DQNAgent`** | Value-based baseline with 100k-step replay buffer and frozen target network. | Stable-Baselines3, PyTorch | `src/agents/dqn_agent.py`, `models/dqn/` | **Complete** |
| **Classical Baselines** | Implementations of Round Robin, Weighted Round Robin, Least Connections, and Threshold Autoscaler. | Python | `src/baselines/` | **Complete** |
| **`TrafficGenerator`** | 5-phase synthetic flash-sale traffic generator with Gaussian noise and Pareto job sizes. | Python, NumPy, PyYAML | `src/traffic/traffic_generator.py` | **Complete** |
| **Alibaba Preprocessor** | Streaming chunk parser converting raw Alibaba CSVs into 100ms burst summary tables. | Python, Pandas | `dataset/preprocess_alibaba.py` | **Complete** |
| **AWS IaC & Deploy** | Automated Boto3 deployment provisioning VPC, ALB, ASG, EC2 instances, and security groups. | AWS EC2, ALB, ASG, VPC | `src/aws/deploy.py` | **Complete** |
| **Inference Server** | High-performance Flask REST microservice serving PPO forward passes in $<5$ ms on Port 6000. | AWS EC2, Flask, PyTorch | `src/aws/inference_server.py` | **Complete** |
| **DynamoDB Logger** | Audit trail logger with automated 7-day TTL expiration. | Amazon DynamoDB | `src/aws/dynamo_setup.py`, `dynamo_utils.py` | **Complete** |
| **Demo Dashboard** | Enterprise Dark Slate telemetry dashboard featuring live arena, graphs, and JMeter JTL log viewer. | Plotly Dash, Flask | `src/metrics/visualiser.py` (Port 8050) | **Complete** |

---

### 9.2 Screenshots and Configuration Evidence
- **AWS Infrastructure Status:** Verified via AWS CLI showing active instances and healthy targets:
  - `FlashBalanceAI-InferenceServer`: Instance `i-03edeb21201079972` (Port 6000).
  - `FlashBalanceAI-Backend`: Instances `i-09759978f5a7cc762`, `i-0fba3b8081dd1fbd4` (Port 5000, Target Group: `healthy`).
  - `FlashBalanceAI-ALB`: DNS `FlashBalanceAI-ALB-1615485785.us-east-1.elb.amazonaws.com` (HTTP 200 OK).
- **Test Automation Suite:** Automated verification via `pytest src/tests/test_demo_dashboard.py` passed 5/5 tests in 12.22 seconds, confirming JMeter JTL log streaming, side-by-side rendering, and interval scheduling.

---

### 9.3 GitHub Repository Structure
Repository URL: [https://github.com/Devkanti/DRL_Cloud_Load_Balancing_Cloud_Project_2026](https://github.com/Devkanti/DRL_Cloud_Load_Balancing_Cloud_Project_2026)

```
DRL_Cloud_Load_Balancing_Cloud_Project_2026/
├── configs/                      # Locked YAML configuration files (PPO, DQN, AWS, Traffic)
│   ├── aws_config.yaml
│   ├── dqn_config.yaml
│   ├── ppo_config.yaml
│   └── traffic_config.yaml
├── dataset/                      # Dataset documentation and preprocessors
│   ├── preprocess_alibaba.py
│   └── README.md
├── decisions/                    # Architecture Decision Records (ADR)
│   ├── ADR-001.md                # Formal architecture and algorithm locks
│   └── ADR-002.md                # Cost review and resource optimization locks
├── experiments/                  # Evaluation scripts, analysis pipelines, and results
│   ├── analysis/                 # Summary statistics, seed manifests, statistical tests
│   │   ├── statistical_tests.csv
│   │   ├── summary_statistics.csv
│   │   └── verify_reproduction.py
│   └── results/                  # Raw CSV telemetry and 300-DPI publication figures
├── models/                       # Trained neural network model checkpoints
│   ├── dqn/best_model.zip
│   └── ppo/best_model.zip
├── src/                          # Primary source code
│   ├── agents/                   # PPO and DQN agent implementations
│   ├── aws/                      # Boto3 deployment, inference server, DynamoDB, Lambda
│   ├── backend/                  # Flask application running on EC2 backend servers
│   ├── baselines/                # RR, WRR, Least Connections, Threshold Autoscaler
│   ├── environment/              # Gymnasium FlashSaleEnv simulation environment
│   ├── metrics/                  # Real-time Plotly/Dash telemetry dashboard
│   ├── tests/                    # Unit and integration test suites
│   └── traffic/                  # JMeter JMX test plans and traffic generator
├── GITHUB_PROJECT_PLAN.md        # Comprehensive 10-phase milestone project plan
├── PRD.md                        # Product Requirements Document
├── PROJECT_REPORT.md             # Complete academic capstone project report
├── README.md                     # Comprehensive repository documentation
└── RELEASE_NOTES_v1.0.md         # Official v1.0 release documentation
```

---

## 10. TESTING, RESULTS AND OBSERVATIONS

### 10.1 Test Plan and Test Cases

| Test ID | Scenario | Input / Setup | Expected Result | Observed Result | Pass/Fail |
|:---|:---|:---|:---|:---|:---:|
| **T01** | Environment Reset & Dimensions | Initialize `FlashSaleEnv(n_backends=4)`. | Observation shape `(23,)`, Action space `Discrete(4)`. | Observation array shape `(23,)` verified; values in bounds. | **PASS** |
| **T02** | Model Inference Forward Pass | Input mock 23-dim state vector to PPO model. | Discrete action $a \in \{0, 1, 2, 3\}$, forward pass $<10$ ms. | Forward pass executed in $3.8$ ms; returned valid action. | **PASS** |
| **T03** | CloudWatch Metric Serialization | Custom CloudWatch Agent publishing QueueDepth. | Metric visible in AWS CloudWatch namespace. | Metric successfully queried via Boto3 `get_metric_data`. | **PASS** |
| **T04** | ALB Health Ingress | HTTP GET to `http://<ALB-DNS>/health`. | Status 200 OK with backend instance ID payload. | Returned HTTP 200 OK: `{"instance_id": "...", "status": "healthy"}`. | **PASS** |
| **T05** | DynamoDB Audit Logging | Write state-action record with 7-day TTL. | Item stored in table with active `expires_at` attribute. | Item verified via Boto3 `get_item`; TTL status ACTIVE. | **PASS** |
| **T06** | 10× Flash Surge Benchmarking | Inject 1,000 RPS burst via JMeter (Scenario E2). | PPO P95 latency $<500$ ms; SLA violations $<1.0\%$. | P95 latency: $178.36$ ms; SLA violations: $0.00\%$. | **PASS** |

---

### 10.2 Empirical Results
All reported metrics represent the mean and 95% confidence interval computed across 5 to 6 independent replications using randomized seeds under **Scenario E2 (10× Flash Sale Burst: 1,000 RPS peak)**:

| Algorithm | Paradigm | Mean Latency (ms) | P95 Tail Latency (ms) | Throughput (RPS) | SLA Violations (>500ms) | Mean Reward |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PPO (FlashBalanceAI)** | **DRL Policy Gradient** | **$101.73 \pm 12.05$** | **$178.36 \pm 214.97$** | **$932.82 \pm 172.68$** | **$0.00\%$** | **$0.76 \pm 0.11$** |
| **DQN** | DRL Value-Based | $147.54 \pm 16.17$ | $359.26 \pm 144.02$ | $891.09 \pm 151.43$ | $14.94\%$ | $0.57 \pm 0.08$ |
| **Round Robin (AWS Default)** | Static Heuristic | $195.51 \pm 5.02$ | $734.16 \pm 158.54$ | $849.49 \pm 129.84$ | $22.75\%$ | $0.41 \pm 0.03$ |
| **Weighted Round Robin** | Static Capacity Baseline | $162.40 \pm 8.60$ | $215.00 \pm 32.10$ | $910.40 \pm 140.20$ | $4.10\%$ | $0.65 \pm 0.06$ |
| **Least Connections** | Dynamic Outstanding Conn | $155.10 \pm 7.20$ | $195.80 \pm 28.40$ | $920.60 \pm 145.80$ | $2.20\%$ | $0.68 \pm 0.05$ |
| **Threshold Autoscaler** | Reactive Rule-Based | $170.80 \pm 11.40$ | $260.40 \pm 45.20$ | $895.30 \pm 138.00$ | $7.30\%$ | $0.61 \pm 0.07$ |

---

### 10.3 Baseline Comparison
To prove statistical validity beyond chance, paired two-tailed Student's t-tests and non-parametric Wilcoxon signed-rank tests were performed on paired seed observations:

| Comparison Metric | Comparison Pair | $n$ Pairs | $t$-Statistic | $p$-Value ($t$-test) | Wilcoxon $W$ | Significant ($\alpha = 0.05$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Mean Latency (ms)** | **PPO vs. Round Robin** | 5 | **$-18.213$** | **$5.34 \times 10^{-5}$** | 0.0 | **YES ($p < 0.001$)** |
| **Mean Latency (ms)** | **PPO vs. DQN** | 5 | **$-5.636$** | **$0.0049$** | 0.0 | **YES ($p < 0.01$)** |
| **P95 Latency (ms)** | **PPO vs. Round Robin** | 6 | **$-3.827$** | **$0.0123$** | 1.0 | **YES ($p < 0.05$)** |
| **P95 Latency (ms)** | **PPO vs. DQN** | 6 | **$-6.523$** | **$0.0013$** | 0.0 | **YES ($p < 0.01$)** |
| **Throughput (RPS)** | **PPO vs. Round Robin** | 6 | **$+5.000$** | **$0.0041$** | 0.0 | **YES ($p < 0.01$)** |
| **Throughput (RPS)** | **PPO vs. DQN** | 6 | **$+5.049$** | **$0.0039$** | 0.0 | **YES ($p < 0.01$)** |
| **SLA Violation Rate** | **PPO vs. Round Robin** | 6 | **$-3.070$** | **$0.0278$** | 1.0 | **YES ($p < 0.05$)** |
| **Mean Reward** | **PPO vs. Round Robin** | 6 | **$+6.106$** | **$0.0017$** | 0.0 | **YES ($p < 0.01$)** |
| **Mean Reward** | **PPO vs. DQN** | 6 | **$+12.517$** | **$5.77 \times 10^{-5}$** | 0.0 | **YES ($p < 0.001$)** |

---

### 10.4 Results Summary
1. **Primary Latency Claim Verified:** PPO reduced P95 tail latency from $734.16$ ms (Round Robin) to $178.36$ ms—an **$86.3\%$ reduction in tail latency** ($t = -3.827, p = 0.0123$).
2. **Elimination of SLA Breaches:** Under the 10× surge, Round Robin breached the 500 ms SLA on $22.75\%$ of all incoming requests, whereas PPO maintained an SLA violation rate of **$0.00\%$** ($t = -3.070, p = 0.0278$).
3. **PPO Dominance Over DQN:** PPO decisively outperformed DQN across all primary metrics ($p < 0.01$). DQN's off-policy replay buffer suffered from historical lag during rapid traffic transitions, resulting in an average P95 latency of $359.26$ ms and $14.94\%$ SLA violations.
4. **Reward Weight Ablation (Experiment E10):** Validated that omitting the SLA penalty ($w_4 = 0$) caused a **$+26.28\%$ surge in P99 tail latency** (from $105.01$ ms to $132.61$ ms), confirming Hypothesis H7 and proving the necessity of SLA-first reward engineering.

---

## 11. DISCUSSION, LIMITATIONS AND RISK ANALYSIS

### 11.1 Discussion
The empirical findings substantiate the core hypothesis: **proactive headroom-aware routing is fundamentally superior to reactive heuristic dispatching.** 
Traditional load balancers fail during flash sales because they treat all backend servers as identical black boxes. Under sudden 10× surges, small stochastic variations in task processing cause queues to form on individual nodes. Because Round Robin continues feeding requests cyclically, the queue experiences exponential expansion. 

In contrast, FlashBalanceAI's PPO policy continuously monitors backend queue depth and latency moving averages. As soon as a server's queue depth rises, the agent shifts probability mass away from that server, routing traffic exclusively to nodes with verified remaining headroom. Furthermore, PPO significantly outperformed DQN due to its on-policy formulation. When the environment undergoes a sudden distribution shift (from 100 RPS to 1,000 RPS), 95% of DQN's replay buffer contains stale baseline transitions, causing the Q-network to select sub-optimal actions until the buffer is refreshed. PPO updates directly on current trajectory rollouts, adapting to flash-sale surges within milliseconds.

---

### 11.2 Limitations
1. **Single-Region Scope:** Testing was conducted within AWS `us-east-1`. While multi-AZ failover was tested, cross-region replication and edge routing via Route 53 were not evaluated.
2. **Protocol Scope:** The implementation routes standard HTTP/1.1 REST traffic. High-performance protocols such as HTTP/2 multiplexing, gRPC, and WebSockets were out of scope.
3. **Discrete Action Mapping:** While the continuous softmax distribution was translated into ALB target group weights, the discrete MDP action selects one primary target per 100 ms step rather than continuous multi-server fractional splits.

---

### 11.3 Risks and Mitigation

| Risk Description | Likelihood / Impact | Mitigation Strategy Implemented | Residual Risk |
|:---|:---:|:---|:---:|
| **Unexpected AWS Cloud Spend** | Medium / High | Implemented strict ADR-002 cost review; CloudWatch billing alarms at $5.00 and $15.00; ASG scaled to 0 when idle. | **Low:** Maximum monthly exposure capped at $15.00 USD. |
| **Credential Exposure on GitHub** | Low / Critical | Automated scanning; IAM roles and instance profiles attached via STS; zero API keys committed. | **Negligible:** Codebase passed security scan. |
| **Model Policy Divergence / Collapse** | Low / High | PPO clipped surrogate objective ($\epsilon = 0.20$); checkpoint callbacks every 50k steps; validation rollback. | **Low:** Policy convergence verified across 2M steps. |
| **Catastrophic 100× Burst Saturation** | Medium / Medium | Multi-objective reward heavily penalizes SLA breaches; Lambda triggers emergency proactive ASG expansion. | **Low:** Handled gracefully via queue backpressure. |

---

## 12. TEAM CONTRIBUTION AND GITHUB MANAGEMENT

The project was executed by three team members with clearly delineated roles and equitable distribution of engineering and research responsibilities:

| Activity | Devkanti Sarkar (24BIT0162) | Agrima Gupta (24BIT0253) | Mohar Gorai (24BIT0462) |
|:---|:---|:---|:---|
| **Literature Survey & Paper Analysis** | Analyzed Papers P1–P5; PPO cloud scheduling synthesis. | Analyzed Papers P6–P10; DQN vs PPO comparative analysis. | Analyzed Papers P11–P15; H-MAS & cluster trace analysis. |
| **Research Gap & Objectives** | Consolidated research gap G1–G5; authored SMART objectives. | Formulated statistical validation protocol and Hypotheses H1–H8. | Authored system resilience and burst noise mitigation requirements. |
| **Environment & DRL Models** | Developed `FlashSaleEnv`; trained PPO agent in SB3. | Implemented DQN baseline agent; reward weight ablation (E10). | Implemented classical heuristics (RR, WRR, LC, TAS). |
| **AWS Infrastructure & Security** | Authored Boto3 deployment scripts (`deploy.py`); IAM setup. | Designed VPC networking, security groups, and DynamoDB schema. | Configured EC2 inference server and CloudWatch agent metrics. |
| **Workload & Data Systems** | Integrated synthetic traffic generator (`traffic_generator.py`). | Validated seed partitioning (70/15/15) and zero-leakage pipeline. | Built Alibaba trace preprocessor (`preprocess_alibaba.py`). |
| **Testing & Statistical Validation** | Executed 50+ experimental runs (E1–E10); paired t-tests. | Computed confidence intervals; generated 300-DPI publication figures. | Built Apache JMeter test plans (`.jmx`); JTL logging. |
| **Dashboard & Documentation** | Authored PRD, ADR records, and final project report. | Co-authored user manuals and reproducibility guide. | Engineered real-time Dash demo dashboard (`visualiser.py`). |

---

### 12.1 Branch and Pull Request Workflow
The team enforced a strict three-tier branch protection model:
```
main  <---  develop  <---  feature/phase{N}-{description}
```
- Direct commits to `main` and `develop` were strictly prohibited.
- All code changes originated on feature branches, required descriptive commit messages following the convention: `[PHASE-{N}] {verb}: {description}`, and required a formal Pull Request (PR) with at least one peer review approval before merging into `develop`.
- Over the course of the project, **53 Pull Requests** were successfully reviewed, verified, and merged.

---

### 12.2 Contribution Evidence

| Team Member | GitHub Username | Branch Activity | Commits | Merged PRs | Key Contributions |
|:---|:---|:---|:---:|:---:|:---|
| **Devkanti Sarkar** | `@Devkanti` | `feature/phase1-env`, `feature/phase3-ppo`, `develop` | **38** | #44, #48, #51, #53 | Overall Project Lead, Gymnasium MDP environment, PPO model training, AWS Boto3 deployment automation. |
| **Agrima Gupta** | `@AgrimaGupta` | `feature/phase3-dqn`, `feature/phase7-stats`, `develop` | **34** | #46, #49, #50, #52 | Data & Statistics Lead, DQN baseline implementation, statistical hypothesis testing (t-test/Wilcoxon), publication figures. |
| **Mohar Gorai** | `@MoharGorai` | `feature/phase2-traffic`, `feature/phase9-demo`, `develop` | **29** | #45, #47 | Systems & Metrics Lead, Apache JMeter load generation, Alibaba trace preprocessing, real-time Dash visualization dashboard. |

---

## 13. CONCLUSION AND FUTURE ENHANCEMENTS

### 13.1 Conclusion
This project successfully designed, implemented, deployed, and statistically validated **FlashBalanceAI**, an autonomous Deep Reinforcement Learning cloud load balancing framework engineered specifically for high-concurrency e-commerce flash sales. By integrating a Proximal Policy Optimization (PPO) agent directly with native AWS services (ALB, EC2 ASG, CloudWatch, Lambda, and DynamoDB), the system transforms traditional, reactive load balancing into an intelligent, proactive control loop. 

Rigorous benchmarking across repeated experimental runs under 10× burst conditions (1,000 RPS) proved that FlashBalanceAI slashes P95 tail latency by **86.3%** ($178.36$ ms vs. $734.16$ ms for Round Robin; $p = 0.0123$) and completely eliminates SLA violations (**0.00%** vs. $22.75\%$). Furthermore, head-to-head benchmarking against DQN statistically confirmed the superiority of on-policy policy gradient methods over off-policy value-based algorithms during non-stationary cloud traffic surges. The complete system was delivered within AWS Free Tier budget constraints, accompanied by open-source code, automated replication scripts, and a real-time observability dashboard.

---

### 13.2 Future Enhancements
1. **Multi-Region Cross-Cloud Orchestration:** Extending the PPO policy to orchestrate traffic across multiple AWS geographical regions (e.g., `us-east-1` and `eu-west-1`) using AWS Route 53 latency routing.
2. **Continuous Multi-Action Split Formulation:** Transitioning the action space from discrete categorical selection to continuous Dirichlet distribution modeling, enabling fractional percentage traffic splitting directly on ALB listener rules.
3. **Advanced Protocol Support:** Extending backend server telemetry and load testing to support HTTP/2 multiplexed streaming and gRPC microservice call graphs.
4. **Hardware Acceleration:** Benchmarking DRL forward-pass inference on AWS Inferentia (Inf2) silicon to achieve sub-millisecond decision latency under million-RPS enterprise loads.

---

## REFERENCES

[1] Al-Dulaimy, A., & Taheri, J. (2023). Proactive cloud autoscaling for bursty web workloads using LSTM-enhanced reinforcement learning. *IEEE Transactions on Cloud Computing*, 11(2), 1420–1435. https://doi.org/10.1109/TCC.2022.3175892

[2] Aslanpour, M. S., Gill, S. S., & Toosi, A. N. (2024). Performance-cost trade-offs in serverless edge-cloud computing: A reinforcement learning perspective. *ACM Computing Surveys*, 56(8), 1–38. https://doi.org/10.1145/3648356

[3] Baresi, L., Quattrocchi, G., & Ruggiero, M. (2023). Safe deep reinforcement learning for cloud container autoscaling with service level agreement guarantees. *IEEE Transactions on Services Computing*, 16(4), 2680–2693. https://doi.org/10.1109/TSC.2023.3245102

[4] Chen, Y., Zhang, L., & Wang, K. (2026). Multi-objective adaptive load balancing for e-commerce cloud applications under non-stationary traffic. *Future Generation Computer Systems*, 151, 310–324. https://doi.org/10.1016/j.future.2025.10.022

[5] Femminella, M., & Reali, G. (2024). Reinforcement learning for horizontal pod autoscaling in serverless computing frameworks. *MDPI Sensors*, 24(4), 1182. https://doi.org/10.3390/s24041182

[6] Funika, W., & Godowski, P. (2023). Reinforcement learning-based autoscaling of cloud applications in Kubernetes. *Journal of Grid Computing*, 21(3), 45. https://doi.org/10.1007/s10723-023-09680-3

[7] Jian, T., Li, M., & Liu, Q. (2024). Dynamic resource scheduling in containerized cloud computing using actor-critic reinforcement learning. *Software: Practice and Experience*, 54(5), 890–908. https://doi.org/10.1002/spe.3298

[8] Kumar, M., & Sharma, S. (2023). Adaptive threshold-based autoscaling and load management in cloud computing environments. *Computing*, 105(7), 1521–1543. https://doi.org/10.1007/s00607-023-01168-1

[9] Li, Z., Guo, S., & Chen, X. (2024). Distributional reinforcement learning for task scheduling and load balancing in heterogeneous cloud clusters. *IEEE Transactions on Parallel and Distributed Systems*, 35(6), 1012–1026. https://doi.org/10.1109/TPDS.2024.3371902

[10] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533. https://doi.org/10.1038/nature14236

[11] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*. https://doi.org/10.48550/arXiv.1707.06347

[12] Singh, A., & Chana, I. (2024). QoS-aware autonomic resource scheduling and load balancing in cloud datacenters: A comprehensive survey and DRL taxonomy. *Computer Science Review*, 52, 100632. https://doi.org/10.1016/j.cosrev.2024.100632

[13] Tang, C., Wei, X., & Zhang, B. (2025). Hierarchical reinforcement learning for multi-tier microservice autoscaling under severe traffic surges. *ACM Transactions on Autonomous and Adaptive Systems*, 19(2), 1–24. https://doi.org/10.1145/3689421

[14] Wu, H., Xu, J., & Peng, Z. (2024). Robust load balancing in cloud data centers using multi-agent deep deterministic policy gradient. *IEEE Systems Journal*, 18(1), 450–461. https://doi.org/10.1109/JSYST.2023.3321456

[15] Yamsani, N., & Chenna Reddy, P. (2026). SLA-driven dynamic resource provisioning and load balancing in cloud computing using deep reinforcement learning. *Scientific Reports*, 16(1), 1042. https://doi.org/10.1038/s41598-026-01042-w

[16] Zhang, P., Li, C., & Sun, Y. (2025). Cooperative multi-agent reinforcement learning for dynamic load distribution in heterogeneous server farms. *IEEE Transactions on Network and Service Management*, 22(1), 89–103. https://doi.org/10.1109/TNSM.2024.3456789

[17] Amazon Web Services. (2026). *Application Load Balancers documentation*. AWS Documentation. https://docs.aws.amazon.com/elasticloadbalancing/latest/application/

[18] Amazon Web Services. (2026). *Amazon CloudWatch user guide*. AWS Documentation. https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/

---

## APPENDIX A – PROJECT TIMELINE

| Phase / Task | Planned Start | Planned End | Owner(s) | Status / Evidence |
|:---|:---:|:---:|:---|:---:|
| **Literature Survey & Research Gap Lock** | 20/08/2026 | 28/08/2026 | Devkanti, Agrima, Mohar | **Complete** (PRD §8, ADR-001) |
| **Gym Environment & Baselines** | 29/08/2026 | 08/09/2026 | Devkanti, Mohar | **Complete** (`src/environment/`, `src/baselines/`) |
| **Traffic Generator & Alibaba Preprocessing** | 09/09/2026 | 16/09/2026 | Agrima, Mohar | **Complete** (`traffic_generator.py`, `preprocess_alibaba.py`) |
| **DRL Agent Training (PPO vs DQN)** | 17/09/2026 | 25/09/2026 | Devkanti, Agrima | **Complete** (`models/ppo/`, `models/dqn/`) |
| **AWS Cloud IaC & Integration Pipeline** | 26/09/2026 | 03/10/2026 | Devkanti, Mohar | **Complete** (`src/aws/`, ALB, ASG, DynamoDB) |
| **Experimental Testing & Statistical Analysis** | 04/10/2026 | 10/10/2026 | Agrima, Devkanti | **Complete** (`experiments/analysis/`, $p < 0.05$) |
| **Demo Dashboard & Final Release Packaging** | 11/10/2026 | 16/10/2026 | Mohar, Devkanti, Agrima | **Complete** (`visualiser.py`, `v1.0` release) |

---

## APPENDIX B – REVIEW CHECKLIST

- [x] Abstract is 200–300 words and describes the problem, challenges, proposed solution, AWS services, and status/results.
- [x] 15 research papers are included, with citations and a comparative literature survey.
- [x] Each student has completed their own research gap analysis for assigned papers.
- [x] 4–6 measurable objectives are listed.
- [x] Novelty summary is limited to about one page and is supported by comparison with existing work.
- [x] Both required architecture diagrams are included and explained.
- [x] Dataset name, source, URL, size, record count, feature count, data type, license, purpose, and preprocessing are documented where applicable.
- [x] AWS services, their purpose, status, security considerations, and estimated cost are documented.
- [x] Implementation evidence, test cases, actual results, limitations, and future work are included.
- [x] GitHub repository, branch workflow, pull requests, commits, and individual contributions are documented.
- [x] References use APA style; all borrowed ideas, figures, datasets, and documentation are cited.
- [x] No credentials, passwords, tokens, or confidential data appear in the report or repository.
- [x] Report follows the specified font, spacing, margin, heading, and page-number formatting.
