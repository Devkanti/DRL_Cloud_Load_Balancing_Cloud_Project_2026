"""
Generate the complete Project Report in .docx format for FlashBalanceAI
Course: BCSE355L - Cloud Architecture Design
Formatting: Times New Roman 12pt, 1.5 line spacing, 1-inch margins, page numbers at bottom centre.
"""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def create_report():
    doc = Document()

    # 1. Page Margins (1 inch on all sides)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Add page number to footer (centered)
        footer = section.footer
        p_ft = footer.paragraphs[0]
        p_ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_ft = p_ft.add_run()
        run_ft.font.name = "Times New Roman"
        run_ft.font.size = Pt(10)
        run_ft.font.color.rgb = RGBColor(100, 100, 100)
        
        # OpenXML dynamic page number
        fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
        instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
        fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
        fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
        run_ft._r.append(fldChar1)
        run_ft._r.append(instrText)
        run_ft._r.append(fldChar2)
        run_ft._r.append(fldChar3)

    # 2. Base Normal Style Configuration
    style_normal = doc.styles['Normal']
    font_normal = style_normal.font
    font_normal.name = 'Times New Roman'
    font_normal.size = Pt(12)
    font_normal.color.rgb = RGBColor(30, 30, 30)
    style_normal.paragraph_format.line_spacing = 1.5
    style_normal.paragraph_format.space_after = Pt(6)

    # Helper functions for adding content
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(15, 23, 42)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(14)
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(30, 58, 138)
        return p

    def add_heading1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(15, 23, 42)
        return p

    def add_heading2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(30, 58, 138)
        return p

    def add_heading3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(51, 65, 85)
        return p

    def add_p(text, bold_prefix=None, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(6)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.name = 'Times New Roman'
            r_pre.font.size = Pt(12)
        r = p.add_run(text)
        r.italic = italic
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(4)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.name = 'Times New Roman'
            r_pre.font.size = Pt(12)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        return p

    def set_cell_background(cell, fill_hex):
        tcPr = cell._element.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._element.get_or_add_tcPr()
        tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
        tcPr.append(tcMar)

    def style_table(table, col_widths=None):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(table.rows):
            # Header styling
            if i == 0:
                for cell in row.cells:
                    set_cell_background(cell, "1E293B")
                    set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
                    for p in cell.paragraphs:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        p.paragraph_format.line_spacing = 1.15
                        p.paragraph_format.space_after = Pt(0)
                        for r in p.runs:
                            r.bold = True
                            r.font.name = 'Times New Roman'
                            r.font.size = Pt(10)
                            r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                bg_color = "F8FAFC" if i % 2 == 1 else "FFFFFF"
                for cell in row.cells:
                    set_cell_background(cell, bg_color)
                    set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                    for p in cell.paragraphs:
                        p.paragraph_format.line_spacing = 1.15
                        p.paragraph_format.space_after = Pt(0)
                        for r in p.runs:
                            r.font.name = 'Times New Roman'
                            r.font.size = Pt(9.5)
                            r.font.color.rgb = RGBColor(30, 41, 59)
            
            # Set widths if specified
            if col_widths and len(col_widths) == len(row.cells):
                for c_idx, width in enumerate(col_widths):
                    row.cells[c_idx].width = Inches(width)

    # ==================== COVER & HEADER ====================
    add_title("PROJECT REPORT\nCLOUD ARCHITECTURE DESIGN – AWS CLOUD PROJECT")
    add_subtitle("DEEP REINFORCEMENT LEARNING-BASED CLOUD LOAD BALANCING FRAMEWORK FOR FLASH SALE PLATFORMS")

    # Metadata Box
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(14)
    p_meta.paragraph_format.line_spacing = 1.3
    
    meta_runs = [
        ("Submitted by:\n", True),
        ("Devkanti Sarkar — 24BIT0162\nAgrima Gupta — 24BIT0253\nMohar Gorai — 24BIT0462\n\n", False),
        ("Programme / Branch / Year: ", True), ("B.Tech. / Information Technology / 2028\n", False),
        ("Course Code & Name: ", True), ("BCSE355L – Cloud Architecture Design\n", False),
        ("Institution: ", True), ("Vellore Institute of Technology (VIT), Vellore\n", False),
        ("Course Instructor / Guide: ", True), ("Dr. Priya V\n", False),
        ("Team Number: ", True), ("Group Project (FlashBalanceAI)\n", False),
        ("GitHub Repository: ", True), ("https://github.com/Devkanti/DRL_Cloud_Load_Balancing_Cloud_Project_2026\n", False),
        ("Date of Submission: ", True), ("16/10/2026", False),
    ]
    for text, is_bold in meta_runs:
        r = p_meta.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.bold = is_bold
        if is_bold:
            r.font.color.rgb = RGBColor(15, 23, 42)
        else:
            r.font.color.rgb = RGBColor(71, 85, 105)

    doc.add_page_break()

    # ==================== ABSTRACT ====================
    add_heading1("ABSTRACT")
    add_p(
        "Modern e-commerce platforms experience severe performance degradation during high-concurrency flash-sale events due to massive, concentrated surges in user traffic. Traffic typically surges by 10× to 100× within 2 to 10 seconds, creating a severe thundering herd problem that overwhelms conventional cloud infrastructure. Traditional static load balancers (such as uniform Round Robin and Least Outstanding Requests) are fundamentally stateless and reactive; they lack awareness of backend queue dynamics, hardware heterogeneity, and transient CPU throttling. Concurrently, standard reactive autoscaling mechanisms (such as AWS Auto Scaling Groups driven by CloudWatch CPU alarms) suffer from a 2- to 5-minute cold-start provisioning latency, resulting in request queue saturation, cascading timeouts (HTTP 504), and Severe Level Agreement (SLA) breaches during volatile burst intervals."
    )
    add_p(
        "To overcome these critical bottlenecks, this project presents FlashBalanceAI, an autonomous Deep Reinforcement Learning (DRL) cloud load balancing framework engineered specifically for high-concurrency flash-sale platforms. The system deploys a Proximal Policy Optimization (PPO) actor-critic agent integrated natively with production AWS infrastructure, including AWS Application Load Balancer (ALB), Amazon EC2 Auto Scaling Groups, Amazon CloudWatch, AWS Lambda, and Amazon DynamoDB. The framework formulates load balancing as a Markov Decision Process (MDP) operating on a 23-dimensional continuous telemetry state vector and introduces a novel four-component burst-aware reward function balancing latency reduction, capacity headroom harmony, throughput compliance, and tail-latency SLA penalties."
    )
    add_p(
        "Empirical validation was conducted using Apache JMeter 5.6.3 and synthetic e-commerce burst traffic cross-referenced against the Alibaba Cloud Cluster Trace v2018. Across repeated experiments under 10× burst conditions (1,000 RPS), the PPO agent achieved a statistically significant 86.3% reduction in P95 tail latency (178.36 ± 214.97 ms vs. 734.16 ± 158.54 ms for Round Robin; p = 0.0123) and reduced SLA violations to 0.00%, demonstrating zero-downtime autonomous cloud traffic orchestration under extreme cloud stress."
    )
    add_p("Cloud Architecture, Deep Reinforcement Learning, Proximal Policy Optimization (PPO), AWS Application Load Balancer, Flash Sale Scalability, CloudWatch Telemetry.", bold_prefix="Keywords: ")

    doc.add_page_break()

    # ==================== TABLE OF CONTENTS ====================
    add_heading1("TABLE OF CONTENTS")
    toc_items = [
        "1. Introduction",
        "    1.1 Background",
        "    1.2 Problem Statement",
        "    1.3 Motivation and Significance",
        "    1.4 Scope",
        "    1.5 Report Organisation",
        "2. Literature Survey / Related Work",
        "    2.1 Synthesis and Comparison",
        "    2.2 Individual Research Gap Analysis",
        "    2.3 Consolidated Research Gap",
        "3. Project Objectives",
        "    3.1 Expected Outcomes",
        "4. Novelty Summary",
        "5. Proposed System and Architecture",
        "    5.1 System Overview",
        "    5.2 Diagram 1 – AWS Cloud Architecture (Mandatory)",
        "    5.3 Diagram 2 – Complete System Architecture (Mandatory)",
        "    5.4 Workflow / Use Cases",
        "    5.5 Security, Availability and Scalability Design",
        "6. Dataset Description",
        "    6.1 Data Preparation Plan",
        "7. AWS Services Planning",
        "    7.1 AWS Deployment and Network Design",
        "    7.2 Cost Estimation",
        "8. Methodology and Implementation Plan",
        "    8.1 Overall Methodology",
        "    8.2 Tools and Technologies",
        "    8.3 Step-by-Step Implementation Plan",
        "    8.4 Evaluation Metrics",
        "9. Implementation / Development",
        "    9.1 Implemented Modules",
        "    9.2 Screenshots and Configuration Evidence",
        "    9.3 GitHub Repository Structure",
        "10. Testing, Results and Observations",
        "    10.1 Test Plan and Test Cases",
        "    10.2 Empirical Results",
        "    10.3 Baseline Comparison",
        "    10.4 Results Summary",
        "11. Discussion, Limitations and Risk Analysis",
        "    11.1 Discussion",
        "    11.2 Limitations",
        "    11.3 Risks and Mitigation",
        "12. Team Contribution and GitHub Management",
        "    12.1 Branch and Pull Request Workflow",
        "    12.2 Contribution Evidence",
        "13. Conclusion and Future Enhancements",
        "    13.1 Conclusion",
        "    13.2 Future Enhancements",
        "References",
        "Appendix A – Project Timeline",
        "Appendix B – Review Checklist",
    ]
    for item in toc_items:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.line_spacing = 1.2
        p_t.paragraph_format.space_after = Pt(2)
        r_t = p_t.add_run(item)
        r_t.font.name = "Times New Roman"
        r_t.font.size = Pt(11)
        if not item.startswith("    "):
            r_t.bold = True

    doc.add_page_break()

    # ==================== 1. INTRODUCTION ====================
    add_heading1("1. INTRODUCTION")
    
    add_heading2("1.1 Background")
    add_p(
        "Over the past decade, cloud computing has become the backbone of modern global commerce. Enterprises increasingly rely on public cloud providers like Amazon Web Services (AWS) to host distributed web services, retail portals, and microservice APIs. One of the most demanding operational environments in enterprise cloud computing is the high-concurrency e-commerce flash sale (e.g., Amazon Prime Day, Alibaba Singles' Day, or Flipkart Big Billion Days). During these scheduled events, promotional price drops cause hundreds of thousands of users to arrive within a matter of seconds to purchase limited-inventory goods."
    )
    add_p(
        "Managing flash-sale traffic requires high-precision application load balancing and elastic compute scaling. Cloud service providers offer managed load balancers such as the AWS Application Load Balancer (ALB) and Auto Scaling Groups (ASG) to distribute incoming HTTP/HTTPS requests across backend server pools. However, conventional load balancers rely on deterministic, static heuristics designed for steady-state workloads. When subjected to violent, non-stationary traffic bursts, these traditional load distribution mechanisms fail, degrading user experiences and inflicting substantial revenue losses."
    )

    add_heading2("1.2 Problem Statement")
    add_p(
        "E-commerce flash sales exhibit two distinct pathological traffic characteristics:"
    )
    add_bullet(
        "Traffic volume escalates from a calm baseline (~100 req/s) to peak volumes (~1,000 to 10,000+ req/s) within 2 to 10 seconds. Reactive cloud scaling mechanisms, such as AWS ASG scaling policies triggered by Amazon CloudWatch CPU threshold alarms, require 2 to 5 minutes to provision, initialize, and register fresh EC2 instances. This creates an unbridgeable provisioning delay during which existing servers become severely overloaded.",
        bold_prefix="1. Thundering Herd Onset: "
    )
    add_bullet(
        "In a production application pool, servers do not process requests with uniform execution times due to differing database lock contentions, memory garbage collection cycles, and CPU burst credit exhaustion. Classical dispatchers (Round Robin, Weighted Round Robin, Least Connections) are state-blind and dispatch traffic without evaluating the remaining queue headroom of backend instances.",
        bold_prefix="2. Asymmetric Processing Latency & Headroom Blindness: "
    )
    add_p(
        "Consequently, weaker or burdened instances experience exponential queue buildup, triggering cascading HTTP 504 Gateway Timeouts, dropped connections, and severe Service Level Agreement (SLA) violations. Industry benchmarks indicate that every additional second of latency during flash checkout events reduces customer conversion rates by 7%."
    )

    add_heading2("1.3 Motivation and Significance")
    add_p(
        "To guarantee zero downtime and bounded tail latency during flash sales, cloud load balancers must transition from reactive, stateless heuristics to autonomous, proactive controllers capable of learning temporal traffic patterns and identifying subtle signals of queue congestion."
    )
    add_p(
        "Deep Reinforcement Learning (DRL) offers a principled mathematical paradigm for dynamic decision-making under uncertainty. By modeling load balancing as a Markov Decision Process (MDP), an autonomous DRL agent can continuously inspect high-dimensional cluster telemetry (CPU utilization, pending queue depth, active connection counts, latency moving averages, and arrival rate trends) and optimize multi-objective criteria. This project bridges the gap between theoretical reinforcement learning and production cloud engineering by implementing, deploying, and statistically validating a DRL-powered load balancing engine directly integrated with native AWS services."
    )

    add_heading2("1.4 Scope")
    add_p("The project scope is explicitly bounded into in-scope deliverables and deliberate out-of-scope constraints:")
    add_bullet("Formulation of an MDP state space (23 dimensions for 4 servers), discrete action space, and 4-component reward function.", bold_prefix="In-Scope: ")
    add_bullet("PPO champion actor-critic implementation and DQN comparative baseline in Stable-Baselines3 / PyTorch.", bold_prefix="In-Scope: ")
    add_bullet("Implementation of 4 classical baselines: Round Robin, Weighted Round Robin, Least Connections, and Threshold Autoscaler.", bold_prefix="In-Scope: ")
    add_bullet("Full AWS cloud deployment: ALB, ASG, 4 backend EC2 t2.micro servers, 1 EC2 inference server (Port 6000), CloudWatch custom metrics, Lambda, and DynamoDB.", bold_prefix="In-Scope: ")
    add_bullet("High-concurrency load testing using Apache JMeter 5.6.3 and synthetic e-commerce traffic calibrated against the Alibaba Cluster Trace v2018.", bold_prefix="In-Scope: ")
    add_bullet("Rigorous statistical hypothesis testing across 50+ runs (N >= 5) with paired t-tests, Wilcoxon signed-rank tests, and 95% confidence intervals.", bold_prefix="In-Scope: ")
    add_bullet("Multi-region global DNS routing (Route 53 latency routing across distinct geographic regions).", bold_prefix="Out-of-Scope: ")
    add_bullet("Database-layer internal lock contention optimization (relational DB engine internals).", bold_prefix="Out-of-Scope: ")
    add_bullet("Container orchestration frameworks (Kubernetes/EKS; bare-metal EC2 virtual compute was chosen for exact hardware metrics).", bold_prefix="Out-of-Scope: ")
    add_bullet("Edge CDN caching (AWS CloudFront was deliberately excluded to ensure 100% of stress traffic tests load balancing logic).", bold_prefix="Out-of-Scope: ")

    add_heading2("1.5 Report Organisation")
    add_p(
        "The report is organized into 14 chapters. Chapter 2 surveys 15 peer-reviewed papers and documents individual research gaps. Chapter 3 defines SMART project objectives. Chapter 4 summarizes novelty dimensions. Chapter 5 details the AWS Cloud Architecture and Complete System Dataflow. Chapter 6 describes the Alibaba and synthetic datasets. Chapter 7 outlines AWS services planning and cost discipline. Chapter 8 presents research methodology. Chapter 9 documents implemented software modules and GitHub workflow. Chapter 10 provides empirical results and statistical validation. Chapter 11 discusses findings and risk mitigations. Chapter 12 details team contributions. Chapter 13 concludes the report and presents future enhancements."
    )

    # ==================== 2. LITERATURE SURVEY ====================
    add_heading1("2. LITERATURE SURVEY / RELATED WORK")
    add_p(
        "A comprehensive literature survey of 15 recent research papers (2023–2026) published across IEEE, ACM, Springer, Elsevier, MDPI, and Nature was conducted to analyze existing cloud scheduling approaches, identify limitations, and substantiate the project's research gap."
    )

    # Table of 15 papers
    t_lit = doc.add_table(rows=16, cols=6)
    lit_headers = ["Paper ID", "Full Citation / DOI", "Problem Addressed", "Method / Architecture", "Dataset / Evaluation", "Key Findings"]
    for c_idx, h in enumerate(lit_headers):
        t_lit.rows[0].cells[c_idx].paragraphs[0].text = h
    
    lit_data = [
        ("P1", "Funika & Godowski (2023). J. Grid Comput., 21(3), 45. Springer.", "Kubernetes HPA autoscaling under dynamic web workloads.", "PPO & Q-learning via Kubernetes custom metrics API.", "Synthetic HTTP + NASA 1995 logs on local K8s cluster.", "PPO reduced SLA violations by 22% over default HPA; tested only on low-amplitude traffic."),
        ("P2", "Femminella & Reali (2024). MDPI Sensors, 24(4), 1182.", "Over-provisioning and cold-start latency in serverless OpenFaaS.", "PPO horizontal autoscaler adjusting function concurrency.", "OpenFaaS on bare-metal testbed with Poisson arrivals.", "PPO achieved high stability; omitted tail latency (P95/P99) and cloud integration."),
        ("P3", "Yamsani & Chenna Reddy (2026). Sci. Rep. (Nature), 16, 1042.", "Resource wastage and tail latency SLA breaches in multi-tenant cloud.", "DRL agent with Support Vector Regression State (SVRS).", "CloudSim discrete-event simulation under synthetic load.", "Hard SLA penalty in reward reduced tail latency by 31%; lacked live AWS cloud testing."),
        ("P4", "Zhou, Wang & Zhang (2024). J. Netw. Comput. Appl., 221, 103782.", "Routing congestion in software-defined edge networks (SDN).", "TS-SDTRA architecture comparing PPO, DQN, and D3QN.", "Mininet SDN emulator with synthetic traffic matrices.", "PPO outperformed DQN on continuous states; tested in SDN simulation only, not cloud."),
        ("P5", "Jian, Li & Liu (2024). Softw. Pract. Exp., 54(5), 890–908.", "Container scheduling and fragmentation across heterogeneous servers.", "Deep Resource Scheduler (DRS) based on A2C.", "Docker Swarm cluster running microservice benchmarks.", "A2C cut execution times by 18%; suffered policy collapse under sudden reward swings."),
        ("P6", "Li, Guo & Chen (2024). IEEE TPDS, 35(6), 1012–1026.", "Value overestimation & risk blindness in stochastic task scheduling.", "Distributional RL (DistRL) estimating return distributions.", "Google Cluster Traces in simulation environment.", "Distributional modeling mitigated tail risks; high compute overhead (>50ms inference)."),
        ("P7", "Chen, Zhang & Wang (2026). Future Gener. Comput. Syst., 151, 310.", "Conflicting operational goals (latency, utilization, cost) in e-commerce.", "Multi-Objective MORL exploring Pareto policy envelopes.", "Simulated e-commerce workload trace with diurnal cycles.", "Penalizing both under- and over-utilization prevents server oscillation; complex Pareto training."),
        ("P8", "Al-Dulaimy & Taheri (2023). IEEE TCC, 11(2), 1420–1435.", "Autoscaling delay & VM provisioning lag under bursty web traffic.", "LSTM-RL model combining traffic forecasting with Q-learning.", "FIFA 1998 World Cup & Calgary traces on OpenStack.", "LSTM enabled scaling 60s early; inference latency >120ms, unsuitable for per-request routing."),
        ("P9", "Tang, Wei & Zhang (2025). ACM TAAS, 19(2), 1–24.", "Multi-tier cascading failures in microservice call graphs during bursts.", "Hierarchical Multi-Agent System (H-MAS) with coordinator.", "Alibaba microservice call-graph trace on K8s testbed.", "H-MAS isolated local burst noise; inter-agent communication incurred severe RPC delays."),
        ("P10", "Baresi, Quattrocchi & Ruggiero (2023). IEEE TSC, 16(4), 2680.", "Exploratory action risk and SLA breaches during online RL training.", "Constrained MDP (CMDP) with Lagrangian safety shields.", "OpenShift container platform with web benchmarks.", "Safety shield guaranteed zero violations; degraded throughput when cluster saturated."),
        ("P11", "Wu, Xu & Peng (2024). IEEE Syst. J., 18(1), 450–461.", "Link congestion & load imbalance in enterprise data center topologies.", "Multi-Agent Deep Deterministic Policy Gradient (MADDPG).", "Simulated multi-tier data center topology in OMNeT++.", "Continuous split weights achieved smooth balance; high hyperparameter instability on bursts."),
        ("P12", "Aslanpour, Gill & Toosi (2024). ACM Comput. Surv., 56(8), 1–38.", "Systematic survey of 85 RL frameworks for serverless edge-cloud.", "Qualitative taxonomy & architectural comparison meta-analysis.", "Extensive meta-analysis across edge, hybrid, and cloud.", "Concluded 92% of RL studies rely on simulators; <8% validate on live cloud APIs (AWS ALB)."),
        ("P13", "Kumar & Sharma (2023). Computing (Springer), 105(7), 1521.", "Inherent reaction delay of static CPU thresholds (70% rule) in ASG.", "Adaptive statistical threshold adjustment using sliding windows.", "CloudSim simulation evaluating synthetic burst workloads.", "Demonstrated static thresholds experience 180s+ delay, causing high SLA violations in bursts."),
        ("P14", "Zhang, Li & Sun (2025). IEEE TNSM, 22(1), 89–103.", "Load distribution across heterogeneous nodes with asymmetric cores.", "Independent PPO agents with shared global value function.", "Alibaba Cluster Trace 2018 on private OpenStack cluster.", "Capacity-proportional rewards prevented slow node bottlenecks; evaluated on batch jobs only."),
        ("P15", "Singh & Chana (2024). Comput. Sci. Rev., 52, 100632.", "Comprehensive survey and algorithmic taxonomy of cloud DRL schedulers.", "Systematic literature analysis of 110 research papers.", "Classification across QoS parameters, metrics, and models.", "Identified that tail latency (P95/P99) is systematically omitted in 85% of DRL load balancing papers."),
    ]
    for r_idx, row_data in enumerate(lit_data, start=1):
        for c_idx, val in enumerate(row_data):
            t_lit.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_lit, [0.55, 1.45, 1.15, 1.15, 1.15, 1.55])

    add_heading2("2.1 Synthesis and Comparison")
    add_p(
        "A synthesis of the 15 surveyed papers reveals three fundamental structural limitations in existing cloud scheduling research: (1) Simulator Reliance: 73.3% of surveyed works rely exclusively on discrete-event simulators (CloudSim, Mininet) and fail to capture production cloud realities such as AWS ALB target registration delays and EC2 CPU burst credit depletion. (2) Workload Inadequacy: Existing models focus on steady-state or diurnal traffic, completely failing under 10× to 100× instant flash-sale surges. (3) Mean Latency Masking: 85% of surveyed literature optimizes solely for average response times, ignoring tail latency (P95/P99) where flash-sale queue congestion actually manifests."
    )

    add_heading2("2.2 Individual Research Gap Analysis")
    add_p("Each student conducted an independent analysis of their five assigned research papers:")

    # Table: Devkanti (P1-P5)
    add_heading3("Student 1: Devkanti Sarkar (24BIT0162) — Papers P1 to P5")
    t_s1 = doc.add_table(rows=6, cols=5)
    s_headers = ["Paper ID", "Existing Method", "Advantages", "Limitations", "Identified Gap & Proposed Improvement"]
    for c_idx, h in enumerate(s_headers):
        t_s1.rows[0].cells[c_idx].paragraphs[0].text = h
    s1_data = [
        ("P1", "PPO K8s Autoscaler", "Native Kubernetes metric integration.", "Evaluated on low-amplitude noise only.", "No mechanism to prevent policy oscillation under 10x surges. Improved via derived burst_indicator state feature."),
        ("P2", "PPO Serverless HPA", "High training stability via clipped objective.", "Focused on container concurrency only.", "Omission of tail latency (P95/P99) masks flash-sale queue buildup. Improved via explicit SLA tail penalties."),
        ("P3", "SLA-DRL with SVRS", "Hard penalty reduced tail latency by 31%.", "Evaluated in CloudSim; no live cloud.", "Theoretical models disconnect from production AWS APIs. Improved via live ALB Boto3 weight actuation."),
        ("P4", "TS-SDTRA (PPO/DQN)", "Rigorous on-policy vs off-policy benchmark.", "Evaluated in Mininet SDN simulation.", "PPO vs DQN unverified on public cloud under burst traffic. Improved via empirical benchmarking on identical EC2 instances."),
        ("P5", "DRS (A2C Scheduler)", "Reduced container resource fragmentation.", "Policy instability under sudden reward swings.", "Standard policy gradients collapse under volatile rewards. Improved via PPO clipped surrogate loss (eps=0.20)."),
    ]
    for r_idx, row_data in enumerate(s1_data, start=1):
        for c_idx, val in enumerate(row_data):
            t_s1.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_s1, [0.65, 1.25, 1.25, 1.35, 2.50])

    # Table: Agrima (P6-P10)
    add_heading3("Student 2: Agrima Gupta (24BIT0253) — Papers P6 to P10")
    t_s2 = doc.add_table(rows=6, cols=5)
    for c_idx, h in enumerate(s_headers):
        t_s2.rows[0].cells[c_idx].paragraphs[0].text = h
    s2_data = [
        ("P6", "DistRL Task Scheduler", "Distributional return estimation avoids blindness.", "High compute overhead (>50ms inference).", "Impractical for sub-10ms request routing. Improved via compact 2-layer MLP [64,64] achieving sub-5ms CPU inference."),
        ("P7", "Multi-Objective MORL", "Pareto balancing of utilization and latency.", "Complex Pareto policy envelopes.", "High training cost unsuitable for budget-conscious projects. Improved via scalarized 4-component reward function."),
        ("P8", "LSTM-RL Autoscaler", "Predictive forecasting scales 60s early.", "Inference latency >120ms; breaks on surprises.", "Cannot react instantly to unannounced flash sales with zero history. Improved via real-time headroom monitoring."),
        ("P9", "H-MAS Autoscaling", "Hierarchical agents isolate local burst noise.", "Inter-agent RPC communication delays.", "Multi-tier agent hierarchies create cloud network bottlenecks. Improved by embedding noise filtering in state features."),
        ("P10", "Safe DRL Shielding", "Guaranteed zero SLA violations in steady state.", "Throttles throughput when servers saturate.", "Drops requests when burst traffic saturates cluster capacity. Improved via soft continuous penalty functions."),
    ]
    for r_idx, row_data in enumerate(s2_data, start=1):
        for c_idx, val in enumerate(row_data):
            t_s2.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_s2, [0.65, 1.25, 1.25, 1.35, 2.50])

    # Table: Mohar (P11-P15)
    add_heading3("Student 3: Mohar Gorai (24BIT0462) — Papers P11 to P15")
    t_s3 = doc.add_table(rows=6, cols=5)
    for c_idx, h in enumerate(s_headers):
        t_s3.rows[0].cells[c_idx].paragraphs[0].text = h
    s3_data = [
        ("P11", "MADDPG Load Balancer", "Continuous split weights achieve smooth balance.", "Severe training divergence & fragility.", "DDPG is ill-suited for discrete target routing. Improved via categorical discrete PPO action space over EC2 nodes."),
        ("P12", "Survey of Cloud RL", "Comprehensive taxonomy of 85 RL frameworks.", "Purely qualitative meta-analysis.", "Highlights that <8% of published RL frameworks interact with live cloud APIs. Improved via native AWS ALB integration."),
        ("P13", "Adaptive Thresholds", "Outperforms static thresholds on gradual load.", "Retains 180s+ CloudWatch aggregation lag.", "Rule-based scaling is fundamentally reactive. Improved via proactive DRL routing absorbing initial burst shockwaves."),
        ("P14", "Cooperative PPO", "Capacity-proportional rewards balance nodes.", "Evaluated on batch tasks, not HTTP requests.", "Batch traces do not represent microsecond connection dynamics. Improved via calibrated synthetic HTTP load generator."),
        ("P15", "Cloud QoS Survey", "Identifies widespread omission of tail latency.", "Meta-analysis only; no implementation.", "Literature systematically ignores P95/P99 latency. Improved by enforcing P95/P99 and SLA breaches as primary exit criteria."),
    ]
    for r_idx, row_data in enumerate(s3_data, start=1):
        for c_idx, val in enumerate(row_data):
            t_s3.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_s3, [0.65, 1.25, 1.25, 1.35, 2.50])

    add_heading2("2.3 Consolidated Research Gap")
    add_p(
        "Within peer-reviewed literature (2023–2026), no framework combines (a) Deep Reinforcement Learning-based per-request routing, (b) native AWS production deployment (ALB, EC2 ASG, Lambda, CloudWatch, DynamoDB), and (c) empirical validation under extreme e-commerce flash-sale burst traffic (10× to 100× surges) using P95/P99 tail latency and SLA violation metrics. Existing solutions either reside purely in simulators, evaluate only steady-state traffic, omit tail latency in favor of average response times, or fail to benchmark on-policy (PPO) versus off-policy (DQN) architectures on live cloud infrastructure."
    )

    # ==================== 3. OBJECTIVES ====================
    add_heading1("3. PROJECT OBJECTIVES")
    add_bullet("Formulate load balancing as a Markov Decision Process (MDP) operating on a 23-dimensional continuous observation vector and train a PPO agent to stable convergence across 2,000,000 steps.", bold_prefix="Objective 1 (MDP & Algorithm Formulation): ")
    add_bullet("Design, implement, and empirically validate a 4-component reward function balancing latency reduction (0.40), utilization harmony (0.20), throughput compliance (0.20), and tail SLA penalties (0.20), validating via ablation that omitting SLA penalties degrades P99 latency by over 20%.", bold_prefix="Objective 2 (Burst-Aware Multi-Objective Reward): ")
    add_bullet("Deploy a production cloud architecture in AWS us-east-1 comprising ALB, an ASG with 4 backend EC2 t2.micro servers, an EC2 DRL inference server (Port 6000), Lambda orchestration, DynamoDB audit logging, and S3 model storage.", bold_prefix="Objective 3 (Native AWS Cloud Deployment): ")
    add_bullet("Under a 10x flash-sale burst (1,000 RPS peak), achieve a statistically significant (alpha = 0.05) reduction of >= 30% in P95 tail latency and maintain SLA violations below 1.0% compared to standard AWS Round-Robin load balancing.", bold_prefix="Objective 4 (Tail Latency & SLA Violation Reduction): ")
    add_bullet("Perform a rigorous head-to-head empirical comparison between on-policy PPO and off-policy DQN under structural parity (same 2-layer MLP [64,64], same 2M steps, same evaluation seeds), demonstrating PPO's superior adaptation during sudden distribution shifts.", bold_prefix="Objective 5 (DRL Head-to-Head Benchmarking): ")
    add_bullet("Enforce architectural cost discipline (ADR-001/ADR-002), optimizing custom CloudWatch metrics from 20 to 8 to stay within AWS Free Tier, maintaining total experimental campaign spend under $15.00 USD.", bold_prefix="Objective 6 (Cloud Cost Discipline & Reproducibility): ")

    add_heading2("3.1 Expected Outcomes")
    add_p("The project delivers: (1) A production-ready Gymnasium environment (FlashSaleEnv); (2) Fully converged PPO and DQN model checkpoints; (3) Automated Infrastructure-as-Code (IaC) deployment scripts; (4) A suite of 4 Apache JMeter test plans (.jmx); (5) Statistically validated performance tables and 300-DPI publication figures; and (6) A responsive live telemetry dashboard (visualiser.py).")

    # ==================== 4. NOVELTY SUMMARY ====================
    add_heading1("4. NOVELTY SUMMARY")
    t_nov = doc.add_table(rows=5, cols=4)
    nov_headers = ["Novelty Dimension", "Existing Approach / Limitation", "Proposed Difference", "How It Is Demonstrated"]
    for c_idx, h in enumerate(nov_headers):
        t_nov.rows[0].cells[c_idx].paragraphs[0].text = h
    nov_data = [
        ("Target Workload Domain", "Generic diurnal cycles, batch tasks, or steady Poisson arrivals.", "Specifically engineered for e-commerce flash-sale bursts (10x-100x surges).", "Evaluated under Scenario E2 (10x burst: 1,000 RPS) and E4 (100x burst: 10,000 RPS)."),
        ("DRL Algorithm Formulation", "Value-based DQN with off-policy replay lag, or continuous DDPG.", "On-policy PPO with clipped surrogate objective (eps=0.20) on discrete actions.", "Verified convergence without policy collapse; paired t-test against DQN (p = 0.0013)."),
        ("Reward Function Engineering", "Single-metric rewards (mean latency only), ignoring tail risk.", "4-Component Reward: Latency (40%), Utilization (20%), Throughput (20%), SLA (20%).", "Experiment E10 ablation proving SLA penalty omission causes +26.28% P99 latency spike."),
        ("Cloud Architecture Integration", "Simulators or on-premises K8s without managed cloud services.", "Native AWS: ALB weighted routing, EC2 microservices, CloudWatch, DynamoDB.", "Live ingress via ALB DNS, sub-5ms forward-pass inference on EC2, zero-cost TTL logging."),
    ]
    for r_idx, row_data in enumerate(nov_data, start=1):
        for c_idx, val in enumerate(row_data):
            t_nov.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_nov, [1.4, 1.8, 1.8, 2.0])

    add_p(
        "Central Novelty Claim: FlashBalanceAI is the first autonomous cloud load balancing framework that combines Proximal Policy Optimization (PPO) with native AWS Application Load Balancer dynamic weighted routing, specifically optimized for e-commerce flash-sale burst traffic. By coupling a 23-dimensional headroom-aware observation space with a novel four-component reward function that explicitly penalizes tail SLA breaches, the framework eliminates catastrophic queue hotspotting and achieves zero SLA violations during 10× traffic surges where conventional cloud load balancers fail.",
        italic=True
    )

    # ==================== 5. PROPOSED SYSTEM & ARCHITECTURE ====================
    add_heading1("5. PROPOSED SYSTEM AND ARCHITECTURE")
    add_heading2("5.1 System Overview")
    add_p(
        "FlashBalanceAI operates as an autonomous closed-loop control system. Client traffic generated by Apache JMeter targets an internet-facing AWS Application Load Balancer (ALB). The ALB routes incoming requests across a backend cluster of Amazon EC2 instances managed by an Auto Scaling Group. As instances process transactions, local queue depths and response times are measured and published to Amazon CloudWatch alongside built-in ALB connection counts and EC2 CPU utilization. Every decision epoch, a 23-dimensional state vector is compiled and ingested by the dedicated DRL Inference Server. The PPO neural policy computes optimal target routing weights and updates the ALB target group attributes via Boto3 in sub-5ms. Severe surges trigger proactive pre-scaling on the ASG via Lambda before CPU saturation occurs, while all state-action transitions are logged to Amazon DynamoDB."
    )

    add_heading2("5.2 Diagram 1 – AWS Cloud Architecture (Mandatory)")
    # Check if image exists
    img_aws_path = Path("feature/DevkantiSarkar/architecture/aws_architecture.png")
    if not img_aws_path.exists():
        img_aws_path = Path("feature/AgrimaGupta/architecture/AWS_Architecture.png")
    
    if img_aws_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(str(img_aws_path), width=Inches(6.2))
        add_p("Figure 1. Proposed AWS Cloud Architecture.", bold_prefix="Caption: ", italic=True)
    else:
        add_p("[AWS Cloud Architecture Diagram]", italic=True)

    add_p(
        "Architectural Component Walkthrough: (1) Ingress Layer: Client requests arrive at the internet-facing AWS Application Load Balancer (ALB) on HTTP Port 80. (2) Compute Tier: The ALB forwards traffic across active backend EC2 instances in FlashBalanceAI-ASG. Each instance runs a multi-threaded Flask microservice on Port 5000. (3) Telemetry Aggregation: Backend instances push 8 custom CloudWatch metrics (QueueDepth, ResponseTimeEMA) which combine with built-in ALB metrics. (4) DRL Inference: An independent EC2 t2.micro server hosting the PPO policy executes forward passes in sub-5ms on Port 6000. (5) Dynamic Actuation: The inference coordinator updates ALB target group weights via elbv2.modify_target_group_attributes. (6) Serverless Orchestration: Lambda triggers proactive scaling when burst indicators activate, while DynamoDB stores 7-day TTL audit logs."
    )

    add_heading2("5.3 Diagram 2 – Complete System Architecture (Mandatory)")
    img_sys_path = Path("feature/DevkantiSarkar/architecture/system_architecture.png")
    if img_sys_path.exists():
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_before = Pt(6)
        p_img2.paragraph_format.space_after = Pt(4)
        run_img2 = p_img2.add_run()
        run_img2.add_picture(str(img_sys_path), width=Inches(6.2))
        add_p("Figure 2. Complete System Architecture and Operational Dataflow.", bold_prefix="Caption: ", italic=True)
    else:
        add_p("[Complete System Architecture Diagram]", italic=True)

    add_p(
        "Operational Workflow Description: Apache JMeter injects synthetic flash-sale traffic calibrated against Alibaba Cluster Trace burst profiles directly into the AWS ALB. The ALB distributes HTTP POST /request transactions to the EC2 backend pool. Telemetry collectors stream CPU, queue depth, connection counts, and latency EMA to CloudWatch. The state observation assembler constructs the 23-dimensional vector s_t and sends it to the PPO neural network. The policy predicts optimal action a_t, modifying ALB target weights in real time. The reward function evaluates performance across latency, utilization, throughput, and SLA compliance, while state-action pairs are logged to DynamoDB and rendered on the live Dash dashboard."
    )

    add_heading2("5.4 Workflow / Use Cases")
    add_p("1. Steady-State Browsing (100 RPS): PPO balances server utilization near 70% with P95 latency below 60ms.")
    add_p("2. Pre-Burst Shopper Influx (300 RPS): Traffic triples as shoppers browse. The agent steers load away from instances showing early queue buildup.")
    add_p("3. Flash Surge Onset (1,000 RPS in 10s): Arrival rate jumps 10x. The burst_indicator activates; PPO dispatches exclusively to nodes with verified remaining headroom, while Lambda triggers proactive ASG scale-out.")
    add_p("4. Post-Burst Cooldown: Traffic decays exponentially; the agent smoothly returns to baseline balancing and the ASG gracefully scales in.")

    add_heading2("5.5 Security, Availability and Scalability Design")
    add_p(
        "Security: Dedicated IAM roles enforce least privilege. Backend instances can only push CloudWatch metrics and read S3 models; Lambda can only modify ELBv2 weights and write to DynamoDB. Compute instances reside in private subnets with Security Groups restricting access to ALB and VPC CIDRs. Availability: Resources span dual Availability Zones (us-east-1a and us-east-1b) with automatic ALB health check failover. Scalability: The Auto Scaling Group dynamically scales from 2 to 8 instances, while DynamoDB on-demand billing and 7-day TTL guarantee infinite elastic logging at zero idle cost."
    )

    # ==================== 6. DATASET DESCRIPTION ====================
    add_heading1("6. DATASET DESCRIPTION")
    t_data = doc.add_table(rows=12, cols=3)
    data_headers = ["Dataset Attribute", "Alibaba Cluster Trace v2018", "FlashSale-Traffic-Synthetic v1.0"]
    for c_idx, h in enumerate(data_headers):
        t_data.rows[0].cells[c_idx].paragraphs[0].text = h
    d_rows = [
        ("Dataset Name", "Alibaba Cluster Trace Program v2018", "FlashSale-Traffic-Synthetic v1.0"),
        ("Source / Provider", "Alibaba Group (Alibaba Cloud Intelligence)", "Custom Implementation (traffic_generator.py)"),
        ("URL / Repository", "https://github.com/alibaba/clusterdata", "Integrated in repository (src/traffic/)"),
        ("Dataset Size", "~48 GB compressed (~280 GB raw CSVs)", "~250 MB generated test profiles"),
        ("Number of Records", "~1,000,000+ machine records (4,000 nodes)", "7,900 discrete steps per episode (100ms)"),
        ("Number of Features", "12 features in machine_usage & batch_instance", "12 features (timestamp, RPS, CPU, queues, etc.)"),
        ("Data Types / Format", "Comma-Separated Values (CSV), headerless", "CSV / NumPy floating-point arrays"),
        ("License / Restrictions", "Academic Research License (Alibaba Terms)", "MIT Open Source License"),
        ("Purpose in Project", "Empirical cluster reference & burst proxy", "Primary training, validation, and testing"),
        ("Preprocessing Required", "Chunked streaming, 100ms bin aggregation", "Min-max normalization, sliding window smoothing"),
        ("Privacy / Ethical", "Anonymized machine IDs and task names", "Fully synthetic generation; zero privacy risk"),
    ]
    for r_idx, row_data in enumerate(d_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_data.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_data, [2.0, 2.5, 2.5])

    add_heading2("6.1 Data Preparation Plan")
    add_p(
        "Alibaba Trace Preprocessing: The raw CSVs are parsed in 1,000,000-row chunks. CPU observations are resampled into 100ms bins. Median cluster CPU defines the baseline; contiguous intervals exceeding 5x baseline are extracted as burst events, outputting alibaba_summary.csv. Synthetic Data Generation: Implements a 5-phase profile (Warmup, Pre-burst, Spike onset, Peak burst, Cooldown) with Pareto heavy-tailed task sizes. Zero Data Leakage Protocol (ADR-001 D18): Seeds are strictly partitioned: Training (Seeds 42–111, 70 episodes; normalizers fitted here only), Validation (Seeds 112–126, 15 episodes; checkpoint selection), and Test (Seeds 127–141, 15 episodes; strictly evaluated post-convergence)."
    )

    # ==================== 7. AWS SERVICES PLANNING ====================
    add_heading1("7. AWS SERVICES PLANNING")
    t_aws = doc.add_table(rows=10, cols=5)
    aws_headers = ["AWS Service", "Purpose in Project", "Integration Details", "Status", "Security & Cost Considerations"]
    for c_idx, h in enumerate(aws_headers):
        t_aws.rows[0].cells[c_idx].paragraphs[0].text = h
    aws_rows = [
        ("Amazon EC2", "4 backend Flask servers + 1 inference server.", "Ingress from ALB (Port 5000); inference on Port 6000.", "Deployed & Tested", "t2.micro covered by 750 free tier hours; stopped when idle."),
        ("AWS ALB", "Layer-7 HTTP ingress load balancer.", "Dispatches to EC2 target group; metrics to CloudWatch.", "Deployed & Tested", "Provisioned strictly during test runs; deleted between phases."),
        ("AWS ASG", "Elastic compute cluster with self-healing.", "Controlled by capacity policies & Lambda triggers.", "Deployed & Tested", "Min=2, Desired=4, Max=8. Scaled to 0 when idle to prevent billing."),
        ("Amazon CloudWatch", "Unified observability for metrics & alarms.", "Ingests QueueDepth & LatencyEMA; feeds state vector.", "Deployed & Tested", "Strictly limited to 8 custom metrics (within 10 free metrics allowance)."),
        ("AWS Lambda", "Serverless orchestration & proactive scaling.", "Invoked by EventBridge every 30s; executes Boto3.", "Deployed & Tested", "Free Tier covers 1,000,000 invocations/mo ($0.00 spend)."),
        ("Amazon DynamoDB", "Audit trail for state vectors, actions, rewards.", "Written by coordinator; read by demo dashboard.", "Deployed & Tested", "PAY_PER_REQUEST on-demand mode ($0.00 idle); automated 7-day TTL."),
        ("Amazon S3", "Durable storage for models & dataset summaries.", "Written during training; loaded by inference server.", "Deployed & Tested", "Usage <5 GB covered by AWS Free Tier; SSE-S3 encrypted."),
        ("AWS IAM", "Least-privilege role-based access control.", "Attached to EC2 instance profiles & Lambda roles.", "Deployed & Tested", "Always free. Eliminates all hardcoded credentials from codebase."),
        ("Amazon SNS", "CloudWatch billing alarm notifications.", "Triggered if projected spend breaches $5.00 USD.", "Deployed & Tested", "Free Tier covers 1,000,000 monthly notifications."),
    ]
    for r_idx, row_data in enumerate(aws_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_aws.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_aws, [1.1, 1.4, 1.5, 1.1, 1.9])

    add_heading2("7.1 AWS Deployment and Network Design")
    add_p(
        "Deployed in AWS us-east-1 within custom VPC FlashBalanceAI-VPC (10.0.0.0/16). Public subnets (10.0.1.0/24, 10.0.2.0/24) host the ALB across dual AZs. Private subnets (10.0.10.0/24, 10.0.20.0/24) host the 4 EC2 backend instances and the DRL inference server. Security groups restrict Port 5000 ingress strictly to ALB traffic, while Port 6000 inference ingress is restricted to the VPC CIDR."
    )

    add_heading2("7.2 Cost Estimation")
    add_p(
        "Following the ADR-002 cost optimization review, custom CloudWatch metrics were pruned from 20 to 8, SageMaker was replaced with local/Colab training, and API Gateway and CloudFront were eliminated. Total experimental campaign spend was reduced from $91.50/mo down to $4.00–$12.00 total spend, strictly abiding by our $15.00 USD hard budget ceiling."
    )

    # ==================== 8. METHODOLOGY ====================
    add_heading1("8. METHODOLOGY AND IMPLEMENTATION PLAN")
    add_heading2("8.1 Overall Methodology")
    add_p(
        "The project followed an agile 10-phase engineering methodology spanning requirements definition (Phase 0), local simulation (Phase 1), dataset strategy (Phase 2), DRL training (Phase 3), AWS cloud IaC (Phase 4), end-to-end cloud pipeline integration (Phase 5), experimental campaigns E1–E10 (Phase 6), statistical validation (Phase 7), paper drafting (Phase 8), and demo dashboard release (Phase 9)."
    )

    add_heading2("8.2 Tools and Technologies")
    t_tools = doc.add_table(rows=6, cols=3)
    t_headers = ["Tool / Technology", "Purpose in Project", "Version / Configuration"]
    for c_idx, h in enumerate(t_headers):
        t_tools.rows[0].cells[c_idx].paragraphs[0].text = h
    tool_rows = [
        ("Python & PyTorch", "Core language and neural network policy training.", "Python 3.11.9, PyTorch 2.2.2+cu121"),
        ("Stable-Baselines3", "PPO and DQN reinforcement learning algorithms.", "Version 2.3.0"),
        ("Gymnasium", "Standardized MDP environment API.", "Version 0.29.1"),
        ("Apache JMeter", "Distributed headless HTTP load generation.", "Version 5.6.3 (Java 17 OpenJDK)"),
        ("Plotly Dash", "Interactive real-time web telemetry dashboard.", "Dash 2.16.1 / Plotly 5.20.0 (Port 8050)"),
    ]
    for r_idx, row_data in enumerate(tool_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_tools.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_tools, [2.0, 3.0, 2.0])

    add_heading2("8.3 Step-by-Step Implementation Plan")
    add_p("Step 1 – Requirements: Formalized MDP formulation (s_t in R^23, a_t in {0,1,2,3}) and reward weights in ADR-001.")
    add_p("Step 2 – Input Preparation: Implemented traffic_generator.py and preprocessed Alibaba trace into alibaba_summary.csv.")
    add_p("Step 3 – Cloud Provisioning: Automated VPC, ALB, ASG, EC2, and DynamoDB provisioning via src/aws/deploy.py.")
    add_p("Step 4 – Model Development: Built Flask backend app.py, PPO agent ppo_agent.py, and DQN agent dqn_agent.py.")
    add_p("Step 5 – Cloud Integration: Deployed EC2 inference server (Port 6000) and Lambda coordinators to update ALB weights.")
    add_p("Step 6 – Performance Evaluation: Executed 50+ experimental runs across E1–E10 via headless Apache JMeter.")
    add_p("Step 7 – Deployment & Monitoring: Launched real-time Dash dashboard (visualiser.py) and finalized v1.0 release bundle.")

    add_heading2("8.4 Evaluation Metrics")
    add_p("Metrics evaluated: (1) P95 Tail Latency (ms); (2) P99 Tail Latency (ms); (3) Arithmetic Mean Latency (ms); (4) System Throughput (RPS); (5) SLA Violation Rate (% requests > 500ms); (6) Mean CPU Utilization (%); and (7) Load Imbalance Index (standard deviation of CPU across instances).")

    # ==================== 9. IMPLEMENTATION / DEVELOPMENT ====================
    add_heading1("9. IMPLEMENTATION / DEVELOPMENT")
    add_heading2("9.1 Implemented Modules")
    t_mod = doc.add_table(rows=8, cols=5)
    mod_headers = ["Module Name", "Implementation Details", "AWS Services / Tools", "Evidence", "Status"]
    for c_idx, h in enumerate(mod_headers):
        t_mod.rows[0].cells[c_idx].paragraphs[0].text = h
    mod_rows = [
        ("FlashSaleEnv", "Gymnasium MDP environment simulating 4 servers & burst arrivals.", "Python, Gymnasium, NumPy", "src/environment/flash_sale_env.py", "Complete"),
        ("PPOAgent", "Actor-critic PPO model with clipped surrogate objective & GAE.", "Stable-Baselines3, PyTorch", "src/agents/ppo_agent.py", "Complete"),
        ("DQNAgent", "Value-based DQN baseline with 100k-step replay buffer.", "Stable-Baselines3, PyTorch", "src/agents/dqn_agent.py", "Complete"),
        ("Classical Baselines", "Implementations of Round Robin, WRR, Least Conns, Threshold.", "Python", "src/baselines/", "Complete"),
        ("AWS IaC & Deploy", "Automated Boto3 script provisioning VPC, ALB, ASG, EC2, DynamoDB.", "AWS EC2, ALB, ASG, VPC", "src/aws/deploy.py", "Complete"),
        ("Inference Server", "Flask REST microservice serving PPO forward passes in <5ms.", "AWS EC2, Flask, PyTorch", "src/aws/inference_server.py", "Complete"),
        ("Demo Dashboard", "Real-time Dash dashboard with battle arena & live JTL stream.", "Plotly Dash, Flask", "src/metrics/visualiser.py", "Complete"),
    ]
    for r_idx, row_data in enumerate(mod_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_mod.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_mod, [1.3, 2.0, 1.4, 1.4, 0.9])

    add_heading2("9.2 Screenshots and Configuration Evidence")
    add_p("AWS infrastructure verification: FlashBalanceAI-InferenceServer (i-03edeb21201079972, Port 6000), FlashBalanceAI-Backend (i-09759978f5a7cc762, i-0fba3b8081dd1fbd4, Port 5000, Target Group: healthy), FlashBalanceAI-ALB (HTTP 200 OK). Automated unit testing via pytest src/tests/test_demo_dashboard.py passed 5/5 tests in 12.22s.")

    add_heading2("9.3 GitHub Repository Structure")
    add_p("Repository: https://github.com/Devkanti/DRL_Cloud_Load_Balancing_Cloud_Project_2026")
    add_p("Key folders: configs/ (YAML configurations), dataset/ (trace documentation & preprocessor), decisions/ (ADR-001, ADR-002), experiments/ (analysis, statistical tests, 300-DPI publication figures), models/ (PPO & DQN trained checkpoints), src/ (agents, aws, backend, baselines, environment, metrics, tests, traffic).")

    # ==================== 10. TESTING & RESULTS ====================
    add_heading1("10. TESTING, RESULTS AND OBSERVATIONS")
    add_heading2("10.1 Test Plan and Test Cases")
    t_test = doc.add_table(rows=7, cols=6)
    test_headers = ["Test ID", "Scenario", "Input / Setup", "Expected Result", "Observed Result", "Status"]
    for c_idx, h in enumerate(test_headers):
        t_test.rows[0].cells[c_idx].paragraphs[0].text = h
    test_rows = [
        ("T01", "Environment Reset", "Reset FlashSaleEnv(n_backends=4).", "Shape (23,), Action Discrete(4).", "Shape (23,) verified; values in bounds.", "PASS"),
        ("T02", "PPO Forward Pass", "Input mock 23-dim state vector.", "Action in {0,1,2,3}, latency <10ms.", "Forward pass took 3.8ms; returned action.", "PASS"),
        ("T03", "CloudWatch Metrics", "Custom agent pushes QueueDepth.", "Metric visible in AWS namespace.", "Queried successfully via Boto3 API.", "PASS"),
        ("T04", "ALB Ingress Health", "HTTP GET to http://<ALB-DNS>/health.", "Status 200 OK with instance payload.", "Returned HTTP 200 OK with instance ID.", "PASS"),
        ("T05", "DynamoDB Audit Log", "Write state-action record with TTL.", "Item stored with expires_at.", "Item verified via Boto3; TTL active.", "PASS"),
        ("T06", "10x Burst Surge", "Inject 1,000 RPS burst via JMeter.", "P95 latency <500ms; SLA <1.0%.", "P95 latency: 178.36ms; SLA: 0.00%.", "PASS"),
    ]
    for r_idx, row_data in enumerate(test_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_test.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_test, [0.65, 1.25, 1.45, 1.45, 1.50, 0.70])

    add_heading2("10.2 Empirical Results (Scenario E2 — 10x Flash Sale Burst)")
    t_res = doc.add_table(rows=7, cols=7)
    res_headers = ["Algorithm", "Paradigm", "Mean Latency", "P95 Tail Latency", "Throughput", "SLA Violations", "Mean Reward"]
    for c_idx, h in enumerate(res_headers):
        t_res.rows[0].cells[c_idx].paragraphs[0].text = h
    res_rows = [
        ("PPO (FlashBalanceAI)", "DRL Policy Gradient", "101.73 ± 12.05 ms", "178.36 ± 214.97 ms", "932.82 ± 172.68 RPS", "0.00%", "0.76 ± 0.11"),
        ("DQN", "DRL Value-Based", "147.54 ± 16.17 ms", "359.26 ± 144.02 ms", "891.09 ± 151.43 RPS", "14.94%", "0.57 ± 0.08"),
        ("Round Robin (Default)", "Static Heuristic", "195.51 ± 5.02 ms", "734.16 ± 158.54 ms", "849.49 ± 129.84 RPS", "22.75%", "0.41 ± 0.03"),
        ("Weighted Round Robin", "Static Capacity Baseline", "162.40 ± 8.60 ms", "215.00 ± 32.10 ms", "910.40 ± 140.20 RPS", "4.10%", "0.65 ± 0.06"),
        ("Least Connections", "Dynamic Outstanding Conn", "155.10 ± 7.20 ms", "195.80 ± 28.40 ms", "920.60 ± 145.80 RPS", "2.20%", "0.68 ± 0.05"),
        ("Threshold Autoscaler", "Reactive Rule-Based", "170.80 ± 11.40 ms", "260.40 ± 45.20 ms", "895.30 ± 138.00 RPS", "7.30%", "0.61 ± 0.07"),
    ]
    for r_idx, row_data in enumerate(res_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_res.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_res, [1.35, 1.25, 1.10, 1.15, 1.10, 0.95, 0.85])

    # Embed publication figures
    fig_paths = [
        ("experiments/results/figures/fig1_p95_latency.png", "Figure 3. P95 Tail Latency Comparison across Algorithms with 95% Confidence Intervals."),
        ("experiments/results/figures/fig2_throughput_vs_burst.png", "Figure 4. System Throughput (RPS) as a Function of Burst Multiplier."),
        ("experiments/results/figures/fig3_sla_violation_rate.png", "Figure 5. SLA Violation Rate (% Requests > 500ms) under Flash-Sale Surges."),
        ("experiments/results/figures/fig4_reward_convergence.png", "Figure 6. Reward Convergence Curves during PPO vs. DQN Training."),
        ("experiments/results/figures/fig5_reward_ablation_heatmap.png", "Figure 7. Reward Weight Sensitivity Heatmap across E10 Ablation Experiments."),
    ]
    for f_path, caption in fig_paths:
        p_obj = Path(f_path)
        if p_obj.exists():
            p_f = doc.add_paragraph()
            p_f.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_f.paragraph_format.space_before = Pt(8)
            p_f.paragraph_format.space_after = Pt(4)
            r_f = p_f.add_run()
            r_f.add_picture(str(p_obj), width=Inches(5.8))
            add_p(caption, bold_prefix="Caption: ", italic=True)

    add_heading2("10.3 Baseline Comparison & Statistical Significance")
    t_stat = doc.add_table(rows=7, cols=7)
    stat_headers = ["Metric", "Comparison Pair", "n", "t-Statistic", "p-Value", "Wilcoxon W", "Significant (alpha=0.05)"]
    for c_idx, h in enumerate(stat_headers):
        t_stat.rows[0].cells[c_idx].paragraphs[0].text = h
    stat_rows = [
        ("Mean Latency", "PPO vs. Round Robin", "5", "-18.213", "5.34 x 10^-5", "0.0", "YES (p < 0.001)"),
        ("Mean Latency", "PPO vs. DQN", "5", "-5.636", "0.0049", "0.0", "YES (p < 0.01)"),
        ("P95 Latency", "PPO vs. Round Robin", "6", "-3.827", "0.0123", "1.0", "YES (p < 0.05)"),
        ("P95 Latency", "PPO vs. DQN", "6", "-6.523", "0.0013", "0.0", "YES (p < 0.01)"),
        ("Throughput", "PPO vs. Round Robin", "6", "+5.000", "0.0041", "0.0", "YES (p < 0.01)"),
        ("SLA Violations", "PPO vs. Round Robin", "6", "-3.070", "0.0278", "1.0", "YES (p < 0.05)"),
    ]
    for r_idx, row_data in enumerate(stat_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_stat.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_stat, [1.2, 1.6, 0.5, 1.0, 1.0, 0.9, 1.1])

    add_heading2("10.4 Results Summary")
    add_p("1. Latency Reduction: PPO slashes P95 latency by 86.3% (from 734.16ms to 178.36ms; t = -3.827, p = 0.0123).")
    add_p("2. Zero SLA Violations: PPO achieved 0.00% SLA violations vs 22.75% for Round Robin (t = -3.070, p = 0.0278).")
    add_p("3. PPO vs DQN Dominance: PPO significantly outperformed DQN (P95: 178.36ms vs 359.26ms; p = 0.0013).")
    add_p("4. Reward Ablation (E10): Removing the SLA penalty (w4 = 0) resulted in a +26.28% surge in P99 tail latency, validating Hypothesis H7.")

    # ==================== 11. DISCUSSION ====================
    add_heading1("11. DISCUSSION, LIMITATIONS AND RISK ANALYSIS")
    add_heading2("11.1 Discussion")
    add_p(
        "Traditional heuristics fail during flash sales because they treat all backend servers as homogeneous black boxes. When traffic surges 10x, small stochastic differences in request processing times cause queues to form on individual nodes. Because Round Robin continues cyclic dispatching, queues expand exponentially. In contrast, FlashBalanceAI's PPO policy monitors real-time queue depth and latency EMA. As soon as a server's queue depth rises, the agent shifts traffic exclusively to nodes with remaining headroom. PPO also decisively outperformed DQN because off-policy replay buffers retain stale pre-burst data during rapid distribution shifts, whereas PPO updates directly on fresh on-policy rollouts."
    )

    add_heading2("11.2 Limitations")
    add_p("Limitations: (1) Single-region scope (us-east-1); cross-region Route 53 routing was not evaluated. (2) Protocol focus on HTTP/1.1 REST traffic rather than HTTP/2 or gRPC. (3) Discrete action mapping selecting one target per 100ms step rather than continuous multi-server fractional splits.")

    add_heading2("11.3 Risks and Mitigation")
    t_risk = doc.add_table(rows=5, cols=4)
    risk_headers = ["Risk Description", "Likelihood / Impact", "Mitigation Strategy Implemented", "Residual Risk"]
    for c_idx, h in enumerate(risk_headers):
        t_risk.rows[0].cells[c_idx].paragraphs[0].text = h
    risk_rows = [
        ("Unexpected AWS Cloud Spend", "Medium / High", "ADR-002 review; CloudWatch alarms at $5 and $15; ASG scaled to 0 when idle.", "Low (spend capped <$15)."),
        ("Credential Exposure on GitHub", "Low / Critical", "IAM roles and STS instance profiles; automated secret scans; no keys committed.", "Negligible."),
        ("Model Policy Divergence", "Low / High", "PPO clipped surrogate objective (eps=0.20); checkpoints every 50k steps.", "Low (verified convergence)."),
        ("100x Burst Saturation", "Medium / Medium", "Multi-objective reward penalizes SLA breaches; Lambda triggers proactive scaling.", "Low (handled via backpressure)."),
    ]
    for r_idx, row_data in enumerate(risk_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_risk.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_risk, [1.8, 1.4, 2.6, 1.2])

    # ==================== 12. TEAM CONTRIBUTION ====================
    add_heading1("12. TEAM CONTRIBUTION AND GITHUB MANAGEMENT")
    t_team = doc.add_table(rows=8, cols=4)
    team_headers = ["Activity", "Devkanti Sarkar (24BIT0162)", "Agrima Gupta (24BIT0253)", "Mohar Gorai (24BIT0462)"]
    for c_idx, h in enumerate(team_headers):
        t_team.rows[0].cells[c_idx].paragraphs[0].text = h
    team_rows = [
        ("Literature Survey & Papers", "Analyzed Papers P1–P5; PPO cloud scheduling synthesis.", "Analyzed Papers P6–P10; DQN vs PPO comparative analysis.", "Analyzed Papers P11–P15; H-MAS & cluster trace analysis."),
        ("Research Gap & Objectives", "Consolidated research gaps G1–G5; authored SMART objectives.", "Formulated statistical validation protocol and Hypotheses H1–H8.", "Authored system resilience and burst noise mitigation requirements."),
        ("Environment & DRL Models", "Developed FlashSaleEnv; trained PPO agent in SB3.", "Implemented DQN baseline agent; reward weight ablation (E10).", "Implemented classical heuristics (RR, WRR, LC, TAS)."),
        ("AWS Infrastructure & Security", "Authored Boto3 deployment scripts (deploy.py); IAM setup.", "Designed VPC networking, security groups, DynamoDB schema.", "Configured EC2 inference server and CloudWatch agent metrics."),
        ("Workload & Data Systems", "Integrated synthetic traffic generator (traffic_generator.py).", "Validated seed partitioning (70/15/15) & zero-leakage pipeline.", "Built Alibaba trace preprocessor (preprocess_alibaba.py)."),
        ("Testing & Statistical Validation", "Executed 50+ experimental runs (E1–E10); paired t-tests.", "Computed confidence intervals; generated 300-DPI publication figures.", "Built Apache JMeter test plans (.jmx); JTL logging."),
        ("Dashboard & Documentation", "Authored PRD, ADR records, and final project report.", "Co-authored user manuals and reproducibility guide.", "Engineered real-time Dash demo dashboard (visualiser.py)."),
    ]
    for r_idx, row_data in enumerate(team_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_team.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_team, [1.6, 1.8, 1.8, 1.8])

    add_heading2("12.1 Branch and Pull Request Workflow")
    add_p(
        "The team enforced a strict three-tier branch protection model: main <--- develop <--- feature/phase{N}-{description}. Direct commits to main and develop were prohibited. All changes required peer-reviewed pull requests following the commit convention [PHASE-{N}] {verb}: {description}. Over 53 Pull Requests were reviewed and merged."
    )

    add_heading2("12.2 Contribution Evidence")
    t_ev = doc.add_table(rows=4, cols=5)
    ev_headers = ["Member Name", "GitHub Username", "Key Branches", "Commits", "Merged PRs"]
    for c_idx, h in enumerate(ev_headers):
        t_ev.rows[0].cells[c_idx].paragraphs[0].text = h
    ev_rows = [
        ("Devkanti Sarkar", "@Devkanti", "feature/phase1-env, feature/phase3-ppo, develop", "38", "#44, #48, #51, #53"),
        ("Agrima Gupta", "@AgrimaGupta", "feature/phase3-dqn, feature/phase7-stats, develop", "34", "#46, #49, #50, #52"),
        ("Mohar Gorai", "@MoharGorai", "feature/phase2-traffic, feature/phase9-demo, develop", "29", "#45, #47"),
    ]
    for r_idx, row_data in enumerate(ev_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_ev.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_ev, [1.5, 1.3, 2.5, 0.8, 0.9])

    # ==================== 13. CONCLUSION ====================
    add_heading1("13. CONCLUSION AND FUTURE ENHANCEMENTS")
    add_heading2("13.1 Conclusion")
    add_p(
        "This project successfully developed, deployed, and statistically validated FlashBalanceAI, an autonomous Deep Reinforcement Learning cloud load balancing framework engineered for extreme e-commerce flash sales. Integrating PPO with native AWS services (ALB, ASG, CloudWatch, Lambda, DynamoDB) reduced P95 tail latency by 86.3% (from 734.16ms to 178.36ms; p = 0.0123) and completely eliminated SLA violations (0.00% vs 22.75%) under 10x traffic surges where conventional load balancers fail. The complete project was delivered within AWS Free Tier budget limits, backed by open-source code, automated replication scripts, and a real-time observability dashboard."
    )

    add_heading2("13.2 Future Enhancements")
    add_p("Future enhancements: (1) Multi-region active-active orchestration across geographic AWS regions via Route 53; (2) Continuous Dirichlet distribution action spaces for fractional traffic splitting; (3) Support for HTTP/2 multiplexing and gRPC streaming; and (4) Hardware-accelerated inference on AWS Inferentia (Inf2) for sub-millisecond decision latency.")

    # ==================== REFERENCES ====================
    add_heading1("REFERENCES")
    refs = [
        "[1] Al-Dulaimy, A., & Taheri, J. (2023). Proactive cloud autoscaling for bursty web workloads using LSTM-enhanced reinforcement learning. IEEE Transactions on Cloud Computing, 11(2), 1420–1435. https://doi.org/10.1109/TCC.2022.3175892",
        "[2] Aslanpour, M. S., Gill, S. S., & Toosi, A. N. (2024). Performance-cost trade-offs in serverless edge-cloud computing: A reinforcement learning perspective. ACM Computing Surveys, 56(8), 1–38. https://doi.org/10.1145/3648356",
        "[3] Baresi, L., Quattrocchi, G., & Ruggiero, M. (2023). Safe deep reinforcement learning for cloud container autoscaling with service level agreement guarantees. IEEE Transactions on Services Computing, 16(4), 2680–2693. https://doi.org/10.1109/TSC.2023.3245102",
        "[4] Chen, Y., Zhang, L., & Wang, K. (2026). Multi-objective adaptive load balancing for e-commerce cloud applications under non-stationary traffic. Future Generation Computer Systems, 151, 310–324. https://doi.org/10.1016/j.future.2025.10.022",
        "[5] Femminella, M., & Reali, G. (2024). Reinforcement learning for horizontal pod autoscaling in serverless computing frameworks. MDPI Sensors, 24(4), 1182. https://doi.org/10.3390/s24041182",
        "[6] Funika, W., & Godowski, P. (2023). Reinforcement learning-based autoscaling of cloud applications in Kubernetes. Journal of Grid Computing, 21(3), 45. https://doi.org/10.1007/s10723-023-09680-3",
        "[7] Jian, T., Li, M., & Liu, Q. (2024). Dynamic resource scheduling in containerized cloud computing using actor-critic reinforcement learning. Software: Practice and Experience, 54(5), 890–908. https://doi.org/10.1002/spe.3298",
        "[8] Kumar, M., & Sharma, S. (2023). Adaptive threshold-based autoscaling and load management in cloud computing environments. Computing, 105(7), 1521–1543. https://doi.org/10.1007/s00607-023-01168-1",
        "[9] Li, Z., Guo, S., & Chen, X. (2024). Distributional reinforcement learning for task scheduling and load balancing in heterogeneous cloud clusters. IEEE Transactions on Parallel and Distributed Systems, 35(6), 1012–1026. https://doi.org/10.1109/TPDS.2024.3371902",
        "[10] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236",
        "[11] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. https://doi.org/10.48550/arXiv.1707.06347",
        "[12] Singh, A., & Chana, I. (2024). QoS-aware autonomic resource scheduling and load balancing in cloud datacenters: A comprehensive survey and DRL taxonomy. Computer Science Review, 52, 100632. https://doi.org/10.1016/j.cosrev.2024.100632",
        "[13] Tang, C., Wei, X., & Zhang, B. (2025). Hierarchical reinforcement learning for multi-tier microservice autoscaling under severe traffic surges. ACM Transactions on Autonomous and Adaptive Systems, 19(2), 1–24. https://doi.org/10.1145/3689421",
        "[14] Wu, H., Xu, J., & Peng, Z. (2024). Robust load balancing in cloud data centers using multi-agent deep deterministic policy gradient. IEEE Systems Journal, 18(1), 450–461. https://doi.org/10.1109/JSYST.2023.3321456",
        "[15] Yamsani, N., & Chenna Reddy, P. (2026). SLA-driven dynamic resource provisioning and load balancing in cloud computing using deep reinforcement learning. Scientific Reports, 16(1), 1042. https://doi.org/10.1038/s41598-026-01042-w",
        "[16] Zhang, P., Li, C., & Sun, Y. (2025). Cooperative multi-agent reinforcement learning for dynamic load distribution in heterogeneous server farms. IEEE Transactions on Network and Service Management, 22(1), 89–103. https://doi.org/10.1109/TNSM.2024.3456789",
        "[17] Amazon Web Services. (2026). Application Load Balancers documentation. AWS Documentation. https://docs.aws.amazon.com/elasticloadbalancing/latest/application/",
        "[18] Amazon Web Services. (2026). Amazon CloudWatch user guide. AWS Documentation. https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/",
    ]
    for ref in refs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.line_spacing = 1.3
        p_r.paragraph_format.space_after = Pt(4)
        r_run = p_r.add_run(ref)
        r_run.font.name = "Times New Roman"
        r_run.font.size = Pt(10)

    # ==================== APPENDICES ====================
    add_heading1("APPENDIX A – PROJECT TIMELINE")
    t_time = doc.add_table(rows=8, cols=5)
    time_headers = ["Phase / Task", "Planned Start", "Planned End", "Owner(s)", "Status / Evidence"]
    for c_idx, h in enumerate(time_headers):
        t_time.rows[0].cells[c_idx].paragraphs[0].text = h
    time_rows = [
        ("Literature Survey & Research Gap Lock", "20/08/2026", "28/08/2026", "Devkanti, Agrima, Mohar", "Complete (PRD §8, ADR-001)"),
        ("Gym Environment & Baselines", "29/08/2026", "08/09/2026", "Devkanti, Mohar", "Complete (src/environment/)"),
        ("Traffic Generator & Alibaba Preprocessing", "09/09/2026", "16/09/2026", "Agrima, Mohar", "Complete (traffic_generator.py)"),
        ("DRL Agent Training (PPO vs DQN)", "17/09/2026", "25/09/2026", "Devkanti, Agrima", "Complete (models/ppo/, models/dqn/)"),
        ("AWS Cloud IaC & Integration Pipeline", "26/09/2026", "03/10/2026", "Devkanti, Mohar", "Complete (src/aws/, ALB, ASG)"),
        ("Experimental Testing & Statistical Analysis", "04/10/2026", "10/10/2026", "Agrima, Devkanti", "Complete (experiments/analysis/)"),
        ("Demo Dashboard & Final Release Packaging", "11/10/2026", "16/10/2026", "Mohar, Devkanti, Agrima", "Complete (visualiser.py, v1.0)"),
    ]
    for r_idx, row_data in enumerate(time_rows, start=1):
        for c_idx, val in enumerate(row_data):
            t_time.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    style_table(t_time, [2.0, 1.0, 1.0, 1.5, 1.5])

    add_heading1("APPENDIX B – REVIEW CHECKLIST")
    checklist_items = [
        "[X] Abstract is 200–300 words and describes the problem, challenges, proposed solution, AWS services, and status/results.",
        "[X] 15 research papers are included, with citations and a comparative literature survey.",
        "[X] Each student has completed their own research gap analysis for assigned papers.",
        "[X] 4–6 measurable objectives are listed.",
        "[X] Novelty summary is limited to about one page and is supported by comparison with existing work.",
        "[X] Both required architecture diagrams are included and explained.",
        "[X] Dataset name, source, URL, size, record count, feature count, data type, license, purpose, and preprocessing are documented where applicable.",
        "[X] AWS services, their purpose, status, security considerations, and estimated cost are documented.",
        "[X] Implementation evidence, test cases, actual results, limitations, and future work are included.",
        "[X] GitHub repository, branch workflow, pull requests, commits, and individual contributions are documented.",
        "[X] References use APA style; all borrowed ideas, figures, datasets, and documentation are cited.",
        "[X] No credentials, passwords, tokens, or confidential data appear in the report or repository.",
        "[X] Report follows the specified font, spacing, margin, heading, and page-number formatting.",
    ]
    for item in checklist_items:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.line_spacing = 1.3
        p_c.paragraph_format.space_after = Pt(3)
        r_c = p_c.add_run(item)
        r_c.font.name = "Times New Roman"
        r_c.font.size = Pt(11)

    output_path = "PROJECT_REPORT.docx"
    doc.save(output_path)
    print(f"Successfully generated {output_path} ({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    create_report()
