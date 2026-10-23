import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "experiments" / "analysis" / "seed_manifest.csv"

def generate_manifest():
    manifest = []
    
    # Train seeds: 42 to 111 (70 seeds)
    for seed in range(42, 112):
        if seed == 42:
            purpose = "Primary Training Seed (seed_train)"
            scenario = "FlashSaleEnv-train"
            rep = 1
            desc = "Master random seed for PPO and DQN training environments (configs/ppo_config.yaml & dqn_config.yaml)"
        elif seed in (43, 44, 45):
            worker_id = seed - 42
            purpose = f"Vectorised Env Worker {worker_id}"
            scenario = "FlashSaleEnv-train"
            rep = worker_id + 1
            desc = f"PPO parallel DummyVecEnv sub-environment worker {worker_id} initialization seed (seed_train + {worker_id})"
        else:
            run_idx = seed - 45
            purpose = "Training Rollout & Seed Exploration"
            scenario = "FlashSaleEnv-train"
            rep = run_idx + 4
            desc = f"Rollout trajectory exploration and environment stochasticity seed for training iterations (run {run_idx})"
        
        manifest.append({
            "seed": seed,
            "split": "train",
            "range_group": "42-111",
            "purpose": purpose,
            "target_scenario": scenario,
            "repetition_index": rep,
            "description": desc,
        })
    
    # Validation seeds: 112 to 126 (15 seeds)
    val_scenarios = ["e1_baseline", "e2_10x", "e3_50x", "e4_100x", "e6_noisy"]
    for idx, seed in enumerate(range(112, 127), start=1):
        assigned_scenario = val_scenarios[(seed - 112) % len(val_scenarios)]
        purpose = "Validation & Local Evaluation Gate"
        desc = (
            f"Validation set seed {seed} for EvalCallback checkpoint selection and pre-AWS local gate "
            f"(src/evaluation/local_gate.py VALIDATION_SEEDS, rotation across {assigned_scenario})"
        )
        manifest.append({
            "seed": seed,
            "split": "val",
            "range_group": "112-126",
            "purpose": purpose,
            "target_scenario": assigned_scenario,
            "repetition_index": idx,
            "description": desc,
        })
        
    # Test seeds: 127 to 141 (15 seeds)
    for seed in range(127, 142):
        if 127 <= seed <= 131:
            rep = seed - 127 + 1
            purpose = "Experiment E1: Baseline 1x Traffic Evaluation"
            scenario = "e1_baseline"
            desc = f"E1 baseline evaluation repetition {rep} (PRD §24 Issue #25, 5 reps test seeds 127-131)"
        elif 132 <= seed <= 136:
            rep = seed - 132 + 1
            purpose = "Experiment E2: 10x Flash-Sale Burst Evaluation (Core Hypothesis)"
            scenario = "e2_10x"
            desc = f"E2 10x burst evaluation repetition {rep} for primary PPO vs RR t-test / Wilcoxon validation (PRD §24 Issue #26)"
        elif 137 <= seed <= 141:
            rep = seed - 137 + 1
            purpose = "Experiment E3: 50x Flash-Sale Burst Extreme Stress Test"
            scenario = "e3_50x"
            desc = f"E3 50x burst evaluation repetition {rep} for peak throughput and SLA degradation under severe load (Issue #27)"
        
        manifest.append({
            "seed": seed,
            "split": "test",
            "range_group": "127-141",
            "purpose": purpose,
            "target_scenario": scenario,
            "repetition_index": rep,
            "description": desc,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "split", "range_group", "purpose", "target_scenario", "repetition_index", "description"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)  
    
    print(f"Generated {len(manifest)} seed entries in {OUTPUT_FILE}")
    
    # Assertions
    train_seeds = [m for m in manifest if m["split"] == "train"]
    val_seeds = [m for m in manifest if m["split"] == "val"]
    test_seeds = [m for m in manifest if m["split"] == "test"]
    assert len(train_seeds) == 70, f"Expected 70 train seeds, got {len(train_seeds)}"
    assert len(val_seeds) == 15, f"Expected 15 val seeds, got {len(val_seeds)}"
    assert len(test_seeds) == 15, f"Expected 15 test seeds, got {len(test_seeds)}"
    assert min(m["seed"] for m in train_seeds) == 42 and max(m["seed"] for m in train_seeds) == 111
    assert min(m["seed"] for m in val_seeds) == 112 and max(m["seed"] for m in val_seeds) == 126
    assert min(m["seed"] for m in test_seeds) == 127 and max(m["seed"] for m in test_seeds) == 141
    print("Seed manifest verified: train 42-111, val 112-126, test 127-141.")

if __name__ == "__main__":
    generate_manifest()
