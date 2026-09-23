"""
Generate and save traffic profile plots for all 5 key scenarios.

Output directory: experiments/results/traffic_profiles/
"""

import os
import sys

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from traffic.traffic_generator import TrafficGenerator

SCENARIOS = {
    "e1_baseline": "E1 — Baseline (1×)",
    "e2_10x":      "E2 — 10× Burst",
    "e3_50x":      "E3 — 50× Burst",
    "e4_100x":     "E4 — 100× Burst",
    "e6_noisy":    "E6 — Noisy (2×, 40% noise)",
}

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "experiments", "results", "traffic_profiles",
)
os.makedirs(OUT_DIR, exist_ok=True)

STEP_MS = 100  # each step = 100 ms

for scenario_key, title in SCENARIOS.items():
    gen = TrafficGenerator.from_config(scenario_key)
    profile = gen.generate()
    time_s = np.arange(len(profile)) * STEP_MS / 1000.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_s, profile, linewidth=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arrival rate (rps)")
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUT_DIR, f"{scenario_key}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

print(f"\nAll {len(SCENARIOS)} scenario plots saved to {OUT_DIR}")
