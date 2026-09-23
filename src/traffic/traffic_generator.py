"""
Synthetic flash-sale traffic generator for FlashBalanceAI.

Produces request-rate time series (arrival rate per 100 ms step) across
five traffic phases:

    warmup → pre-burst → spike onset → peak → cooldown

All scenario parameters are loaded from configs/traffic_config.yaml.
"""

import os
import numpy as np
import yaml
from typing import Optional, Dict, Any


class TrafficGenerator:
    """
    Generates a synthetic flash-sale traffic profile as a 1-D numpy array
    of arrival rates (requests per second) at 100 ms resolution.

    Args:
        burst_multiplier: Peak traffic as a multiple of baseline_rps.
        noise_std_fraction: Gaussian noise σ expressed as a fraction of the
            instantaneous rate (e.g. 0.10 → 10 % noise).
        seed: Random seed for reproducibility.
        baseline_rps: Baseline request rate (default 100).
        warmup_steps: Number of steps at baseline (default 1200).
        pre_burst_steps: Number of steps at 3× baseline (default 600).
        spike_onset_steps: Number of steps for linear ramp (default 100).
        peak_steps: Number of steps at peak rate (default 3000).
        cooldown_steps: Number of steps for exponential decay (default 3000).
    """

    def __init__(
        self,
        burst_multiplier: float = 10.0,
        noise_std_fraction: float = 0.10,
        seed: int = 42,
        baseline_rps: float = 100.0,
        warmup_steps: int = 1200,
        pre_burst_steps: int = 600,
        spike_onset_steps: int = 100,
        peak_steps: int = 3000,
        cooldown_steps: int = 3000,
    ):
        self.burst_multiplier = burst_multiplier
        self.noise_std_fraction = noise_std_fraction
        self.seed = seed
        self.baseline_rps = baseline_rps
        self.warmup_steps = warmup_steps
        self.pre_burst_steps = pre_burst_steps
        self.spike_onset_steps = spike_onset_steps
        self.peak_steps = peak_steps
        self.cooldown_steps = cooldown_steps

        self.total_steps = (
            warmup_steps + pre_burst_steps + spike_onset_steps
            + peak_steps + cooldown_steps
        )

    def generate(self) -> np.ndarray:
        """
        Build the full traffic profile.

        Returns:
            np.ndarray of shape (total_steps,) with arrival rates in rps.
        """
        rng = np.random.default_rng(self.seed)

        peak_rate = self.baseline_rps * self.burst_multiplier
        pre_burst_rate = 3.0 * self.baseline_rps  # 300 rps

        # --- Phase 1: Warmup (constant baseline) ---
        warmup = np.full(self.warmup_steps, self.baseline_rps)

        # --- Phase 2: Pre-burst (constant 3× baseline) ---
        pre_burst = np.full(self.pre_burst_steps, pre_burst_rate)

        # --- Phase 3: Spike onset (linear ramp from pre-burst to peak) ---
        spike_onset = np.linspace(pre_burst_rate, peak_rate, self.spike_onset_steps)

        # --- Phase 4: Peak (constant at peak rate) ---
        peak = np.full(self.peak_steps, peak_rate)

        # --- Phase 5: Cooldown (exponential decay back to baseline) ---
        # Solve for τ so that the last step ≈ baseline_rps:
        #   peak_rate * exp(-cooldown_steps / τ) ≈ baseline_rps
        #   τ = cooldown_steps / ln(peak_rate / baseline_rps)
        ratio = max(peak_rate / self.baseline_rps, 1.001)  # avoid log(1)
        tau = self.cooldown_steps / np.log(ratio)
        t_cool = np.arange(self.cooldown_steps)
        cooldown = self.baseline_rps + (peak_rate - self.baseline_rps) * np.exp(-t_cool / tau)

        # --- Concatenate ---
        profile = np.concatenate([warmup, pre_burst, spike_onset, peak, cooldown])

        # --- Add Gaussian noise ---
        noise = rng.normal(0, self.noise_std_fraction * profile)
        profile = profile + noise

        # Clamp to non-negative
        profile = np.maximum(profile, 0.0)

        return profile

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        scenario_name: str,
        config_path: Optional[str] = None,
    ) -> "TrafficGenerator":
        """
        Construct a TrafficGenerator for a named scenario in
        configs/traffic_config.yaml.

        Args:
            scenario_name: Key under ``scenarios:`` in the YAML
                           (e.g. ``"e2_10x"``).
            config_path:   Override path to the YAML config file.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "../../configs/traffic_config.yaml"
            )

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        scenario = cfg["scenarios"][scenario_name]

        return cls(
            burst_multiplier=scenario["burst_multiplier"],
            noise_std_fraction=scenario["noise_std_fraction"],
            seed=scenario["seed"],
            baseline_rps=cfg.get("baseline_rps", 100),
            warmup_steps=cfg.get("warmup_steps", 1200),
            pre_burst_steps=cfg.get("pre_burst_steps", 600),
            spike_onset_steps=cfg.get("spike_onset_steps", 100),
            peak_steps=cfg.get("peak_steps", 3000),
            cooldown_steps=cfg.get("cooldown_steps", 3000),
        )
