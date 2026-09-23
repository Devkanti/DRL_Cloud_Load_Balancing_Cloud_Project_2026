"""Tests for the synthetic TrafficGenerator."""

import numpy as np
import pytest

from traffic.traffic_generator import TrafficGenerator


# ── Constants from configs/traffic_config.yaml ──────────────────────

TOTAL_STEPS = 7900
BASELINE_RPS = 100


# ── Core shape / length tests ──────────────────────────────────────

class TestGenerateShape:

    def test_output_length_is_7900(self):
        gen = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.10, seed=42)
        profile = gen.generate()
        assert profile.shape == (TOTAL_STEPS,)

    def test_output_non_negative(self):
        gen = TrafficGenerator(burst_multiplier=50.0, noise_std_fraction=0.10, seed=42)
        profile = gen.generate()
        assert np.all(profile >= 0.0)


# ── Scenario-specific tests ────────────────────────────────────────

class TestScenarioPeaks:

    def test_e2_10x_peak_within_5_percent(self):
        """e2_10x scenario mean peak rate is 100 * 10 = 1000 rps ± 5 %."""
        gen = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.10, seed=44)
        profile = gen.generate()
        # Peak phase: after warmup(1200) + pre_burst(600) + spike_onset(100)
        peak_start = 1200 + 600 + 100
        peak_end = peak_start + 3000
        mean_peak = np.mean(profile[peak_start:peak_end])
        expected = BASELINE_RPS * 10  # 1000
        assert abs(mean_peak - expected) / expected < 0.05, (
            f"Mean peak {mean_peak:.1f} is more than 5 % off from {expected}"
        )

    def test_e3_50x_peak(self):
        gen = TrafficGenerator(burst_multiplier=50.0, noise_std_fraction=0.10, seed=44)
        profile = gen.generate()
        peak_start = 1200 + 600 + 100
        peak_end = peak_start + 3000
        mean_peak = np.mean(profile[peak_start:peak_end])
        expected = BASELINE_RPS * 50
        assert abs(mean_peak - expected) / expected < 0.05

    def test_e1_baseline_stays_flat(self):
        gen = TrafficGenerator(burst_multiplier=1.0, noise_std_fraction=0.05, seed=44)
        profile = gen.generate()
        # With burst_multiplier=1, peak == 100 rps and pre-burst is 300.
        # The pre-burst phase actually goes to 300, so max should be near 300.
        # But the important thing is that the "peak" portion is at baseline.
        # Check the peak phase region (after warmup+pre_burst+spike_onset)
        peak_start = 1200 + 600 + 100
        peak_end = peak_start + 3000
        peak_region = profile[peak_start:peak_end]
        mean_peak = np.mean(peak_region)
        # burst_multiplier=1 → peak = 100, so mean should be ~100
        assert abs(mean_peak - BASELINE_RPS) / BASELINE_RPS < 0.10


# ── Noise tests ────────────────────────────────────────────────────

class TestNoise:

    def test_noise_adds_variance(self):
        gen_no_noise = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.0, seed=42)
        gen_with_noise = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.10, seed=42)

        p_clean = gen_no_noise.generate()
        p_noisy = gen_with_noise.generate()

        # With noise, the two should differ
        assert not np.allclose(p_clean, p_noisy)

    def test_zero_noise_deterministic(self):
        gen1 = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.0, seed=42)
        gen2 = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.0, seed=42)
        assert np.allclose(gen1.generate(), gen2.generate())


# ── Seed reproducibility ──────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_output(self):
        gen1 = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.10, seed=99)
        gen2 = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.10, seed=99)
        assert np.allclose(gen1.generate(), gen2.generate())

    def test_different_seed_different_output(self):
        gen1 = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.10, seed=99)
        gen2 = TrafficGenerator(burst_multiplier=10.0, noise_std_fraction=0.10, seed=100)
        assert not np.allclose(gen1.generate(), gen2.generate())


# ── Config factory ─────────────────────────────────────────────────

class TestFromConfig:

    def test_from_config_e2_10x(self):
        gen = TrafficGenerator.from_config("e2_10x")
        profile = gen.generate()
        assert profile.shape == (TOTAL_STEPS,)
        assert gen.burst_multiplier == 10.0
        assert gen.noise_std_fraction == 0.10

    def test_from_config_e6_noisy(self):
        gen = TrafficGenerator.from_config("e6_noisy")
        assert gen.burst_multiplier == 2.0
        assert gen.noise_std_fraction == 0.40
