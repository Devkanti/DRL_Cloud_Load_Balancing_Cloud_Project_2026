"""Tests for MetricsCollector and visualiser module."""

import csv
import json
import os
import tempfile
import pytest
import numpy as np

from metrics.collector import MetricsCollector


# ── Helper ──────────────────────────────────────────────────────────

def _run_fake_episode(collector: MetricsCollector, num_steps: int = 50):
    """Feed synthetic step data into the collector and finalise the episode."""
    for i in range(num_steps):
        collector.record_step(
            latency_ms=50.0 + i * 0.5,
            cpu_util=0.3 + (i / num_steps) * 0.4,
            throughput=10.0,
            sla_violations=1 if i > 40 else 0,
            total_requests=10,
            reward=0.8 - i * 0.005,
        )
    return collector.end_episode()


# ── Tests ───────────────────────────────────────────────────────────

class TestMetricsCollector:

    def test_end_episode_returns_all_fields(self):
        mc = MetricsCollector()
        summary = _run_fake_episode(mc)

        required = [
            "p95_latency",
            "p99_latency",
            "throughput_rps",
            "cpu_utilisation_mean",
            "sla_violation_rate",
            "reward_episode_mean",
        ]
        for field in required:
            assert field in summary, f"Missing field: {field}"

    def test_p95_p99_ordering(self):
        mc = MetricsCollector()
        summary = _run_fake_episode(mc)
        assert summary["p99_latency"] >= summary["p95_latency"]

    def test_sla_violation_rate_range(self):
        mc = MetricsCollector()
        summary = _run_fake_episode(mc)
        assert 0.0 <= summary["sla_violation_rate"] <= 1.0

    def test_multiple_episodes(self):
        mc = MetricsCollector()
        _run_fake_episode(mc, 30)
        _run_fake_episode(mc, 60)
        _run_fake_episode(mc, 10)
        assert len(mc.episodes) == 3

    def test_buffers_reset_between_episodes(self):
        mc = MetricsCollector()
        _run_fake_episode(mc, 20)
        assert len(mc._step_latencies) == 0
        assert len(mc._step_rewards) == 0


class TestExportCSV:

    def test_csv_has_all_columns(self):
        mc = MetricsCollector()
        _run_fake_episode(mc)
        _run_fake_episode(mc)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            path = f.name

        try:
            mc.export_csv(path)

            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            expected_cols = {
                "p95_latency",
                "p99_latency",
                "throughput_rps",
                "cpu_utilisation_mean",
                "sla_violation_rate",
                "reward_episode_mean",
            }
            assert set(rows[0].keys()) == expected_cols
        finally:
            os.unlink(path)

    def test_csv_values_are_numeric(self):
        mc = MetricsCollector()
        _run_fake_episode(mc)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            path = f.name

        try:
            mc.export_csv(path)
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for val in row.values():
                        float(val)  # should not raise
        finally:
            os.unlink(path)


class TestJSONSerialisation:

    def test_round_trip(self):
        mc = MetricsCollector()
        _run_fake_episode(mc, 40)
        _run_fake_episode(mc, 20)

        json_str = mc.to_json()
        restored = MetricsCollector.from_json(json_str)

        assert len(restored.episodes) == 2
        for orig, rest in zip(mc.episodes, restored.episodes):
            for key in orig:
                assert abs(orig[key] - rest[key]) < 1e-9

    def test_json_is_valid(self):
        mc = MetricsCollector()
        _run_fake_episode(mc)
        data = json.loads(mc.to_json())
        assert "episodes" in data
        assert isinstance(data["episodes"], list)


class TestVisualiserImport:

    def test_visualiser_importable(self):
        """Verify that the visualiser module can be imported without errors."""
        import metrics.visualiser  # noqa: F401
