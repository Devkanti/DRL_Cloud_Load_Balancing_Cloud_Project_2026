"""Tests for Issue #13 local evaluation-gate mechanics."""

import csv

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from evaluation.local_gate import RESULT_COLUMNS, _weighted_percentile, run_gate, simulate_episode


class _FirstBackend:
    def select_backend(self, observation):
        return 0


def test_gate_episode_reports_required_metrics():
    metrics = simulate_episode(_FirstBackend(), np.array([100.0, 100.0, 1000.0]))
    assert set(metrics) == set(RESULT_COLUMNS[3:])
    assert metrics["p95_latency_ms"] > 0
    assert metrics["throughput_rps"] > 0


def test_weighted_p95_prioritises_high_request_bins():
    assert _weighted_percentile(np.array([50.0, 500.0]), np.array([100, 1]), 95) == 50.0


def test_gate_exports_one_row_per_agent_with_saved_models(tmp_path):
    ppo, dqn = PPOAgent(), DQNAgent()
    ppo.model, dqn.model = ppo._build_model(), dqn._build_model()
    try:
        ppo_path = ppo.save(tmp_path / "ppo")
        dqn_path = dqn.save(tmp_path / "dqn")
        output = tmp_path / "local_gate_results.csv"
        rows = run_gate(ppo_path, dqn_path, output, seeds=(112,), scenarios=("e1_baseline",), max_steps=3)
        with output.open(newline="", encoding="utf-8") as file:
            exported = list(csv.DictReader(file))
        assert len(rows) == len(exported) == 6
        assert list(exported[0]) == RESULT_COLUMNS
    finally:
        ppo.close()
        dqn.close()
