"""
MetricsCollector — aggregates per-step metrics during experiment runs.

Tracks six key metrics per episode:
  - p95_latency
  - p99_latency
  - throughput_rps
  - cpu_utilisation_mean
  - sla_violation_rate
  - reward_episode_mean

Supports CSV export (compatible with notebooks/03_results_analysis.ipynb)
and JSON serialisation (for Lambda/S3 upload).
"""

import json
import csv
import numpy as np
from typing import List, Dict, Any, Optional


class MetricsCollector:
    """
    Collects per-step latency, CPU, throughput, SLA, and reward data,
    then computes episode-level summary statistics.
    """

    def __init__(self):
        # Per-step buffers for the current episode
        self._step_latencies: List[float] = []
        self._step_cpu_utils: List[float] = []
        self._step_throughputs: List[float] = []
        self._step_sla_violations: List[int] = []
        self._step_total_requests: List[int] = []
        self._step_rewards: List[float] = []

        # Completed episode summaries
        self.episodes: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Step-level recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        latency_ms: float,
        cpu_util: float,
        throughput: float,
        sla_violations: int,
        total_requests: int,
        reward: float,
    ) -> None:
        """
        Record metrics for a single environment step.

        Args:
            latency_ms: Response-time EMA (or raw latency) for this step.
            cpu_util: Mean CPU utilisation across backends for this step.
            throughput: Number of requests processed in this step.
            sla_violations: Number of SLA-violating requests in this step.
            total_requests: Total requests handled in this step.
            reward: Scalar reward returned by FlashSaleEnv.step().
        """
        self._step_latencies.append(latency_ms)
        self._step_cpu_utils.append(cpu_util)
        self._step_throughputs.append(throughput)
        self._step_sla_violations.append(sla_violations)
        self._step_total_requests.append(total_requests)
        self._step_rewards.append(reward)

    # ------------------------------------------------------------------
    # Episode-level aggregation
    # ------------------------------------------------------------------

    def end_episode(self) -> Dict[str, float]:
        """
        Finalise the current episode: compute summary statistics from the
        step buffers, store the summary, and reset the buffers.

        Returns:
            A dict with the six episode-level metrics.
        """
        latencies = np.array(self._step_latencies) if self._step_latencies else np.array([0.0])
        rewards = np.array(self._step_rewards) if self._step_rewards else np.array([0.0])

        total_violations = sum(self._step_sla_violations)
        total_requests = sum(self._step_total_requests)
        total_throughput = sum(self._step_throughputs)
        num_steps = max(len(self._step_latencies), 1)

        summary: Dict[str, float] = {
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99)),
            "throughput_rps": float(total_throughput / num_steps * 10),  # steps are 100ms
            "cpu_utilisation_mean": float(np.mean(self._step_cpu_utils)) if self._step_cpu_utils else 0.0,
            "sla_violation_rate": float(total_violations / max(total_requests, 1)),
            "reward_episode_mean": float(np.mean(rewards)),
        }

        self.episodes.append(summary)
        self._reset_buffers()
        return summary

    def _reset_buffers(self) -> None:
        self._step_latencies.clear()
        self._step_cpu_utils.clear()
        self._step_throughputs.clear()
        self._step_sla_violations.clear()
        self._step_total_requests.clear()
        self._step_rewards.clear()

    # ------------------------------------------------------------------
    # Export / serialisation
    # ------------------------------------------------------------------

    def export_csv(self, path: str) -> None:
        """
        Write all completed episode summaries to a CSV file.

        Columns: p95_latency, p99_latency, throughput_rps,
                 cpu_utilisation_mean, sla_violation_rate, reward_episode_mean

        Args:
            path: Filesystem path for the output CSV.
        """
        fieldnames = [
            "p95_latency",
            "p99_latency",
            "throughput_rps",
            "cpu_utilisation_mean",
            "sla_violation_rate",
            "reward_episode_mean",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in self.episodes:
                writer.writerow(ep)

    def to_json(self) -> str:
        """
        Serialise all episode summaries to a JSON string.
        Compatible with Lambda/S3 upload workflows.
        """
        return json.dumps({"episodes": self.episodes}, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MetricsCollector":
        """
        Reconstruct a MetricsCollector from a JSON string produced by to_json().
        """
        data = json.loads(json_str)
        collector = cls()
        collector.episodes = data.get("episodes", [])
        return collector
