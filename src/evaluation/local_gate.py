"""Fast, deterministic local pre-AWS evaluation gate for all traffic scenarios."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from baselines.least_connections import LeastConnections
from baselines.round_robin import RoundRobin
from baselines.threshold_autoscaler import ThresholdAutoscaler
from baselines.weighted_round_robin import WeightedRoundRobin
from traffic.traffic_generator import TrafficGenerator


VALIDATION_SEEDS = tuple(range(112, 127))
SCENARIOS = ("e1_baseline", "e2_10x", "e3_50x", "e4_100x", "e6_noisy")
RESULT_COLUMNS = [
    "agent", "scenario", "seed", "p95_latency_ms", "throughput_rps",
    "sla_violation_rate", "mean_reward",
]


@dataclass
class _State:
    active: np.ndarray
    queues: np.ndarray
    ema_latency: np.ndarray


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, percentile / 100 * cumulative[-1])])


def _observation(state: _State, arrival_rate: float, time_since_spike: float) -> np.ndarray:
    values: list[float] = []
    for index in range(4):
        values.extend([
            min(state.active[index] / 100.0, 1.0), state.active[index], state.queues[index],
            state.ema_latency[index], 1.0,
        ])
    values.extend([arrival_rate / 1000.0, float(arrival_rate > 100.0), time_since_spike])
    return np.asarray(values, dtype=np.float32)


def _select_action(policy: Any, observation: np.ndarray) -> int:
    if hasattr(policy, "predict"):
        action, _ = policy.predict(observation, deterministic=True)
        return int(action)
    return int(policy.select_backend(observation))


def simulate_episode(policy: Any, profile: np.ndarray) -> dict[str, float]:
    """Simulate one 100 ms episode using an aggregate four-backend queue model.

    The model has 100 request slots per backend per 100 ms interval, matching
    FlashSaleEnv's backend connection limit. It computes one latency per time
    bin and uses that bin's request count as the P95 weight.
    """
    state = _State(np.zeros(4), np.zeros(4), np.full(4, 50.0))
    latencies: list[float] = []
    latency_weights: list[int] = []
    total_processed = total_arrivals = total_violations = 0
    rewards: list[float] = []
    since_spike = 0.0
    capacity = 100

    for arrival_rate in profile:
        arrivals = max(1, int(round(float(arrival_rate) * 0.1)))
        observation = _observation(state, float(arrival_rate), since_spike)
        action = _select_action(policy, observation)
        if action not in range(4):
            raise ValueError(f"Policy returned invalid backend index: {action}")

        queued_before = state.queues[action]
        offered = queued_before + arrivals
        processed = min(offered, capacity)
        state.queues[action] = offered - processed
        state.active *= 0.0
        state.active[action] = min(offered, capacity)
        cpu = state.active[action] / capacity
        latency = 50.0 * (1.0 + cpu) + queued_before / capacity * 100.0
        state.ema_latency[action] = 0.1 * latency + 0.9 * state.ema_latency[action]

        latencies.append(latency)
        latency_weights.append(arrivals)
        total_processed += int(processed)
        total_arrivals += arrivals
        if latency > 200.0:
            total_violations += arrivals

        average_latency = float(np.mean(state.ema_latency))
        r_lat = max(0.0, 1.0 - average_latency / 200.0)
        r_util = float(np.mean(state.active / capacity))
        r_tput = min(1.0, processed / capacity)
        r_sla = 1.0 - (arrivals if latency > 200.0 else 0.0) / arrivals
        rewards.append(0.4 * r_lat + 0.2 * r_util + 0.2 * r_tput + 0.2 * r_sla)
        since_spike = 0.0 if arrival_rate > 100.0 else since_spike + 0.1

    return {
        "p95_latency_ms": _weighted_percentile(np.asarray(latencies), np.asarray(latency_weights), 95),
        "throughput_rps": total_processed / (len(profile) * 0.1),
        "sla_violation_rate": total_violations / total_arrivals,
        "mean_reward": float(np.mean(rewards)),
    }


def _policy_factories(ppo_path: Path, dqn_path: Path) -> dict[str, Callable[[], Any]]:
    ppo, dqn = PPOAgent(), DQNAgent()
    ppo.load(ppo_path)
    dqn.load(dqn_path)
    return {
        "PPO": lambda: ppo.model,
        "DQN": lambda: dqn.model,
        "RoundRobin": RoundRobin,
        "WeightedRoundRobin": WeightedRoundRobin,
        "LeastConnections": LeastConnections,
        "ThresholdAutoscaler": ThresholdAutoscaler,
    }


def run_gate(
    ppo_path: str | Path,
    dqn_path: str | Path,
    output_path: str | Path,
    seeds: Iterable[int] = VALIDATION_SEEDS,
    scenarios: Iterable[str] = SCENARIOS,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every agent on every requested scenario and export the CSV."""
    factories = _policy_factories(Path(ppo_path), Path(dqn_path))
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            generator = TrafficGenerator.from_config(scenario)
            generator.seed = int(seed)
            profile = generator.generate()
            if max_steps is not None:
                profile = profile[:max_steps]
            for name, factory in factories.items():
                row = {"agent": name, "scenario": scenario, "seed": int(seed)}
                row.update(simulate_episode(factory(), profile))
                rows.append(row)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def assert_gate(rows: Iterable[dict[str, Any]]) -> None:
    """Enforce the required aggregate PPO-versus-baseline gate conditions."""
    records = list(rows)
    means = {
        agent: {
            metric: float(np.mean([row[metric] for row in records if row["agent"] == agent]))
            for metric in ("p95_latency_ms", "mean_reward")
        }
        for agent in {row["agent"] for row in records}
    }
    for baseline in ("RoundRobin", "WeightedRoundRobin", "LeastConnections", "ThresholdAutoscaler"):
        if means["PPO"]["p95_latency_ms"] >= means[baseline]["p95_latency_ms"]:
            raise AssertionError(f"Gate failed: PPO mean P95 latency is not below {baseline}.")
    if means["PPO"]["mean_reward"] <= means["DQN"]["mean_reward"]:
        raise AssertionError("Gate failed: PPO mean reward is not above DQN.")
