"""Run Issue #13 after the full PPO and DQN checkpoints are available."""

from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.local_gate import assert_gate, run_gate


def main() -> None:
    rows = run_gate(
        ROOT / "models" / "ppo" / "best_model.zip",
        ROOT / "models" / "dqn" / "best_model.zip",
        ROOT / "experiments" / "results" / "local_gate_results.csv",
    )
    assert_gate(rows)
    print("Local evaluation gate passed")


if __name__ == "__main__":
    main()
