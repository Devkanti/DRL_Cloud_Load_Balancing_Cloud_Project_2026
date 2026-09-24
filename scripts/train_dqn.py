"""Run the locked DQN training after PPO validation has completed."""

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.dqn_agent import DQNAgent


def main() -> None:
    ppo_report = ROOT / "models" / "ppo" / "validation_report.json"
    while not ppo_report.exists():
        print("Waiting for PPO validation to finish...", flush=True)
        time.sleep(60)

    agent = DQNAgent()
    try:
        agent.train()
        report = {"dqn": agent.evaluate()}
        report_path = ROOT / "models" / "dqn" / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        agent.close()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_drl_validation.py")], check=True)
    print("DQN training and paired validation curve completed")


if __name__ == "__main__":
    main()
