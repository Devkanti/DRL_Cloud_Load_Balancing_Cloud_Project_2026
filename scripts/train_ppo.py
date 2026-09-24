"""Run the complete locked PPO training and local reward gate."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.ppo_agent import PPOAgent


def main() -> None:
    agent = PPOAgent()
    try:
        agent.train()
        ppo = agent.evaluate()
        round_robin = agent.evaluate_round_robin()
        report = {"ppo": ppo, "round_robin": round_robin}
        report_path = ROOT / "models" / "ppo" / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if ppo["mean_episode_reward"] <= round_robin["mean_episode_reward"]:
            raise SystemExit("PPO local reward gate failed; see validation_report.json")
        print("PPO local reward gate passed")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
