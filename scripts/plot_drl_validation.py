"""Create a same-seed PPO-versus-DQN validation reward curve."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).parents[1]


def main() -> None:
    with (ROOT / "models" / "ppo" / "validation_report.json").open(encoding="utf-8") as file:
        ppo = json.load(file)["ppo"]
    with (ROOT / "models" / "dqn" / "validation_report.json").open(encoding="utf-8") as file:
        dqn = json.load(file)["dqn"]
    if ppo["seeds"] != dqn["seeds"]:
        raise ValueError("PPO and DQN validation reports use different seed sets")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(ppo["seeds"], ppo["episode_rewards"], marker="o", label="PPO")
    axis.plot(dqn["seeds"], dqn["episode_rewards"], marker="o", label="DQN")
    axis.set(xlabel="Validation seed", ylabel="Episode reward", title="PPO vs DQN validation rewards")
    axis.legend()
    axis.grid(alpha=0.3)
    output = ROOT / "models" / "drl_validation_rewards.png"
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
