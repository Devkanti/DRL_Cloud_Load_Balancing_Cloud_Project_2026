"""Stable-Baselines3 DQN baseline training for FlashBalanceAI."""

from __future__ import annotations

from importlib.util import find_spec
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from environment.flash_sale_env import FlashSaleEnv


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dqn_config.yaml"
VALIDATION_SEEDS = tuple(range(112, 127))


class _SeedCyclingFlashSaleEnv(FlashSaleEnv):
    """Cycles the validation seed across evaluation episodes."""

    def __init__(self, seeds: Iterable[int] = VALIDATION_SEEDS) -> None:
        self._seeds = cycle(seeds)
        super().__init__()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return super().reset(seed=next(self._seeds) if seed is None else seed, options=options)


class DQNAgent:
    """Owns the DQN baseline, evaluation callback, and model persistence."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open(encoding="utf-8") as file:
            self.config: dict[str, Any] = yaml.safe_load(file)
        self.save_dir = PROJECT_ROOT / self.config["save_dir"]
        self.tensorboard_log = PROJECT_ROOT / self.config["tensorboard_log"]
        self.model: DQN | None = None
        self.train_env: Monitor | None = None
        self.eval_env: Monitor | None = None
        self.eval_callback: EvalCallback | None = None
        self.checkpoint_callback: CheckpointCallback | None = None

    def _build_environments(self) -> None:
        if self.train_env is None:
            train = FlashSaleEnv()
            train.reset(seed=int(self.config["seed_train"]))
            self.train_env = Monitor(train)
        if self.eval_env is None:
            self.eval_env = Monitor(_SeedCyclingFlashSaleEnv())

    def _build_model(self) -> DQN:
        self._build_environments()
        assert self.train_env is not None
        return DQN(
            "MlpPolicy",
            self.train_env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            learning_starts=self.config["learning_starts"],
            batch_size=self.config["batch_size"],
            gamma=self.config["gamma"],
            target_update_interval=self.config["target_update_interval"],
            train_freq=self.config["train_freq"],
            exploration_fraction=self.config["exploration_fraction"],
            exploration_final_eps=self.config["exploration_final_eps"],
            policy_kwargs={"net_arch": self.config["net_arch"]},
            seed=self.config["seed_train"],
            tensorboard_log=str(self.tensorboard_log) if find_spec("tensorboard") else None,
            verbose=0,
        )

    def _build_callbacks(self) -> list:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log.mkdir(parents=True, exist_ok=True)
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.save_dir),
            log_path=str(self.save_dir / "evaluations"),
            eval_freq=int(self.config["eval_freq"]),
            n_eval_episodes=int(self.config["n_eval_episodes"]),
            deterministic=True,
            render=False,
        )
        self.checkpoint_callback = CheckpointCallback(
            save_freq=int(self.config["checkpoint_freq"]),
            save_path=str(self.save_dir),
            name_prefix="dqn_checkpoint",
        )
        return [self.eval_callback, self.checkpoint_callback]

    def train(self, total_timesteps: int | None = None) -> DQN:
        """Train for the configured duration and save the best available model."""
        self._build_environments()
        if self.model is None:
            self.model = self._build_model()
        self.model.learn(
            total_timesteps=int(total_timesteps or self.config["total_timesteps"]),
            callback=self._build_callbacks(),
            progress_bar=False,
        )
        best_path = self.save_dir / "best_model.zip"
        if not best_path.exists():
            self.model.save(str(best_path))
        return self.model

    def evaluate(self, seeds: Iterable[int] = VALIDATION_SEEDS) -> dict[str, Any]:
        """Evaluate deterministic DQN episodes over the supplied seed set."""
        if self.model is None:
            raise RuntimeError("Train or load a DQN model before evaluation.")
        seed_list = list(seeds)
        rewards = [self._episode_reward(seed) for seed in seed_list]
        return {
            "seeds": seed_list,
            "episode_rewards": rewards,
            "mean_episode_reward": float(np.mean(rewards)),
            "std_episode_reward": float(np.std(rewards)),
        }

    def _episode_reward(self, seed: int) -> float:
        assert self.model is not None
        env = FlashSaleEnv()
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
        env.close()
        return total_reward

    def save(self, path: str | Path | None = None) -> Path:
        """Save the current DQN model and return its ``.zip`` path."""
        if self.model is None:
            raise RuntimeError("Train or load a DQN model before saving.")
        output = Path(path) if path is not None else self.save_dir / "best_model.zip"
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output))
        return output if output.suffix == ".zip" else output.with_suffix(".zip")

    def load(self, path: str | Path) -> DQN:
        """Load a checkpoint into this agent."""
        self._build_environments()
        assert self.train_env is not None
        self.model = DQN.load(str(path), env=self.train_env)
        return self.model

    def close(self) -> None:
        """Release environment resources."""
        if self.train_env is not None:
            self.train_env.close()
        if self.eval_env is not None:
            self.eval_env.close()
