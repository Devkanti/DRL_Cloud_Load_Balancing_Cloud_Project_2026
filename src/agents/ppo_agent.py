"""Stable-Baselines3 PPO training and evaluation for FlashBalanceAI."""

from __future__ import annotations

from importlib.util import find_spec
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines.round_robin import RoundRobin
from environment.flash_sale_env import FlashSaleEnv


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ppo_config.yaml"
VALIDATION_SEEDS = tuple(range(112, 127))


class _SeedCyclingFlashSaleEnv(FlashSaleEnv):
    """An evaluation environment that advances through the validation seeds."""

    def __init__(self, seeds: Iterable[int] = VALIDATION_SEEDS) -> None:
        self._seeds = cycle(seeds)
        super().__init__()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return super().reset(seed=next(self._seeds) if seed is None else seed, options=options)


class PPOAgent:
    """Owns the PPO model, vectorised environments, callbacks, and evaluation."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open(encoding="utf-8") as file:
            self.config: dict[str, Any] = yaml.safe_load(file)
        self.save_dir = PROJECT_ROOT / self.config["save_dir"]
        self.tensorboard_log = PROJECT_ROOT / self.config["tensorboard_log"]
        self.model: PPO | None = None
        self.train_env: DummyVecEnv | None = None
        self.eval_env: DummyVecEnv | None = None
        self.eval_callback: EvalCallback | None = None
        self.checkpoint_callback: CheckpointCallback | None = None

    def _make_training_env(self, index: int):
        seed = int(self.config["seed_train"]) + index

        def factory():
            env = FlashSaleEnv()
            env.reset(seed=seed)
            return Monitor(env)

        return factory

    def _build_environments(self) -> None:
        if self.train_env is None:
            self.train_env = DummyVecEnv(
                [self._make_training_env(index) for index in range(int(self.config["n_envs"]))]
            )
        if self.eval_env is None:
            self.eval_env = DummyVecEnv([lambda: Monitor(_SeedCyclingFlashSaleEnv())])

    def _build_model(self) -> PPO:
        self._build_environments()
        assert self.train_env is not None
        return PPO(
            "MlpPolicy",
            self.train_env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            policy_kwargs={"net_arch": self.config["net_arch"]},
            seed=self.config["seed_train"],
            # TensorBoard is optional in the project environment. EvalCallback
            # still persists reward histories when it is unavailable.
            tensorboard_log=str(self.tensorboard_log) if find_spec("tensorboard") else None,
            verbose=0,
        )

    def _build_callbacks(self) -> list:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log.mkdir(parents=True, exist_ok=True)
        # Callback calls occur once per vector step. Scale frequencies so the
        # configured values remain total environment timesteps.
        n_envs = int(self.config["n_envs"])
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.save_dir),
            log_path=str(self.save_dir / "evaluations"),
            eval_freq=max(1, int(self.config["eval_freq"]) // n_envs),
            n_eval_episodes=int(self.config["n_eval_episodes"]),
            deterministic=True,
            render=False,
        )
        self.checkpoint_callback = CheckpointCallback(
            save_freq=max(1, int(self.config["checkpoint_freq"]) // n_envs),
            save_path=str(self.save_dir),
            name_prefix="ppo_checkpoint",
        )
        return [self.eval_callback, self.checkpoint_callback]

    def train(self, total_timesteps: int | None = None) -> PPO:
        """Train for the configured duration and save the best observed model."""
        self._build_environments()
        if self.model is None:
            self.model = self._build_model()
        callbacks = self._build_callbacks()
        self.model.learn(
            total_timesteps=int(total_timesteps or self.config["total_timesteps"]),
            callback=callbacks,
            progress_bar=False,
        )
        best_path = self.save_dir / "best_model.zip"
        if not best_path.exists():
            # A very short smoke run may finish before the first evaluation.
            self.model.save(str(best_path))
        return self.model

    def evaluate(self, seeds: Iterable[int] = VALIDATION_SEEDS) -> dict[str, Any]:
        """Evaluate deterministic PPO episodes on each supplied validation seed."""
        if self.model is None:
            raise RuntimeError("Train or load a PPO model before evaluation.")
        rewards = [self._episode_reward(self.model, seed) for seed in seeds]
        return {
            "seeds": list(seeds),
            "episode_rewards": rewards,
            "mean_episode_reward": float(np.mean(rewards)),
            "std_episode_reward": float(np.std(rewards)),
        }

    def evaluate_round_robin(self, seeds: Iterable[int] = VALIDATION_SEEDS) -> dict[str, Any]:
        """Evaluate the Round Robin baseline on the same seed set."""
        rewards = [self._episode_reward(RoundRobin(), seed) for seed in seeds]
        return {
            "seeds": list(seeds),
            "episode_rewards": rewards,
            "mean_episode_reward": float(np.mean(rewards)),
            "std_episode_reward": float(np.std(rewards)),
        }

    @staticmethod
    def _episode_reward(policy: PPO | RoundRobin, seed: int) -> float:
        env = FlashSaleEnv()
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if isinstance(policy, PPO):
                action, _ = policy.predict(observation, deterministic=True)
            else:
                action = policy.select_backend(observation)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
        env.close()
        return total_reward

    def save(self, path: str | Path | None = None) -> Path:
        """Save the current model and return the resulting ``.zip`` path."""
        if self.model is None:
            raise RuntimeError("Train or load a PPO model before saving.")
        output = Path(path) if path is not None else self.save_dir / "best_model.zip"
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output))
        return output if output.suffix == ".zip" else output.with_suffix(".zip")

    def load(self, path: str | Path) -> PPO:
        """Load a previously saved PPO checkpoint into this agent."""
        self._build_environments()
        assert self.train_env is not None
        self.model = PPO.load(str(path), env=self.train_env)
        return self.model

    def close(self) -> None:
        """Release vectorised-environment resources."""
        if self.train_env is not None:
            self.train_env.close()
        if self.eval_env is not None:
            self.eval_env.close()
