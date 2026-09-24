"""Unit and smoke tests for the Issue #11 PPO implementation."""

from pathlib import Path

from agents.ppo_agent import PPOAgent, VALIDATION_SEEDS


def test_agent_uses_locked_configuration():
    agent = PPOAgent()
    model = agent._build_model()
    try:
        assert agent.config["n_envs"] == 4
        assert agent.config["net_arch"] == [64, 64]
        assert model.n_steps == agent.config["n_steps"]
        assert model.batch_size == agent.config["batch_size"]
        assert tuple(VALIDATION_SEEDS) == tuple(range(112, 127))
    finally:
        agent.close()


def test_save_and_load_untrained_model(tmp_path):
    agent = PPOAgent()
    agent.model = agent._build_model()
    checkpoint = agent.save(tmp_path / "ppo_smoke_model")
    try:
        assert checkpoint.is_file()
        restored = agent.load(checkpoint)
        assert restored.policy is not None
    finally:
        agent.close()
