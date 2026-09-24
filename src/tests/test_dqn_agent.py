"""Unit and smoke tests for the Issue #12 DQN implementation."""

from agents.dqn_agent import DQNAgent, VALIDATION_SEEDS


def test_agent_uses_locked_configuration_and_fairness_settings():
    agent = DQNAgent()
    model = agent._build_model()
    try:
        assert agent.config["seed_train"] == 42
        assert agent.config["net_arch"] == [64, 64]
        assert model.batch_size == agent.config["batch_size"]
        assert tuple(VALIDATION_SEEDS) == tuple(range(112, 127))
    finally:
        agent.close()


def test_save_and_load_untrained_model(tmp_path):
    agent = DQNAgent()
    agent.model = agent._build_model()
    checkpoint = agent.save(tmp_path / "dqn_smoke_model")
    try:
        assert checkpoint.is_file()
        restored = agent.load(checkpoint)
        assert restored.policy is not None
    finally:
        agent.close()
