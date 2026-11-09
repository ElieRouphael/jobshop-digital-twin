"""DQN agent: learning rule on hand-made tensors, reward bookkeeping, and a short training run."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from jobshop.checker import check_schedule
from jobshop.dqn import (
    ACTIONS,
    STATE_DIM,
    DQNConfig,
    DQNController,
    Learner,
    QNetwork,
    compute_td_targets,
    epsilon_by_step,
    estimated_tardiness,
    features,
    soft_update,
    total_tardiness,
    train,
)
from jobshop.scenario import make_scenario, make_workshop
from jobshop.shop import Shop

WS = make_workshop()


class FixedQ(torch.nn.Module):
    """Returns the same Q-values whatever the state."""

    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor([values], dtype=torch.float32)

    def forward(self, x):
        return self.values.expand(x.shape[0], -1)


def test_td_target_stops_at_the_end_of_the_run():
    rewards = torch.tensor([[-1.0], [-2.0]])
    terminals = torch.tensor([[0.0], [1.0]])
    net = FixedQ([1.0, 5.0, 3.0, 2.0])
    targets = compute_td_targets(rewards, torch.zeros(2, STATE_DIM), terminals, net, net, 0.9, double_dqn=False)
    assert targets.squeeze(1).tolist() == pytest.approx([-1.0 + 0.9 * 5.0, -2.0])


def test_double_dqn_selects_with_the_online_net_and_evaluates_with_the_target_net():
    online, target = FixedQ([0.0, 9.0, 0.0, 0.0]), FixedQ([4.0, 1.0, 7.0, 0.0])
    args = (torch.zeros(1, 1), torch.zeros(1, STATE_DIM), torch.zeros(1, 1), online, target, 1.0)
    assert compute_td_targets(*args, double_dqn=True).item() == pytest.approx(1.0)  # target value of action 1
    assert compute_td_targets(*args, double_dqn=False).item() == pytest.approx(7.0)  # max of the target net


def test_soft_update_moves_the_target_towards_the_online_net():
    a, b = QNetwork(STATE_DIM, len(ACTIONS), (8,)), QNetwork(STATE_DIM, len(ACTIONS), (8,))
    before = [p.clone() for p in b.parameters()]
    soft_update(b, a, 0.25)
    for old, new, src in zip(before, b.parameters(), a.parameters()):
        assert torch.allclose(new, old + 0.25 * (src - old), atol=1e-6)


def test_epsilon_decays_from_start_to_end():
    config = DQNConfig()
    assert epsilon_by_step(0, config) == pytest.approx(config.epsilon_start)
    assert epsilon_by_step(10**7, config) == pytest.approx(config.epsilon_end)


class Probe(DQNController):
    """Random policy that records the state and the committed tardiness at each decision."""

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(0)
        self.states, self.committed = [], []

    def sequence(self, shop, m):
        if len(m.queue) == 1:
            return m.queue[0]
        self.states.append(features(shop, m))
        self.committed.append(estimated_tardiness(shop))
        from jobshop.rules import pick

        return pick(ACTIONS[int(self.rng.integers(len(ACTIONS)))], shop, m)


def test_features_are_finite_and_committed_tardiness_never_decreases():
    probe = Probe()
    shop = Shop(WS, make_scenario(WS, 90, n_jobs=40, mtbf=120.0), probe)
    shop.run()
    states = np.array(probe.states)
    assert states.shape[1] == STATE_DIM and len(states) > 20
    assert np.isfinite(states).all() and np.abs(states).max() <= 10
    assert all(b >= a for a, b in zip(probe.committed, probe.committed[1:]))
    assert probe.committed[-1] <= total_tardiness(shop) == estimated_tardiness(shop)


def test_rewards_of_an_episode_sum_to_minus_the_total_tardiness():
    config = replace(DQNConfig(), min_buffer_size=10**9)  # collect only, no optimisation
    learner = Learner(config, seed=0)
    shop = Shop(WS, make_scenario(WS, 91, n_jobs=40, mtbf=120.0), DQNController(learner=learner))
    shop.run()
    learner.end_episode(total_tardiness(shop))
    transitions = list(learner.buffer._buffer)
    assert total_tardiness(shop) > 0
    assert sum(t.reward for t in transitions) * config.reward_scale == pytest.approx(-total_tardiness(shop))
    assert [t.terminal for t in transitions] == [False] * (len(transitions) - 1) + [True]
    assert all(t.reward <= 0 for t in transitions)


def test_short_training_run_and_greedy_agent_is_valid_and_deterministic():
    config = replace(DQNConfig(), episodes=4, validate_every=2, n_jobs=15, min_buffer_size=32, batch_size=16)
    result = train(WS, config, seed=0, log=lambda *_: None)
    assert result["state_dict"] is not None and result["steps"] > 0
    net = QNetwork(STATE_DIM, len(ACTIONS), config.hidden_sizes)
    net.load_state_dict(result["state_dict"])
    scenario = make_scenario(WS, 92, n_jobs=30, mtbf=120.0)
    logs = []
    for _ in range(2):
        shop = Shop(WS, scenario, DQNController(net))
        logs.append(shop.run())
        assert check_schedule(WS, scenario, logs[-1]) == []
    assert logs[0].segments == logs[1].segments
