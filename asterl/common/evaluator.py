import gymnasium as gym
import numpy as np


def make_env(env_id, seed, env_kwargs=None, max_episode_steps=None):
    kwargs = dict(env_kwargs or {})
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def rollout_episode(
    env,
    policy,
    *,
    max_action,
    action_dim,
    expl_noise=0.0,
    random_actions=False,
):
    """Run one episode; returns (total_reward, transitions).

    Transitions are (state, action, next_state, reward, terminated) with the
    bootstrap mask taken from `terminated` only — a truncated episode still
    bootstraps, unlike the old gym done_bool logic.
    """
    state, _ = env.reset()
    transitions = []
    total_reward = 0.0
    done = False
    while not done:
        if random_actions:
            action = env.action_space.sample()
        elif expl_noise > 0:
            action = (
                policy.select_action(np.asarray(state))
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
        else:
            action = policy.select_action(np.asarray(state))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transitions.append((state, action, next_state, reward, float(terminated)))
        total_reward += reward
        state = next_state
    return total_reward, transitions


def eval_policy(env, policy, episodes=5):
    total = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.asarray(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
    return total / episodes
