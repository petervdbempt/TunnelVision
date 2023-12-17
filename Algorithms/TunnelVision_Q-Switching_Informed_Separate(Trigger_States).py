from Environment.TunnelVision import TunnelVision
from AgentFunctions import (epsilon_greedy, greedy, visualize_visitation_counts, visualize_q_values,
                            save_rewards_to_file, plot_rewards, plot_combined_subplots, evaluate_policy,
                            epsilon_switching, trigger_state_switching)
import numpy as np
import math


def q_learning(env, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight, decay_rate):
    mode = 'explore'
    intra_episodic_step_count = 0
    trigger_states = []

    Qe = np.zeros((env.num_states, env.num_actions), dtype=np.float64)
    Qi = np.zeros((env.num_states, env.num_actions), dtype=np.float64)
    visitation_count = np.zeros(env.num_states, dtype=np.int32)
    rewards = []
    training = []
    epsilons = []
    modes = []

    for episode in range(num_episodes):
        if episode % 10 == 0:
            rewards.append(evaluate_policy(Qe, gamma, env))

        env.reset()
        state = 0
        total_reward = 0
        episodic_step_count = 0
        terminated = False
        action = epsilon_greedy(Qe, state, epsilon)

        while not terminated:
            episodic_step_count += 1
            intra_episodic_step_count += 1

            # Informed switching
            # mode = epsilon_switching(mode, epsilon, visitation_count[state])
            mode, trigger_states = trigger_state_switching(mode, visitation_count[state], state, trigger_states)

            next_state, reward, terminated, truncated = env.step(action)

            if mode == 'explore':
                intrinsic_reward = visitation_bonus_weight / (visitation_count[state] + 1)
                next_action = epsilon_greedy(Qi, next_state, epsilon)
                modes.append(1)
            elif mode == 'exploit':
                intrinsic_reward = 0
                next_action = greedy(Qe, next_state)
                modes.append(0)

            visitation_count[state] += 1
            total_reward += (gamma ** episodic_step_count) * reward

            # Q-learning updates (max Q-value of next state)
            max_next_action = np.argmax(Qe[next_state, :])
            Qe[state, action] = Qe[state, action] + alpha * (
                    reward + gamma * Qe[next_state, max_next_action] - Qe[state, action])

            max_next_action = np.argmax(Qi[next_state, :])
            Qi[state, action] = Qi[state, action] + alpha * (
                    intrinsic_reward + gamma * Qi[next_state, max_next_action] - Qi[state, action])

            state = next_state
            action = next_action

        visitation_count[state] += 1
        epsilons.append(epsilon)
        epsilon = 0.01 + (1 - 0.01) * math.exp(-decay_rate * episode)
        training.append(total_reward)
        # if total_reward >= 0.25:
        #     print(total_reward, episode)
        # if episode % 100 == 0:
        #     visualize_q_values(env, Qi)

    # visualize_visitation_counts(env, visitation_count)
    print(trigger_states)
    visualize_q_values(env, Qi)
    visualize_q_values(env, Qe)

    return Qe, rewards, training, epsilons, modes


def run_experiment(env, num_runs, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight, decay_rate):
    all_rewards = []
    all_training = []
    all_epsilon = []
    all_modes = []

    for run in range(num_runs):
        Q, rewards, training, epsilons, modes = q_learning(env, num_episodes, alpha, gamma, epsilon,
                                                           visitation_bonus_weight, decay_rate)
        all_rewards.append(rewards)
        all_training.append(training)
        all_epsilon.append(epsilons)
        all_modes.append(modes)
        print('Run completed')

    pX = np.sum(np.concatenate(all_modes) == 1) / np.size(np.concatenate(all_modes))
    return all_rewards, all_training, all_epsilon, pX


if __name__ == "__main__":
    algorithm = "Informed Q-Switching (Trigger States) With Separate Q-Tables"
    env = TunnelVision(env='Standard')
    num_runs = 1
    num_episodes = 100000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1  # Initial epsilon value
    decay_rate = 0.0001
    visitation_bonus_weight = 1
    np.random.seed(42)

    (average_rewards, average_training, average_epsilon, pX) = (
        run_experiment(env, num_runs, num_episodes, alpha, gamma, epsilon,
                       visitation_bonus_weight, decay_rate))

    # Print or save the average rewards for comparison
    plot_combined_subplots(average_rewards, average_training, average_epsilon,
                           num_runs, num_episodes, alpha, gamma, algorithm, pX)

    # Save the average rewards to a file
    save_rewards_to_file("../TunnelVision/Data/TV_Q-Switching_Informed_Separate_TS_EVAL.csv", average_rewards)
