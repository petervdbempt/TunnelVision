from Environment.TunnelVision import TunnelVision
from AgentFunctions import (epsilon_greedy, greedy, visualize_visitation_counts, visualize_q_values,
                            save_rewards_to_file, plot_combined_subplots, evaluate_policy)
import numpy as np
import math


def q_learning(env, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight, decay_rate):
    Q = np.zeros((env.num_states, env.num_actions), dtype=np.float64)
    visitation_count = np.zeros(env.num_states, dtype=np.int32)
    rewards = []
    training = []
    epsilons = []

    for episode in range(num_episodes):
        if episode % 10 == 0:
            rewards.append(evaluate_policy(Q, gamma, env))

        env.reset()
        state = 0
        total_reward = 0
        episodic_step_count = 0
        terminated = False
        action = epsilon_greedy(Q, state, epsilon)

        while not terminated:
            episodic_step_count += 1
            next_state, reward, terminated, truncated = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            visitation_count[state] += 1
            total_reward += (gamma ** episodic_step_count) * reward
            reward += visitation_bonus_weight / (visitation_count[state] + 1)

            # Q-learning update (max Q-value of next state)
            max_next_action = np.argmax(Q[next_state, :])
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, max_next_action] - Q[state, action])

            state = next_state
            action = next_action

        visitation_count[state] += 1
        epsilons.append(epsilon)
        epsilon = 0.01 + (1 - 0.01) * math.exp(-decay_rate * episode)
        training.append(total_reward)
        if total_reward >= 0.25:
            print(total_reward, episode)
        # if episode >= 110 and run == 5:
        #     if episode % 100 == 0:
        #         visualize_q_values(env, Q)

    # visualize_visitation_counts(env, visitation_count)
    return Q, rewards, training, epsilons


def run_experiment(env, num_runs, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight, decay_rate):
    all_rewards = []
    all_training = []
    all_epsilon = []

    for run in range(num_runs):
        Q, rewards, training, epsilons = q_learning(env, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight, decay_rate)
        all_rewards.append(rewards)
        all_training.append(training)
        all_epsilon.append(epsilons)
        print('Run completed')

    return all_rewards, all_training, all_epsilon


if __name__ == "__main__":
    algorithm = "Q-Learning with Epsilon Decay"
    env = TunnelVision(env='Standard')
    num_runs = 50
    num_episodes = 100000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1  # Starting epsilon value
    decay_rate = 0.0001
    visitation_bonus_weight = 0
    np.random.seed(42)

    average_rewards, average_training, average_epsilon = run_experiment(env, num_runs, num_episodes, alpha, gamma,
                                                                        epsilon, visitation_bonus_weight, decay_rate)

    # Print or save the average rewards for comparison
    plot_combined_subplots(average_rewards, average_training, average_epsilon,
                           num_runs, num_episodes, alpha, gamma, algorithm, 0)

    # Save the average rewards to a file
    save_rewards_to_file("../TunnelVision/Data/TV_Q-Learning_Regular_EVAL.csv", average_rewards)
