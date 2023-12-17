from Environment.TunnelVision import TunnelVision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os


def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        # Check if all values are the same
        if len(np.unique(Q[state, :])) == 1:
            return np.random.randint(Q.shape[1])
        else:
            return np.argmax(Q[state, :])


def greedy(Q, state):
    # Check if all values are the same
    if len(np.unique(Q[state, :])) == 1:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state, :])


def visualize_visitation_counts(env, visitation_count):
    visitation_counts_grid = np.zeros((env.rows, env.cols), dtype=int)

    for i in range(env.rows):
        for j in range(env.cols):
            state_index = env.coordinates_to_index(i, j)
            visitation_counts_grid[i, j] = visitation_count[state_index]

    plt.figure(figsize=(10, 8))
    sns.heatmap(visitation_counts_grid, annot=True, cmap="viridis", cbar_kws={'label': 'Visitation Counts'})
    plt.title('Visitation Counts per State')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()


def q_learning(env, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight):
    mode = 'explore'
    intra_episodic_step_count = 0

    Q = np.zeros((env.num_states, env.num_actions), dtype=np.float64)
    visitation_count = np.zeros(env.num_states, dtype=np.int32)
    rewards = []
    training = []
    epsilons = []

    for episode in range(num_episodes):
        if episode % 10 == 0:
            rewards.append(evaluate_policy(Q, env))

        env.reset()
        state = 0
        total_reward = 0
        episodic_step_count = 0
        terminated = False
        action = epsilon_greedy(Q, state, epsilon)

        while not terminated:
            episodic_step_count += 1
            intra_episodic_step_count += 1

            # Blind switching
            if mode == 'explore' and intra_episodic_step_count >= 100000:
                intra_episodic_step_count = 0
                mode = 'exploit'
            elif mode == 'exploit' and intra_episodic_step_count >= 1000000:
                intra_episodic_step_count = 0
                mode = 'explore'

            next_state, reward, terminated, truncated = env.step(action)

            if mode == 'explore':
                visitation_bonus_weight = 1
            elif mode == 'exploit':
                visitation_bonus_weight = 0
            next_action = epsilon_greedy(Q, next_state, 0.1)

            visitation_count[state] += 1
            total_reward += (gamma ** episodic_step_count) * reward
            reward += visitation_bonus_weight / (visitation_count[state] + 1)

            # Q-learning update (max Q-value of next state)
            max_next_action = np.argmax(Q[next_state, :])
            Q[state, action] = Q[state, action] + alpha * (
                        reward + gamma * Q[next_state, max_next_action] - Q[state, action])

            state = next_state
            action = next_action

        visitation_count[state] += 1
        epsilons.append(epsilon)
        epsilon = 0.01 + (1 - 0.01) * math.exp(-0.001 * episode)
        training.append(total_reward)

    # visualize_visitation_counts(env, visitation_count)
    return Q, rewards, training, epsilons


def print_q_values(Q, num_states, num_actions):
    for state in range(num_states):
        print(f"Q-values for State {state}:")

        for action in range(num_actions):
            print(f"Action {action}: {Q[state, action]}")

        print("\n")


def save_rewards_to_file(filename, average_rewards):
    data_directory = os.path.join(os.path.dirname(__file__), '..', 'Data')
    full_filepath = os.path.join(data_directory, os.path.basename(filename))
    np.savetxt(full_filepath, average_rewards, delimiter=',')


def plot_rewards(data, num_runs):
    std_dev = np.std(data, axis=0)
    average = np.mean(data, axis=0)

    plt.fill_between(range(len(average)), average - std_dev, average + std_dev,
                     alpha=0.3, label='Standard Deviation')
    plt.plot(average)
    plt.title(f'Average Over {num_runs} Run(s)')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.grid()
    # plt.show()


def evaluate_policy(Q, env):
    env.reset()
    state = 0
    eval_reward = 0
    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        if step_count >= 5000:
            truncated = True
            # Break statement is necessary, I don't know why
            break
        step_count += 1

        action = greedy(Q, state)
        next_state, reward, terminated, truncated = env.step(action)

        eval_reward += (gamma ** step_count) * reward
        state = next_state

    return eval_reward


def run_experiment(env, num_runs, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight):
    all_rewards = []
    all_training = []
    all_epsilon = []

    for run in range(num_runs):
        Q, rewards, training, epsilons = q_learning(env, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight)
        all_rewards.append(rewards)
        all_training.append(training)
        all_epsilon.append(epsilons)
        print('run completed')

    return all_rewards, all_training, all_epsilon


def plot_combined_subplots(average_rewards, average_training, average_epsilon, num_runs):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_rewards(average_rewards, num_runs)
    plt.title('Average Evaluation Rewards')

    plt.subplot(1, 3, 2)
    plot_rewards(average_training, num_runs)
    plt.title('Average Training Rewards')

    plt.subplot(1, 3, 3)
    plot_rewards(average_epsilon, num_runs)
    plt.title('Epsilon Decay')

    # min_y, max_y = 0, 0.25
    min_y, max_y = 0, max(max(np.mean(average_rewards, axis=0)), max(np.mean(average_training, axis=0)))
    plt.subplot(1, 3, 1).set_ylim(min_y, max_y)
    plt.subplot(1, 3, 2).set_ylim(min_y, max_y)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = TunnelVision(env='Standard')
    num_runs = 10
    num_episodes = 10000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1  # Initial epsilon value
    visitation_bonus_weight = 1

    average_rewards, average_training, average_epsilon = run_experiment(env, num_runs, num_episodes, alpha, gamma,
                                                                        epsilon, visitation_bonus_weight)

    # Print or save the average rewards for comparison
    plot_combined_subplots(average_rewards, average_training, average_epsilon, num_runs)

    # Save the average rewards to a file
    save_rewards_to_file("../TunnelVision/Data/TV_Q-Switching_Blind_EVAL.csv", average_rewards)
