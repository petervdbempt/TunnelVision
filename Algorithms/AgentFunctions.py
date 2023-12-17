from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import os


def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        max_values = np.where(Q[state, :] == np.max(Q[state, :]))[0]
        if len(max_values) == 1:
            return max_values[0]
        else:
            # Handle ties by randomly choosing one of the maximum values
            return np.random.choice(max_values)


def greedy(Q, state):
    # Check if all values are the same
    max_values = np.where(Q[state, :] == np.max(Q[state, :]))[0]
    if len(max_values) == 1:
        return max_values[0]
    else:
        # Handle ties by randomly choosing one of the maximum values
        return np.random.choice(max_values)

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


def visualize_q_values(env, q):
    max_q_values = np.max(q, axis=1)  # Find the max Q-value for each state
    q_values_grid = np.zeros((env.rows, env.cols), dtype=float)
    for i in range(env.rows):
        for j in range(env.cols):
            state_index = env.coordinates_to_index(i, j)
            q_values_grid[i, j] = max_q_values[state_index]

    plt.figure(figsize=(10, 8))
    sns.heatmap(q_values_grid, annot=True, cmap="viridis", cbar_kws={'label': 'Q-values (max action)'})
    plt.title('Q-values per State(max action)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()


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


def plot_combined_subplots(average_rewards, average_training, average_epsilon,
                           num_runs, num_episodes, alpha, gamma, algorithm, pX):
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

    min_y, max_y = 0, 1
    plt.subplot(1, 3, 1).set_ylim(min_y, max_y)
    plt.subplot(1, 3, 2).set_ylim(min_y, max_y)

    # Add text to the plot
    algorithm_info = (f"Algorithm: {algorithm}\n"
                      f"Number of Runs: {num_runs}\n"
                      f"Number of Episodes per Run: {num_episodes}\n"
                      f"Learning Rate (Alpha): {alpha}\n"
                      f"Discount Factor (Gamma): {gamma}\n"
                      f"Ratio of Exploration (pX): {pX:.5f}")

    plt.gcf().text(0.06, 0.75, algorithm_info, fontsize=8)
    plt.tight_layout()
    plt.show()


def evaluate_policy(Q, gamma, env):
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


def epsilon_switching(mode, epsilon, visitation_count):
    # Exploration mode
    if mode == 'explore':
        if np.random.rand() < epsilon:
            mode = 'exploit'
        else:
            if visitation_count >= 1024:
                mode = 'exploit'

    # Exploitation mode
    elif mode == 'exploit':
        if np.random.rand() < epsilon:
            mode = 'explore'
        else:
            if visitation_count <= 32:
                mode = 'explore'
    return mode


def blind_switching(mode, intra_episodic_step_count):
    if mode == 'explore' and intra_episodic_step_count >= 10000:
        intra_episodic_step_count = 0
        mode = 'exploit'
        # visualize_q_values(env, Qe)
    elif mode == 'exploit' and intra_episodic_step_count >= 1000:
        intra_episodic_step_count = 0
        mode = 'explore'
    return mode, intra_episodic_step_count


def trigger_state_switching(mode, visitation_count, state, trigger_states):
    # print(trigger_states)
    if mode == 'explore':
        if visitation_count > 1 and state not in trigger_states:
            mode = 'exploit'
            trigger_states.append(state)
    elif mode == 'exploit':  # No need to check for trigger states here since visitation counts never go down
        if visitation_count <= 3200:  # 200 for non-separate, 3200 for separate
            mode = 'explore'
    return mode, trigger_states
