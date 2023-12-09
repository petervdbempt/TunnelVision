from TunnelVision import TunnelVision
import numpy as np
import matplotlib.pyplot as plt


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


def q_learning(env, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight):
    mode = 'explore'

    Qe = np.zeros((env.num_states, env.num_actions), dtype=np.float64)
    Qi = np.zeros((env.num_states, env.num_actions), dtype=np.float64)
    visitation_count = np.zeros(env.num_states, dtype=np.int32)
    rewards = []

    for episode in range(num_episodes):
        env.reset()
        state = 0
        total_reward = 0
        episodic_step_count = 0
        terminated = False
        action = epsilon_greedy(Qe, state, epsilon)

        if episode % 10 == 0:
            rewards.append(evaluate_policy(Qe, env))

        while not terminated:
            episodic_step_count += 1

            # Informed switching
            if mode == 'explore' and visitation_count[state] >= 3:
                mode = 'exploit'
                print('exploit')
            elif mode == 'exploit' and visitation_count[state] < 3:
                mode = 'explore'
                print('explore')

            next_state, reward, terminated, truncated = env.step(action)

            if mode == 'explore':
                intrinsic_reward = visitation_bonus_weight / (visitation_count[state] + 1)
                next_action = epsilon_greedy(Qi, next_state, 0.1)
            elif mode == 'exploit':
                intrinsic_reward = 0
                next_action = epsilon_greedy(Qe, next_state, 0.1)

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
        # Epsilon decay here if needed
        # print(total_reward, epsilon)

    return Qe, rewards


def print_q_values(Q, num_states, num_actions):
    for state in range(num_states):
        print(f"Q-values for State {state}:")

        for action in range(num_actions):
            print(f"Action {action}: {Q[state, action]}")

        print("\n")


def save_rewards_to_file(filename, average_rewards):
    np.savetxt(filename, average_rewards, delimiter=',')


def plot_rewards(average_rewards):
    plt.plot(average_rewards)
    plt.title('Average Evaluation Rewards Over Runs')
    plt.xlabel('Episodes')
    plt.ylabel('Average Evaluation Reward')
    plt.show()


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

    for run in range(num_runs):
        Q, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight)
        all_rewards.append(rewards)
        print('run completed')

    average_rewards = np.mean(all_rewards, axis=0)
    return average_rewards


if __name__ == "__main__":
    env = TunnelVision(env='Standard')
    num_runs = 10
    num_episodes = 10000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    visitation_bonus_weight = 1

    average_rewards = run_experiment(env, num_runs, num_episodes, alpha, gamma, epsilon, visitation_bonus_weight)

    # Print or save the average rewards for comparison
    plot_rewards(average_rewards)

    # Save the average rewards to a file
    save_rewards_to_file("TV_Q-Switching_Informed_Separate_EVAL.csv", average_rewards)
