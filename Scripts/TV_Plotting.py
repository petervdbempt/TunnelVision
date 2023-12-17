import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rewards(csv_files, labels):
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Plot each CSV file with a different color
    for i, csv_file in enumerate(csv_files):
        full_path = os.path.join(script_dir, "..", "Data", csv_file)
        data = pd.read_csv(full_path, header=None)

        std_dev = np.std(data, axis=0)
        average = np.mean(data, axis=0)
        print(max(average))

        plt.fill_between(range(len(average)), average - std_dev, average + std_dev, alpha=0.3)
        plt.plot(average, label=labels[i])

    # Add labels and title
    plt.title('Comparison of Algorithms')
    plt.xlabel('Episodes')
    plt.ylabel('Average Eval Reward')
    plt.grid()

    # Add legend
    plt.legend(loc='lower right')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Specify the CSV files and labels for each algorithm
    csv_files = ["TV_Q-Learning_Regular_EVAL.csv"]
    labels = ["Q-Learning"]

    # Call the function to plot the data
    plot_rewards(csv_files, labels)

#     csv_files = ["TV_Q-Learning_Regular_EVAL.csv", "TV_Q-Switching_Blind_EVAL.csv",
#                  "TV_Q-Switching_Blind_Separate_EVAL.csv", "TV_Q-Switching_Informed_EVAL.csv",
#                  "TV_Q-Switching_Informed_Separate_EVAL.csv"]
#     labels = ["Q-Learning", "Q-Switching_Blind", "Q-Switching_Blind_Separate", "Q-Switching_Informed",
#               "Q-Switching_Informed_Separate"]
