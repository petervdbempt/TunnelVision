import matplotlib.pyplot as plt
import pandas as pd

def plot_rewards(csv_files, labels):
    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot each CSV file with a different color
    for i, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file, header=None)
        plt.plot(data, label=labels[i])

    # Add labels and title
    plt.title('Comparison of Algorithms')
    plt.xlabel('Episodes')
    plt.ylabel('Average Eval Reward')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Specify the CSV files and labels for each algorithm
    csv_files = ["TV_Q_Regular_EVAL.csv", "TV_Q-Switching_Blind_EVAL.csv", "TV_Q-Switching_Blind_Separate_EVAL.csv",
                 "TV_Q-Switching_Informed_EVAL.csv", "TV_Q-Switching_Informed_Separate_EVAL.csv"]
    labels = ["Q-Learning", "Q-Switching_Blind", "Q-Switching_Blind_Separate", "Q-Switching_Informed",
              "Q-Switching_Informed_Separate"]

    # Call the function to plot the data
    plot_rewards(csv_files, labels)
