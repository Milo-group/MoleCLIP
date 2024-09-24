import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file, index_col=[0, 1])

    # Extract actual and predicted values
    actual = df.iloc[0, :].values
    predicted = df.iloc[1, :].values

    # Plot the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5, color='blue', edgecolors='black')
    
    # Plot the 1:1 line
    plt.plot([-0.1, 1.1], [-0.1, 1.1], color='red', linestyle='--', linewidth=2)

    # Set axis limits
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    # Set labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Scatter Plot')

    # Show the plot
    plt.grid(True)
    plt.savefig('out.png')


def parse_args():

    # Argument parsing
    parser = argparse.ArgumentParser(description='K-means clustering')
    parser.add_argument('-file_path', type=str, default="", help='dataset path')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    plot_actual_vs_predicted(args.file_path)