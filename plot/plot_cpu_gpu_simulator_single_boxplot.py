import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import numpy as np

def plot_cpu_gpu(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Define categories and percentages for CPU and GPU
    cpu_categories = ['MISC', 'V100', 'P100', 'T4']
    gpu_categories = ['MISC', 'V100', 'P100', 'T4']
    cpu_percentages = [96, 96, 64, 96]
    gpu_percentages = [8, 8, 2, 2]

    # Initialize lists to store data for each category
    cpu_data = {}
    gpu_data = {}

    # Calculate confidence intervals for each category
    for category, cpu_perc, gpu_perc in zip(cpu_categories, cpu_percentages, gpu_percentages):
        cpu_cols = [col for col in df.columns if 'cpu' in col and category in col]
        gpu_cols = [col for col in df.columns if 'gpu' in col and category in col]

        cpu_data[category] = df[cpu_cols].values.flatten() * 100 / cpu_perc
        gpu_data[category] = df[gpu_cols].values.flatten() * 100 / gpu_perc

    # Plot the confidence intervals
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Add a shift to each box
    box_width = 0.2
    positions = [1, 2, 3, 4]
    box_colors = ['red', 'green', 'blue', 'orange']

    # Store legend patches
    legend_patches = []

    # # Modify the boxplot calls to include colors and legend patches
    for i, category in enumerate(cpu_categories):
        for j, position in enumerate(positions):
            color = box_colors[i % len(box_colors)]  # Use modulo to cycle through colors for each position
            if j % 4 == 0:
                patch = mpatches.Patch(color=color, label=category)
                legend_patches.append(patch)
        axs[0].boxplot([cpu_data[category]], positions=[positions[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))
        axs[1].boxplot([gpu_data[category]], positions=[positions[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))



    # Add legends
    axs[0].legend(handles=legend_patches, title='Legend', loc='upper right')
    axs[1].legend(handles=legend_patches, title='Legend', loc='upper right')

    axs[0].set_title('CPU')
    axs[0].set_xticks(positions)
    axs[0].set_xticklabels(cpu_categories)

    axs[1].set_title('GPU')
    axs[1].set_xticks(positions)
    axs[1].set_xticklabels(gpu_categories)

    plt.savefig("boxcpu_gpu.png")

# Example usage
csv_file = 'data_0_3.csv'
plot_cpu_gpu(csv_file)
