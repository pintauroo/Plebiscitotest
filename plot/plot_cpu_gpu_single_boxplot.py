import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import numpy as np
from scipy.stats import t

def plot_cpu_gpu(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Initialize lists to store data for each category
    cpu_ci = {}
    gpu_ci = {}
    # Define categories and percentages for CPU and GPU
    cpu_categories = ['MISC', 'V100', 'P100', 'T4']
    gpu_categories = ['MISC', 'V100', 'P100', 'T4']
    cpu_percentages = [96, 96, 64, 96]
    gpu_percentages = [8, 8, 2, 2]

    if '1_UTIL_FIFO_0_nosplit' in csv_file:
        
        cpu_columns = [col for col in df.columns if 'used_cpu' in col]
        gpu_columns = [col for col in df.columns if 'used_gpu' in col]

        misc_cpu_cols, v100_cpu_cols, p100_cpu_cols, t4_cpu_cols = [], [], [], []
        misc_gpu_cols, v100_gpu_cols, p100_gpu_cols, t4_gpu_cols = [], [], [], []
        misc_cpu_data = []
        v100_cpu_data = []
        p100_cpu_data = []
        t4_cpu_data = []

        misc_gpu_data = []
        v100_gpu_data = []
        p100_gpu_data = []
        t4_gpu_data = []
        
        gpu_type = {}
        for i in range(len(gpu_columns)):
            gpu_type['node_'+str(i)+'_gpu_type'] = df['node_'+str(i)+'_gpu_type'][1]

        for i in range(len(gpu_columns)):
            if gpu_type['node_'+str(i)+'_gpu_type'] == 'MISC':
                misc_gpu_cols.append(gpu_columns[i])
            elif gpu_type['node_'+str(i)+'_gpu_type'] == 'V100':        
                v100_gpu_cols.append(gpu_columns[i])
            elif gpu_type['node_'+str(i)+'_gpu_type'] == 'P100':
                p100_gpu_cols.append(gpu_columns[i])
            elif gpu_type['node_'+str(i)+'_gpu_type'] == 'T4':
                t4_gpu_cols  .append(gpu_columns[i])

            if gpu_type['node_'+str(i)+'_gpu_type'] == 'MISC':
                misc_cpu_cols.append(cpu_columns[i])
            elif gpu_type['node_'+str(i)+'_gpu_type'] == 'V100':
                v100_cpu_cols.append(cpu_columns[i])
            elif gpu_type['node_'+str(i)+'_gpu_type'] == 'P100':
                p100_cpu_cols.append(cpu_columns[i])
            elif gpu_type['node_'+str(i)+'_gpu_type'] == 'T4':
                t4_cpu_cols.append(cpu_columns[i])


        for category, columns, perc in [('MISC', misc_cpu_cols, 96), ('V100', v100_cpu_cols, 96), ('P100', p100_cpu_cols, 64), ('T4', t4_cpu_cols, 96)]:
            cpu_ci[category] = df[columns].values.flatten() * 100 / perc
            
        for category, columns, perc in [('MISC', misc_gpu_cols, 8), ('V100', v100_gpu_cols, 8), ('P100', p100_gpu_cols, 2), ('T4', t4_gpu_cols, 2)]:
            gpu_ci[category] = df[columns].values.flatten() * 100 / perc




    else:
        cpu_columns = [col for col in df.columns if 'cpu' in col]
        gpu_columns = [col for col in df.columns if 'gpu' in col]

        misc_gpu_cols = [col for col in gpu_columns if 'MISC' in col]
        v100_gpu_cols = [col for col in gpu_columns if 'V100' in col]
        p100_gpu_cols = [col for col in gpu_columns if 'P100' in col]
        t4_gpu_cols = [col for col in gpu_columns if 'T4' in col]

        misc_cpu_cols = [col for col in cpu_columns if 'MISC' in col]
        v100_cpu_cols = [col for col in cpu_columns if 'V100' in col]
        p100_cpu_cols = [col for col in cpu_columns if 'P100' in col]
        t4_cpu_cols = [col for col in cpu_columns if 'T4' in col]


        # Calculate confidence intervals for each category
        for category, cpu_perc, gpu_perc in zip(cpu_categories, cpu_percentages, gpu_percentages):
            cpu_cols = [col for col in df.columns if 'cpu' in col and category in col]
            gpu_cols = [col for col in df.columns if 'gpu' in col and category in col]

            cpu_ci[category] = df[cpu_cols].values.flatten() * 100 / cpu_perc
            gpu_ci[category] = df[gpu_cols].values.flatten() * 100 / gpu_perc

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
        axs[0].boxplot([cpu_ci[category]], positions=[positions[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))
        axs[1].boxplot([gpu_ci[category]], positions=[positions[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))



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
csv_file = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/1_UTIL_FIFO_0_nosplit.csv'
# csv_file = '1-1_LGF_FIFO_0.25_split.csv'
plot_cpu_gpu(csv_file)
