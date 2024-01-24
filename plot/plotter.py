
import pandas as pd
from scipy.stats import t
import pandas as pd
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import t

def preprocess_data(series, threshold=1000):
    # Set a threshold to remove waiting time values beyond the limit
    processed_data = series[series <= threshold]
    return processed_data

def plot_waiting_time_confidence_interval1(csv_files):
    # Initialize lists to store data for each category
    misc_data = []
    v100_data = []
    p100_data = []
    t4_data = []

    # Iterate over CSV files
    for csv_file in csv_files:
        # read csv file into pandas dataframe
        df = pd.read_csv(csv_file)

        # Extract waiting_time data for each category and preprocess it
        misc_waiting_time = preprocess_data(df[df['gpu_type'].str.contains('MISC')]['waiting_time'])
        v100_waiting_time = preprocess_data(df[df['gpu_type'].str.contains('V100')]['waiting_time'])
        p100_waiting_time = preprocess_data(df[df['gpu_type'].str.contains('P100')]['waiting_time'])
        t4_waiting_time = preprocess_data(df[df['gpu_type'].str.contains('T4')]['waiting_time'])

        # Append data to the corresponding category list
        misc_data.append(misc_waiting_time)
        v100_data.append(v100_waiting_time)
        p100_data.append(p100_waiting_time)
        t4_data.append(t4_waiting_time)

    # Plot the boxplots
    fig, axs = plt.subplots(figsize=(10, 5))

    # Add a shift to each box
    box_width = 0.2
    positions = [1, 2, 3, 4]
    positions_shift = [-0.4, -0.2, 0, 0.2]
    box_colors = ['red', 'green', 'blue', 'orange']

    # Store legend patches
    legend_patches = []

    # Modify the boxplot calls to include colors and legend patches
    for i, waiting_time_data in enumerate([misc_data, v100_data, p100_data, t4_data]):
        for j, position in enumerate(positions):
            color = box_colors[i % len(box_colors)]  # Use modulo to cycle through colors for each position
            if j % 4 == 0:
                if i == 0:
                    patch = mpatches.Patch(color=color, label='ID')
                elif i == 1:
                    patch = mpatches.Patch(color=color, label='SGF')
                elif i == 2:
                    patch = mpatches.Patch(color=color, label='LGF')
                elif i == 3:
                    patch = mpatches.Patch(color=color, label='UTIL')
                legend_patches.append(patch)

            axs.boxplot(waiting_time_data[j], positions=[position + positions_shift[i]],
                        widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))

    # Add legend
    axs.legend(handles=legend_patches, title='Legend', loc='upper right')

    axs.set_title('Waiting Time Distribution (Outliers Removed)')
    axs.set_xticks(positions)
    axs.set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    axs.set_xlabel('GPU Type')
    axs.set_ylabel('Waiting Time (s)')

    plt.savefig("box_waiting_time_preprocessed.png")



def plot_cpu_gpu(csv_files):
    # Initialize lists to store data for each category
    misc_cpu_data = []
    v100_cpu_data = []
    p100_cpu_data = []
    t4_cpu_data = []

    misc_gpu_data = []
    v100_gpu_data = []
    p100_gpu_data = []
    t4_gpu_data = []

    

    # Iterate over CSV files
    for csv_file in csv_files:
        # read csv file into pandas dataframe

        if 'split' in csv_file:
            print('plebiscito')
            df = pd.read_csv(basepath_plebi + csv_file + '.csv')

            cpu_columns = [col for col in df.columns if 'used_cpu' in col]
            gpu_columns = [col for col in df.columns if 'used_gpu' in col]

            misc_cpu_cols, v100_cpu_cols, p100_cpu_cols, t4_cpu_cols = [], [], [], []
            misc_gpu_cols, v100_gpu_cols, p100_gpu_cols, t4_gpu_cols = [], [], [], []
            
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




        else:
            print('simulator')
            df = pd.read_csv(basepath_simulator + csv_file + '.csv')
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
        cpu_ci = {}
        gpu_ci = {}

        for category, columns, perc in [('MISC', misc_cpu_cols, 96), ('V100', v100_cpu_cols, 96), ('P100', p100_cpu_cols, 64), ('T4', t4_cpu_cols, 96)]:
            cpu_data = df[columns].values.flatten() * 100 / perc
            # cpu_mean = cpu_data.mean()
            # cpu_std = cpu_data.std(ddof=1)
            # cpu_n = len(cpu_data)
            # cpu_se = cpu_std / cpu_n**0.5
            # cpu_ci[category] = t.interval(0.95, cpu_n - 1, cpu_mean, cpu_se)
            # df[columns] = cpu_data.reshape(df[columns].shape)

            # Append data to the corresponding category list
            if category == 'MISC':
                misc_cpu_data.append(cpu_data)
            elif category == 'V100':
                v100_cpu_data.append(cpu_data)
            elif category == 'P100':
                p100_cpu_data.append(cpu_data)
            elif category == 'T4':
                t4_cpu_data.append(cpu_data)

        for category, columns, perc in [('MISC', misc_gpu_cols, 8), ('V100', v100_gpu_cols, 8), ('P100', p100_gpu_cols, 2), ('T4', t4_gpu_cols, 2)]:
            gpu_data = df[columns].values.flatten() * 100 / perc
            # gpu_mean = gpu_data.mean()
            # gpu_std = gpu_data.std(ddof=1)
            # gpu_n = len(gpu_data)
            # gpu_se = gpu_std / gpu_n**0.5
            # gpu_ci[category] = t.interval(0.95, gpu_n - 1, gpu_mean, gpu_se)
            # df[columns] = gpu_data.reshape(df[columns].shape)

            # Append data to the corresponding category list
            if category == 'MISC':
                misc_gpu_data.append(gpu_data)
            elif category == 'V100':
                v100_gpu_data.append(gpu_data)
            elif category == 'P100':
                p100_gpu_data.append(gpu_data)
            elif category == 'T4':
                t4_gpu_data.append(gpu_data)

    # Your existing code setup
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    box_width = 0.2
    positions = [1, 2, 3, 4]
    positions_shift = [-0.4, -0.2, 0, 0.2]
    box_colors = ['red', 'green', 'blue', 'orange']
    legend_patches = []

    for i, (cpu_data, gpu_data) in enumerate(zip([misc_cpu_data, v100_cpu_data, p100_cpu_data, t4_cpu_data],
                                                [misc_gpu_data, v100_gpu_data, p100_gpu_data, t4_gpu_data])):
        for j, position in enumerate(positions):
            color = box_colors[i % len(box_colors)]
            if j % 4 == 0:
                # Define legend patches
                if i == 0:
                    patch = mpatches.Patch(color=color, label='MISC')
                elif i == 1:
                    patch = mpatches.Patch(color=color, label='V100')
                elif i == 2:
                    patch = mpatches.Patch(color=color, label='P100')
                elif i == 3:
                    patch = mpatches.Patch(color=color, label='T4') 
                legend_patches.append(patch)
            
            # Plotting the boxplot
            axs[0].boxplot(cpu_data[j], positions=[position + positions_shift[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))
            axs[1].boxplot(gpu_data[j], positions=[position + positions_shift[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))

        # Set x-tick labels for each position with a shift
        # shifted_positions = [p + shift for p in positions for shift in positions_shift]
        # repeated_labels = ['MISC', 'V100', 'P100', 'T4'] * len(positions_shift)
        # axs[0].set_xticks(shifted_positions)
        # axs[0].set_xticklabels(repeated_labels, rotation=90)
        # axs[1].set_xticks(shifted_positions)
        # axs[1].set_xticklabels(repeated_labels, rotation=90)

        # Set x-tick labels for each position
        axs[0].set_xticks(positions)
        axs[0].set_xticklabels(csvs)
        axs[1].set_xticks(positions)
        axs[1].set_xticklabels(csvs)

    # Add legends
    axs[0].legend(handles=legend_patches, title='Legend', loc='upper right')
    axs[1].legend(handles=legend_patches, title='Legend', loc='upper right')

    axs[0].set_title('CPU')
    axs[1].set_title('GPU')

    plt.savefig("boxcpu_gpu_fifo.png")

basepath_simulator = '/home/crownlabs/Plebiscitotest/res/'
basepath_plebi = '/home/crownlabs/Plebiscitotest/'
# csvs = ['1_UTIL_FIFO_0_nosplit','2_LGF_FIFO_0_nosplit','data_FIFO_LGF','data_FIFO_UTIL']
csvs = ['1_0_UTIL_FIFO_0_split','1_1_UTIL_FIFO_0_split','1_1_UTIL_FIFO_0_split','1_3_UTIL_FIFO_0_split']
# plot_cpu_gpu(['data_0_2.csv','data_0_2.csv','data_0_2.csv','data_0_2.csv'])
plot_cpu_gpu(csvs)
# plot_cpu_gpu(['data_SDF_ID.csv','data_SDF_SGF.csv','data_SDF_LGF.csv','data_SDF_UTIL.csv'])
# plot_cpu_gpu(['data_0_0.csv', 'data_0_1.csv', 'data_0_2.csv', 'data_0_3.csv'])
# plot_cpu_gpu(['data_8_0.csv', 'data_8_1.csv', 'data_8_2.csv', 'data_8_3.csv'])


# plot_waiting_time_confidence_interval1(['jobs_0_0.csv', 'jobs_0_1.csv', 'jobs_0_2.csv', 'jobs_0_3.csv'])
# plot_waiting_time_confidence_interval1(['jobs_8_0.csv', 'jobs_8_1.csv', 'jobs_8_2.csv', 'jobs_8_3.csv'])
