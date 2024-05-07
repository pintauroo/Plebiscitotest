
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



def plot_cpu_gpu(csv_files, completed_number):
    # Initialize lists to store data for each category
    misc_cpu_data = []
    v100_cpu_data = []
    p100_cpu_data = []
    t4_cpu_data = []

    misc_gpu_data = []
    v100_gpu_data = []
    p100_gpu_data = []
    t4_gpu_data = []

    lables = []
    

    # Iterate over CSV files
    for csv_file in csv_files:
        # read csv file into pandas dataframe
        df = pd.read_csv('/home/andrea/projects/Plebiscitotest/'+csv_file)
        num_rows, num_columns = df.shape
        print(csv_file)
        print("Number of rows:", num_rows)
        print("Number of columns:", num_columns)
        df = df.loc[0:1000].reset_index(drop=True)


        if 'bid' in csv_file:

            if 'FIFO' in csv_file:
                lables.append('Plebi_FIFO ' + str(completed_number['Plebi_FIFO']))
            elif 'SDF' in csv_file:
                lables.append('Plebi_SDF ' + str(completed_number['Plebi_SDF']))
            
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
            if 'FIFO' in csv_file:
                lables.append('FIFO ' + str(completed_number['FIFO']))
            elif 'SDF' in csv_file:
                lables.append('SDF ' + str(completed_number['SDF']))
            
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
            cpu_mean = cpu_data.mean()
            cpu_std = cpu_data.std(ddof=1)
            cpu_n = len(cpu_data)
            cpu_se = cpu_std / cpu_n**0.5
            cpu_ci[category] = t.interval(0.95, cpu_n - 1, cpu_mean, cpu_se)
            df[columns] = cpu_data.reshape(df[columns].shape)

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
            gpu_mean = gpu_data.mean()
            gpu_std = gpu_data.std(ddof=1)
            gpu_n = len(gpu_data)
            gpu_se = gpu_std / gpu_n**0.5
            gpu_ci[category] = t.interval(0.95, gpu_n - 1, gpu_mean, gpu_se)
            df[columns] = gpu_data.reshape(df[columns].shape)

            # Append data to the corresponding category list
            if category == 'MISC':
                misc_gpu_data.append(gpu_data)
            elif category == 'V100':
                v100_gpu_data.append(gpu_data)
            elif category == 'P100':
                p100_gpu_data.append(gpu_data)
            elif category == 'T4':
                t4_gpu_data.append(gpu_data)
    # Settings for the plots
    box_width = 0.2
    positions = [1, 2, 3, 4]
    positions_shift = [-0.4, -0.2, 0, 0.2]
    box_colors = ['red', 'green', 'blue', 'orange']

    # Plot for CPU
    fig_cpu, ax_cpu = plt.subplots(figsize=(10, 5))
    legend_patches = []

    for i, cpu_data in enumerate([misc_cpu_data, v100_cpu_data, p100_cpu_data, t4_cpu_data]):
        # for j, position in enumerate(positions):
            color = box_colors[i % len(box_colors)]
            if i % 4 == 0:
                patch = mpatches.Patch(color=color, 
                                       label=lables[0])
                                    #    label=['FIFO ', 'SDF', 'P_FIFO', 'P_SDF'][i])
                legend_patches.append(patch)
            ax_cpu.boxplot(cpu_data[0], positions=[positions[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))

    ax_cpu.legend(handles=legend_patches, title='Legend', loc='lower left')
    ax_cpu.set_title('CPU Utilization by Model')
    ax_cpu.set_xticks(positions)
    ax_cpu.set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    plt.savefig("cpu_boxplot_s.png")

    # Plot for GPU
    fig_gpu, ax_gpu = plt.subplots(figsize=(10, 5))
    legend_patches = []

    for i, gpu_data in enumerate([misc_gpu_data, v100_gpu_data, p100_gpu_data, t4_gpu_data]):
        # if i % 4 == 0:

        # for j, position in enumerate(positions):
            color = box_colors[0]
            
            if i%4==0:
                patch = mpatches.Patch(color=color, label=lables[0])
                legend_patches.append(patch)

            ax_gpu.boxplot(gpu_data[0], positions=[positions[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))

    ax_gpu.legend(handles=legend_patches, title='Legend', loc='lower left')
    ax_gpu.set_title('GPU Utilization by Model')
    ax_gpu.set_xticks(positions)
    ax_gpu.set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    plt.savefig("gpu_boxplot_s.png")





# csvs = ['1_LGF_FIFO_0_nosplit_norebid_jobs_report.csv','1jobs_FIFO_LGF.csv','1_LGF_SDF_0_nosplit_norebid_jobs_report.csv','1jobs_SDF_LGF.csv']
csvs = ['1_UTIL_FIFO_0_nosplit_norebid_jobs_report.csv','1jobs_FIFO_UTIL.csv','1_UTIL_SDF_0_nosplit_norebid_jobs_report.csv','1jobs_SDF_UTIL.csv']

completed_number = {}
for csv_file in csvs:
    df = pd.read_csv('/home/andrea/projects/Plebiscitotest/'+csv_file)
    if 'report' in csv_file:
        filtered_df = df[df['submit_time'] + df['duration'] < 6000]
        count = filtered_df.shape[0]
        print(csv_file, count)
        if 'FIFO' in csv_file:
            completed_number['Plebi_FIFO'] = count
        elif 'SDF' in csv_file:
            completed_number['Plebi_SDF'] = count

        
    else:
        filtered_df = df[df['allocated_at'] + df['duration'] < 6000]
        count = filtered_df.shape[0]
        print(csv_file,count)
        if 'FIFO' in csv_file:
            completed_number['FIFO'] = count
        elif 'SDF' in csv_file:
            completed_number['SDF'] = count

csvs = ['1_UTIL_FIFO_0_nosplit_norebid.csv']

plot_cpu_gpu(csvs, completed_number)



