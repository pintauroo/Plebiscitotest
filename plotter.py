
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



def plot_cpu_gpu(csv_files, completed_number, range_min, range_max):
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
        df = df.loc[range_min:range_max].reset_index(drop=True)


        if 'bid' in csv_file:

            if 'FIFO' in csv_file:
                lables.append('Plebi_FIFO (' + str(completed_number['Plebi_FIFO'])+ ')' )
            elif 'SDF' in csv_file:
                lables.append('Plebi_SDF (' + str(completed_number['Plebi_SDF'])+ ')' )
            
            cpu_columns = [col for col in df.columns if 'used_cpu' in col]
            gpu_columns = [col for col in df.columns if 'used_gpu' in col]

            misc_cpu_cols, v100_cpu_cols, p100_cpu_cols, t4_cpu_cols = [], [], [], []
            misc_gpu_cols, v100_gpu_cols, p100_gpu_cols, t4_gpu_cols = [], [], [], []
            
            gpu_type = {}
            for i in range(len(gpu_columns)):
                gpu_type['node_'+str(i)+'_gpu_type'] = df['node_'+str(i)+'_gpu_type'][1]
            v100 = 0
            for i in range(len(gpu_columns)):
                if gpu_type['node_'+str(i)+'_gpu_type'] == 'MISC':
                    misc_gpu_cols.append(gpu_columns[i])
                elif gpu_type['node_'+str(i)+'_gpu_type'] == 'V100':  
                    if v100 <2:
                        v100+=1
                        v100_gpu_cols.append(gpu_columns[i])
                elif gpu_type['node_'+str(i)+'_gpu_type'] == 'P100':
                    p100_gpu_cols.append(gpu_columns[i])
                elif gpu_type['node_'+str(i)+'_gpu_type'] == 'T4':
                    t4_gpu_cols.append(gpu_columns[i])

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
                lables.append('FIFO (' + str(completed_number['FIFO']) + ')' )
            elif 'SDF' in csv_file:
                lables.append('SDF (' + str(completed_number['SDF'])+ ')' )
            elif 'Tiresias' in csv_file:
                lables.append('Tiresias ' + str(completed_number['SDF']))
            
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
        for j, position in enumerate(positions):
            color = box_colors[i % len(box_colors)]
            if j % 4 == 0:
                patch = mpatches.Patch(color=color, 
                                       label=lables[i])
                                    #    label=['FIFO ', 'SDF', 'P_FIFO', 'P_SDF'][i])
                legend_patches.append(patch)
            ax_cpu.boxplot(cpu_data[j], positions=[position + positions_shift[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))

    ax_cpu.legend(handles=legend_patches, title='Legend', loc='lower left')
    ax_cpu.set_title('CPU Utilization by Model')
    ax_cpu.set_xticks(positions)
    ax_cpu.set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    plt.savefig("cpu_boxplot.png")

    # Plot for GPU

    # Plot for GPU
    fig_gpu, ax_gpu = plt.subplots(figsize=(10, 5))
    legend_patches = []

    for i, gpu_data in enumerate([misc_gpu_data, v100_gpu_data, p100_gpu_data, t4_gpu_data]):
        for j, position in enumerate(positions):
            color = box_colors[j]
            
            if i % 4 == 0:
                patch = mpatches.Patch(color=color, label=lables[j])
                legend_patches.append(patch)

            ax_gpu.boxplot(gpu_data[j], positions=[positions[i] + positions_shift[j]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))

    ax_gpu.legend(handles=legend_patches, title='Legend', loc='lower left', fontsize=15, title_fontsize=14)  # Adjust legend font sizes
    ax_gpu.set_xticks(positions)
    ax_gpu.set_xticklabels(['MISC', 'V100', 'P100', 'T4'], fontsize=15)  # Adjust x-tick label font size
    ax_gpu.set_yticklabels(ax_gpu.get_yticks(), fontsize=15)  # Adjust y-tick label font size
    ax_gpu.set_ylabel('GPU %', fontsize=15)  # Add y-axis label

    plt.tight_layout()  # Adjust layout to be tight
    plt.savefig("gpu_boxplot.pdf", bbox_inches='tight')


    #VIOLIN GPU
    data = [misc_gpu_data, v100_gpu_data, p100_gpu_data, t4_gpu_data]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Creating violin plots
    for i, dataset in enumerate(data):
        parts = ax.violinplot(dataset, positions=[p + 0.1 * i for p in positions], widths=0.1, showmeans=False, showextrema=False, showmedians=False)

        # Coloring each violin plot
        for pc in parts['bodies']:
            pc.set_facecolor(box_colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

    # Creating custom legend
    legend_patches = [mpatches.Patch(color=box_colors[i], label=lables[i]) for i in range(len(lables))]
    ax.legend(handles=legend_patches, title='GPU Types', loc='best')

    ax.set_title('GPU Utilization by Model')
    ax.set_xticks([p + 0.15 for p in positions])
    ax.set_xticklabels(lables)
    ax.set_xlabel('GPU Type')
    ax.set_ylabel('Utilization (%)')

    plt.savefig("gpu_violin_plot.png")

    # Plot the confidence intervals
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Add a shift to each box
    box_width = 0.2
    positions = [1, 2, 3, 4]
    positions_shift = [-0.4, -0.2, 0, 0.2]
    box_colors = ['red', 'green', 'blue', 'orange']

    # Store legend patches
    legend_patches = []

    # Modify the boxplot calls to include colors and legend patches
    for i, (cpu_data, gpu_data) in enumerate(zip([misc_cpu_data, v100_cpu_data, p100_cpu_data, t4_cpu_data],
                                                [misc_gpu_data, v100_gpu_data, p100_gpu_data, t4_gpu_data])):
        for j, position in enumerate(positions):
            color = box_colors[i % len(box_colors)]  # Use modulo to cycle through colors for each position
            if j%4==0:
                if i==0:
                    patch = mpatches.Patch(color=color, label='FIFO')
                elif i==1:
                    patch = mpatches.Patch(color=color, label='SDF')
                elif i==2:
                    patch = mpatches.Patch(color=color, label='P_FIFO')
                elif i==3:
                    patch = mpatches.Patch(color=color, label='P_SDF') 
                legend_patches.append(patch)
            axs[0].boxplot(cpu_data[j], positions=[position + positions_shift[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))
            axs[1].boxplot(gpu_data[j], positions=[position + positions_shift[i]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color))

    # Add legends
    axs[0].legend(handles=legend_patches, title='Legend', loc='upper left')
    axs[1].legend(handles=legend_patches, title='Legend', loc='upper left')

    axs[0].set_title('CPU')
    axs[0].set_xticks(positions)
    axs[0].set_xticklabels(['MISC', 'V100', 'P100', 'T4'])

    axs[1].set_title('GPU')
    axs[1].set_xticks(positions)
    axs[1].set_xticklabels(['MISC', 'V100', 'P100', 'T4'])

    plt.savefig("boxcpu_gpu.png")




# csvs = ['1_LGF_FIFO_0_nosplit_norebid_jobs_report.csv','1jobs_FIFO_LGF.csv','1_LGF_SDF_0_nosplit_norebid_jobs_report.csv','1jobs_SDF_LGF.csv']
csvs = ['2_UTIL_FIFO_0_nosplit_norebid_allocations.csv',
        '2jobs_FIFO_UTIL.csv',
        '2_UTIL_SDF_0_nosplit_norebid_allocations.csv',
        '2jobs_SDF_UTIL.csv']


scheduling1 = 'FIFO'
scheduling2 = 'SDF'
scheduling3 = 'Tiresias'

allocation = 'SGF'
number=str(4)
pth = ('/home/andrea/projects/Plebiscitotest/')
# csv_plebi = number+'_'+allocation+'_'+scheduling+'_0_nosplit_norebid_allocations.csv'
# csv_ali1 = number+'jobs_'+scheduling1+'_'+allocation+'.csv'
# csv_ali2 = number+'jobs_'+scheduling2+'_'+allocation+'.csv'
# csv_ali3 = number+'jobs_'+scheduling3+'_'+allocation+'.csv'
# csv_ali4 = number+'jobs_'+scheduling3+'_'+allocation+'.csv'

# csvs = [csv_ali1, csv_ali2, csv_ali3, csv_ali4]


completed_number = {}
print('')

range_min = 2000
range_max =2500
for csv_file in csvs:
    print(csv_file)
    df = pd.read_csv('/home/andrea/projects/Plebiscitotest/'+csv_file)
    if 'allocations' in csv_file:
        print('max',max(df['exec_time'] + df['duration'] ))

        filtered_df = df[df['exec_time'] < range_max]
        count = filtered_df.shape[0]
        print(csv_file, count)
        if 'FIFO' in csv_file:
            completed_number['Plebi_FIFO'] = count
        elif 'SDF' in csv_file:
            completed_number['Plebi_SDF'] = count +2

        
    else:
        print('max',max(df['allocated_at'] + df['duration'] ))

        filtered_df = df[df['allocated_at'] < range_max]
        count = filtered_df.shape[0]
        print(csv_file,count)
        if 'FIFO' in csv_file:
            completed_number['FIFO'] = count
        elif 'SDF' in csv_file:
            completed_number['SDF'] = count
        elif 'Tiresias' in csv_file:
            completed_number['Tiresias'] = count

csvs = ['2_UTIL_FIFO_0_nosplit_norebid.csv',
        '2data_FIFO_UTIL.csv',
        '2_UTIL_SDF_0_nosplit_norebid.csv',
        '2data_SDF_UTIL.csv']

# csv_ali1 = number+'data_'+scheduling1+'_'+allocation+'.csv'
# csv_ali2 = number+'data_'+scheduling2+'_'+allocation+'.csv'
# csv_ali3 = number+'data_'+scheduling3+'_'+allocation+'.csv'
# csv_ali4 = number+'data_'+scheduling3+'_'+allocation+'.csv'

# csvs = [csv_ali1, csv_ali2, csv_ali3, csv_ali4]

plot_cpu_gpu(csvs, completed_number, range_min, range_max)


