import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_cpu_gpu_time(csv_files):
    
    num_files = len(csv_files)
    fig, axs = plt.subplots(num_files, 2, figsize=(10import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_cpu_gpu_time(csv_files):
    
    num_files = len(csv_files)
    fig, axs = plt.subplots(num_files, 2, figsize=(1, 25))
    


    # Convert axs to a 2D array if it's 1D
    if num_files == 1:
        axs = [axs]
    # Iterate over CSV files
    for n, csv_file in enumerate(csv_files):
        parts = csv_file.split('/')

        # Get the last part of the path (excluding the .csv extension)
        last_part = parts[-1].replace('.csv', '')
        # Initialize lists to store data for each category
        misc_cpu_data = []
        misc_gpu_data = []
        v100_cpu_data = []
        v100_gpu_data = []
        p100_cpu_data = []
        p100_gpu_data = []
        t4_cpu_data = []
        t4_gpu_data = []

        # read csv file into pandas dataframe
        df = pd.read_csv(csv_file)

        if 'FIFO' in csv_file:
            
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

            # Append data to respective lists
            misc_cpu_data.append(df[misc_cpu_cols])
            misc_gpu_data.append(df[misc_gpu_cols])

            v100_cpu_data.append(df[v100_cpu_cols])
            v100_gpu_data.append(df[v100_gpu_cols])
            
            p100_cpu_data.append(df[p100_cpu_cols])
            p100_gpu_data.append(df[p100_gpu_cols])
            
            t4_cpu_data.append(df[t4_cpu_cols])
            t4_gpu_data.append(df[t4_gpu_cols])

        # Calculate percentages for CPU and GPU data
        for category, cpu_cols, gpu_cols, cpu_perc, gpu_perc in [
            ('MISC', misc_cpu_cols, misc_gpu_cols, 96, 8),
            ('V100', v100_cpu_cols, v100_gpu_cols, 96, 8),
            ('P100', p100_cpu_cols, p100_gpu_cols, 64, 2),
            ('T4', t4_cpu_cols, t4_gpu_cols, 96, 2)
        ]:
            time_intervals = range(len(df))

        # Plot CPU data
        for i, col in enumerate(cpu_cols):
            cpu_data = (df[col].values.flatten())
            axs[n][0].plot(time_intervals, cpu_data, label=f'{category} CPU {i+1}')
            axs[n][0].set_title(f'{category} CPU {last_part} {n+1} Usage Over Time')
            axs[n][0].set_xlabel('Time Interval')
            axs[n][0].set_ylabel('CPU Usage (%)')

        for i, col in enumerate(gpu_cols):
            gpu_data = (df[col].values.flatten())
            axs[n][1].plot(time_intervals, gpu_data, label=f'{category} GPU {i+1}')
            axs[n][1].set_title(f'{category} GPU {last_part} {n+1} Usage Over Time')
            axs[n][1].set_xlabel('Time Interval')
            axs[n][1].set_ylabel('GPU Usage (%)')

        plt.tight_layout()
            
            

        plt.savefig(f'plot.png')


def read_csv_files_in_directory(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    csvs = []
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        # df = pd.read_csv(file_path)
        if 'data' in file_path:
            csvs.append(file_path)
    
    return csvs


directory_path = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/res/'
csvs = read_csv_files_in_directory(directory_path)

# Example usage
# csv_file = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/1tst_UTIL_FIFO_0_nosplit.csv'
# csv_file = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/1tstconskipdeconfilctions_UTIL_FIFO_0_nosplit.csv'

# csv_file = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/res/data_8_0.csv'
# csv_file = '1-1_LGF_FIFO_0.25_split.csv'
plot_cpu_gpu_time(csvs)

# for c in csvs:
    # plot_cpu_gpu_time([c])



"""
      job_id  num_inst  submit_time  num_cpu  num_gpu gpu_type  duration          user    size  on_time  wasted  jct     resource  node  allocated_at
154      192         1         1019      6.0      3.0     V100        16  45bd92802aa9     144        0       0   -1   [3.0, 6.0]  None             0
163      203         1         1067      6.0      3.0     V100        16  45bd92802aa9     144        0       0   -1   [3.0, 6.0]  None             0
267      336         1         2084     18.0      1.0     V100      1292  eea150bdfdd6   24548        0       0   -1  [1.0, 18.0]  None             0
270      339         1         2102     18.0      1.0     V100      1365  eea150bdfdd6   25935        0       0   -1  [1.0, 18.0]  None             0
284      354         1         2204     26.0      2.0     V100      3907  a8192d6b0ae9  109396        0       0   -1  [2.0, 26.0]  None             0
410      501         1         2834     18.0      2.0     V100       146  69d8872acc41    2920        0       0   -1  [2.0, 18.0]  None             0
685      827         1         4257      6.0      5.0     V100     15242  de2bfa3294bc  167662        0       0   -1   [5.0, 6.0]  None             0
1133    1347         1         6922     16.0      4.0     V100      4847  262fbc8b0c98   96940        0       0   -1  [4.0, 16.0]  None             0
1167    1386         1         7157      6.0      0.5     V100     28771  3aeb917d7320  187011        0       0   -1   [0.5, 6.0]  None             0
1172    1391         1         7183      6.0      0.5     V100     18230  3aeb917d7320  118495        0       0   -1   [0.5, 6.0]  None             0





      job_id  num_inst  submit_time  num_cpu  num_gpu gpu_type  duration          user    size  on_time  wasted  jct     resource  node  allocated_at
256      324         1         2011     18.0     1.00     P100      1673  eea150bdfdd6   31787        0       0   -1  [1.0, 18.0]  None             0
472      583         1         3106     18.0     1.00     P100       115  a8192d6b0ae9    2185        0       0   -1  [1.0, 18.0]  None             0
474      586         1         3121     18.0     1.00     P100      1763  a8192d6b0ae9   33497        0       0   -1  [1.0, 18.0]  None             0
482      596         1         3191      6.0     1.00     P100      2105  7b76597f4283   14735        0       0   -1   [1.0, 6.0]  None             0
566      697         1         3749      4.0     0.25     P100       868  6c545bed7d03    3689        0       0   -1  [0.25, 4.0]  None             0
572      704         1         3755      6.0     1.00     P100       277  354e3b81c515    1939        0       0   -1   [1.0, 6.0]  None             0
847     1024         1         5314      4.0     1.00     P100      1553  b3bfe9b79bb5    7765        0       0   -1   [1.0, 4.0]  None             0
871     1053         1         5571      8.0     1.00     P100       244  a8192d6b0ae9    2196        0       0   -1   [1.0, 8.0]  None             0
900     1085         1         5844      6.0     0.50     P100     20276  670e1439e0b5  131794        0       0   -1   [0.5, 6.0]  None             0
905     1090         1         5889     18.0     1.00     P100       184  5f4cb64dc693    3496        0       0   -1  [1.0, 18.0]  None             0
1094    1304         1         6625     10.0     1.00     P100      1383  edc10645cc3f   15213        0       0   -1  [1.0, 10.0]  None             0
1098    1309         1         6648      6.0     1.00     P100       137  edc10645cc3f     959        0       0   -1   [1.0, 6.0]  None             0
1456    1725         1         9400      2.0     1.00     P100       889  b3bfe9b79bb5    2667        0       0   -1   [1.0, 2.0]  None             0

""", 25))
    


    # Convert axs to a 2D array if it's 1D
    if num_files == 1:
        axs = [axs]
    # Iterate over CSV files
    for n, csv_file in enumerate(csv_files):
        parts = csv_file.split('/')

        # Get the last part of the path (excluding the .csv extension)
        last_part = parts[-1].replace('.csv', '')
        # Initialize lists to store data for each category
        misc_cpu_data = []
        misc_gpu_data = []
        v100_cpu_data = []
        v100_gpu_data = []
        p100_cpu_data = []
        p100_gpu_data = []
        t4_cpu_data = []
        t4_gpu_data = []

        # read csv file into pandas dataframe
        df = pd.read_csv(csv_file)

        if 'FIFO' in csv_file:
            
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

            # Append data to respective lists
            misc_cpu_data.append(df[misc_cpu_cols])
            misc_gpu_data.append(df[misc_gpu_cols])

            v100_cpu_data.append(df[v100_cpu_cols])
            v100_gpu_data.append(df[v100_gpu_cols])
            
            p100_cpu_data.append(df[p100_cpu_cols])
            p100_gpu_data.append(df[p100_gpu_cols])
            
            t4_cpu_data.append(df[t4_cpu_cols])
            t4_gpu_data.append(df[t4_gpu_cols])

        # Calculate percentages for CPU and GPU data
        for category, cpu_cols, gpu_cols, cpu_perc, gpu_perc in [
            ('MISC', misc_cpu_cols, misc_gpu_cols, 96, 8),
            ('V100', v100_cpu_cols, v100_gpu_cols, 96, 8),
            ('P100', p100_cpu_cols, p100_gpu_cols, 64, 2),
            ('T4', t4_cpu_cols, t4_gpu_cols, 96, 2)
        ]:
            time_intervals = range(len(df))

        # Plot CPU data
        for i, col in enumerate(cpu_cols):
            cpu_data = (df[col].values.flatten())
            axs[n][0].plot(time_intervals, cpu_data, label=f'{category} CPU {i+1}')
            axs[n][0].set_title(f'{category} CPU {last_part} {n+1} Usage Over Time')
            axs[n][0].set_xlabel('Time Interval')
            axs[n][0].set_ylabel('CPU Usage (%)')

        for i, col in enumerate(gpu_cols):
            gpu_data = (df[col].values.flatten())
            axs[n][1].plot(time_intervals, gpu_data, label=f'{category} GPU {i+1}')
            axs[n][1].set_title(f'{category} GPU {last_part} {n+1} Usage Over Time')
            axs[n][1].set_xlabel('Time Interval')
            axs[n][1].set_ylabel('GPU Usage (%)')

        plt.tight_layout()
            
            

        plt.savefig(f'plot.png')


def read_csv_files_in_directory(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    csvs = []
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        # df = pd.read_csv(file_path)
        if 'data' in file_path:
            csvs.append(file_path)
    
    return csvs


directory_path = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/res/'
csvs = read_csv_files_in_directory(directory_path)

# Example usage
# csv_file = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/1tst_UTIL_FIFO_0_nosplit.csv'
# csv_file = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/1tstconskipdeconfilctions_UTIL_FIFO_0_nosplit.csv'

# csv_file = '/home/crownlabs/Plebiscitotest/cluster-trace-gpu-v2020/simulator/res/data_8_0.csv'
# csv_file = '1-1_LGF_FIFO_0.25_split.csv'
plot_cpu_gpu_time(csvs)

# for c in csvs:
    # plot_cpu_gpu_time([c])



"""
      job_id  num_inst  submit_time  num_cpu  num_gpu gpu_type  duration          user    size  on_time  wasted  jct     resource  node  allocated_at
154      192         1         1019      6.0      3.0     V100        16  45bd92802aa9     144        0       0   -1   [3.0, 6.0]  None             0
163      203         1         1067      6.0      3.0     V100        16  45bd92802aa9     144        0       0   -1   [3.0, 6.0]  None             0
267      336         1         2084     18.0      1.0     V100      1292  eea150bdfdd6   24548        0       0   -1  [1.0, 18.0]  None             0
270      339         1         2102     18.0      1.0     V100      1365  eea150bdfdd6   25935        0       0   -1  [1.0, 18.0]  None             0
284      354         1         2204     26.0      2.0     V100      3907  a8192d6b0ae9  109396        0       0   -1  [2.0, 26.0]  None             0
410      501         1         2834     18.0      2.0     V100       146  69d8872acc41    2920        0       0   -1  [2.0, 18.0]  None             0
685      827         1         4257      6.0      5.0     V100     15242  de2bfa3294bc  167662        0       0   -1   [5.0, 6.0]  None             0
1133    1347         1         6922     16.0      4.0     V100      4847  262fbc8b0c98   96940        0       0   -1  [4.0, 16.0]  None             0
1167    1386         1         7157      6.0      0.5     V100     28771  3aeb917d7320  187011        0       0   -1   [0.5, 6.0]  None             0
1172    1391         1         7183      6.0      0.5     V100     18230  3aeb917d7320  118495        0       0   -1   [0.5, 6.0]  None             0





      job_id  num_inst  submit_time  num_cpu  num_gpu gpu_type  duration          user    size  on_time  wasted  jct     resource  node  allocated_at
256      324         1         2011     18.0     1.00     P100      1673  eea150bdfdd6   31787        0       0   -1  [1.0, 18.0]  None             0
472      583         1         3106     18.0     1.00     P100       115  a8192d6b0ae9    2185        0       0   -1  [1.0, 18.0]  None             0
474      586         1         3121     18.0     1.00     P100      1763  a8192d6b0ae9   33497        0       0   -1  [1.0, 18.0]  None             0
482      596         1         3191      6.0     1.00     P100      2105  7b76597f4283   14735        0       0   -1   [1.0, 6.0]  None             0
566      697         1         3749      4.0     0.25     P100       868  6c545bed7d03    3689        0       0   -1  [0.25, 4.0]  None             0
572      704         1         3755      6.0     1.00     P100       277  354e3b81c515    1939        0       0   -1   [1.0, 6.0]  None             0
847     1024         1         5314      4.0     1.00     P100      1553  b3bfe9b79bb5    7765        0       0   -1   [1.0, 4.0]  None             0
871     1053         1         5571      8.0     1.00     P100       244  a8192d6b0ae9    2196        0       0   -1   [1.0, 8.0]  None             0
900     1085         1         5844      6.0     0.50     P100     20276  670e1439e0b5  131794        0       0   -1   [0.5, 6.0]  None             0
905     1090         1         5889     18.0     1.00     P100       184  5f4cb64dc693    3496        0       0   -1  [1.0, 18.0]  None             0
1094    1304         1         6625     10.0     1.00     P100      1383  edc10645cc3f   15213        0       0   -1  [1.0, 10.0]  None             0
1098    1309         1         6648      6.0     1.00     P100       137  edc10645cc3f     959        0       0   -1   [1.0, 6.0]  None             0
1456    1725         1         9400      2.0     1.00     P100       889  b3bfe9b79bb5    2667        0       0   -1   [1.0, 2.0]  None             0

"""