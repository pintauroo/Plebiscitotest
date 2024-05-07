import pandas as pd
import matplotlib.pyplot as plt

def plot_cpu_gpu_time(csv_file):
    print("Processing:", csv_file)

    # Read csv file into pandas dataframe
    df = pd.read_csv(csv_file)

    df = df.loc[0:1000].reset_index(drop=True)


    cpu_columns = [col for col in df.columns if 'used_cpu' in col]
    gpu_columns = [col for col in df.columns if 'used_gpu' in col]

    # Initialize storage for columns by GPU type
    gpu_data = {
        'MISC': {'cpu': [], 'gpu': []},
        'V100': {'cpu': [], 'gpu': []},
        'P100': {'cpu': [], 'gpu': []},
        'T4': {'cpu': [], 'gpu': []}
    }

    # Classify columns by GPU type
    for i, col in enumerate(gpu_columns):
        gpu_type = df['node_' + str(i) + '_gpu_type'][1]  # Assuming consistent type across the DataFrame
        if gpu_type in ['MISC', 'V100', 'P100', 'T4']:  # Ensure the GPU type is one of the expected types
            gpu_data[gpu_type]['gpu'].append(col)
            corresponding_cpu_col = cpu_columns[i]  # Match CPU column by index
            gpu_data[gpu_type]['cpu'].append(corresponding_cpu_col)

    # Plotting data for each GPU type
    for gpu_type in ['MISC', 'V100', 'P100', 'T4']:


        plt.figure(figsize=(20, 10))
        for i, col in enumerate(gpu_columns):
            gpu_type_node = df['node_' + str(i) + '_gpu_type'][1]
            if gpu_type_node == gpu_type:



                plt.plot(df.index, df[col], label=f'GPU {col}')
                plt.title(f'{gpu_type} GPU Usage Over Time')
                plt.xlabel('Time Interval')
                plt.ylabel('GPU Usage (%)')
                plt.legend()

        plt.tight_layout()
        output_path = f"{gpu_type}_usage_plot.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to {output_path}")

# Example usage
csv_file = '/home/andrea/projects/Plebiscitotest/1_SGF_FIFO_0_nosplit_norebid.csv'
plot_cpu_gpu_time(csv_file)
