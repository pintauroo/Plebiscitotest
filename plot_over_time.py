import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_filter_data(directory, csv_files):
    data = {'FIFO': [], 'SDF': []}
    for csv_file in csv_files:
        full_path = os.path.join(directory, csv_file)  # Properly joining paths
        df = pd.read_csv(full_path)
        if 'report' in csv_file:
            df['end_time'] = df['submit_time'] + df['duration']
            # Make sure that operations involving columns result in the correct data type
            filtered_df = df[(df['end_time'] < 16407) & (df['num_gpu'] > 0)]
        else:
            df['end_time'] = df['allocated_at'] + df['duration']
            # Check sum of GPU columns is greater than zero, ensuring data types are handled correctly
            gpu_sum = df.filter(like='gpu').sum(axis=1)
            filtered_df = df[(df['end_time'] < 16407) & (gpu_sum > 0)]

        key = 'FIFO' if 'FIFO' in csv_file else 'SDF'
        data[key].append(filtered_df)

    return data

def aggregate_gpu_utilization(data):
    utilization = {'FIFO': [], 'SDF': []}
    for key, dfs in data.items():
        for df in dfs:
            if 'num_gpu' in df.columns:
                df['total_gpu'] = df['num_gpu'] * df['duration']
                utilization[key].append(df.groupby('allocated_at')['total_gpu'].sum())
            else:
                gpu_cols = [col for col in df.columns if 'gpu' in col and not col.endswith('MISC')]
                for col in gpu_cols:
                    df[col] = df[col] * (df['end_time'] - df['allocated_at'])
                utilization[key].append(df[gpu_cols].sum())

    final_util = {key: pd.concat(vals).groupby(pd.concat(vals).index).mean() for key, vals in utilization.items()}
    return final_util

def plot_gpu_utilization(utilization, output_dir):
    plt.figure(figsize=(10, 6))
    for key, util in utilization.items():
        plt.plot(util.index, util, label=f"{key} Utilization")

    plt.title('GPU Utilization Over Time')
    plt.xlabel('Time')
    plt.ylabel('GPU Utilization')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'gpu_utilization_over_time.png'))
    plt.close()

def plot_average_utilization_comparison(utilization, output_dir):
    averages = {key: util.mean() for key, util in utilization.items()}
    plt.bar(averages.keys(), averages.values(), color=['blue', 'green'])
    plt.title('Average GPU Utilization Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Average Utilization')
    plt.savefig(os.path.join(output_dir, 'average_gpu_utilization_comparison.png'))
    plt.close()

# Example usage



directory = '/home/andrea/projects/Plebiscitotest/'

csv_files = ['1_LGF_FIFO_0_nosplit_norebid_jobs_report.csv','1jobs_FIFO_LGF.csv','1_LGF_SDF_0_nosplit_norebid_jobs_report.csv','1jobs_SDF_LGF.csv']
output_directory = '/home/andrea/projects/Plebiscitotest/output'
data = load_and_filter_data(directory, csv_files)
gpu_utilization = aggregate_gpu_utilization(data)

plot_gpu_utilization(gpu_utilization, output_directory)
plot_average_utilization_comparison(gpu_utilization, output_directory)

