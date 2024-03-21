import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read the CSV file and calculate waiting times
def calculate_waiting_times(csv_file):
    df = pd.read_csv(directory+ '/' + csv_file)
    # df['waiting_time'] = pd.to_datetime(df['allocated_at'] if 'split' not in csv_file else (df['deadline']).max()) - pd.to_datetime(df['submit_time'])
    df['waiting_time'] = (df['allocated_at'] if 'split' not in csv_file else df['exec_time']) - df['submit_time']
    # df['waiting_time'] = df['waiting_time'].dt.total_seconds() / 60  # Convert to minutes
    return df['waiting_time']

# Function to plot the CDF of waiting times for multiple groups
def plot_multiple_cdfs(data_dict):
    plt.figure(figsize=(10, 8))
    
    for label, data in data_dict.items():
        # Aggregate waiting times from all CSVs in the list
        all_waiting_times = []
        for csv_file in data:
            waiting_times = calculate_waiting_times(csv_file)
            all_waiting_times.extend(waiting_times.tolist())
        
        # Calculate and plot the CDF for this group
        data_sorted = np.sort(all_waiting_times)
        cdf = np.arange(1, len(data_sorted)+1) / len(data_sorted)
        plt.plot(data_sorted, cdf, marker='.', linestyle='none', label=label)
    
    plt.title('CDF of Job Waiting Times by Group')
    plt.xlabel('Waiting Time (minutes)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('multiple_cdfs_waiting_times.png')



def find_csvs():

    # Keywords for scheduling and allocation
    # scheduling_keywords = ["FIFO", "SDF"]
    scheduling_keywords = ["SDF"]
    allocation_keywords = ["LGF", "SGF", "UTIL"]

    # Dictionary to store combinations
    combinations = {}

    # Function to ensure the key exists in the combinations dictionary
    def add_to_combinations(key, filename):
        if key not in combinations:
            combinations[key] = []
        combinations[key].append(filename)

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and 'jobs' in filename:
            if 'nosplit' in filename:
                # Add filename under the 'split' key
                for scheduling in scheduling_keywords:
                    for allocation in allocation_keywords:
                        if scheduling in filename and allocation in filename:
                            # Save the filename under the specific combination
                            combination_key = scheduling + '_' + allocation
                            add_to_combinations('nosplit_'+combination_key, filename)
            elif 'split' in filename:
                pass
                # for scheduling in scheduling_keywords:
                #     for allocation in allocation_keywords:
                #         if scheduling in filename and allocation in filename:
                #             # Save the filename under the specific combination
                #             combination_key = scheduling + '_' + allocation
                #             add_to_combinations('split_'+combination_key, filename)
            else:
                # Iterate through scheduling and allocation keywords
                for scheduling in scheduling_keywords:
                    for allocation in allocation_keywords:
                        if scheduling in filename and allocation in filename:
                            # Save the filename under the specific combination
                            combination_key = scheduling + '_' + allocation
                            add_to_combinations(combination_key, filename)

    # Print or save the unique combinations
    print(combinations)

    return combinations


directory = "."

csv_files_dict = find_csvs()


plot_multiple_cdfs(csv_files_dict)