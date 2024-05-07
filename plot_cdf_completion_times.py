import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from itertools import cycle

def find_files(mypath, keyword='1'):
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.csv' in f and keyword in f]
    jobs_report = [f for f in file_list if "jobs" in f]
    simulation_report = [f for f in file_list if "jobs" not in f]
    return jobs_report, simulation_report

# Use your path to find CSV files
csv_file_jobs, simulation_report = find_files('/home/andrea/projects/Plebiscitotest/')

# Combining lists of jobs and simulations
csvs = csv_file_jobs

completed_number = {}
# Use itertools.cycle to handle an arbitrary number of files with repeatable patterns
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  # Added more colors
markers = cycle(['o', '^', 's', 'p', 'x', '*', '+'])  # Added more markers

plt.figure(figsize=(10, 6))

for csv_file in csvs:
    color = next(colors)
    marker = next(markers)
    df = pd.read_csv('/home/andrea/projects/Plebiscitotest/' + csv_file)
    if 'UTIL' in csv_file:

        if 'allocations' in csv_file:
            # Filter DataFrame based on a realistic condition
            filtered_df = df[df['submit_time'] != df['allocated_at']]
            completion_time = filtered_df['submit_time'] + filtered_df['duration']
        else:
            # Properly filter and calculate completion time for non-report files
            filtered_df = df[df['allocated_at'] > 0]  # Example condition, adjust as necessary
            completion_time = filtered_df['allocated_at'] + filtered_df['duration']
        
        count = filtered_df.shape[0]
        print(csv_file, count)

        # Track the count based on the file naming
        if 'FIFO' in csv_file:
            completed_number['Plebi_FIFO' if 'report' in csv_file else 'FIFO'] = count
        elif 'SDF' in csv_file:
            completed_number['Plebi_SDF' if 'report' in csv_file else 'SDF'] = count

        # Plotting the CDF for the filtered DataFrame
        data = completion_time.sort_values()
        cdf = np.arange(1, len(data) + 1) / len(data)

        plt.plot(data, cdf, marker=marker, linestyle='-', color=color, label=csv_file)

plt.title('CDF of Completion Times for Different Files')
plt.xlabel('Completion Time')
plt.ylabel('CDF')
plt.grid(True)
plt.legend(loc='best', fontsize='small')
plt.savefig('cdf.png')
plt.close()  # Ensure the plot is closed after saving
