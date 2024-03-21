import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Function to calculate waiting jobs over time for a given DataFrame
def calculate_waiting_jobs(df):
    start_time = df['submit_time'].min() if not df.empty else 0
    end_time = (df['allocated_at'] + df['duration']).max() if not df.empty else 0
    timeline = range(start_time, end_time + 1)

    waiting_jobs = defaultdict(list)

    for _, row in df.iterrows():
        for time in range(row['submit_time'], row['allocated_at'] + 1):
            waiting_jobs[time].append(1)  # Add 1 for each job waiting at 'time'

    # Summarize the counts at each time point
    for time in waiting_jobs:
        waiting_jobs[time] = sum(waiting_jobs[time])

    return waiting_jobs

# List of CSV files to process
csv_files = [
            #  '/home/crownlabs/Plebiscitotest/1_LGF_FIFO_1_nosplit_jobs_report.csv',
             '/home/crownlabs/Plebiscitotest/1jobs_FIFO_LGF.csv',
            #  '/home/crownlabs/Plebiscitotest/1jobs_FIFO_SGF.csv',
            #  '/home/crownlabs/Plebiscitotest/1jobs_FIFO_UTIL.csv',
             ]
# This will hold the waiting jobs counts from all files, keyed by time
aggregate_waiting_jobs = defaultdict(list)

# Process each file
for file in csv_files:
    df = pd.read_csv(file)
    waiting_jobs = calculate_waiting_jobs(df)
    
    # Aggregate counts by time across all files
    for time, count in waiting_jobs.items():
        aggregate_waiting_jobs[time].append(count)

# Prepare data for boxplot
times = sorted(aggregate_waiting_jobs.keys())
data_to_plot = [aggregate_waiting_jobs[time] for time in times]

# Plotting
plt.figure(figsize=(12, 8))

# Creating boxplot
plt.boxplot(data_to_plot, positions=times, showmeans=True)

plt.title('Distribution of Number of Jobs Waiting Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Waiting Jobs')
plt.grid(True)
plt.savefig('box.png')
