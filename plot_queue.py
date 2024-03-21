import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate waiting jobs over time for a given DataFrame
def calculate_waiting_jobs(df, file):
    start_time = df['submit_time'].min()
    end_time = (df['allocated_at']).max() if 'split' not in file else (df['exec_time']).max()
    timeline = range(start_time, end_time + 1)

    waiting_jobs = {time: 0 for time in timeline}

    for _, row in df.iterrows():
        for time in range(row['submit_time'], ((row['allocated_at']) if 'split' not in file else (row['exec_time'])) + 1):
            if time in waiting_jobs:
                waiting_jobs[time] += 1

    if waiting_jobs:
        times, waiting_counts = zip(*sorted(waiting_jobs.items()))
    else:
        print('Handle the case where waiting_jobs is empty')
        times, waiting_counts = [], []
    
    return times, waiting_counts

# List of CSV files to process
csv_files = [
            #  '/home/crownlabs/Plebiscitotest/1_LGF_FIFO_1_nosplit_jobs_report.csv',
             '/home/crownlabs/Plebiscitotest/1_LGF_FIFO_1_nosplit_jobs_report.csv',
             '/home/crownlabs/Plebiscitotest/1jobs_FIFO_LGF.csv',
             ] # Add more file names as needed

plt.figure(figsize=(12, 8))

# Process each file
for file in csv_files:
    df = pd.read_csv(file)
    times, waiting_counts = calculate_waiting_jobs(df, file)
    # plt.plot(times, waiting_counts, label=f'{file}', marker='o', linestyle='-', linewidth=1, markersize=4)
    plt.plot(times, waiting_counts, label=f'{file}', linewidth=1, markersize=4)

plt.title('Comparison of Number of Jobs Waiting Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Waiting Jobs')
plt.legend()
plt.grid(True)
plt.savefig('tst.png')
