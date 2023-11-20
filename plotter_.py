
import pandas as pd
from scipy.stats import t
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt


def plot_cpu_gpu(csv_files):
    # read csv files into pandas dataframes
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(csv_file))

    # get list of cpu and gpu columns
    cpu_columns = [col for col in dfs[0].columns if 'cpu' in col]
    gpu_columns = [col for col in dfs[0].columns if 'gpu' in col]

    # group dataframes by gpu_type
    grouped = []
    for df in dfs:
        grouped.append(df.groupby('gpu_type'))

    # Calculate confidence intervals for each category
    cpu_ci = {}
    gpu_ci = {}

    for category, perc in [('MISC', 96), ('V100', 96), ('P100', 64), ('T4', 96)]:
        cpu_data = []
        gpu_data = []
        for group in grouped:
            # extract cpu and gpu data for the current category
            cpu_cols = [col for col in cpu_columns if category in col]
            gpu_cols = [col for col in gpu_columns if category in col]
            cpu_data.append(group[cpu_cols].values.flatten() * 100 / perc)
            gpu_data.append(group[gpu_cols].values.flatten() * 100 / perc)

        cpu_data = np.concatenate(cpu_data)
        cpu_mean = cpu_data.mean()
        cpu_std = cpu_data.std(ddof=1)
        cpu_n = len(cpu_data)
        cpu_se = cpu_std / cpu_n**0.5
        cpu_ci[category] = t.interval(0.95, cpu_n - 1, cpu_mean, cpu_se)

        gpu_data = np.concatenate(gpu_data)
        gpu_mean = gpu_data.mean()
        gpu_std = gpu_data.std(ddof=1)
        gpu_n = len(gpu_data)
        gpu_se = gpu_std / gpu_n**0.5
        gpu_ci[category] = t.interval(0.95, gpu_n - 1, gpu_mean, gpu_se)

    # Plot the boxplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    cpu_data = []
    gpu_data = []
    for group in grouped:
        cpu_group_data = []
        gpu_group_data = []
        for category in ['MISC', 'V100', 'P100', 'T4']:
            # extract cpu and gpu data for the current category
            cpu_cols = [col for col in cpu_columns if category in col]
            gpu_cols = [col for col in gpu_columns if category in col]
            cpu_group_data.append(group[cpu_cols].values.flatten() * 100 / perc)
            gpu_group_data.append(group[gpu_cols].values.flatten() * 100 / perc)

        cpu_data.append(cpu_group_data)
        gpu_data.append(gpu_group_data)

    axs[0].boxplot(cpu_data)
    axs[0].set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    axs[0].set_title('CPU')

    axs[1].boxplot(gpu_data)
    axs[1].set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    axs[1].set_title('GPU')

    plt.savefig("boxcpu_gpu.png")

    # Add a new plot for the CDF
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))

    for group in grouped:
        cpu_group_data = []
        gpu_group_data = []
        for category in ['MISC', 'V100', 'P100', 'T4']:
            # extract cpu and gpu data for the current category
            cpu_cols = [col for col in cpu_columns if category in col]
            gpu_cols = [col for col in gpu_columns if category in col]
            cpu_group_data.append(group[cpu_cols].values.flatten() * 100 / perc)
            gpu_group_data.append(group[gpu_cols].values.flatten() * 100 / perc)

        axs2[0].hist(cpu_group_data, cumulative=True, density=True, histtype='step', bins=100, label=['MISC', 'V100', 'P100', 'T4'])
        axs2[1].hist(gpu_group_data, cumulative=True, density=True, histtype='step', bins=100, label=['MISC', 'V100', 'P100', 'T4'])

    axs2[0].set_title('CPU CDF')
    axs2[0].legend()

    axs2[1].set_title('GPU CDF')
    axs2[1].legend()

    plt.savefig("cdfcpu_gpu.png")

    # Add a new plot for the time series
    fig3, axs3 = plt.subplots(4, 1, figsize=(30, 10), sharex=True, sharey=False)

    for i, category in enumerate(['MISC', 'V100', 'P100', 'T4']):
        for group in grouped:
            # extract cpu and gpu data for the current category
            cpu_cols = [col for col in cpu_columns if category in col]
            gpu_cols = [col for col in gpu_columns if category in col]
            axs3[i].plot(group.index, group[cpu_cols].values.flatten() * 100 / perc, label=f'CPU {group.name}')
            axs3[i].plot(group.index, group[gpu_cols].values.flatten() * 100 / perc, label=f'GPU {group.name}')
        axs3[i].set_title(f'{category}')
        axs3[i].set_xlabel('Time')
        axs3[i].set_ylabel('Utilization')
        axs3[i].legend(loc='upper right')
        axs3[i].set_ylim([0, 100])

    fig3.subplots_adjust(hspace=0.5)

    plt.savefig("time_series_cpu_gpu.png")




def plot_dedlays(csv_file):
    # read csv file into pandas dataframe
    df = pd.read_csv(csv_file)


    MISC_rows = df[df['gpu_type'] == 'MISC']
    V100_rows = df[df['gpu_type'] == 'V100']
    P100_rows = df[df['gpu_type'] == 'P100']
    T4_rows = df[df['gpu_type'] == 'T4']

import matplotlib.pyplot as plt


def plot_waiting_time_confidence_interval(csv_file):
    # read csv file into pandas dataframe
    df = pd.read_csv(csv_file)

    # group data by gpu_type
    grouped = df.groupby('gpu_type')

    # loop through each group and generate boxplot
    data = []
    labels = []
    for gpu_type, group in grouped:
        # extract waiting_time data for the current group
        waiting_time = group['waiting_time']
        data.append(waiting_time)
        labels.append(gpu_type)

    # plot the data as a boxplot
    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.title('Waiting Time Distribution')
    plt.xlabel('GPU Type')
    plt.ylabel('Waiting Time (s)')
    plt.savefig('jobs')

def plot_waiting_time_confidence_interval1(csv_file):
    # read csv file into pandas dataframe
    df = pd.read_csv(csv_file)

    # group data by gpu_type
    grouped = df.groupby('gpu_type')

    # loop through each group and generate confidence interval plot
    for gpu_type, group in grouped:
        # extract waiting_time data for the current group
        waiting_time = group['waiting_time']

        # calculate mean and standard deviation
        mean = waiting_time.mean()
        std = waiting_time.std()

        # calculate confidence interval
        n = len(waiting_time)
        dof = n - 1
        alpha = 0.95
        t_value = t.ppf(alpha, dof)
        margin_of_error = t_value * std / (n ** 0.5)
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        # plot the data and confidence interval
        plt.figure()
        plt.hist(waiting_time, bins=20, density=True)
        plt.axvline(mean, color='red', label='mean')
        plt.axvline(lower_bound, color='green', linestyle='--', label='95% CI')
        plt.axvline(upper_bound, color='green', linestyle='--')
        plt.title(f'Waiting Time Distribution for {gpu_type}')
        plt.xlabel('Waiting Time (s)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('jobs')

    






# plot_waiting_time_confidence_interval('jobs.csv')
plot_cpu_gpu(['data0.csv', 'data1.csv', 'data2.csv', 'data4.csv'])


# def extract_categories(csv_file):
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_file)

#     # Define the categories for CPU and GPU
#     cpu_categories = ['MISC', 'T4', 'P100', 'V100']
#     gpu_categories = ['MISC', 'T4', 'P100', 'V100']

#     # Initialize dictionaries to store DataFrames for each category
#     cpu_dfs = {category: pd.DataFrame() for category in cpu_categories}
#     gpu_dfs = {category: pd.DataFrame() for category in gpu_categories}

#     # Loop through the columns and extract values based on categories
#     for column in df.columns:
#         # Extract category and device type from column name
#         _, device, category = column.split('_')

#         # Check if the category is for CPU or GPU
#         if device.startswith('cpu') and category in cpu_categories:
#             cpu_dfs[category][device] = df[column]
#         elif device.startswith('gpu') and category in gpu_categories:
#             gpu_dfs[category][device] = df[column]

#     return cpu_dfs, gpu_dfs

# # Example usage
# csv_file_path = 'data.csv'  # Replace with the actual path to your CSV file
# cpu_data, gpu_data = extract_categories(csv_file_path)
# print(cpu_data)

# Now, cpu_data and gpu_data contain DataFrames for each CPU and GPU category
# You can access them using cpu_data['MISC'], cpu_data['T4'], etc. for CPU
# and gpu_data['MISC'], gpu_data['T4'], etc. for GPU.



