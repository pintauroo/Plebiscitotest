
import pandas as pd
from scipy.stats import t
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt


def plot_cpu_gpu(csv_file):
    # read csv file into pandas dataframe

    df = pd.read_csv(csv_file)

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

    for category, columns, perc in [('MISC', misc_gpu_cols, 8), ('V100', v100_gpu_cols, 8), ('P100', p100_gpu_cols, 2), ('T4', t4_gpu_cols, 2)]:
        gpu_data = df[columns].values.flatten() * 100 / perc
        gpu_mean = gpu_data.mean()
        gpu_std = gpu_data.std(ddof=1)
        gpu_n = len(gpu_data)
        gpu_se = gpu_std / gpu_n**0.5
        gpu_ci[category] = t.interval(0.95, gpu_n - 1, gpu_mean, gpu_se)
        df[columns] = gpu_data.reshape(df[columns].shape)

    # Plot the confidence intervals
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].boxplot([df[cols].values.flatten() for cols in [misc_cpu_cols, v100_cpu_cols, p100_cpu_cols, t4_cpu_cols]])
    axs[0].set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    axs[0].set_title('CPU')

    axs[1].boxplot([df[cols].values.flatten() for cols in [misc_gpu_cols, v100_gpu_cols, p100_gpu_cols, t4_gpu_cols]])
    axs[1].set_xticklabels(['MISC', 'V100', 'P100', 'T4'])
    axs[1].set_title('GPU')

    plt.savefig("boxcpu_gpu.png")


    # Add a new plot for the CDF
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))

    axs2[0].hist([df[cols].values.flatten() for cols in [misc_cpu_cols, v100_cpu_cols, p100_cpu_cols, t4_cpu_cols]], cumulative=True, density=True, histtype='step', bins=100, label=['MISC', 'V100', 'P100', 'T4'])
    axs2[0].set_title('CPU CDF')
    axs2[0].legend()

    axs2[1].hist([df[cols].values.flatten() for cols in [misc_gpu_cols, v100_gpu_cols, p100_gpu_cols, t4_gpu_cols]], cumulative=True, density=True, histtype='step', bins=100, label=['MISC', 'V100', 'P100', 'T4'])
    axs2[1].set_title('GPU CDF')
    axs2[1].legend()

    plt.savefig("cdfcpu_gpu.png")

    # Add a new plot for the time series
    fig3, axs3 = plt.subplots(4, 1, figsize=(30, 10), sharex=True, sharey=False)

    for i, (category, columns) in enumerate([('MISC', misc_cpu_cols), ('V100', v100_cpu_cols), ('P100', p100_cpu_cols), ('T4', t4_cpu_cols)]):
        for col in columns:
            axs3[i].plot(df.index, df[col], label=col)
        axs3[i].set_title(f'CPU {category}')
        axs3[i].set_xlabel('Time')
        axs3[i].set_ylabel('Utilization')
        axs3[i].legend(loc='upper right')
        axs3[i].set_ylim([0, 100])

    fig3.subplots_adjust(hspace=0.5)
    

    plt.savefig("time_series_cpu.png")

    fig4, axs4 = plt.subplots(4, 1, figsize=(30, 10), sharex=True, sharey=False)

    for i, (category, columns) in enumerate([('MISC', misc_gpu_cols), ('V100', v100_gpu_cols), ('P100', p100_gpu_cols), ('T4', t4_gpu_cols)]):
        for col in columns:
            axs4[i].plot(df.index, df[col], label=col)
        axs4[i].set_title(f'GPU {category}')
        axs4[i].set_xlabel('Time')
        axs4[i].set_ylabel('Utilization')
        axs4[i].legend(loc='upper right')
        axs4[i].set_ylim([0, 100])

    fig4.subplots_adjust(hspace=0.5)

    plt.savefig("time_series_gpu.png")




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
# plot_cpu_gpu(['data0.csv', 'data1.csv', 'data2.csv', 'data3.csv'])
plot_cpu_gpu('data3.csv')


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



