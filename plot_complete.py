from os import listdir
from os.path import isfile, join
import re
import sys
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns

# > 0 means job is completed before expected duration
# < 0 means job is completed after expected duration
def calculate_actual_duration(csv_file, directory, start_submit, end_submit):
    df = pd.read_csv(directory+ '/' + csv_file)
    df = df[df.submit_time >= start_submit]
    df = df[df.submit_time < end_submit]
    df["saved_seconds"] = df["duration"] - (df["complete_time"] - df["exec_time"])
    return df["saved_seconds"]

def calculate_executed_jobs(csv_file, directory, start_submit, end_submit):
    df = pd.read_csv(directory+ '/' + csv_file)
    df = df[df.exec_time >= start_submit]
    df = df[df.exec_time < end_submit]
    return len(df.index)

def calculate_waiting_times(csv_file, directory, start_submit=0, end_submit=99999999999999999999):
    # print(csv_file)
    df = pd.read_csv(directory + '/' + csv_file)

    df = df[(df['submit_time'] >= start_submit) & (df['submit_time'] < end_submit)]

    # Assuming 'exec_time' and 'submit_time' are in a suitable format (e.g., timestamps)
    df['waiting_time'] = df['exec_time'] - df['submit_time'] if 'allocations' in csv_file else df['waiting_time']
    completed = max(df['complete_time'])  if 'report' in csv_file else max(df['allocated_at'] + df['duration']) 
    # Uncomment if conversion to minutes is needed
    # df['waiting_time'] = (df['waiting_time'].dt.total_seconds()) / 60  # Convert to minutes

    # Define GPU categories
    categories = {
        "0 to 1 GPU": (0, 1),
        "1 to 2 GPUs": (1, 2),
        "More than 2 GPUs": (2, np.inf)  # assuming inf for any higher value
    }

    # Prepare data for the boxplot
    waiting_times = []
    labels = []
    for label, (low, high) in categories.items():
        if high == np.inf:
            subset = df[df['num_gpu'] > low]
        else:
            subset = df[(df['num_gpu'] > low) & (df['num_gpu'] <= high)]

        if not subset.empty:
            subset = subset[(subset['waiting_time']<10000)]
            waiting_times.append(subset['waiting_time'])
            labels.append(label)

    

    # Set up the plot
    # plt.figure(figsize=(10,10))

    # # Create a boxplot
    # plt.boxplot(waiting_times, labels=labels, notch=True, patch_artist=True)
    # plt.xticks(rotation=45)  # Rotate labels if they overlap

    # # Customize the plot
    # plt.xlabel('GPU Category')
    # plt.ylabel('Waiting Time')
    # plt.title('Boxplot of Waiting Time by GPU Usage between ' + str(start_submit) +'-'+str(end_submit) )
    # plt.savefig(str(end_submit) + csv_file + '_boxplot.png')  # Save the figur

    return waiting_times, completed

def find_files(mypath):
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    jobs_report = []
    simulation_report = []
    for f in file_list:
            if '.csv' in f:
                if "jobs" in f or "allocations" in f:
                    jobs_report.append(f)
                else:
                    simulation_report.append(f)
            
    return jobs_report, simulation_report

def indetify_last_exec_time(csv_files, directory):
    f = csv_files[0]
    
    df = pd.read_csv(directory+ '/' + f)
    
    return df["exec_time"].max()

def indentify_thresholds(csv_files, directory, threshold):
    f = csv_files[0]
    
    df = pd.read_csv(directory+ '/' + f)
    
    for index, row in df.iterrows():
        sum = 0
                        
        for i in range(int(row["n_nodes"])):
            sum += (float(row["node_" + str(i) + "_used_gpu"]/row["node_" + str(i) + "_initial_gpu"]))
        
        if sum/int(row["n_nodes"]) > threshold:
            return index
        
def generate_label(filename):
    return " ".join(filename.split("_")[1:]).replace("jobs report.csv", "").replace("nosplit", "").replace("rebid", "r")


def boxplot(csv_files, directory, start_submit, end_submit):

    # ---------------------------------------------------
    # Boxplot 
    # ---------------------------------------------------

    scheduling1 = 'FIFO'
    scheduling2 = 'SDF'
    allocation = 'UTIL'

    plebi = defaultdict(list)
    plebi_completed = {scheduling1+'_'+allocation:[], scheduling2+'_'+allocation:[]}
    alibaba = defaultdict(list)
    alibaba_completed = {scheduling1+'_'+allocation:[], scheduling2+'_'+allocation:[]}



    for f in csv_files:
        p, completed = calculate_waiting_times(f, directory, start_submit, end_submit)
        if 'report' in f:  # PLEBI
            if scheduling1 in f and allocation in f:
                plebi_completed[scheduling1+'_'+allocation].append(completed)
            if scheduling2 in f and allocation in f:
                plebi_completed[scheduling2+'_'+allocation].append(completed)

        else:  # ALIBABA
            if scheduling1 in f and allocation in f:
                alibaba_completed[scheduling1+'_'+allocation].append(completed)
            if scheduling2 in f and allocation in f:
                alibaba_completed[scheduling2+'_'+allocation].append(completed)


    for lab in plebi_completed:
        plebi_mean = np.mean(plebi_completed[lab])
        plebi_sem = np.std(plebi_completed[lab]) / np.sqrt(len(plebi_completed[lab]))
        max_plebi = max(plebi_completed[lab])
        max_alibaba = max(alibaba_completed[lab])



        normalized_plebi = [x / max_plebi for x in plebi_completed[lab]]
        normalized_alibaba = [x / max_alibaba for x in alibaba_completed[lab]]

        # Calculate mean and standard error of the mean (SEM) for each normalized dataset
        plebi_mean = np.mean(normalized_plebi)
        plebi_sem = np.std(normalized_plebi) / np.sqrt(len(normalized_plebi))

        alibaba_mean = np.mean(normalized_alibaba)
        alibaba_sem = np.std(normalized_alibaba) / np.sqrt(len(normalized_alibaba))

        # Plotting the bar plots with error bars for confidence intervals
        fig, ax = plt.subplots()

        bar_width = 0.35
        index = np.arange(1)

        bar1 = ax.bar(index, plebi_mean, bar_width, yerr=plebi_sem, label='Plebi Completed', capsize=5)
        bar2 = ax.bar(index + bar_width, alibaba_mean, bar_width, yerr=alibaba_sem, label='Alibaba Completed', capsize=5)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Normalized Values')
        ax.set_title('Comparison of Normalized Plebi and Alibaba with Confidence Intervals')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(['Mean Values'])
        ax.legend()

        plt.savefig(str(lab) + '_barplot.png')



def compact_gpu_data(csv_file, directory, start_submit=0, end_submit=99999999999999999999):
    # print(csv_file)
    df = pd.read_csv(directory + '/' + csv_file)

    # if df.count().nunique() == 1:
    #     print("All columns have the same length.")
    # else:
    #     print("Columns have different lengths.")






    df_section = df.loc[start_submit:end_submit].reset_index(drop=True)
    
    # Initialize lists to store values for each GPU type
    gpu_misc = []
    gpu_t4 = []
    gpu_p100 = []
    gpu_v100 = []

    # Process data based on the file name and column content
    if 'data' in csv_file:
        # Loop over each column in the DataFrame and append the relevant data to each list
        for column in df_section.columns:
            if 'gpu' in column:
                if 'MISC' in column:
                    gpu_misc.extend(df_section[column].tolist())
                elif 'T4' in column:
                    gpu_t4.extend(df_section[column].tolist())
                elif 'P100' in column:
                    gpu_p100.extend(df_section[column].tolist())
                elif 'V100' in column:
                    gpu_v100.extend(df_section[column].tolist())
    else:
        for column in df_section.columns:
            if 'gpu_type' in column:
                first_value = df_section[column][0]
                if not isinstance(first_value, int):
                    str_first_value = str(first_value)
                    number_match = re.search(r'\d+', column)
                    if number_match:
                        number = number_match.group()
                        target_col = 'node_' + number + '_used_gpu'
                        # print(f"Found numeric value in column name: {column}",   len(df_section[target_col].tolist()) )

                        if target_col in df_section.columns:
                            values_list = df_section[target_col].tolist()

                            # Check if any value in the list is less than zero
                            has_negative_values = any(value < 0 for value in values_list)
                            if has_negative_values:
                                print(f"Negative values found in column: {column}")
                            if 'MISC' in str_first_value:
                                gpu_misc.extend(df_section[target_col].tolist())
                                # print(len(gpu_misc), len(gpu_t4), len(gpu_p100), len(gpu_v100))
                            elif 'T4' in str_first_value:
                                gpu_t4.extend(df_section[target_col].tolist())
                                # print(len(gpu_misc), len(gpu_t4), len(gpu_p100), len(gpu_v100))

                            elif 'P100' in str_first_value:
                                gpu_p100.extend(df_section[target_col].tolist())
                                # print(len(gpu_misc), len(gpu_t4), len(gpu_p100), len(gpu_v100))
                            elif 'V100' in str_first_value:
                                gpu_v100.extend(df_section[target_col].tolist())
                                # print(len(gpu_misc), len(gpu_t4), len(gpu_p100), len(gpu_v100))
                    else:
                        print(f"No numeric value found in column name: {column}")



    
    # Optionally, flatten the lists if needed (depends on whether you want each type as a list of lists or a single list)
    gpu_misc = [ sublist for sublist in gpu_misc]
    gpu_t4 =   [ sublist for sublist in gpu_t4]
    gpu_p100 = [ sublist for sublist in gpu_p100]
    gpu_v100 = [ sublist for sublist in gpu_v100]


    return gpu_misc, gpu_t4, gpu_p100, gpu_v100


def plot_gpu_utilization(data, name, start_submit, end_submit):
    plt.figure(figsize=(10, 6))

    # Prepare data for plotting
    util_data = []
    gpu_labels = []
    for gpu, utilizations in data.items():
        util_data.extend(utilizations)
        gpu_labels.extend([gpu] * len(utilizations))

    # Create a DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame({
        'GPU Type': gpu_labels,
        'Utilization (%)': util_data
    })

    # Create a box plot
    sns.boxplot(x='GPU Type', y='Utilization (%)', data=df)

    # Adding plot title and labels
    plt.title('GPU Utilization')
    plt.xlabel('GPU Type')
    plt.ylabel('Utilization (%)')
    plt.xticks(rotation=45)  # Rotate labels for better visibility

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(f'ktm_{name}_{start_submit}_{end_submit}.png')




def plot_simulationreport(csv_files, directory, start_submit, end_submit, it):


    alibaba_gpu_utilization = {'MISC':[], 'V100':[], 'P100':[], 'T4':[]}
    plebi_gpu_utilization = {'MISC':[], 'V100':[], 'P100':[], 'T4':[]}


    for f in csv_files:
        if 'FIFO' in f and 'UTIL' in f and 'allocations' not in f:
        # if '1_LGF_FIFO_0_nosplit_norebid' in f or '1data_FIFO_LGF' in f:

            tmp = {}
            tmp['MISC'] = []
            tmp['P100'] = []
            tmp['T4'] = []
            tmp['V100'] = []
            if 'data' not in f:  # PLEBI
                print('plebi', f)
                (tmp['MISC'], tmp['T4'], 
                tmp['P100'], tmp['V100']) = \
                        compact_gpu_data(f, directory, start_submit, end_submit)
            
                plebi_gpu_utilization['MISC'].extend(tmp['MISC'])
                plebi_gpu_utilization['T4'].extend(tmp['T4'])
                plebi_gpu_utilization['P100'].extend(tmp['P100'])
                plebi_gpu_utilization['V100'].extend(tmp['V100'])
            else:  # ALIBABA
                    print('alibaba', f)
                    (tmp['MISC'], tmp['T4'], 
                    tmp['P100'], tmp['V100']) = \
                        compact_gpu_data(f, directory, start_submit, end_submit)
                    alibaba_gpu_utilization['MISC'].extend(tmp['MISC'])
                    alibaba_gpu_utilization['T4'].extend(tmp['T4'])
                    alibaba_gpu_utilization['P100'].extend(tmp['P100'])
                    alibaba_gpu_utilization['V100'].extend(tmp['V100'])
                    # break
                    
    alibaba_gpu_utilization['MISC'] = [x / 8 * 100 for x in alibaba_gpu_utilization['MISC']]
    alibaba_gpu_utilization['T4']   = [x / 2 * 100 for x in alibaba_gpu_utilization['T4']]
    alibaba_gpu_utilization['P100'] = [x / 2 * 100 for x in alibaba_gpu_utilization['P100']]
    alibaba_gpu_utilization['V100'] = [x / 8 * 100 for x in alibaba_gpu_utilization['V100']]
    plebi_gpu_utilization['MISC'] = [x / 8 * 100 for x in plebi_gpu_utilization['MISC']]
    plebi_gpu_utilization['T4']   = [x / 2 * 100 for x in plebi_gpu_utilization['T4']]
    plebi_gpu_utilization['P100'] = [x / 2 * 100 for x in plebi_gpu_utilization['P100']]
    plebi_gpu_utilization['V100'] = [x / 8 * 100 for x in plebi_gpu_utilization['V100']]

    if any(x < 0 for x in plebi_gpu_utilization['MISC']) or any(x < 0 for x in plebi_gpu_utilization['T4']) or any(x < 0 for x in plebi_gpu_utilization['P100']) or any(x < 0 for x in plebi_gpu_utilization['V100']):
        print('Negative values found')
    
    # print(max(plebi_gpu_utilization['MISC']), max(plebi_gpu_utilization['T4']), 
    #         max(plebi_gpu_utilization['P100']), max(plebi_gpu_utilization['V100']))
    # print(max(alibaba_gpu_utilization['MISC']), max(alibaba_gpu_utilization['T4']), 
    #         max(alibaba_gpu_utilization['P100']), max(alibaba_gpu_utilization['V100']))
            
    print('here')
                
    plot_gpu_utilization(plebi_gpu_utilization, 'plebi', start_submit, end_submit)
    plot_gpu_utilization(alibaba_gpu_utilization, 'alibaba', start_submit, end_submit)

def plot_jobsreport(csv_files, directory, start_submit, end_submit, it):

    scheduling = 'FIFO'
    allocation = 'UTIL'

    plebi = defaultdict(list)
    plebi_completed = {scheduling+'_'+allocation:[]}
    plebi_gpu_utilization = {'MISC':[], 'V100':[], 'P100':[], 'T4':[]}
    alibaba = defaultdict(list)
    alibaba_completed = {scheduling+'_'+allocation:[]}
    alibaba_gpu_utilization = {'MISC':[], 'V100':[], 'P100':[], 'T4':[]}



    for f in csv_files:
        if scheduling in f and allocation in f and '1' in f:
            if 'allocations' in f: # PLEBI
                p, completed = calculate_waiting_times(f, directory, start_submit, end_submit)
                print('PLEBI', f)

                plebi_completed[scheduling+'_'+allocation].append(completed)
                for i, v in enumerate(p):
                    plebi[i].extend(v)
            elif '1jobs_'+scheduling+'_'+allocation in f:  # ALIBABA
                p, completed = calculate_waiting_times(f, directory, start_submit, end_submit)
                print('ALIBABA', f)
                alibaba_completed[scheduling+'_'+allocation].append(completed)
                # alibaba_gpu_utilization['MISC'], 
                # alibaba_gpu_utilization['T4'], 
                # alibaba_gpu_utilization['P100'], 
                # alibaba_gpu_utilization['V100'] =  compact_gpu_data(f, directory, start_submit, end_submit)
                for i, v in enumerate(p):

                    alibaba[i].extend(v)






    
    fig, ax = plt.subplots()

    # Collect all box data in a list for plotting
    box_data = [plebi[i] for i in plebi]
    labels = [str(i) for i in plebi]  # Create labels from the dictionary keys

    # Plot all box plots on the same axes
    ax.boxplot(box_data, notch=True, patch_artist=True, labels=labels)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels if they overlap

    # Customize the plot
    ax.set_xlabel('Index from "plebi"')
    ax.set_ylabel('Waiting Time')
    ax.set_title('Boxplot of Waiting Time by GPU Usage between ' + str(start_submit) + '-' + str(end_submit))

    # cdf Plebi
    plt.savefig(str(end_submit) + '_boxplot.png')

    fig, ax = plt.subplots()

    # Collect all data in a list for plotting
    data_list = [plebi[i] for i in plebi]
    labels = [str(i) for i in plebi]  # Use keys from `plebi` for labels

    # Plot CDF for each set of data
    for data, label in zip(data_list, labels):
        sorted_data = np.sort(data)  # Sort the data
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)  # Calculate CDF values
        ax.plot(sorted_data, yvals, label=label)

    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels if they overlap

    # Customize the plot
    ax.set_xlabel('Waiting Time')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Waiting Time by GPU Usage between ' + str(start_submit) + '-' + str(end_submit))
    ax.legend()

    # Save the figure
    plt.savefig(str(end_submit) + '_cdf_plebi.png')



    fig, ax = plt.subplots()

    # Collect all box data in a list for plotting
    box_data = [alibaba[i] for i in alibaba]
    labels = [str(i) for i in alibaba]  # Create labels from the dictionary keys

    # Plot all box plots on the same axes
    ax.boxplot(box_data, notch=True, patch_artist=True, labels=labels)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels if they overlap

    # Customize the plot
    ax.set_xlabel('Index from "plebi"')
    ax.set_ylabel('Waiting Time')
    ax.set_title('Boxplot of Waiting Time by GPU Usage between ' + str(start_submit) + '-' + str(end_submit))

    # Save the figure
    plt.savefig(str(end_submit) + '_boxplot_alibaba.png')

    #CDF
    fig, ax = plt.subplots()

    # Collect all data in a list for plotting
    data_list = [alibaba[i] for i in alibaba]
    labels = [str(i) for i in alibaba]  # Use keys from `alibaba` for labels

    # Plot CDF for each set of data
    for data, label in zip(data_list, labels):
        sorted_data = np.sort(data)  # Sort the data
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)  # Calculate CDF values
        ax.plot(sorted_data, yvals, label=label)

    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels if they overlap

    # Customize the plot
    ax.set_xlabel('Waiting Time')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Waiting Time by GPU Usage between ' + str(start_submit) + '-' + str(end_submit))
    ax.legend()

    # Save the figure
    plt.savefig(str(end_submit) + '_cdf_alibaba.png')

        # wt2 = calculate_actual_duration(f, directory, start_submit, end_submit)
        
        # d[generate_label(f)] = wt.to_list()
        # d2[generate_label(f)] = wt2.to_list()
        # d3[generate_label(f)] = calculate_executed_jobs(f, directory, start_submit, end_submit)
                
    # df = pd.DataFrame(d)
    # df.boxplot(figsize=(30, 20), rot=90, showfliers=True, ax=axes[0, it])
    # axes[0, it].title.set_text(f"Waiting time [{start_submit}-{end_submit}]")
    # axes[0, it].set_ylabel("Waiting time (s)")
    # plt.ylabel("Waiting time (s)")
    # plt.savefig(f"waiting_time[{start_submit}-{end_submit}].pdf")
    # plt.clf()
    
    # df2 = pd.DataFrame(d2)
    # df2.boxplot(figsize=(30, 20), rot=90, showfliers=True, ax=axes[1, it])
    # axes[1, it].title.set_text(f"Actual_duration [{start_submit}-{end_submit}]")
    # axes[1, it].set_ylabel("Actual duration (s). NOTE: > 0 job is completed before expected duration, < 0 job is completed after.")
    # # plt.ylabel("Actual duration (s). NOTE: > 0 job is completed before expected duration, < 0 job is completed after expected duration")
    # # plt.savefig(f"actual_duration[{start_submit}-{end_submit}].pdf")
    # # plt.clf()
    
    # df3 = pd.DataFrame(d3, index=[0])
    # df3.boxplot(figsize=(30, 20), rot=90, ax=axes[2, it])
    # axes[2, it].title.set_text(f"Executed jobs [{start_submit}-{end_submit}]")
    # axes[2, it].set_ylabel("Executed jobs")
    return plebi_completed, alibaba_completed
    
def check_if_checkpoint_are_consistent(checkpoints):
    for i in range(1, len(checkpoints)):
        if checkpoints[i] < checkpoints[i-1]:
            sys.exit("Checkpoints are not consistent")

def plot_simulation_time(files, directory, axes):
    d = {}
    for f in files:
        with open(directory+ '/' + f, 'r') as fp:
            lines = len(fp.readlines())
            d[generate_label(f)] = lines

    df = pd.DataFrame(d, index=[0])
    df.boxplot(figsize=(30, 20), rot=90, ax=axes[3, 0])
    axes[3, 0].title.set_text(f"Simulation Time")
    axes[3, 0].set_ylabel("Elaposed simulation time (s)")
        
if __name__ == '__main__':
    mypath = "/home/andrea/projects/Plebiscitotest"
    
    occupation_threshold = 0.75
    
    checkpoints = [0]
    
    jobs_report, simulation_report = find_files(mypath)
    # for j in jobs_report:
    #     # print(j)
    #     # if 'LGF' in j and 'FIFO' in j:
    #     if 'FIFO' in j and 'LGF' in j:
    #         print(j)

    jobs_report = sorted(jobs_report)
    simulation_report = sorted(simulation_report)
    
    # checkpoints.append(indentify_thresholds(simulation_report, mypath, occupation_threshold))
    # checkpoints.append(indetify_last_exec_time(jobs_report, mypath))
    
    # decommenta se vuoi intervalli fissi
    checkpoints.append(1000)
    # checkpoints.append(3000)
    checkpoints.append(1500)
    checkpoints.append(2000)
    checkpoints.append(2500)
    
    check_if_checkpoint_are_consistent(checkpoints)
    
    # fig, axes = plt.subplots(nrows=4, ncols=len(checkpoints)-1, figsize=(35, 20))
    for i in range(1, len(checkpoints)):
        start_submit = checkpoints[i-1]
        end_submit = checkpoints[i]
        
        
        
        
        plot_jobsreport(jobs_report, mypath, start_submit, end_submit, i-1)
        # plot_simulationreport(simulation_report, mypath, start_submit, end_submit, i-1)
    #     # plot_simulation_time(simulation_report, mypath, axes)

        # boxplot(jobs_report, mypath, start_submit, end_submit)
 