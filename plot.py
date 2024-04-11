from os import listdir
from os.path import isfile, join
import sys
from matplotlib import pyplot as plt
import pandas as pd

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

def calculate_waiting_times(csv_file, directory, start_submit, end_submit):
    df = pd.read_csv(directory+ '/' + csv_file)
    df = df[df.submit_time >= start_submit]
    df = df[df.submit_time < end_submit]
    # df['waiting_time'] = pd.to_datetime(df['allocated_at'] if 'split' not in csv_file else (df['deadline']).max()) - pd.to_datetime(df['submit_time'])
    df['waiting_time'] = df['exec_time'] - df['submit_time']
    # df['waiting_time'] = df['waiting_time'].dt.total_seconds() / 60  # Convert to minutes
    return df["waiting_time"]

def find_files(mypath):
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    jobs_report = []
    simulation_report = []
    for f in file_list:
        if "jobs_report" in f:
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

def plot(csv_files, directory, start_submit, end_submit, axes, it):
    d = {}
    d2 = {}
    d3 = {}

    for f in csv_files:
        wt = calculate_waiting_times(f, directory, start_submit, end_submit)
        wt2 = calculate_actual_duration(f, directory, start_submit, end_submit)
        
        d[generate_label(f)] = wt.to_list()
        d2[generate_label(f)] = wt2.to_list()
        d3[generate_label(f)] = calculate_executed_jobs(f, directory, start_submit, end_submit)
                
    df = pd.DataFrame(d)
    df.boxplot(figsize=(30, 20), rot=90, showfliers=True, ax=axes[0, it])
    axes[0, it].title.set_text(f"Waiting time [{start_submit}-{end_submit}]")
    axes[0, it].set_ylabel("Waiting time (s)")
    # plt.ylabel("Waiting time (s)")
    # plt.savefig(f"waiting_time[{start_submit}-{end_submit}].pdf")
    # plt.clf()
    
    df2 = pd.DataFrame(d2)
    df2.boxplot(figsize=(30, 20), rot=90, showfliers=True, ax=axes[1, it])
    axes[1, it].title.set_text(f"Actual_duration [{start_submit}-{end_submit}]")
    axes[1, it].set_ylabel("Actual duration (s). NOTE: > 0 job is completed before expected duration, < 0 job is completed after.")
    # plt.ylabel("Actual duration (s). NOTE: > 0 job is completed before expected duration, < 0 job is completed after expected duration")
    # plt.savefig(f"actual_duration[{start_submit}-{end_submit}].pdf")
    # plt.clf()
    
    df3 = pd.DataFrame(d3, index=[0])
    df3.boxplot(figsize=(30, 20), rot=90, ax=axes[2, it])
    axes[2, it].title.set_text(f"Executed jobs [{start_submit}-{end_submit}]")
    axes[2, it].set_ylabel("Executed jobs")
    
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
    mypath = "src/res/to-plot"
    
    occupation_threshold = 0.75
    
    checkpoints = [0]
    
    jobs_report, simulation_report = find_files(mypath)
    jobs_report = sorted(jobs_report)
    simulation_report = sorted(simulation_report)
    
    # checkpoints.append(indentify_thresholds(simulation_report, mypath, occupation_threshold))
    # checkpoints.append(indetify_last_exec_time(jobs_report, mypath))
    
    # decommenta se vuoi intervalli fissi
    checkpoints.append(1500)
    checkpoints.append(3000)
    checkpoints.append(4500)
    checkpoints.append(6000)
    
    check_if_checkpoint_are_consistent(checkpoints)
    
    fig, axes = plt.subplots(nrows=4, ncols=len(checkpoints)-1, figsize=(35, 20))
    for i in range(1, len(checkpoints)):
        start_submit = checkpoints[i-1]
        end_submit = checkpoints[i]
        
        plot(jobs_report, mypath, start_submit, end_submit, axes, i-1)
        plot_simulation_time(simulation_report, mypath, axes)
    
    fig.tight_layout()
    fig.savefig("results.pdf")
