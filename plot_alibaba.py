from os import listdir
import os
from os.path import isfile, join
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 16}

matplotlib.rc('font', **font)

def cleanup_data(x, y):
    res_x = []
    res_y = []

    prev_x = int(x[0])
    count = 0
    cum = 0

    for id, x_i in enumerate(x):
        if int(x_i) != prev_x:
            res_x.append(prev_x)
            res_y.append(cum/count)

            prev_x = int(x_i)
            count = 1
            cum = y[id]
        else:
            count += 1
            cum += y[id]

    return res_x, res_y

def merge_data(x, y):
    res_x = []

    for l in x:
        res_x += l
    
    res_x = list(set(res_x))
    count_y = [0 for _ in range(len(res_x))]
    res_y = [0 for _ in range(len(res_x))]

    for id, l in enumerate(y):
        for x_id, li in enumerate(l):
            index = -1
            for i in range(len(count_y)):
                if res_x[i] == x[id][x_id]:
                    index = i
                    break
            # print(x_id)
            count_y[index] += 1
            res_y[index] += li
    
    for i in range(len(res_y)):
        res_y[i] = res_y[i]/count_y[i]

    return res_x, res_y

def plot_unallocated_gpu_Plebi(directory, simulation_file, labels, ax, repetitions):
    for id, f in enumerate(simulation_file):
        xs = []
        ys = []

        for i in range(1, repetitions+1):
            df_simulation = pd.read_csv(directory + '/' + str(i) + "_" + f + "_nosplit_norebid.csv")
            df_report = pd.read_csv(directory + '/' + str(i) + "_" + f + "_nosplit_norebid_jobs_report.csv")
            times = list(df_report["submit_time"])
            res_y = []
            res_x = []
            cumulative_GPU = 0
            first = False
            second = False

            for index, row in df_simulation.iterrows():
                if index not in times:
                    continue

                sum = 0
                tot_gpu = 0

                for i in range(50):
                    sum += (float(row["node_" + str(i) + "_initial_gpu"] - row["node_" + str(i) + "_used_gpu"]))
                    tot_gpu += float(row["node_" + str(i) + "_initial_gpu"])
            
                res_y.append(sum)

                for c in df_report[df_report.submit_time == index]["num_gpu"]:
                    job_gpu = float(c)
                    cumulative_GPU += job_gpu

                if cumulative_GPU/tot_gpu*100 > 90 and not first:
                    first = True
                    print(index)
                if cumulative_GPU/tot_gpu*100 > 110 and not second:
                    second = True
                    print(index)

                res_x.append(cumulative_GPU/tot_gpu*100)
            
            res_x, res_y = cleanup_data(res_x, res_y)

            xs.append(res_x)
            ys.append(res_y)
            
        x, y = merge_data(xs, ys)
        ax.plot(x, y, label="Plebi " + labels[id])

def plot_unallocated_gpu_Alibaba(directory, simulation_file, labels, ax, repetitions):
    for id, f in enumerate(simulation_file):
        xs = []
        ys = []

        for i in range(1, repetitions+1):
            # df_simulation = pd.read_csv(directory + '/' + str(i) + "_data_" + f + ".csv")
            # df_report = pd.read_csv(directory + '/' + str(i) + "_jobs_" + f + ".csv")
            df_simulation = pd.read_csv(directory + '/' + str(i) + "_" + f + ".csv")
            df_report = pd.read_csv(directory + '/' + str(i) + "_" + f + "_jobs_report.csv")
            times = list(df_report["submit_time"])
            res_y = []
            res_x = []
            cumulative_GPU = 0

            for index, row in df_simulation.iterrows():
                if index not in times:
                    continue

                sum = 0
                tot_gpu = 0

                for i in range(50):
                    sum += (float(row["node_" + str(i) + "_initial_gpu"] - row["node_" + str(i) + "_used_gpu"]))
                    tot_gpu += float(row["node_" + str(i) + "_initial_gpu"])
            
                res_y.append(sum)

                for c in df_report[df_report.submit_time == index]["num_gpu"]:
                    job_gpu = float(c)
                    cumulative_GPU += job_gpu

                res_x.append(cumulative_GPU/tot_gpu*100)
            
            res_x, res_y = cleanup_data(res_x, res_y)

            xs.append(res_x)
            ys.append(res_y)

        x, y = merge_data(xs, ys)
        ax.plot(x, y, label="Alibaba " + labels[id])

def compute_jain_intervals(df, n_nodes):
    jini = {}
    jini["jini_cpu"] = []
    jini["jini_gpu"] = []
    
    for _, row in df.iloc[:2125].iterrows():
    # for _, row in df.iterrows():

        sum_cpu = 0
        sum_cpu_square = 0
        sum_gpu = 0
        sum_gpu_square = 0
                        
        for i in range(n_nodes):
            sum_cpu += (float(row["node_" + str(i) + "_initial_cpu"]) - float(row["node_" + str(i) + "_used_cpu"]))
            sum_cpu_square += (float(row["node_" + str(i) + "_initial_cpu"]) - float(row["node_" + str(i) + "_used_cpu"]))**2
            sum_gpu += (float(row["node_" + str(i) + "_initial_gpu"]) - float(row["node_" + str(i) + "_used_gpu"]))
            sum_gpu_square += (float(row["node_" + str(i) + "_initial_gpu"]) - float(row["node_" + str(i) + "_used_gpu"]))**2
            
        jini["jini_cpu"].append(sum_cpu**2 / (n_nodes* sum_cpu_square))
        jini["jini_gpu"].append(sum_gpu**2 / (n_nodes* sum_gpu_square))
    
    df_f = pd.DataFrame(jini)

    lower_cpu = df_f["jini_cpu"].quantile(0.05)
    higher_cpu = df_f["jini_cpu"].quantile(0.95)
    lower_gpu = df_f["jini_gpu"].quantile(0.05)
    higher_gpu = df_f["jini_gpu"].quantile(0.95)

    return lower_cpu, higher_cpu, lower_gpu, higher_gpu

def plot_Jain(alibaba_file, alibaba_label, plebiscito_file, plebiscito_label, directory, repetitions):

    for i in range(len(alibaba_file)):
        fig, ax = plt.subplots()

        # a_file = os.path.join(directory, str(repetitions) + "_data_" + str(alibaba_file[i])+'.csv') 
        a_file = os.path.join(directory, str(repetitions) + "_" + str(alibaba_file[i])+'.csv') 

        p_file = os.path.join(directory, str(repetitions) + "_" + str(plebiscito_file[i])+'_nosplit_norebid.csv') 

        a_df = pd.read_csv(a_file)
        p_df = pd.read_csv(p_file)

        lower_cpu, higher_cpu, lower_gpu, higher_gpu = compute_jain_intervals(a_df, 30)

        ax.add_patch(Rectangle((lower_cpu, lower_gpu), higher_cpu - lower_cpu, higher_gpu - lower_gpu, alpha=0.5, label="Alibaba " + alibaba_label[i], hatch='\\'))
        
        lower_cpu, higher_cpu, lower_gpu, higher_gpu = compute_jain_intervals(p_df, 30)

        ax.add_patch(Rectangle((lower_cpu, lower_gpu), higher_cpu - lower_cpu, higher_gpu - lower_gpu, alpha=0.5, label="Plebi " + plebiscito_label[i], hatch='//'))
        
        ax.set_ylabel('GPU')
        ax.set_xlabel('CPU')

        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.95))
        fig.savefig(f"{alibaba_label[i]}_jain.pdf")

        plt.clf()

if __name__ == '__main__':
    mypath = "src/res/to-plot"
    fig, ax = plt.subplots()
    repetitions = 1
    
    # Alibaba = ["SDF_LGF", "SDF_SGF", "SDF_UTIL"]
    # Alibaba_label = ["LGF", "SGF", "UTIL"]
    # Plebiscito = ["LGF_SDF_0", "SGF_SDF_0", "UTIL_SDF_0"]
    # Plebiscito_label = ["LGF", "SGF", "UTIL"]
    
    Alibaba = ["FGD", "FGD_fail"]
    Alibaba_label = ["FGD", "FGD_fail"]
    Plebiscito = ["FGD_SDF_0"]
    Plebiscito_label = ["FGD"]

    #plot_unallocated_gpu_Plebi(mypath, simulation_report, jobs_report, ax)
    plot_unallocated_gpu_Alibaba(mypath, Alibaba, Alibaba_label, ax, repetitions)
    plot_unallocated_gpu_Plebi(mypath, Plebiscito, Plebiscito_label, ax, repetitions)

    # ax.set_xlim(80, 130)
    # ax.set_ylim(0, 40)
    ax.set_xlabel('Arrived workloads (in % of cluster GPU capacity)')
    ax.set_ylabel('Unallocated GPUs')
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    fig.savefig("allocation" + ".pdf")

    plot_Jain(Alibaba, Alibaba_label, Plebiscito, Plebiscito_label, mypath, 1)