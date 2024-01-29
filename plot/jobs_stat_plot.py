import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_plot_folder(dirname):
    # check if the plot directory exists, if not create it
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def plot_node_resource_usage_box(filename, res_type, n_nodes, dir_name):
    """
    Plots the resource usage of nodes in the form of a boxplot and saves the plot to a file.

    Args:
        filename (str): The name of the file containing the data to plot.
        res_type (str): The type of resource to plot (e.g. "cpu", "gpu").
        n_nodes (int): The number of nodes to plot.
        dir_name (str): The name of the directory to save the plot file in.
    """
    # plot node resource usage using data from filename
    df = pd.read_csv(filename + ".csv")
    
    # select only the columns matching the pattern node_*_updated_gpu
    df2 = df.filter(regex=("node.*"+res_type))
    
    d = {}
    for i in range(n_nodes):
        gpu_type = df['node_'+str(i)+'_gpu_type'].iloc[1]
        if gpu_type not in d:
            d[str(gpu_type)] = []
        d[str(gpu_type)] += list(df2["node_" + str(i) + "_used_" + res_type] / df2["node_" + str(i) + "_initial_" + res_type])
    
    # use matplotlib to plot the data and save the plot to a file
    plt.boxplot(d.values())
    plt.xticks(range(1, len(d.keys()) + 1), d.keys())

    
    plt.ylabel(f"{res_type} usage")
    plt.xlabel("GPU type")
    plt.savefig(os.path.join(dir_name, 'node_' + res_type + '_resource_usage_box.png'))
    # ticks = [i+1 for i in range(len(d.keys()))]
    # plt.xticks(ticks, d.keys())
    
    # clear plot
    plt.clf()

def plot_node_resource_usage(filename, res_type, n_nodes, dir_name):
    """
    Plots the resource usage of nodes over time and saves the plot to a file.

    Args:
        filename (str): The name of the file containing the data to plot.
        res_type (str): The type of resource to plot (e.g. "cpu", "gpu").
        n_nodes (int): The number of nodes to plot.
        dir_name (str): The name of the directory to save the plot file in.
    """
    # plot node resource usage using data from filename
    df = pd.read_csv(filename + ".csv")
    
    # select only the columns matching the pattern node_*_updated_gpu
    df2 = df.filter(regex=("node.*"+res_type))
    
    d = {}
    for i in range(n_nodes):
        gpu_type = df['node_'+str(i)+'_gpu_type'].iloc[0]
        d["node_" + str(i) + "_" + str(gpu_type)] = df2["node_" + str(i) + "_used_" + res_type] / df2["node_" + str(i) + "_initial_" + res_type]
    
    df_2 = pd.DataFrame(d)
    
    # use matplotlib to plot the data and save the plot to a file
    df_2.plot(legend=None)
    
    plt.ylabel(f"{res_type} usage")
    plt.xlabel("time")
    plt.savefig(os.path.join(dir_name, 'node_' + res_type + '_resource_usage.png'))
    
    # clear plot
    plt.clf()
    plt.close()
    

def plot_job_execution_delay(filename, dir_name):
    """
    Plots a histogram and a boxplot of job execution delays and saves the plots to files.

    Args:
        filename (str): The name of the CSV file containing job data.
        dir_name (str): The name of the directory where the plots will be saved.
    """
    if 'report' in filename:
        try:
            df = pd.read_csv(basepath + filename + '.csv')
        except:
            return
        res = df["exec_time"] - df["submit_time"]
    else:
        try:
            df = pd.read_csv(basepath + filename + '.csv')
        except:
            return
        res = df["allocated_at"] - df["submit_time"]


  
    # plot histogram using the res variable
    res.astype(int).hist()
    
    # save the plot to a file
    plt.ylabel(f"Occurrences")
    plt.xlabel("Job execution delay (s)")
    plt.savefig(os.path.join(dir_name, 'job_execution_delay.png'))
    
    # clear plot
    plt.clf()
    plt.close()



def plot_job_execution_delay_cdf(file_list, dir_name):
    """
    Plots the CDF of job execution delays for multiple files in the same plot and saves the plot to a file.

    Args:
        file_list (list): A list of CSV file names containing job data.
        dir_name (str): The name of the directory where the plot will be saved.
    """
    # Create a figure for the combined CDF plot
    plt.figure(figsize=(10, 6))

    for filename in file_list:
        if 'report' in filename:
            try:
                df = pd.read_csv(basepath + filename + '.csv')
            except:
                continue
            res = df["exec_time"] - df["submit_time"]
        else:
            try:
                df = pd.read_csv(basepath + filename + '.csv')
            except:
                continue
            res = df["allocated_at"] - df["submit_time"]

        # Calculate the CDF
        res = res.dropna().astype(int)  # Remove NaNs and convert to integers
        res = res[res >= 0]  # Filter out negative delays, if any
        res.sort_values(inplace=True)  # Sort the delays
        cdf = res.rank(method='average') / len(res)  # Compute the CDF

        # Plot the CDF for each file
        plt.plot(res, cdf, label=filename)

    # Customize the plot
    plt.ylabel("CDF")
    plt.xlabel("Job execution delay (s)")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(os.path.join(dir_name, 'job_execution_delay_cdf_combined.png'))
    
    # Clear plot
    plt.clf()
    plt.close()






def plot_job_execution_delay_cdf_filter(file_list, dir_name):
    """
    Plots the CDF of job execution delays for multiple files in the same plot and saves the plot to a file.

    Args:
        file_list (list): A list of CSV file names containing job data.
        dir_name (str): The name of the directory where the plot will be saved.
    """
    # Create a figure for the combined CDF plot
    plt.figure(figsize=(10, 6))
    gpu_type = 'T4'
    for filename in file_list:
        if 'report' in filename:
            try:
                df = pd.read_csv(basepath + filename + '.csv')
            except:
                print('error')
                continue

            # df = df[df['gpu_type'] == gpu_type]

            res = df["exec_time"] - df["submit_time"]
        else:
            try:

                df = pd.read_csv(basepath + filename + '.csv')
            except:
                print('error')
                continue

            # df = df[df['gpu_type'] == gpu_type]

            res = df["allocated_at"] - df["submit_time"]

        # Calculate the CDF
        res = res.dropna().astype(int)  # Remove NaNs and convert to integers
        res = res[res >= 0]  # Filter out negative delays, if any
        res.sort_values(inplace=True)  # Sort the delays
        cdf = res.rank(method='average') / len(res)  # Compute the CDF

        # Plot the CDF for each file
        plt.plot(res, cdf, label=filename)

    # Customize the plot
    plt.ylabel("CDF")
    plt.xlabel("Job execution delay (s)")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(os.path.join(dir_name, 'job_execution_delay_cdf_combined.png'))
    
    # Clear plot
    plt.clf()
    plt.close()





def plot_job_execution_delay_submit_clt(simulation_name):
    """
    Plots the CDF of job execution delays for multiple files in the same plot and saves the plot to a file.

    Args:
        file_list (list): A list of CSV file names containing job data.
        dir_name (str): The name of the directory where the plot will be saved.
    """

    # folder_path = '/home/crownlabs/Plebiscitotest/res'  # Replace with the actual path to your folder
    folder_path = '/home/andrea/projects/Plebiscitotest/res/try2'  # Replace with the actual path to your folder

    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)

    # Filter for CSV files
    csv_files = [file for file in all_files if file.endswith('.csv')]

    # Now, 'csv_files' contains a list of all CSV files in the folder
    print("CSV Files in the folder:")
    file_list = []

    # for i in range(20):
    #     # if '1_'+str(i)+'_'+simulation_name+'.csv' in csv_files:
    #     if 'data' in csv_files:
    #         file_list.append('1_'+str(i)+'_'+simulation_name)
    for i in csv_files:
        # if 'jobs' in i and 'report' not in i:
        if 'report' in i:
            file_list.append(i)




    print(file_list)


    # Create a figure for the combined CDF plot
    gpu_types = ['T4','MISC','V100','P100']
    # gpu_types = ['MISC']
    # gpu_type = 'MISC'
    # gpu_type = 'V100'
    # gpu_type = 'P100'
    for gpu_type in gpu_types:
        plt.figure(figsize=(10, 6))

        means = []
        for filename in file_list:
            if 'report' in filename:
                try:
                    print(folder_path + filename)
                    df = pd.read_csv(folder_path + '/'+filename)
                except:
                    print('error')
                    continue

                # df = df[df['gpu_type'] == gpu_type]

                res = df["exec_time"] - df["submit_time"]
            else:
                try:

                    df = pd.read_csv(folder_path +'/'+ filename)
                except:
                    print('error')
                    continue

                df = df[df['gpu_type'] == gpu_type]

                res = df["allocated_at"] - df["submit_time"]

            res = res.dropna().astype(int)  # Remove NaNs and convert to integers
            res = res[res >= 0]  # Filter out negative delays, if any

            mean = sum(res) / len(res)
            # median = np.median(res)
            means.append(mean)        



        plt.hist(means, bins=20, edgecolor='black')
        plt.title('Distribution of Means')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('plebi_'+str(gpu_type)+str(simulation_name)+'.png')
        # Clear plot
        plt.clf()
        plt.close()







    
def plot_job_deadline(filename, dir_name):
    """
    Plots a histogram of job deadline exceeded times based on the given CSV file.

    Args:
        filename (str): The name of the CSV file (without the .csv extension).
        dir_name (str): The name of the directory where the plot will be saved.

    Returns:
        None
    """
    try:
        df = pd.read_csv(filename + "_jobs_report.csv")
    except:
        return
        
    res = df["exec_time"] + df["duration"] - df["deadline"]
        
    # plot histogram using the res variable
    res.astype(int).hist()
    
    plt.ylabel(f"Occurrences")
    plt.xlabel("Job deadline exceeded (s)")
    
    # save the plot to a file
    plt.savefig(os.path.join(dir_name, 'job_deadline_exceeded.png'))
    
    # clear plot
    plt.clf()
    plt.close()

    
def plot_job_messages_exchanged(job_count, dir_name):
    """
    Generate a boxplot of the number of messages exchanged by each job and save the plot to a file.

    Args:
        job_count (dict): A dictionary containing the number of messages exchanged by each job.
        dir_name (str): The directory where the plot will be saved.

    Returns:
        None
    """
    data = list(job_count.values())
    
    _ = plt.figure()
 
    # Creating plot
    plt.boxplot(data)
    
    plt.savefig(os.path.join(dir_name, 'number_messages_job.png'))
    
    # clear plot
    plt.clf()
    plt.close()

    
def plot_all(n_edges, filename, job_count, dir_name):
    """
    Plots all the relevant graphs for the given parameters.

    Args:
        n_edges (int): Number of edges in the graph.
        filename (str): Name of the file containing the data.
        job_count (dict): Jobs in the system.
        dir_name (str): Name of the directory where the plots will be saved.
    """
    generate_plot_folder(dir_name)
    
    plot_node_resource_usage(filename, "gpu", n_edges, dir_name)
    plot_node_resource_usage(filename, "cpu", n_edges, dir_name)
    
    plot_node_resource_usage_box(filename, "gpu", n_edges, dir_name)
    plot_node_resource_usage_box(filename, "cpu", n_edges, dir_name)
    
    plot_job_execution_delay(filename, dir_name)
    plot_job_deadline(filename, dir_name)
    
    plot_job_messages_exchanged(job_count, dir_name)
    
if __name__ == "__main__":
    
    dir_name = "plot"
    generate_plot_folder(dir_name)
        
    # plot_node_resource_usage("GPU", "gpu", 20, dir_name)
    # plot_node_resource_usage("GPU", "cpu", 20, dir_name)
    
    # plot_node_resource_usage_box("GPU", "gpu", 20, dir_name)
    # plot_node_resource_usage_box("GPU", "cpu", 20, dir_name)
    
    # plot_job_execution_delay("jobs_report", dir_name)
    # plot_job_deadline("jobs_report", dir_name)

    basepath = '/home/andrea/projects/Plebiscitotest/'
    # plot_job_execution_delay_cdf(['1_UTIL_FIFO_0_nosplit_jobs_report', '2_LGF_FIFO_0_nosplit_jobs_report'], dir_name)


    # plot_job_execution_delay_cdf_filter(['1_UTIL_FIFO_0_nosplit_jobs_report',
    #                               '2_LGF_FIFO_0_nosplit_jobs_report',
    #                               'jobs_FIFO_LGF',
    #                             #   'jobs_FIFO_SGF',
    #                               'jobs_FIFO_UTIL'
    #                               ], dir_name)
    
    # plot_job_execution_delay_cdf_filter(['1_UTIL_FIFO_0_split_jobs_report',
    #                               '2_LGF_FIFO_0_split_jobs_report',
    #                               '3_UTIL_SDF_0_split_jobs_report',
    #                             #   'jobs_FIFO_SGF',
    #                               '4_LGF_SDF_0_split_jobs_report'
    #                               ], dir_name)
    


    unique_simulation_names = ['jobs_FIFO_ID', 
                               'jobs_FIFO_LGF',
                               'jobs_FIFO_SGF',
                               'jobs_FIFO_UTIL',
                               'jobs_SDF_ID', 
                               'jobs_SDF_SGF', 
                               'jobs_SDF_LGF', 
                               'jobs_SDF_UTIL']# Add all unique names
    
    unique_simulation_names = ['UTIL_FIFO_0_split_jobs_report']

    # Run the function for each unique simulation name
    for simulation_name in unique_simulation_names:
        plot_job_execution_delay_submit_clt(simulation_name)

