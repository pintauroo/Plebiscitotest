import random
import pandas as pd
import sys
import pandas as pd
import os
import time
import logging
import random
from pathlib import Path
from os import path

from Plebiscito.src.simulator import Simulator_Plebiscito
from Plebiscito.src.config import ApplicationGraphType, DebugLevel, SchedulingAlgorithm, Utility
from Plebiscito.src.dataset_builder import generate_dataset
from kubernetes.kubernetes_scheduler import KubernetesScheduler

# from Alibaba.simulator import Simulator
# from Alibaba.utils import print_fn, ALLOC_POLICY_DICT, PREEMPT_POLICY_DICT


def generate_node_failures(n_nodes, n_failures, n_jobs):
    if n_failures == 0:
        return {}
    
    time = [i for i in range(n_jobs)]
    nodes = [i for i in range(n_nodes)]
    
    failure_time = random.choices(time, k=n_failures)
    failure_nodes = random.choices(nodes, k=n_failures)
    
    return {"time": failure_time, "nodes": failure_nodes}

if __name__ == '__main__':
    NUM_JOBS = 400 #args.num_jobs
    n_nodes = 30
    n_failure = 0
    
    # # ------ START FROM ALIBABA -------
    
    # DATE = "%02d%02d" % (time.localtime().tm_mon, time.localtime().tm_mday)

    # # INPUT TRACE FILE
    # CSV_FILE_PATH = Path(__file__).parent / 'traces/pai/'
    # DESCRIBE_FILE = None
    # CSV_FILE = 'df_dataset.csv'
    # rep = 1
    
    # ARRIVAL_RATE =0 # args.arrival_rate
    # NUM_GPUS = 0 #args.num_gpus
    # REPEAT =1 # args.repeat
    # SORT_NODE_POLICY = 3
    # MAX_TIME = int(1e9)
    # VERBOSE = 0
    # LOG_LEVEL = logging.WARNING
    # NUM_NODES = 1
    # NUM_CPUS = round(23.22 * NUM_GPUS)  # 23.22 * num_gpus 156576/6742
    # HETERO = True  # heterogeneous cluster
    # PATTERN = 0  # Cluster capacity varying pattern
    # GPU_TYPE_MATCHING = 1 # GPU type perfect match
    # EXPORT_JOB_STATS = True
    # EXPORT_CLUSTER_UTIL = True
    # RANDOM_SEED = 42
    # NUM_SPARE_NODE = 0
    # SORT_BY_JCT = True

    # # Logging in directory
    # LOG_DIR = Path(__file__).parent / 'logs'

    # comments = '%dg_%dn_h%d_%dp_%dsn_%dgt-%dar-%dj-%dx-%dr' % (NUM_GPUS, NUM_NODES, HETERO, PATTERN, SORT_NODE_POLICY, GPU_TYPE_MATCHING, ARRIVAL_RATE, NUM_JOBS, REPEAT, RANDOM_SEED)

    # log_time = int(time.time() % 100000)
    # if not os.path.exists(LOG_DIR):
    #     os.makedirs(LOG_DIR)

    # log_file = LOG_DIR / ("%s-%s-%s-%s.log" % (DATE, CSV_FILE, log_time, comments))
    # logging.basicConfig(level=LOG_LEVEL, format="%(message)s", filename=log_file, filemode='a')
    # describe_file = CSV_FILE_PATH / DESCRIBE_FILE if DESCRIBE_FILE is not None else None
    
    # ------ END FROM ALIBABA -------
    
    dataset = generate_dataset(entries_num=NUM_JOBS)
    failures = generate_node_failures(n_nodes, n_failure, NUM_JOBS)
    
    # # ------ START ALIBABA SIMULATION -------
    
    # # for alloc_policy in [0, 1, 2, 4, 8]:  # 0SDF, 1SJU, 2SJG, 4SJGG, 8FIFO (see utils.py)
    # # for alloc_policy in [0, 8]:  # 0SDF, 1SJU, 2SJG, 4SJGG, 8FIFO (see utils.py)
    # for alloc_policy in [8]:  # 0SDF, 1SJU, 2SJG, 4SJGG, 8FIFO (see utils.py)
    #     # for preempt_policy in [2]:  # 2LGF
    #     preempt_policy =2
    #     # for sorting_policy in [0, 1, 2, 3]:  
    #     for sorting_policy in [3]:  
    #         print('INIT,', str(alloc_policy),', ', str(sorting_policy))

    #         key = (alloc_policy, preempt_policy)
    #         print_key = "(%-4s,%4s)" % (ALLOC_POLICY_DICT.get(key[0]), PREEMPT_POLICY_DICT.get(key[1]))

    #         # running
    #         start_time = time.time()
    #         print_fn("\n###### %s ######" % print_key)

    #         simulator = Simulator(
    #             csv_file=CSV_FILE_PATH / CSV_FILE,
    #             alloc_policy=alloc_policy,
    #             preempt_policy=preempt_policy,
    #             sort_node_policy=sorting_policy,
    #             num_nodes=NUM_NODES,
    #             random_seed=RANDOM_SEED,
    #             max_time=MAX_TIME,
    #             num_spare_node=NUM_SPARE_NODE,
    #             pattern=PATTERN,
    #             hetero=HETERO,
    #             num_gpus=NUM_GPUS,
    #             num_cpus=NUM_CPUS,
    #             describe_file=describe_file,
    #             log_file=log_file,
    #             export_job_stats=EXPORT_JOB_STATS,
    #             export_cluster_util=EXPORT_CLUSTER_UTIL,
    #             arrival_rate=ARRIVAL_RATE,
    #             num_jobs_limit=NUM_JOBS,
    #             gpu_type_matching=GPU_TYPE_MATCHING,
    #             verbose=VERBOSE,
    #             dataset=dataset,
    #             repetition=rep)
    #         results = simulator.simulator_go(repeat=REPEAT)
    #         print('done,', str(alloc_policy),', ', str(sorting_policy))
            
    # # ------ END ALIBABA SIMULATION -------
    
    # ------ START PLEBISCITO SIMULATION -------
    
    for job_dict in dataset:
        job_dict['submit_time'] += 1
        job_dict['bw'] = 0
        job_dict["exec_time"] = -1
        #job_dict["bw"] = 0 #float(job_dict["write_count"])
        job_dict["final_node_allocation"] = []
        job_dict["final_gpu_allocation"] = []
        job_dict["deadline"] = job_dict['submit_time'] + job_dict['duration'] * (1 + 0.1 * random.random()) # 10% deadline slack
        
    dataset = pd.DataFrame(dataset)
    
    simulator1 = Simulator_Plebiscito(filename="1",
                          n_nodes=n_nodes,
                          n_jobs=NUM_JOBS,
                          dataset=dataset,
                          failures=failures,
                          logical_topology="ring_graph",
                          scheduling_algorithm=SchedulingAlgorithm.FIFO,
                          #debug_level=DebugLevel.TRACE,
                          #enable_logging=True,
                          split=False)
    
    simulator2 = Simulator_Plebiscito(filename="2",
                          n_nodes=n_nodes,
                          n_jobs=NUM_JOBS,
                          dataset=dataset,
                          failures=failures,
                          logical_topology="ring_graph",
                          scheduling_algorithm=SchedulingAlgorithm.SDF,
                          #debug_level=DebugLevel.TRACE,
                          #enable_logging=True,
                          split=False)
    
    simulator3 = Simulator_Plebiscito(filename="3",
                          n_nodes=n_nodes,
                          n_jobs=NUM_JOBS,
                          dataset=dataset,
                          failures=failures,
                          logical_topology="ring_graph",
                          scheduling_algorithm=SchedulingAlgorithm.FIFO,
                          #debug_level=DebugLevel.TRACE,
                          #enable_logging=True,
                          split=True)
    
    simulator4 = Simulator_Plebiscito(filename="4",
                          n_nodes=n_nodes,
                          n_jobs=NUM_JOBS,
                          dataset=dataset,
                          failures=failures,
                          logical_topology="ring_graph",
                          scheduling_algorithm=SchedulingAlgorithm.SDF,
                          #debug_level=DebugLevel.TRACE,
                          #enable_logging=True,
                          split=True)
    
    nodes = simulator1.get_nodes()
    adj = simulator1.get_adjacency_matrix()
    
    simulator_kubernetes = KubernetesScheduler(nodes, dataset, "kubernetes", ApplicationGraphType.LINEAR, True, adj, failures)
    
    simulator1.run()
    simulator2.run()
    simulator3.run()
    simulator4.run()
    
    simulator_kubernetes.run()
    
    # ------ END PLEBISCITO SIMULATION -------
    
    
    
    
    
    
    









