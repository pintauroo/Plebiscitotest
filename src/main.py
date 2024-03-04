import random

import pandas as pd
from Plebiscito.src.simulator import Simulator_Plebiscito
from Plebiscito.src.config import ApplicationGraphType, DebugLevel
from Plebiscito.src.dataset_builder import generate_dataset
from kubernetes.kubernetes_scheduler import KubernetesScheduler

def generate_node_failures(n_nodes, n_failures, n_jobs):
    if n_failures == 0:
        return {}
    
    time = [i for i in range(n_jobs)]
    nodes = [i for i in range(n_nodes)]
    
    failure_time = random.choices(time, k=n_failures)
    failure_nodes = random.choices(nodes, k=n_failures)
    
    return {"time": failure_time, "nodes": failure_nodes}

if __name__ == '__main__':
    n_jobs = 50
    n_nodes = 10
    n_failure = 0
    
    dataset = generate_dataset(entries_num=n_jobs)
    failures = generate_node_failures(n_nodes, n_failure, n_jobs)
    
    for job_dict in dataset:
        job_dict['submit_time'] += 1
        job_dict['bw'] = 0
        job_dict["exec_time"] = -1
        #job_dict["bw"] = 0 #float(job_dict["write_count"])
        job_dict["final_node_allocation"] = []
        job_dict["final_gpu_allocation"] = []
        job_dict["deadline"] = job_dict['submit_time'] + job_dict['duration'] * (1 + 0.1 * random.random()) # 10% deadline slack
        
    dataset = pd.DataFrame(dataset)
    
    simulator = Simulator_Plebiscito(filename="prova",
                          n_nodes=n_nodes,
                          n_jobs=n_jobs,
                          dataset=dataset,
                          failures=failures,
                          logical_topology="linear_topology",
                          #debug_level=DebugLevel.TRACE,
                          #enable_logging=True,
                          split=False)
    
    nodes = simulator.get_nodes()
    adj = simulator.get_adjacency_matrix()
    
    simulator_kubernetes = KubernetesScheduler(nodes, dataset, "kubernetes", ApplicationGraphType.LINEAR, True, adj, failures)
    
    simulator.run()
    simulator_kubernetes.run()
    
    