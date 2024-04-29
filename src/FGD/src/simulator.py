import copy
import os
import random
from FGD.src.node import Node
from Plebiscito.src.jobs_handler import message_data
from Plebiscito.src.utils import generate_gpu_types



import pandas as pd

num_layers = None
allocation = None
best_allocation = None
best_power_consumption = None

def dfs(adj, start):
    """
    Performs a depth-first search on a graph represented by an adjacency matrix.

    Args:
        matrix: A 2D list representing the adjacency matrix of the graph.
        start: The starting node for the DFS.

    Returns:
        The list of connected vertices
    """
    n = len(adj)  # Number of nodes in the graph
    visited = [False] * n  # Keep track of visited nodes

    def dfs_recursive(node):
        visited[node] = True  # Mark the current node as visited

        for neighbor in range(n):
            if adj[node][neighbor] and not visited[neighbor]:
                dfs_recursive(neighbor)  # Recursive call for unvisited neighbors

    dfs_recursive(start)
    
    ret = []
    for i in range(len(visited)):
        if visited[i]:
            ret.append(i)
    
    return ret

class FGD:
    def __init__(self, nodes, dataset, filename, application_graph_type, split, adjacency_matrix, failures = {}):
        self.dataset = dataset.sort_values(by=["submit_time"])
        self.compute_nodes = []
        self.filename = filename
        self.application_type = application_graph_type
        self.split = split
        self.adjacency_matrix = adjacency_matrix
        self.failures = failures
        self.leader_id = random.randint(0, nodes-1)
        self.gpu_types = generate_gpu_types(nodes)
        self.processed_jobs = []
        
        for g in self.gpu_types:
            self.compute_nodes.append(Node(g))
            
        print("FGD initialized")
            
    def save_node_state(self):
        d = {}
        
        for i, n in enumerate(self.compute_nodes):
            d["node_" + str(i) + "_cpu"] = n.used_cpu
            d["node_" + str(i) + "_gpu"] = n.used_gpu
            d["node_" + str(i) + "_bw"] = n.used_bw
            
        # append dictionary to exixting csv file
        df = pd.DataFrame(d, index=[0])
        
        if os.path.exists(self.filename + ".csv"):
            df.to_csv(self.filename + ".csv", mode='a', header=False, index=False)
        else:
            df.to_csv(self.filename + ".csv", mode='a', header=True, index=False)
            
    def save_job_state(self):            
        df = pd.DataFrame(self.processed_jobs)
        
        df.to_csv(self.filename + "_jobs_report.csv", mode='a', header=True, index=False)
            
    def detach_node(self, id):
        self.adjacency_matrix[:][id] = 0
        self.adjacency_matrix[id][:] = 0
            
    def run(self):
        time_instant = 1
        running_jobs = []
        
        print("FGD started")
        
        self.save_node_state()
        
        while len(self.dataset) > 0:
            id = -1
            if bool(self.failures):
                for i in range(len(self.failures["time"])):
                    if time_instant == self.failures["time"][i]:
                        id = self.failures["nodes"][i]
                        break
                if id != -1:
                    self.detach_node(id)
            
            self.deallocate(time_instant, running_jobs)
                
            jobs = self.dataset[self.dataset['submit_time'] <= time_instant]
            self.dataset.drop(self.dataset[self.dataset['submit_time'] <= time_instant].index, inplace = True)
            
            for _, job in jobs.iterrows():
                self.allocate(job, running_jobs, time_instant)
                
            print(f"Remaining jobs: {len(self.dataset)}")
            print(f"Running jobs: {len(running_jobs)}")
                
            self.save_node_state()
            time_instant += 1
            
        while len(running_jobs) > 0:
            self.deallocate(time_instant, running_jobs)
            self.save_node_state()
            time_instant += 1
        
        self.save_node_state()
        self.save_job_state()
        #print(self.processed_jobs)

    def deallocate(self, time_instant, running_jobs):
        end = False
        
        while not end:
            end = True
            for id, j in enumerate(running_jobs):
                if j["duration"] + j["exec_time"] < time_instant:
                    for i in range(len(self.compute_nodes)):
                        self.compute_nodes[i].deallocate(j["job_id"], j["NN_cpu"], j["NN_gpu"])
                    print(f"Deallocated job {j['job_id']}")
                    running_jobs[id]["complete_time"] = time_instant
                    self.processed_jobs.append(copy.deepcopy(running_jobs[id]))
                    del running_jobs[id]
                    end = False
                    break

    def allocate(self, job, running_jobs, time_instant):                
        data = message_data(
                    job['job_id'],
                    job['user'],
                    job['num_gpu'],
                    job['num_cpu'],
                    job['duration'],
                    job['bw'],
                    job['gpu_type'],
                    deallocate=False,
                    split=self.split,
                    app_type=self.application_type
                )
        
        global best_allocation
        best_allocation = [-1 for i in range(data['N_layer'])]
                
        self.compute_allocation(data)
        
        if -1 in best_allocation:
            self.dataset = pd.concat([self.dataset, pd.DataFrame([job])], sort=False)
            #print(f"Failed to allocate job {job['job_id']}")
            return            
            
        j = {}
        j['job_id'] = job['job_id']
        j['submit_time'] = job['submit_time']
        j['duration'] = job['duration']
        j['exec_time'] =  time_instant
        j['NN_cpu'] = data['NN_cpu']
        j['NN_gpu'] = data['NN_gpu']
        running_jobs.append(j)
        
    def compute_allocation(self, job):
        global best_allocation
                
        connected_nodes = dfs(self.adjacency_matrix, self.leader_id)
        #print(connected_nodes)
        
        for i in range(len(job["NN_cpu"])):
            node_scores = []
            
            for id, n in enumerate(self.compute_nodes):
                if id in connected_nodes:
                    frag = n.compute_fragmentation(job["NN_cpu"][i], job["NN_gpu"][i], 1/len(job["NN_cpu"], job["job_id"]), i)
                    if frag != None:
                        node_scores.append(frag)
                    else:
                        node_scores.append(float('inf'))
                else:
                    node_scores.append(float('inf'))
        
            best_score = float('inf')
            best_location = -1
            for id, s in enumerate(node_scores):
                if s < best_score:
                    best_score = s
                    best_location = id
                
            if best_location != -1:
                self.compute_nodes[best_location].allocate(job["job_id"], job["NN_cpu"][i], job["NN_gpu"][i], i)
                best_allocation[i] = best_location
            else:
                for j in range(i):
                    self.compute_nodes[best_allocation[j]].deallocate(job["job_id"], job["NN_cpu"][j], job["NN_gpu"][j], j)
                    best_allocation[j] = -1
                return 

