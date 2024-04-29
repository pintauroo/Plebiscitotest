import copy
from Plebiscito.src.config import Utility, GPUType, GPUSupport
from FGD.src.utils import Quadrant

class Node:
    def __init__(self, gpu_type):
        self.gpu_type = gpu_type
        self.total_cpu, self.total_gpu = GPUSupport.get_compute_resources(gpu_type)
        self.gpus = []
        self.used_cpu = 0
        self.used_gpu = 0
        self.used_bw = 0
        self.allocated_on = {}
        
        for _ in range(self.total_gpu):
            self.gpus.append(1)
            
    def deallocate(self, job_id, cpu, gpu, id):
        self.gpus[self.allocated_on[job_id][id]] += gpu
    
        self.used_cpu -= cpu
            
    def allocate(self, job_id, cpu, gpu, id):
        self.gpus[self.allocated_on[job_id][id]] -= gpu
    
        self.used_cpu += cpu
    
    def compute_quadrant(self, cpu, gpu, u):
        if gpu == 0:
            return Quadrant.OTHER
        
        if cpu > self.total_cpu - self.used_cpu or gpu > u:
            return Quadrant.Q124
        else:
            return Quadrant.Q3
        
    def can_host(self, cpus):
        if cpus > self.total_cpu - self.used_cpu:
            return False
        
        return True
    
    def compute_fragmentation(self, workload_cpus, workload_gpus, popularity, job_id, id):
        node_gpus = copy.deepcopy(self.gpus) 
        
        if job_id not in self.allocated_on:
            self.allocated_on[job_id] = [-1 for _ in range(1000)]
        
        if not self.can_host(workload_cpus[id]):
            return None
        
        fragmentation = 0
        
        # for each task in the workload
        best_frag = float('inf')
        best_id = -1
                    
        for j in range(len(node_gpus)):
            if node_gpus[j] < workload_gpus[id]:
                continue
            
            f_before = self._compute_fragmentation(workload_cpus, workload_gpus, node_gpus)
            node_gpus[j] -= workload_gpus[id]
            f_after = self._compute_fragmentation(workload_cpus, workload_gpus, node_gpus)
            frag = f_after - f_before
            
            if frag < best_frag:
                best_frag = frag
                best_id = j
                
            node_gpus[j] += workload_gpus[id]
            
        if best_frag == float('inf'):
            return None
        
        self.allocated_on[job_id][id] = best_id
        fragmentation += best_frag * popularity
            
        return fragmentation

    def _compute_fragmentation(self, workload_cpus, workload_gpus, node_gpus):
        u = self.compute_u(node_gpus)
        f = 0
        
        for i in range(len(workload_cpus)):
            quadrant = self.compute_quadrant(workload_cpus[i], workload_gpus[i], u)
                
            if quadrant == Quadrant.Q124:
                for g in node_gpus:
                    f += g
            elif quadrant == Quadrant.Q3:
                for g in node_gpus:
                    if g < min(workload_gpus[i], 1):
                        f += g
            else:
                for g in node_gpus:
                    f += g
        
        return f

    def compute_u(self, node_gpus):
        fully_unallocated = 0
        maximum_partial = 0
                
        for i in range(len(node_gpus)):
            if node_gpus[i] == 1:
                fully_unallocated += 1
            elif node_gpus[i] > maximum_partial:
                maximum_partial = node_gpus[i]
                        
        u = fully_unallocated + maximum_partial
        return u
            