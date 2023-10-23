import time
from causallearn.search.ConstraintBased.FCI import fci
from typing import Dict, Tuple
import numpy as np

def run_fci(data: Dict) -> Tuple:
    start_time = time.time()
    Data = data['cd-nod']['data']
    G, _ = fci(Data)
    estimated = estimate_causal_structure_fci(G.graph)
    end_time = time.time()
    return estimated, end_time - start_time

def estimate_causal_structure_fci(adjmatrix):
    # identify as a directed edge, if node i is a direct cause of node j, or node j is not an ancestor of node i
    # see doc in https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html#id4
    nodes = adjmatrix.shape[0]
    edges = []
    for i in range(nodes):
        for j in range(i+1, nodes):
            if adjmatrix[i, j] == 2 and adjmatrix[j, i] == 1:
                edges.append((i, j))
            if adjmatrix[i, j] == -1 and adjmatrix[j, i] == 1:
                edges.append((i, j))
    return set(edges)


