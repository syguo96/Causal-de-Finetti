import time
from typing import Dict, Tuple
from causallearn.search.ScoreBased.GES import ges

def run_ges(data: Dict) -> Tuple:
    start_time = time.time()
    Data = data['cd-nod']['data']
    record = ges(Data)
    graph = record['G'].graph
    estimated = estimate_causal_structure_ges(graph)
    end_time = time.time()
    return estimated, end_time - start_time

def estimate_causal_structure_ges(adjmatrix):
    # unless clearly identify as a directed edge, else output default solution
    nodes = adjmatrix.shape[0]
    edges = []
    for i in range(nodes):
        for j in range(i+1, nodes):
            if adjmatrix[i, j] == -1 and adjmatrix[j, i] == 1:
                edges.append((i, j))
    return set(edges)
