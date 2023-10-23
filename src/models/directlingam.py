import time
import lingam
from typing import Dict, Tuple
import numpy as np

def run_directLinGAM(data: Dict) -> Tuple:
    start_time = time.time()
    Data = data['cd-nod']['data']
    model = lingam.DirectLiNGAM()
    model.fit(Data)
    adjmatrix = model.adjacency_matrix_
    estimated = estimate_causal_structure_lingam(adjmatrix)
    end_time = time.time()
    return estimated, end_time - start_time

def estimate_causal_structure_lingam(adjmatrix):
    adjmatrix = np.where(adjmatrix!= 0, 1, 0)
    if (adjmatrix == np.array([[0, 0], [1, 0]])).all():
        estimated = set([(0, 1)])
    elif (adjmatrix == np.array([[0, 1], [0, 0]])).all():
        estimated = set([(1, 0)])
    elif (adjmatrix == np.zeros(2)).all():
        estimated = set([])
    else:
        estimated = 'inconclusive'
    return estimated