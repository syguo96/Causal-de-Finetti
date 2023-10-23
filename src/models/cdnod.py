import time
from typing import Dict, Tuple
import matlab.engine
eng = matlab.engine.start_matlab()
path = './src/models/cdnod-matlab'
eng.cd(path, nargout=0)

def run_cd_nod(data: Dict) -> Tuple:
    start_time = time.time()
    Data = data['cd-nod']['data'].astype(float)
    c_indx = data['cd-nod']['c_indx']
    cond_ind_test = 'indtest_new_t'
    maxFanIn = 2
    alpha = 0.05
    Type = 1
    pars = {}
    pars['pairwise'] = False
    g = eng.nonsta_cd_new(Data, cond_ind_test, c_indx, maxFanIn, alpha, Type, pars)
    estimated_cd_nod = estimate_causal_structure_cdnod(g)
    end_time = time.time()
    return estimated_cd_nod, end_time - start_time

def estimate_causal_structure_cdnod(g):
    num_var = len(g) - 1
    edge_dict = []
    for d in range(num_var):
        for k in range(num_var):
            if g[d][k] == 1:
                edge_dict.append((d, k))
    return set(edge_dict)
