import time
from typing import Dict, Tuple
from causalnex.structure.notears import from_numpy

def run_notears(data: Dict) -> Tuple:
    start_time = time.time()
    Data = data['cd-nod']['data']
    sm = from_numpy(Data)
    if list(sm.edges) == [(0, 1), (1, 0)]:
        # when there is bidrected edges, turn to default solution
        estimated = set()
    else:
        estimated = set(sm.edges)
    end_time = time.time()
    return estimated, end_time - start_time


