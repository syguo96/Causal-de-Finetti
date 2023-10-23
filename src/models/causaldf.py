import time
from src.utils.conditional_independence_tests import *
from typing import Dict, Tuple
from causallearn.utils.cit import CIT
from gsq.ci_tests import ci_test_bin, ci_test_dis
from pcalg import estimate_cpdag
from pcalg import estimate_skeleton


def test_topological_order(var_dict):
    """
    Here focus on 2 samples in each environment.
    Input: var_dict = {var_name: var_value}, where var_value is a num_sample * num_env matrix
    Output: sink_dict = {i: []}
            keys: ranging from 0 to n, where n is number of variables
            values: is the list of var_names that belong to S_i
    """
    # print("sink_dict:")
    n = len(var_dict.keys())
    var_names = set(var_dict.keys())
    sink_dict = {}
    for key in range(n+1):
        sink_dict[key] = []
    for i in range(1, n+1, 1):
        # print(i)
        # Find which variables need to be tested.
        test_vars = var_names
        for j in range(i):
            test_vars = test_vars ^ set(sink_dict[j]) # Exclude all that have been classified before
        # print(test_vars)
        for var in test_vars:
            # print(var)
            Xji = var_dict[var][0,:]
            # Look for valid set for S>=n
            exc_vars = test_vars ^ set([var])
            is_var_in_cur_sink = False
            if len(exc_vars) == 0:
                is_var_in_cur_sink = True
            else:
                sigtable = {}
                for k in exc_vars:
                    Xkl = var_dict[k][1,:]
                    con_vars = np.stack([var_dict[m][0,:] for m in exc_vars], axis = 1)
                    if Xkl.dtype == int:
                        _, sig = gtest(Xji, Xkl, con_vars)
                    else:
                        D = np.hstack((Xji.reshape(-1, 1), Xkl.reshape(-1, 1), con_vars))
                        kci_obj = CIT(D, "kci", approx=True)  # construct a CIT instance with data and method name
                        sig = kci_obj([0], [1], list(range(2, D.shape[1])))  # X_1 \ind Y_2 | X_2
                        print(var, k, exc_vars, sig)
                    sigtable[k] = sig
                if all(v >= 0.05 for v in list(sigtable.values())):
                    is_var_in_cur_sink = True
            if is_var_in_cur_sink:
                sink_dict[i].append(var)
    sink_dict = {k: set(v) for k, v in sink_dict.items()}
    return sink_dict


def test_edge(sink_dict, var_dict):
    """
    Input: sink_dict
           var_dict
    Output: edge_dict = {var_name: [list of var names that is its children]}
    """
    N = len(var_dict.keys())
    edge_dict = {}
    for key in var_dict.keys():
        edge_dict[key] = []
    for d in range(1, N, 1):
        for n in range(1 + d, N + 1, 1):
            if d == 1:
                # print(d, n)
                bigger_n_keys = [k for k in sink_dict.keys() if k > n]
                if len(bigger_n_keys) == 0:
                    S_bigger_n = []
                else:
                    S_bigger_n = [v for k in bigger_n_keys for v in sink_dict[k]]
                for Xji_name in sink_dict[n]:
                    for Xki_name in sink_dict[n - d]:
                        Xji = var_dict[Xji_name][0, :]
                        Xki = var_dict[Xki_name][0, :]
                        if Xji.dtype == int:
                            if len(S_bigger_n) == 0:
                                con_vars = np.matrix([])
                            else:
                                con_vars = np.stack([var_dict[v][0,:] for v in S_bigger_n], axis = 1)
                            _, sig = gtest(Xji, Xki, con_vars)
                        else:
                            if len(S_bigger_n) == 0:
                                D = np.hstack((Xji.reshape(-1, 1), Xki.reshape(-1, 1)))
                            else:
                                con_vars = np.stack([var_dict[v][0, :] for v in S_bigger_n], axis=1)
                                D = np.hstack((Xji.reshape(-1, 1), Xki.reshape(-1, 1), con_vars))
                            kci_obj = CIT(D, "kci", approx = True)  # construct a CIT instance with data and method name
                            sig = kci_obj([0], [1], list(range(2, D.shape[1])))  # X_1 \ind Y_2 | X_2
                            print(Xji_name, Xki_name, S_bigger_n, sig)
                        if sig <= 0.05:
                            edge_dict[Xji_name].append(Xki_name)
            if d >= 2:
                # print(d, n)
                # Need to condition on S>n set
                bigger_n_keys = [k for k in sink_dict.keys() if k > n]
                if len(bigger_n_keys) == 0:
                    S_bigger_n = []
                else:
                    S_bigger_n = [v for k in bigger_n_keys for v in sink_dict[k]]
                n_keys = [k for k in sink_dict.keys() if k == n]
                if len(n_keys) == 0:
                    S_n = []
                else:
                    S_n = [v for k in n_keys for v in sink_dict[k]]
                # print(S_bigger_n)
                for Xji_name in sink_dict[n]:
                    for Xki_name in sink_dict[n - d]:
                        Xji = var_dict[Xji_name][0, :]
                        Xki = var_dict[Xki_name][0, :]
                        # Need to condition on pa(ch(Xji)) interesect Sn
                        coparent = S_n.remove(Xji_name)
                        if coparent == None:
                            coparent = []
                        # coparent = []
                        # for child in edge_dict[Xji_name]:
                        #     for k, v in edge_dict.items():
                        #         if child in v:
                        #             if k in sink_dict[n] and k != Xji_name:
                        #                 coparent.append(k)
                        # print(coparent)
                        # Need to condition on non-collider intermediary nodes pa(Xki)
                        IM = []
                        for k, v in edge_dict.items():
                            if Xki_name in v:
                                if k not in S_bigger_n and k not in sink_dict[n]:
                                    IM.append(k)
                        # print(IM)
                        con_vars_name = list(set(S_bigger_n) | set(coparent) | set(IM))
                        if Xji.dtype == int:
                            if len(con_vars_name) == 0:
                                con_vars = np.matrix([])
                            else:
                                con_vars = np.stack([var_dict[v][0, :] for v in con_vars_name], axis=1)

                            _, sig = gtest(Xji, Xki, con_vars)
                        else:
                            if len(con_vars_name) == 0:
                                D = np.hstack((Xji.reshape(-1, 1), Xki.reshape(-1, 1), con_vars))
                            else:
                                con_vars = np.stack([var_dict[v][0, :] for v in con_vars_name], axis=1)
                                D = np.hstack((Xji.reshape(-1, 1), Xki.reshape(-1, 1), con_vars))
                            kci_obj = CIT(D, "kci", approx=True)  # construct a CIT instance with data and method name
                            sig = kci_obj([0], [1], list(range(2, D.shape[1])))  # X_1 \ind Y_2 | X_2
                            print(Xji_name, Xki_name, con_vars_name, sig)
                        if sig <= 0.05:
                            edge_dict[Xji_name].append(Xki_name)
    out = []
    for k in edge_dict.keys():
        for v in edge_dict[k]:
            out.append((int(k), int(v)))
    return set(out)

def run_causaldf_multivariate(data: Dict) -> Tuple:
    start_time = time.time()
    sink_dict = test_topological_order(data['data'])
    edge_dict = test_edge(sink_dict, data['data'])
    end_time = time.time()
    return edge_dict, end_time - start_time

def run_causaldf_bivariate(data: Dict) -> Tuple:
    start_time = time.time()
    X = data['x']
    Y = data['y']

    # if X.dtype == 'int':
    #     _, sigxtoy = gtest(X[0, :], Y[1, :], X[1, :].reshape(-1, 1)) # X_1 \ind Y_2 | X_2
    #     _, sigytox = gtest(Y[0, :], X[1, :], Y[1, :].reshape(-1, 1)) # Y_1 \ind X_2 | Y_2
    #     _, sigind = gtest(X[0, :], Y[0, :], np.matrix([])) # X_1 \ind Y_1
    # else:
    D = np.hstack((X, Y))
    kci_obj = CIT(D, "kci", approx=True)  # construct a CIT instance with data and method name
    sigxtoy = kci_obj([0], [3], [1]) # X_1 \ind Y_2 | X_2
    sigytox = kci_obj([2], [1], [3]) # Y_1 \ind X_2 | Y_2
    sigind = kci_obj([0], [2])

    estimated = estimate_causal_structure_causaldf_bivariate(sigxtoy, sigytox, sigind, threshold=0.05)
    end_time = time.time()
    return estimated, end_time - start_time

def estimate_causal_structure_causaldf_bivariate(sigxtoy, sigytox, sigind, threshold):
    estimated_dict = {'0': set([(0, 1)]),
                      '1': set([(1, 0)]),
                      '2': set([])}
    if sigxtoy >= threshold and sigytox <= threshold and sigind <= threshold:
        estimated = set([(0, 1)])
    elif sigxtoy <= threshold and sigytox >= threshold and sigind <= threshold:
        estimated = set([(1, 0)])
    elif sigxtoy >= threshold and sigytox >= threshold and sigind >= threshold:
        estimated = set([])
    else:
        estimated_index = np.argmax([sigxtoy, sigytox, sigind], axis=0)
        estimated = estimated_dict[str(estimated_index)]

    return estimated

def structural_hamming_distance(estimated, true):
    if len(estimated) != 0:
        return (len(estimated.difference(true)) + len(true.difference(estimated)))/(len(true) + len(estimated))
    return len(true.difference(estimated))/len(true)

def run_pc(data: Dict) -> Tuple:
    start_time = time.time()
    Data = data['cd-nod']['data']
    indep_test_func = ci_test_bin if Data.dtype == int else ci_test_dis
    alpha = 0.01
    (graph, sep_set) = estimate_skeleton(indep_test_func=indep_test_func,
                                         data_matrix=Data,
                                         alpha=alpha)

    graph = estimate_cpdag(skel_graph=graph, sep_set=sep_set)
    end_time = time.time()
    return set(graph.edges), end_time - start_time


def run_random_bivariate(data: Dict) -> Tuple:
    start_time = time.time()
    mode = np.random.randint(3)
    estimated = [set([(0, 1)]), set([(1, 0)]), set([])]
    end_time = time.time()
    return estimated[mode], end_time - start_time

def run_random_multivariate(data: Dict) -> Tuple:
    start_time = time.time()
    mode = np.random.randint(4)
    estimated = [set([(0, 2), (1, 2)]), set([(0, 1), (1, 2)]), set([(0, 1), (0, 2), (1, 2)]), set([(0, 1), (0, 2)])]
    end_time = time.time()
    return estimated[mode], end_time - start_time