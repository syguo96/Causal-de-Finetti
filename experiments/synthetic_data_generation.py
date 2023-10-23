import numpy as np
from typing import Dict
import networkx as nx
import random

def scm_bivariate_binary(seed: int, num_env: int) -> Dict:
    """
    Create Binary Exchangeable Data for Bivariate Graph
    :param seed:
    :param num_env:
    :return:
    """

    # Instantiating the output results: data
    data = {}

    num_sample = 2
    alpha_theta = 1
    beta_theta = 3
    alpha_psi = 2
    beta_psi = 4

    np.random.seed(seed)
    theta = np.random.beta(alpha_theta, beta_theta, num_env)
    psi = np.random.beta(alpha_psi, beta_psi, num_env)
    # mode decides the true causal structure 0: X->Y; 1: Y->X; 2: X \ind Y
    mode = np.random.choice(3)
    if mode == 0:
        X = np.random.binomial(1, p=theta, size=(num_sample, num_env))
        psi_ind = np.random.binomial(1, p=psi, size=(num_sample, num_env))
        Y = (psi_ind != X).astype(int)
        data['true_structure'] = set([(0, 1)])
    elif mode == 1:
        Y = np.random.binomial(1, p=theta, size=(num_sample, num_env))
        psi_ind = np.random.binomial(1, p=psi, size=(num_sample, num_env))
        X = (psi_ind != Y).astype(int)
        data['true_structure'] = set([(1, 0)])
    elif mode == 2:
        X = np.random.binomial(1, p=theta, size=(num_sample, num_env))
        Y = np.random.binomial(1, p=psi, size=(num_sample, num_env))
        data['true_structure'] = set([])


    data['x'] = X
    data['y'] = Y

    return data

def scm_bivariate_continuous(num_env: int, num_sample: int) -> Dict:
    """
    Create Binary Exchangeable Data for Bivariate Graph
    :param seed:
    :param num_env:
    :return:
    """

    # Instantiating the output results: data
    data = {}

    num_var = 2
    N = np.random.uniform(-1, 1, (num_var, num_env))
    N = np.repeat(N[:, :, np.newaxis], num_sample, axis=2)
    Nprime = np.random.laplace(N)

    mode = np.random.choice(3)
    if mode == 0:
        A = np.tril(np.random.randint(1, 10, size=(num_var, num_var)), 0)
        A[0, 0] = 1
        data['true_structure'] = set([(0, 1)])
    elif mode == 1:
        A = np.triu(np.random.randint(1, 10, size=(num_var, num_var)), 0)
        A[1, 1] = 1
        data['true_structure'] = set([(1, 0)])
    elif mode == 2:
        A = np.eye(2)
        data['true_structure'] = set([])

    D = np.einsum('ij, jkh->ikh', A, Nprime)

    env_idx = np.random.choice(num_env, int(num_env/2), replace = False)
    env_mask = np.zeros((num_var, num_env, num_sample), dtype = bool)
    env_mask[:, env_idx, :] = True
    B = (A - np.eye(num_var))*np.random.randint(1, 10)
    D += np.einsum('ij, jkh->ikh', B, Nprime**2) * env_mask


    Data = D.reshape(num_var, -1).T
    c_indx = np.repeat(range(1, num_env + 1), num_sample).reshape(-1, 1).astype(float)

    X = D[0, :, :2] #num_env, num_sample
    Y = D[1, :, :2]
    data['causal-de-finetti'] = {
        'x': X,
        'y': Y,
    }
    data['cd-nod'] = {
        'data': Data,
        'c_indx': c_indx,
    }
    data['x'] = X
    data['y'] = Y

    return data


def scm_multivariate_binary(num_env: int, num_sample: int, num_var: int) -> Dict:
    """
    Create Binary Exchangeable Data for Bivariate Graph
    :param seed:
    :param num_env:
    :return:
    """

    # Instantiating the output results: data
    data = {}

    alpha = 1
    beta = 3

    # Ensure there are specified num_var in the generated DAG
    nodes_generated = []
    while len(nodes_generated) != num_var or len(nodes_generated) == 0:
        G = nx.gnp_random_graph(num_var, 0.5, directed=True)
        DAG = nx.DiGraph([(u, v, {'weight': random.randint(-10, 10)}) for (u, v) in G.edges() if u < v])
        nodes_generated = DAG.nodes

    thetas = np.random.beta(alpha, beta, (num_env, num_var))
    # Generate Data
    Data = np.zeros((num_sample*num_env, num_var))
    var_dict = {str(x): None for x in range(num_var)}
    var_dict['0'] = np.random.binomial(1, p = thetas[:, 0], size = (num_sample, num_env))
    Data[:, 0] = var_dict['0'].T.reshape(-1)
    for d in range(1, num_var):
        factor = np.random.binomial(1, p = thetas[:, d], size = (num_sample, num_env))
        predecessors = [x for x in DAG.predecessors(d)]
        if predecessors != []:
            multi = 1
            for x in DAG.predecessors(d):
                multi *= var_dict[str(x)]
            var_dict[str(d)] = (factor != multi).astype(int)
        else:
            var_dict[str(d)] = factor
        Data[:, d] = var_dict[str(d)].T.reshape(-1)

    data['true_structure'] = set(DAG.edges)
    data['data'] = var_dict

    Data = Data.astype(int)
    c_indx = np.repeat(range(1, num_env + 1), num_sample).reshape(-1, 1).astype(float)
    data['cd-nod'] = {
        'data': Data,
        'c_indx': c_indx,
        'true_structure': set(DAG.edges)
    }

    return data

def scm_multivariate_continuous(num_env: int, num_sample: int, num_var: int) -> Dict:
    """
    Create Binary Exchangeable Data for Bivariate Graph
    :param seed:
    :param num_env:
    :return:
    """

    # Instantiating the output results: data
    data = {}
    data['data'] = {}

    # Ensure there are specified num_var in the generated DAG
    nodes_generated = []
    while len(nodes_generated) != num_var or len(nodes_generated) == 0:
        G = nx.gnp_random_graph(num_var, 0.5, directed=True)
        DAG = nx.DiGraph([(u, v, {'weight': random.randint(-10, 10)}) for (u, v) in G.edges() if u < v])
        nodes_generated = DAG.nodes

    data['true_structure'] = set(DAG.edges)

    N = np.random.uniform(-10, 10, (num_var, num_env))
    N = np.repeat(N[:, :, np.newaxis], num_sample, axis=2)
    Nprime = np.random.laplace(N)

    A = np.eye(num_var)
    for (v1, v2) in DAG.edges:
        A[v2, v1] = np.random.randint(low=1, high=10)

    D = np.einsum('ij, jkh->ikh', A, Nprime)
    var_dict = {}
    for v in range(num_var):
        var_dict[str(v)] = D[v, :].T

    data['data'] = var_dict
    Data = D.reshape(num_var, -1).T
    c_indx = np.repeat(range(1, num_env + 1), num_sample).reshape(-1, 1).astype(float)


    data['cd-nod'] = {
        'data': Data,
        'c_indx': c_indx,
        'true_structure': set(DAG.edges)
    }
    return data
