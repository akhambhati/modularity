from tqdm import tqdm
import numpy as np
from .genlouvain import genlouvain 


def comms_to_coassign(comms):
    comms_matr = np.zeros((len(comms), len(np.unique(comms))))
    comms_matr[np.arange(len(comms)), comms] = 1
    return (comms_matr @ comms_matr.T)


def gen_consensus(A, P_estimator, n_consensus, max_tries=10): 
    n_tries = 0
    last_uncertain = np.inf
    A_cons_best = None 
    while True: 
        A_cons = np.zeros_like(A)
        for n_iter in range(n_consensus):
            B = A - P_estimator(A)
            comms, _ = genlouvain(B, limit=1000, verbose=False)
            A_cons += comms_to_assign(comms)
        A_cons /= n_iter
        A_cons[np.diag_indices_from(A_cons)] = 0

        n_tries += 1
        if n_tries >= max_tries:
            break

        n_uncertain = ((A_cons > 0) & (A_cons < 1)).sum()
        if n_uncertain < last_uncertain:
            A_cons_best = A_cons.copy()
            last_uncertain = n_uncertain
        if last_uncertain == 0:
            break
    comms, _ = genlouvain(A_cons_best, limit=1000, verbose=False) 

    return comms, A_cons_best


def recursive_clustering(
        A,
        P_estimator,
        n_consensus,
        n_survival_perm,
        survival_pval):

    n_layers = 5
    comm_layers = [np.zeros(len(A))]
    for layer in range(n_layers):
        new_comm_layer = np.nan*np.zeros(len(A))
        comm_ids = np.unique(comm_layers[-1])
        for c_id in comm_ids:
            A_comm = A[comm_layers[-1]==c_id, :][:, comm_layers[-1]==c_id]

            subcomms, subA_cons = gen_consensus(A_comm, P_estimator, n_consensus)
            new_comm_layer[comm_layers[-1]==c_id] = subcomms
        comm_layers.append(new_comm_layer)
    return np.array(comm_layers)
