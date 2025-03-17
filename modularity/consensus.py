from tqdm import tqdm
import numpy as np
from .genlouvain import genlouvain 
from .stats import comms_to_coassign 


louvain_helper = lambda x: genlouvain(x, limit=1000, verbose=False)


def rewire_helper(A):
    A_null = np.zeros_like(A)
    A_null[*np.triu_indices_from(A, k=1)] = np.random.permutation(
            A[*np.triu_indices_from(A, k=1)])
    A_null += A_null.T
    return A_null


def gen_consensus(A, n_consensus, max_tries=10, P_estimator=None, modularity_fn=None): 
    A = A.copy()

    A_cons_init = None 
    for n_tries in range(max_tries):
        A_cons = np.zeros_like(A)
        for n_iter in range(n_consensus):
            if P_estimator is None:
                B = A - rewire_helper(A)
            else:
                B = A - P_estimator(A)
            if modularity_fn is None:
                comms = louvain_helper(B)
            else:
                comms = modularity_fn(B)
            A_cons += comms_to_coassign(comms)
        A_cons[*np.diag_indices_from(A_cons)] = 0

        if A_cons_init is None:
            A_cons_init = A_cons / n_consensus

        n_uncertain = ((A_cons > 0) & (A_cons < n_iter)).sum()
        if n_uncertain == 0:
            break

        A = A_cons.copy()

    return comms, A_cons_init
