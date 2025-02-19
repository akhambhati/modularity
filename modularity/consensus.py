from tqdm import tqdm
import numpy as np
from .genlouvain import genlouvain 


louvain_helper = lambda x: genlouvain(x, limit=1000, verbose=False)


def gen_consensus(B, n_consensus, max_tries=10, modularity_fn=louvain_helper): 
    n_tries = 0
    for n_tries in range(max_tries):
        A_cons = np.zeros_like(B)
        for n_iter in range(n_consensus):
            comms = louvain_helper(B)
            A_cons += comms_to_coassign(comms)
        A_cons /= n_iter
        A_cons[np.diag_indices_from(A_cons)] = 0
        
        n_uncertain = ((A_cons > 0) & (A_cons < 1)).sum()
        if n_uncertain == 0:
            break

        B = A_cons - np.diag(np.mean(A_cons, axis=0))

    return comms
