from tqdm import tqdm
import numpy as np
from .genlouvain import genlouvain 
from .stats import comms_to_coassign 

louvain_helper = lambda x: genlouvain(x, limit=1000, verbose=False)


def gen_consensus(B, n_consensus, max_tries=10, modularity_fn=None): 
    n_tries = 0
    for n_tries in range(max_tries):
        A_cons = np.zeros_like(B)
        for n_iter in range(n_consensus):
            if modularity_fn is None:
                comms = louvain_helper(B)
            else:
                comms = modularity_fn(B)
            A_cons += comms_to_coassign(comms)
        
        n_uncertain = ((A_cons > 0) & (A_cons < n_iter)).sum()
        if n_uncertain == 0:
            break

        B = A_cons - A_cons[*np.triu_indices_from(A_cons, k=1)].mean()

    return comms
