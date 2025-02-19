from tqdm import tqdm
import numpy as np
from .genlouvain import genlouvain 


def comms_to_coassign(comms):
    comms_matr = np.zeros((len(comms), len(np.unique(comms))))
    comms_matr[np.arange(len(comms)), comms] = 1
    return (comms_matr @ comms_matr.T)


def modularity_statistic(B, partition, n_perm):
    partition = partition.astype(int)
   
    Q = {}
    for p_id in np.unique(partition):
        qc = B[partition == p_id, :][:, partition == p_id].mean()
        qc_null = []
        for i in range(n_perm):
            partition2 = np.random.permutation(partition) 
            qc_null.append(B[partition2 == p_id, :][:, partition2 == p_id].mean())
        qc_pv = np.mean(np.array(qc_null) >= qc)
        
        Q[p_id] = {'Qc': qc, 'Qc_pv': qc_pv}
    return Q


def pac_statistic(consensus):
    E = consensus[*np.triu_indices_from(consensus, k=1)]
    return np.mean((E > 0) & (E < 1))


def jaccard_statistic(partitions):
    partitions = partitions.astype(int)

    n_p = len(partitions)
    jac_matrix = np.zeros((n_p, n_p))
    
    for i, p_i in enumerate(partitions):
        p_i_matr = comms_to_coassign(p_i) 
        for j, p_j in enumerate(partitions):
            p_j_matr = comms_to_coassign(p_j)

            I = (p_i_matr*p_j_matr).sum()
            U = p_i_matr.sum() + p_j_matr.sum() - I
            J = I / U 

            jac_matrix[i, j] = J

    return jac_matrix
