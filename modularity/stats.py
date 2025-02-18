from tqdm import tqdm
import numpy as np
from .genlouvain import genlouvain 


def comms_to_coassign(comms):
    comms_matr = np.zeros((len(comms), len(np.unique(comms))))
    comms_matr[np.arange(len(comms)), comms] = 1
    return (comms_matr @ comms_matr.T)


def modularity_statistic(B, partition):
    partition = partition.astype(int)

    Q = np.sum(B*comms_to_coassign(partition))
    return Q


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
