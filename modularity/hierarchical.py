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

    A_cons = np.zeros_like(A)
    for n_iter in range(n_consensus):
        B = A - P_estimator(A)
        comms, Q = genlouvain(B, limit=1000, verbose=False)
        comms_matr = np.zeros((len(comms), len(np.unique(comms))))
        comms_matr[np.arange(len(comms)), comms] = 1
        A_cons += (comms_matr @ comms_matr.T)
    A_cons /= n_iter
    A_cons[np.diag_indices_from(A_cons)] = 0
        

def temporal_state_graph(feature_matrix, nonneg):
    """
    Construct a temporal adjacency matrix from an observed set of features. 
    
    Parameters
    ----------
        feature_matrix: numpy.ndarray, shape=[n_timepts x n_feats]
            Array of features (n_feats) observed at different time points (n_timepts).
            Assumes that features are normalized such that values are of comparable scale.
        
        nonneg: bool
            Indicates whether the features should be considered nonnegative in value.       
        
    Returns
    -------
        M: numpy.ndarray, shape=[n_timepts x n_timepts]
            Array containing temporal adjacency matrix. Rows and columns encode
            similarity in the observed feature profile at two different time points.        
    """

    M = np.zeros((feature_matrix.shape[0], feature_matrix.shape[0]))
    for ii, (tr_ix, tr_iy) in enumerate(zip(*np.triu_indices(feature_matrix.shape[0], k=1))):
        ft_pair = feature_matrix[[tr_ix, tr_iy]]
        ft_pair = ft_pair[:, ~np.isnan(ft_pair).any(axis=0)]
        ft_pair = (ft_pair.T / np.linalg.norm(ft_pair, axis=1)).T
        #ma_corr = np.corrcoef(ft_pair)[0,1]
        ma_corr = (ft_pair @ ft_pair.T)[0,1]
        M[tr_ix, tr_iy] = ma_corr
    M += M.T
    M[np.diag_indices_from(M)] = 0
    return M


def temporal_proximity_graph(feature_times, tau):
    """
    Construct a temporal proximity matrix based on the time vector. 
    Assumes that temporal dependency falls exponentially as exp(-t/tau).
    
    Parameters
    ----------
        feature_times: numpy.ndarray, shape=[n_timepts]
            Vector of times corresponding to the measured features. 
        
        tau: float
            Time constant of the minimum duration temporal process to consider
            in the model. Should be in the same units as the feature_times.
        
    Returns
    -------
        Mt: numpy.ndarray, shape=[n_timepts x n_timepts]
            Array containing temporal proximity matrix. Rows and columns encode
            the likelihood that the features at two different time points were
            generated by the same process.
    """
    Mt = pairwise.euclidean_distances(feature_times.reshape(-1,1))
    Mt = np.exp(-Mt/tau)
    Mt[np.diag_indices_from(Mt)] = 0
    
    return Mt


def temporal_surrogate_state_graph(feature_matrix, Mt, nonneg, perm_axis=0):
    """
    Non-informative uniform shuffle of feature values.

    Parameters
    ----------
        feature_matrix: numpy.ndarray, shape=[n_timepts x n_feats]
            Array of features (n_feats) observed at different time points (n_timepts).
            Assumes that features are normalized such that values are of comparable scale.

        Mt: numpy.ndarray, shape=[n_timepts x n_timepts]
            Array containing temporal proximity matrix. Rows and columns encode
            the likelihood that the features at two different time points were
            generated by the same process.

        nonneg: bool
            Indicates whether the features should be considered nonnegative in value.

        perm_axis: int
            Iterate over perm_axis and shuffle the other dimension.

    Returns
    -------
        M: numpy.ndarray, shape=[n_timepts x n_timepts]
            Array containing temporal adjacency matrix. Rows and columns encode
            similarity in the observed feature profile at two different time points.
    """

    if perm_axis == 0:
        # randomize features per observation
        feature_matrix_surr = np.array([np.random.permutation(m)
                                        for m in feature_matrix])
    else:
        # randomize observations per feature
        feature_matrix_surr = np.array([np.random.permutation(m)
                                        for m in feature_matrix.T]).T
 

    # reweight randomized features according to temporal proximity
    # preserve temporal dependeny in surrogate data
    Mr = temporal_state_graph(Mt@feature_matrix_surr, nonneg=nonneg)

    return Mr  


def compare_true_surrogate_graph(M, Mr):
    """
    Compare the true temporal graph to surrogate temporal graphs that
    preserve temporal dependence.
    
    Parameters
    ----------
        M: numpy.ndarray, shape=[n_timepts x n_timepts]
            Array containing temporal adjacency matrix. Rows and columns encode
            similarity in the observed feature profile at two different time points.  
        
        Mr: numpy.ndarray, shape=[n_timepts x n_timepts]
            Array containing surrogate temporal adjacency matrix. Rows and columns encode
            similarity in the observed feature profile at two different time points.  

    Returns
    -------
        res: float
            Correlation between the true graph and null graph.
    """
   
    res = pearsonr(convert_adj_matr_to_conn_vec(M),
                   convert_adj_matr_to_conn_vec(Mr))[0]

    return res 
