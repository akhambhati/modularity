'''
Manipulate adjacency and modularity matrices.

Author: Ankit N. Khambhati <akhambhati@gmail.com>

Last Updated: 2018/11/07
'''

import numpy as np
import scipy.sparse as sp

from . import errors as err


def convert_conn_vec_to_adj_matr(conn_vec):
    '''
    Convert connections to adjacency matrix,
    assuming symmetric connectivity

    Parameters
    ----------
        conn_vec: numpy.ndarray
            Vector with shape (n_conn,) specifying unique connections

    Returns
    -------
        adj_matr: numpy.ndarray
            Symmetric matrix with shape (n_node, n_node)
    '''

    # Standard param checks
    err.check_type(conn_vec, np.ndarray)
    if not len(conn_vec.shape) == 1:
        raise ValueError('%r has more than 1-dimension')

    # Compute number of nodes
    n_node = int(np.floor(np.sqrt(2 * len(conn_vec))) + 1)

    # Compute upper triangle indices (by convention)
    triu_ix, triu_iy = np.triu_indices(n_node, k=1)

    # Convert to adjacency matrix
    adj_matr = np.zeros((n_node, n_node))
    adj_matr[triu_ix, triu_iy] = conn_vec

    adj_matr += adj_matr.T

    return adj_matr


def convert_adj_matr_to_conn_vec(adj_matr):
    '''
    Convert adjacency matrix to connectivity vector,
    assuming symmetric connectivity.

    Parameters
    ----------
        adj_matr: numpy.ndarray
            Matrix with shape (n_win, n_node, n_node)

    Returns
    -------
        conn_vec: numpy.ndarray
            Vector with shape (n_conn,)
    '''

    # Standard param checks
    err.check_type(adj_matr, np.ndarray)
    if not len(adj_matr.shape) == 2:
        raise ValueError('%r requires 2-dimensions (n_node, n_node)')

    # Compute number of nodes
    n_node = adj_matr.shape[1]

    # Compute upper triangle indices (by convention)
    triu_ix, triu_iy = np.triu_indices(n_node, k=1)

    # Convert to connectivity vector
    conn_vec = adj_matr[triu_ix, triu_iy]

    return conn_vec


def super_modularity_matr(conn_matr, gamma, omega, null='None'):
    """
    Find the super modularity matrix of a network with single or multiple
    layers. Current implementation assumes sequential linking between layers
    with homogenous weights.

    Parameters
    ----------
        conn_matr: numpy.ndarray
            Connection matrix over multiple layers
            Has shape: [n_layers x n_conns]

        gamma: float
            Intra-layer resolution parameter, typical values around 1.0

        omega: float
            Inter-layer resolution parameter, typical values around 1.0

        null: str
            Choose a null mode type: 
                ['None', 'sequential', 'connectional', 'nodal']

    Returns
    -------
        ml_mod_matr: numpy.ndarray
            Multilayer modularity matrix
            Has shape: [n_nodes*n_layers x n_nodes*n_layers]

        twomu: float
            Total edge weight in the network
    """
    # Standard param checks
    err.check_type(conn_matr, np.ndarray)
    err.check_type(gamma, float)
    err.check_type(omega, float)
    err.check_type(null, str)

    # Check conn_matr dimensions
    if not len(conn_matr.shape) == 2:
        raise ValueError('%r does not have two-dimensions' % conn_matr)
    n_layers = conn_matr.shape[0]
    n_conns = conn_matr.shape[1]
    n_nodes = int(np.floor(np.sqrt(2 * n_conns)) + 1)

    # Check null model specomm_initfication
    valid_null_types = ['none', 'sequential', 'connectional', 'nodal']
    null = null.lower()
    if null not in valid_null_types:
        raise ValueError('%r is not on of %r' % (null, valid_null_types))

    # Initialize multilayer matrix
    B = np.zeros((n_nodes * n_layers, n_nodes * n_layers))
    twomu = 0

    if null == 'sequential':
        rnd_layer_ix = np.random.permutation(n_layers)
        conn_matr = conn_matr[rnd_layer_ix, :]

    if null == 'connectional':
        rnd_node_ix = np.random.permutation(n_nodes)
        rnd_node_iy = np.random.permutation(n_nodes)
        ix, iy = np.mgrid[0:n_nodes, 0:n_nodes]

    for ll, conn_vec in enumerate(conn_matr):
        A = convert_conn_vec_to_adj_matr(conn_vec)
        if null == 'connectional':
            A = A[rnd_node_ix[ix], rnd_node_iy[iy]]
            A = np.triu(A, k=1)
            A += A.T

        # Compute node degree
        k = np.sum(A, axis=0)
        twom = np.sum(k)  # Intra-layer average node degree
        twomu += twom  # Inter-layer accumulated node degree

        # NG Null-model
        if twom < 1e-6:
            P = np.dot(k.reshape(-1, 1), k.reshape(1, -1)) / 1.0
        else:
            P = np.dot(k.reshape(-1, 1), k.reshape(1, -1)) / twom

        # Multi-slice modularity matrix
        start_ix = ll * n_nodes
        end_ix = (ll + 1) * n_nodes
        B[start_ix:end_ix, start_ix:end_ix] = A - gamma * P

    # Add inter-slice degree
    twomu += twomu + 2 * omega * n_nodes * (n_layers - 1)

    # Add the sequential inter-layer model
    interlayer = sp.spdiags(
        np.ones((2, n_nodes * n_layers)), [-n_nodes, n_nodes],
        n_nodes * n_layers, n_nodes * n_layers).toarray()
    if null == 'nodal':
        null_layer = np.random.permutation(np.diag(np.ones(n_nodes)))
        for ll in range(n_layers - 1):
            interlayer[ll * n_nodes:(ll + 1) * n_nodes, (ll + 1) * n_nodes:(
                ll + 2) * n_nodes] = null_layer
        interlayer = np.triu(interlayer, k=1)
        interlayer += interlayer.T

    B = B + omega * interlayer
    B = np.triu(B, k=1)
    B += B.T
    ml_mod_matr = B

    return ml_mod_matr, twomu
