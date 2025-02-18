'''
Manipulate adjacency and modularity matrices.

Author: Ankit N. Khambhati <akhambhati@gmail.com>

Last Updated: 2018/11/07
'''

import numpy as np
import scipy.sparse as sp

from . import errors as err

MAX_DENSE = 10000

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


def super_modularity_matr(ml_adj_matr, layer_weight_matr, null_ml_adj_matr=None):
    """
    Returns a super modularity matrix of a network based on
    constraints provided by a topological null model and inter/intra-layer
    interactivity.

    Parameters
    ----------
        ml_adj_matr: numpy.ndarray
            Multilayer adjacency matrix
            Has shape: [n_layers x n_nodes x n_nodes]

        layer_weight_matr: numpy.ndarray
            Diagonal contains structural resolution weights specific to 
            the intra-layer connectivity. Off-diagonal contains weights
            linking nodes across layers.
            Has shape: [n_layers x n_layers]

        null_ml_adj_matr: numpy.ndarray
            Null multilayer adjacency matrix containing edge interactions
            derived from a surrogate model of the network.
            Has shape: [n_layers x n_nodes x n_nodes]

    Returns
    -------
        B_super: numpy.ndarray
            Super multilayer modularity matrix
            Has shape: [n_nodes*n_layers x n_nodes*n_layers]
    """
    # Standard param checks
    err.check_type(ml_adj_matr, np.ndarray)
    err.check_type(layer_weight_matr, np.ndarray)

    # Check ml_adj_matr dimensions
    if not len(ml_adj_matr.shape) == 3:
        raise ValueError('%r does not have three-dimensions' % ml_adj_matr)
    n_layers, n_nodes_x, n_nodes_y = ml_adj_matr.shape
    if not n_nodes_x == n_nodes_y:
        raise ValueError('%r is not symmetric' % ml_adj_matr)
    n_nodes = n_nodes_x

    # Check null ml_adj_matr dimensions
    if null_ml_adj_matr is not None:
        if not ml_adj_matr.shape == null_ml_adj_matr.shape:
            raise ValueError('null_ml_adj_matr is not of compatible size with ml_adj_matr')

        B_intra_layer = ml_adj_matr - null_ml_adj_matr
    else:
        B_intra_layer = ml_adj_matr.copy()

    # Check layer_weight_matr dimensions
    if not len(layer_weight_matr.shape) == 2:
        raise ValueError('%r does not have two-dimensions' % layer_weight_matr)
    n_layers_x, n_layers_y = layer_weight_matr.shape
    if not n_layers_x == n_layers_y:
        raise ValueError('%r is not symmetric' % layer_weight_matr)
    if not n_layers == n_layers_x:
        raise ValueError('layer_weight_matr is not of compatible size with ml_adj_matr')

    # Apply intra-layer weights
    B_intra_layer = B_intra_layer - np.diag(layer_weight_matr).reshape(-1,1,1)

    # Apply inter-layer weights
    if (n_nodes * n_layers) > MAX_DENSE:
        B_super = sp.lil_array((n_nodes * n_layers,  n_nodes * n_layers))
    else:
        B_super = np.zeros((n_nodes * n_layers, n_nodes * n_layers))

    # Populate with intra-layer modularity matrices
    for ll in range(n_layers):
        start_ix = ll * n_nodes
        end_ix = (ll + 1) * n_nodes
        B_super[start_ix:end_ix, start_ix:end_ix] = B_intra_layer[ll]

    # Populate with inter-layer weights
    for l1 in range(n_layers):
        for l2 in range(n_layers):
            if l1 == l2:
                continue
            if layer_weight_matr[l1, l2] == 0:
                continue
            row_start_ix = l1 * n_nodes
            row_end_ix = (l1 + 1) * n_nodes
            col_start_ix = l2 * n_nodes
            col_end_ix = (l2 + 1) * n_nodes

            B_super[row_start_ix:row_end_ix,
                    col_start_ix:col_end_ix] = (np.eye(n_nodes) *
                                                layer_weight_matr[l1, l2])

    return B_super
