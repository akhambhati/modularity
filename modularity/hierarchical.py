from tqdm import tqdm
import numpy as np
from .genlouvain import genlouvain 
from .stats import comms_to_coassign, modularity_statistic
from .consensus import gen_consensus


def recursive_clustering(
        A,
        n_consensus,
        initial_comms=None):

    # Assign all nodes to same community if at the top of the hierarchy
    if initial_comms is None:
        initial_comms = np.zeros(len(A)).astype(int).astype(str)
   
    # Make sure init comms are not all singletons
    comm_ids = np.unique(initial_comms)
    if len(comm_ids) == len(A):
        break

    # Placeholder for the next layer in the hierarchy
    next_comms = initial_comms.copy()

    # Iterate over communities in init layer
    for c_id in comm_ids:

        # Grab the sub-network belonging to the current community
        A_comm = A[initial_comms==c_id, :][:, initial_comms==c_id]

        # Find subnetwork communities using consensus clustering with 
        # a permutational null model
        subcomms = gen_consensus(A_comm, n_consensus)[0].astype(int).astype(str)

        # Insert community hierarchy label
        next_comms[initial_comms == c_id] += ("." + subcomms)
    return np.array(comm_layers)


def _nested_dict(dd, kk, vals=None):
    if not kk:
        dd['N'] = vals
        return dd
    key = int(kk[0])
    return {key: _nested_dict(dd, kk[1:], vals)}


def _merge_dicts(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            _merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1


def build_tree(comm_layers):
    tree = {}
    for c_id in np.unique(comm_layers):
        if np.isnan(c_id):
            continue
        layer, node = np.nonzero(comm_layers == c_id)
        l0 = layer[0]
        n0 = node[0]
        tree = _merge_dicts(
                tree, _nested_dict({}, list(comm_layers[:l0+1, n0]), vals=node))
    return tree


def _tree_all_keys(tree, keys=None):
    """
    """
    if keys is None:
        keys = []

    for key in tree:
        if not str(key).isnumeric():
            continue
        keys.append(key)
        if isinstance(tree[key], dict):
            _tree_all_keys(tree[key], keys)
    return keys


def _get_branch(tree, target_value, branch=()):
    """
    Recursively retrieves the child-parent relationship for a given value in a dictionary.
    """
    if branch is None:
        branch = ()

    for key, value in tree.items():
        if key == target_value:
            return branch + (key,)
        elif isinstance(value, dict):
            result = _get_branch(value, target_value, branch + (key,))
            if result:
                return result
    return None


def _get_branch_item(tree, branch):
    """
    Recursively retrieves the child-parent relationship for a given value in a dictionary.
    """
    for key in branch:
        if isinstance(tree[key], dict):
            tree = tree[key]
    return tree


def update_tree_modularity(A, P_estimator, tree, n_survival_perm):
    tree_nodes = np.unique(_tree_all_keys(tree))
    for node in tqdm(tree_nodes):
        branch = _get_branch(tree, node)
        if len(branch) == 1:
            continue
        parent = _get_branch_item(tree, branch[:-1])
        child = _get_branch_item(tree, branch)

        if len(child['N']) == 1:
            Q, Q_pv = (0.0, 1.0)
        else:
            A_parent = A[parent['N'], :][:, parent['N']]
            child_partition = np.zeros(len(parent['N']))
            child_partition[
                    np.intersect1d(parent['N'], child['N'],
                        return_indices=True)[1]] = 1
            Q, Q_pv = calc_modularity(A_parent, P_estimator, child_partition, n_survival_perm)
        child['Q'] = {'Qv': Q, 'Qp': Q_pv}


def prune_tree(tree, survival_pval):
    tree_nodes = np.unique(_tree_all_keys(tree))
    for node in tqdm(tree_nodes):
        branch = _get_branch(tree, node)
        if branch is None:
            continue
        if len(branch) == 1:
            continue
        child = _get_branch_item(tree, branch)
        if child['Q']['Qp'] >= survival_pval:
            parent = _get_branch_item(tree, branch[:-1])
            parent.pop(node, None)
