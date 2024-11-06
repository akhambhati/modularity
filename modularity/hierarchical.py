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
    A_cons_perm_best = None
    while True: 
        A_cons = np.zeros_like(A)
        A_cons_perm = np.zeros_like(A)
        for n_iter in range(n_consensus):
            B = A - P_estimator(A)
            comms, _ = genlouvain(B, limit=1000, verbose=False)
            A_cons += comms_to_coassign(comms)
            A_cons_perm += comms_to_coassign(np.random.permutation(comms))
        A_cons /= n_iter
        A_cons[np.diag_indices_from(A_cons)] = 0
        A_cons_perm /= n_iter
        A_cons_perm[np.diag_indices_from(A_cons_perm)] = 0

        n_tries += 1
        if n_tries >= max_tries:
            break

        n_uncertain = ((A_cons > 0) & (A_cons < 1)).sum()
        if n_uncertain < last_uncertain:
            A_cons_best = A_cons.copy()
            A_cons_perm_best = A_cons_perm.copy()
            last_uncertain = n_uncertain
        if last_uncertain == 0:
            break
    comms, _ = genlouvain(A_cons_best-A_cons_perm_best,
            limit=1000, verbose=False) 

    return comms, A_cons_best


def calc_modularity(A, P_estimator, partition, n_perm):
    B = A - P_estimator(A)
    partition = partition.astype(int)
    Q = np.sum(B*comms_to_coassign(partition))
    Q_null = np.array([
        np.sum(B*comms_to_coassign(np.random.permutation(partition)))
        for i in range(n_perm)])
    return Q, np.mean(Q_null >= Q)


def recursive_clustering(
        A,
        P_estimator,
        n_consensus,
        min_comm_size=0):

    comm_layers = [np.zeros(len(A))]
    last_comm_id = 0
    while True:
        print('Optimizing Layer: {}'.format(len(comm_layers)))
        new_comm_layer = np.nan*np.zeros(len(A))
        comm_ids = np.unique(comm_layers[-1])
        if len(comm_ids) == len(A):
            break
        for c_id in comm_ids:
            if np.isnan(c_id):
                continue
            A_comm = A[comm_layers[-1]==c_id, :][:, comm_layers[-1]==c_id]
            subcomms, subA_cons = gen_consensus(A_comm, P_estimator, n_consensus)
            subcomms = subcomms.astype(float)
            for sub_c_id in np.unique(subcomms):
                if len(subcomms[subcomms == sub_c_id]) < min_comm_size:
                    subcomms[subcomms == sub_c_id] = np.nan
            subcomms += (last_comm_id + 1)
            new_comm_layer[comm_layers[-1]==c_id] = subcomms
            last_comm_id = max(subcomms)
        if np.isnan(new_comm_layer).all():
            break
        comm_layers.append(new_comm_layer)
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


def update_tree_modularity(A, P_estimator, tree, n_survival_perm, survival_pval):
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
