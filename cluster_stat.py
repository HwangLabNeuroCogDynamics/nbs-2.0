from __future__ import division, print_function
import multiprocessing
import numpy as np
import os
import functools
import itertools

from bct.utils import BCTParamError, get_rng, binarize

# FIXME considerable gains could be realized using vectorization, although
# generating the null distribution will take a while


def ttest2_stat_only(x, y, tail):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y : type
        Description of parameter `y`.
    tail : type
        Description of parameter `tail`.

    Returns
    -------
    type
        Description of returned object.

    """
    t = np.mean(x) - np.mean(y)
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                 * np.var(y, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1 / n1 + 1 / n2)
    if denom == 0:
        return 0
    if tail == 'both':
        return np.abs(t / denom)
    if tail == 'left':
        return -t / denom
    else:
        return t / denom


def ttest_paired_stat_only(A, B, tail):
    """Short summary.

    Parameters
    ----------
    A : type
        Description of parameter `A`.
    B : type
        Description of parameter `B`.
    tail : type
        Description of parameter `tail`.

    Returns
    -------
    type
        Description of returned object.

    """
    n = len(A - B)
    df = n - 1
    sample_ss = np.sum((A - B)**2) - np.sum(A - B)**2 / n
    unbiased_std = np.sqrt(sample_ss / (n - 1))
    z = np.mean(A - B) / unbiased_std
    t = z * np.sqrt(n)
    if tail == 'both':
        return np.abs(t)
    if tail == 'left':
        return -t
    else:
        return t


def get_edge_components(A, no_depend=False):
    '''Short summary.

    Returns the components of an undirected graph specified by the binary and
    undirected matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.
    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected matrix
    no_depend : Any
        Does nothing, included for backwards compatibility
    Returns
    -------
    matrix with assigned edges -- sum all edges in component to get vector of edges size
    comps : ([N], [M]) tuple of lists
        tuple of vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes
    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1
    Note: The identity of each component (i.e. its numerical value in the
    result) is not guaranteed to be identical the value returned in BCT,
    matlab code, although the component topology is.
    Many thanks to Nick Cullen for providing this implementation
    '''

    # if not np.all(A == A.T):  # ensure matrix is undirected
    #     raise BCTParamError('get_components can only be computed for undirected'
    #                         ' matrices.  If your matrix is noisy, correct it with np.around')

    A = binarize(A, copy=True)
    n = A.shape[0]
    m = A.shape[1]

    edge_map = [(u, v) for u in range(n) for v in range(m) if A[u, v] == 1]
    comp_list = []

    # iterate through edge map and sort into components
    for item in edge_map:
        new_comp = [[item]]

        # find connected edges
        for s in comp_list:
            for edge in s:
                if item[0] == edge[0] or item[1] == edge[1]:
                    new_comp.append(s)
                    break

        # remove merged sets from component list
        for s in new_comp:
            if s in comp_list:
                comp_list.remove(s)

        # merge component and add new component to component list
        new_comp = list(itertools.chain.from_iterable(new_comp))
        comp_list.append(new_comp)

    # create list of component sizes
    edge_sizes = np.array([len(s) for s in comp_list])

    return comp_list, edge_sizes




def nbs_rect(x, y, thresh, k=1000, tail='both', paired=False, verbose=False, seed=None, cores=os.cpu_count()):
    '''
    Performs the NBS for populations X and Y for a t-statistic threshold of
    alpha.
    Parameters
    ----------
    x : NxMxP np.ndarray
        matrix representing the first population with P subjects. must be
        symmetric.
    y : NxMxQ np.ndarray
        matrix representing the second population with Q subjects. Q need not
        equal P. must be symmetric.
    thresh : float
        minimum t-value used as threshold
    k : int
        number of permutations used to estimate the empirical null
        distribution
    tail : {'left', 'right', 'both'}
        enables specification of particular alternative hypothesis
        'left' : mean population of X < mean population of Y
        'right' : mean population of Y < mean population of X
        'both' : means are unequal (default)
    paired : bool
        use paired sample t-test instead of population t-test. requires both
        subject populations to have equal N. default value = False
    verbose : bool
        print some extra information each iteration. defaults value = False
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    pval : Cx1 np.ndarray
        A vector of corrected p-values for each component of the networks
        identified. If at least one p-value is less than alpha, the omnibus
        null hypothesis can be rejected at alpha significance. The null
        hypothesis is that the value of the connectivity from each edge has
        equal mean across the two populations.
    adj : IxIxC np.ndarray
        an adjacency matrix identifying the edges comprising each component.
        edges are assigned indexed values.
    null : Kx1 np.ndarray
        A vector of K sampled from the null distribution of maximal component
        size.
    Notes
    -----
    ALGORITHM DESCRIPTION
    The NBS is a nonparametric statistical test used to isolate the
    components of an N x N undirected connectivity matrix that differ
    significantly between two distinct populations. Each element of the
    connectivity matrix stores a connectivity value and each member of
    the two populations possesses a distinct connectivity matrix. A
    component of a connectivity matrix is defined as a set of
    interconnected edges.
    The NBS is essentially a procedure to control the family-wise error
    rate, in the weak sense, when the null hypothesis is tested
    independently at each of the N(N-1)/2 edges comprising the undirected
    connectivity matrix. The NBS can provide greater statistical power
    than conventional procedures for controlling the family-wise error
    rate, such as the false discovery rate, if the set of edges at which
    the null hypothesis is rejected constitus
    ses a large component or
    components.
    The NBS comprises fours steps:
    1. Perform a two-sample T-test at each edge indepedently to test the
       hypothesis that the value of connectivity between the two
       populations come from distributions with equal means.
    2. Threshold the T-statistic available at each edge to form a set of
       suprathreshold edges.
    3. Identify any components in the adjacency matrix defined by the set
       of suprathreshold edges. These are referred to as observed
       components. Compute the size of each observed component
       identified; that is, the number of edges it comprises.
    4. Repeat K times steps 1-3, each time randomly permuting members of
       the two populations and storing the size of the largest component
       identified for each permuation. This yields an empirical estimate
       of the null distribution of maximal component size. A corrected
       p-value for each observed component is then calculated using this
       null distribution.
    [1] Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic:
        Identifying differences in brain networks. NeuroImage.
        10.1016/j.neuroimage.2010.06.041
    '''
    rng = get_rng(seed)

    if tail not in ('both', 'left', 'right'):
        raise BCTParamError('Tail must be both, left, right')

    n, m, num_sub_x = x.shape
    num_sub_y = y.shape[2]

    # Want to make algorthim able to handle matrices of different sizes
    if not y.shape[0] == n or not y.shape[1] == m:
        raise BCTParamError('Connectivity matrices must have equal N x M')
    elif paired and num_sub_y != num_sub_x:
        raise BCTParamError('Population matrices must be an equal size')

    # consider n x m edges
    ixes = np.ones([n, m])

    # number of edges
    num_edges = n * m
    print(num_edges)

    # vectorize connectivity matrices for speed
    xmat, ymat = np.zeros((num_edges, num_sub_x)), np.zeros(
        (num_edges, num_sub_y))

    for i in range(num_sub_x):
        xmat[:, i] = x[:, :, i].flatten()
    for i in range(num_sub_y):
        ymat[:, i] = y[:, :, i].flatten()
    del x, y

    # perform t-test at each edge
    t_stat = np.zeros((num_edges))
    for i in range(num_edges):
        if paired:
            t_stat[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
        else:
            t_stat[i] = ttest2_stat_only(xmat[i, :], ymat[i, :], tail)

    # threshold
    ind_t, = np.where(t_stat > thresh)

    if len(ind_t) == 0:
        raise BCTParamError("Unsuitable threshold")

    # suprathreshold adjacency matrix
    adj = np.zeros([num_edges])
    adj[ind_t] = 1
    adj.resize([n, m])
    print('suprathreshold adjacency matrix')
    print(adj)
    comps, sz_comps = get_edge_components(adj)
    nr_components = np.size(sz_comps)

    # get max component size
    if np.size(sz_comps):
        max_sz = np.max(sz_comps)
    else:
        raise BCTParamError('True matrix is degenerate')
    print('max component size is %i' % max_sz)

    # estimate empirical null distribution of maximum component size by
    # generating k independent permutations
    print('estimating null distribution with %i permutations' % k)
    null_arr = np.zeros((k))

    # parallel processing of null distribution (drastically speeds up comp time)
    pool = multiprocessing.Pool(cores)
    null_arr = pool.map(functools.partial(permute, n, m, num_sub_x, num_sub_y, xmat, ymat, paired, tail, thresh, rng), null_arr)

    # calculate p-vals
    pvals = np.zeros((nr_components))
    for i in range(nr_components):
        print(f'component index: {i}')
        print(f'comp size: {sz_comps[i]}')
        pvals[i] = np.size(np.where(null_arr >= sz_comps[i])) / k
        print(f'p-value: {pvals[i]:.4f}')

    return pvals, adj, null_arr, sz_comps


def permute(n, m, num_sub_x, num_sub_y, xmat, ymat, paired, tail, thresh, rng, null):
    num_edges = n * m

    # randomize
    if paired:
        indperm = np.sign(0.5 - rng.rand(1, num_sub_x))
        d = np.hstack((xmat, ymat)) * np.hstack((indperm, indperm))
    else:
        d = np.hstack((xmat, ymat))[:, rng.permutation(num_sub_x + num_sub_y)]

    # ttest on each edge to get suprathresholded edges
    t_stat_perm = np.zeros((num_edges))
    for i in range(num_edges):
        if paired:
            t_stat_perm[i] = ttest_paired_stat_only(
                d[i, :num_sub_x], d[i, -num_sub_x:], tail)
        else:
            t_stat_perm[i] = ttest2_stat_only(
                d[i, :num_sub_x], d[i, -num_sub_y:], tail)
    ind_t, = np.where(t_stat_perm > thresh)

    # adjacency matrix with ones denoting suprathresholded edges
    adj_perm = np.zeros((num_edges))
    adj_perm[ind_t] = 1
    adj_perm.resize([n, m])

    # isolate components and find size
    comps, sz_comps = get_edge_components(adj_perm)

    # max size of null components
    if np.size(sz_comps):
        null = np.max(sz_comps)
    else:
        null = 0

    return null
