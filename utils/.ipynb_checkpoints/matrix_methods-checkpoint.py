################################################################################  
import numpy as np

################################################################################
def get_spectral_radius(A):
    '''Given an adjacency matrix, computes and returns the maximum absolute
    value of its eigenvalues.
    
    Parameters
    ----------
    A : 2D numpy array
        An adjacency matrix.
    
    Returns
    -------
    r : float
        The spectral radius (i.e. largest absolute value of eigenvalues) of A.
    '''
    
    try:
        r = np.max(np.abs(np.linalg.eig(A)[0]))
    except:
        r = np.nan
        
    return r


################################################################################
def negate(A):
    '''Given a matrix A of 1s and 0s, returns the negation of A such that all
    0s in A are changed to 1s and vice versa.
    
    Parameters
    ----------
    A : 2D numpy array
        2D-array with integer values in [0,1].
        
    Returns
    -------
    A_neg : 2D numpy array
        Negation of A.
    '''

    A_neg = np.array(A == 0, dtype=int)

    return A_neg


################################################################################
def random_adjacency(n, p, normalize=True):
    '''Make an adjacency matrix of a directed G(n,p) random graph.

    DEPRECATED.
    
    Parametes
    ---------
    n : int
        Number of nodes in the network.
        
    p : float
        Connection probability for node pairs.
        
    normalize : bool (default=`True`)
        If set to `True`, normalize adjacency matrix by its spectral radius
        (i.e., largest absolute value of eigenvalues). If the spectral radius is
        0, do not normalize.
        
    Returns
    -------
    A : 2D numpy array
        An adjacency matrix of a directed G(n,p) random graph.
    '''

    eig_max = 0 
    while eig_max==0:
        A = np.random.uniform(size=(n,n))
        A[A<(1-p)] = 0
        A[A>=(1-p)] = 1
        np.fill_diagonal(A, 0)
        eig_max = np.max(np.abs(np.linalg.eig(A)[0]))
    
    return A/eig_max


################################################################################
def assign_delays(adj, all_delays):
    ''' Given an adjacency matrix adj and a list of integer delays all_delays, 
    returns the list of adjaceny matrices all_adj, where all_adj[i] consists of 
    the edges of adj that correspond with the i-th delay in all_delays (note: 
    the sum of all_adj[i] over all i is adj).
    
    Parameters
    ----------
    adj : 2D numpy array
        An adjacency matrix.
    
    Returns
    -------
    As : list
        A list of adjacency matrices.
    '''

    num_edges = int(np.sum(adj))
    # Randomly assign each edge in adj a delay index
    rand_assignments = np.random.randint(len(all_delays), size=num_edges)
    As = []
    for k in range(len(all_delays)): # k = delay index
        adj_k = np.copy(adj)
        assignment_ind = 0 # track which edge assignment index we are at
        for i in range(adj_k.shape[0]):
            for j in range(adj_k.shape[1]):
                # If the original adj matrix has an edge here
                if (adj_k[i, j] == 1):
                    # And if this edge doesn't correspond with the kth delay, 
                    # exclude the edge
                    if (rand_assignments[assignment_ind] != k):
                        adj_k[i, j] = 0
                    # We have encountered another edge. Increase
                    # the assignment_ind.
                    assignment_ind += 1
        As.append(adj_k)
    return As


################################################################################
def feedforward_multiplicities(arr, add_wedges=True, add_paths=True):
    '''Compute the feedforward multiplicities of edges an non-edges in a 
    network. The feedforward multiplicity of an edge is the number of 
    feedforward loops (i.e., directed 3-cycles with one edge directionality 
    flipped) that include the edge. The feedforward multiplicity of a non-edge 
    is the number of feedforward loops that would exist if this non-edge were
    an edge.

    Parameters
    ----------
    arr : 2D numpy array
        An adjacency matrix.

    Returns
    -------
    ffm_e : 1D array
        An array of feedforward multiplicities of edges.

    ffm_ne : 1D array
        An array of feedforward multiplicities of non-edges.
    '''

    # compute feedfoward multiplicities
    ffm = np.zeros(arr.shape)
    if add_wedges:
        ffm += np.matmul(arr, arr.T)
    if add_paths:
        ffm += np.matmul(arr, arr)

    # feedforward multiplicities of edges
    ffm_e = np.ravel(ffm[arr>0])

    # feedforward multiplicites of non-edges
    ffm_ne = np.ravel(ffm[arr==0])
    
    return ffm_e, ffm_ne


################################################################################    
def feedback_multiplicities(arr):
    '''Compute the feedback multiplicities of edges an non-edges in a network.
    The feedback multiplicity of an edge is the number of feedback loops (i.e.,
    directed 3-cycles) that include the edge. The feedback multiplicity of a 
    non-edge is the number of feedback loops that would exist if this non-edge
    were an edge.

    Parameters
    ----------
    arr : 2D numpy array
        An adjacency matrix.

    Returns
    -------
    fbm_e : 1D array
        An array of feedback multiplicities of edges.

    fbm_ne : 1D array
        An array of feedback multiplicities of non-edges.
    '''

    # compute feedback multiplicities
    fbm = np.zeros(arr.shape)
    fbm += np.matmul(arr.T, arr.T)

    # feedback multiplicities of edges
    fbm_e = np.ravel(fbm[arr>0])    

    # feedback multiplicities of non-edges
    fbm_ne = np.ravel(fbm[arr==0])

    return fbm, fbm_ne
