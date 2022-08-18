###############################################################################
import random
import numpy as np
import networkx as nx

from matrix_methods import get_spectral_radius

###############################################################################
def get_adjacency(G):
    '''Given a graph G, returns the adjacency matrix associated with G
    (adj[i, j] gives connection from node i to node j).
    
    Parameters
    ----------
    G : networkX Graph or networkX DiGraph
        The graph for which an adjacency matrix should be returned.
    
    Returns
    -------
    A : 2D numpy array (dtype=float)
        The adjacency matrix of `G`.
    '''

    adj = nx.adjacency_matrix(G).todense()
    A = np.array(adj, dtype='float')
    # transpose A because networkx define a_{ij} as an edge from i to j; we
    # use dynamical systems convention that defines a_{ij} as edge from j to i
    A = A.T 
    
    return A


############################################################################### 
def make_adjacency(n, d, reciprocity=None, model='ER', force_recurrent=True,
    force_dag=False, normalize=False, rewiring_probability=0.1):
    '''Make an adjacency matrix for a realization of a specified
    graph model.
    
    Parameters
    ----------
    n : int
        Number of nodes.
        
    d : float
        Network density.
        
    reciprocity : float (default=d)
        Parameter in [0,1] that tunes how many edges (u,v) have a corresponding
        reverse edge (v,u) in the network.
        
    model : string (default='ER')
        2-character string indicating a (random-)graph model ('ER': Erdos-Renyi 
        random graph, 'BA': Barabasi-Albert graph, 'RL': regular lattice graph, 
        'RR': regular ring graph, 'SW': Watts-Strogatz small world graph).
        
    force_recurrent : bool (default=True)
        If True, force adjacency matrix to have a non-zero spectral radius.
        
    normalize : bool (default=False)
        If True, normalize elements of the adjacency matrix such that the 
        spectral radius is 1 for recurrent networks.
        
    rewiring_probability : float (default=0.1)
        If `model=='SW'`, edges in the ring will be rewired according to the 
        rewiring algorithm for the Watts-Strogatz model with this probability.
        
    Returns
    -------
    A : 2D numpy array
        An adjacency matrix of a realization of a graph model.
    '''

    if reciprocity is None:
        reciprocity = d
    # not all combinations of density and reciprocity are possible
    # if an impossible one is given, set reciprocity to a lower value
    reciprocity = max([reciprocity, 2*d-1])
    # compute density that will lead to the correct effective density after 
    # making some edges unidirectional
    d0 = min([1, d/((reciprocity)+(1-reciprocity)/2.0)]) 
    #TODO: Do I need to make any correction to the line above?
    
    if model == 'ER':
        def network_model():
            G = nx.gnm_random_graph(n, int(d0*(n-1)*n/2), directed=False)   
            return G
    elif model == 'BA':
        def network_model():
            G = fixed_density_ba_graph(n, d0, directed=False)
            return G
    elif model == 'RL':
        def network_model():
            G = lattice_with_fixed_density(n, d0, directed=False)
            return G
    elif model == 'RR':
        def network_model():
            G = ring_with_fixed_density(n, d0, p=0, directed=False)
            return G
    elif model == 'SW':
        def network_model():
            G = ring_with_fixed_density(n, d0, p=rewiring_probability, 
                directed=False)
            return G
    else:
        print("Network type "+model+" is not in ['ER','BA','RL','RR','SW'].")
        return None
    
    # force directed network to not be a DAG
    if force_recurrent:
        eig_max = 0.0
        while eig_max == 0.0:
            G = network_model()
            A = get_adjacency(G)
            #plt.imshow(A)  
            A = adj_to_directed_adj(A, reciprocity)
            eig_max = get_spectral_radius(A)
    else:
        if force_dag:
            eig_max = 1.0
            while eig_max != 0.0:
                G = network_model()
                A = get_adjacency(G)
                A = adj_to_directed_adj(A, reciprocity)
                eig_max = get_spectral_radius(A)
        else:
            G = network_model()
            A = get_adjacency(G)
            A = adj_to_directed_adj(A, reciprocity)
            
    # normalize adjacency matrix
    if normalize:
        eig_max = get_spectral_radius(get_adjacency(G))
        if eig_max > 0:
            A = A/eig_max
            
    return A


###############################################################################
def adj_to_directed_adj(A, r):
    '''Turn the adjacency matrix of an undirected graph into the adjacency 
    matrix of a directed graph by removing one edge of each pair of reciprocal 
    directed edges probability `1-r`.
    
    Parameters
    ----------
    A : 2D numpy array
        an adjacency matrix of an undirected graph.
        
    r : float
        Desired edge reciprocity.
        
    Returns
    -------
    A_dir : 2D numpy array
        An adjacency matrix of a directed graph.
    '''

    # select edges to be made unidirectional:
    mask = np.random.choice([0,1], p=[r, 1-r], size=A.shape)
    #A2 = mask*np.triu(A) # adjacency matrix with selected edges
    selected_edges = np.nonzero(mask*np.triu(A, k=1))

    # for each edge that we are going to make unidirectional, select a 
    # directionality
    directions = np.random.choice([0,1], size=len(selected_edges[0]))

    # make a copy of the adjacency matrix
    A_dir = np.copy(A)

    # set directions of unidirectional edges in the copied adjacency matrix
    A_dir[selected_edges] = directions
    A_dir[selected_edges[1],selected_edges[0]] = 1-directions
    
    return A_dir


###############################################################################
def lattice_with_fixed_density(n, d, directed=False):
    '''Create a 2D square lattice graph with a fixed network density `d`. Nodes 
    are arranged on a square lattice. If there are more lattice positions than
    nodes, the last lattice positions remain unoccupied. All nodes are 
    connected to all nodes that are adjacent on the lattice. Then, the function 
    adds edges between node pairs with a lattice distance of 2, 3, and so on 
    until the desired network density is reached.
    
    Parameters
    ----------
    n : int
        Number of nodes.
        
    d : float
        Network density.
        
    directed : bool (default=False)
        If True, return a directed graph. This option is not implemented.
        
    Returns
    -------
    G : networkX Graph
        A 2D lattice graph with additional random edges.
    '''
    
    if directed:
        print('NotImplementedError in lattice_with_fixed_density.')
        return None
    
    # make graph
    dim = int(np.ceil(np.sqrt(n)))
    G = nx.grid_2d_graph(dim,dim)
    G.remove_nodes_from(list(G.nodes())[n:])
    G = nx.from_numpy_array(nx.to_numpy_array(G))
    
    # add edges of increasing length to create specified density
    extra_edges = d*n*(n-1)//2 - G.number_of_edges()
    if extra_edges >= 0:
        k = 1
        A = get_adjacency(G)
        powers = [A+np.eye(len(A))] # add eye to avoid self-edges
        while True:
            k += 1
            # add A**k to list of matrix powers
            powers += [np.sign(np.linalg.matrix_power(powers[0],k))]
            # positive elements of B correspond to new edges of length k
            B = powers[-1] - np.sum(powers[:-1], axis=0)
            B[B<=0] = 0
            B[B>0] = 1
            extra_edges -= np.sum(B)
            if extra_edges >= 0:
                # add all edges of length k
                G.add_edges_from(list(np.transpose(np.nonzero(B))))
            else: 
                # add some edges of length k    
                G.add_edges_from(random.sample(list(np.transpose(
                    np.nonzero(B))), int(extra_edges + np.sum(B))))
                break
    else:
        G.remove_edges_from(random.sample(list(G.edges()), -int(extra_edges)))
    return G

###############################################################################
def ring_with_fixed_density(n, d, p=0, directed=False):
    '''Create a ring graph with a fixed network density d. Nodes 
    are arranged on a circle. All nodes are connected to their first nearest
    neighbors, than second nearest neighbors, and so on ... until the desired 
    network density is reached.
    
    Parameters
    ----------
    n : int
        Number of nodes.
        
    d : float
        Network density.
        
    p : float (default=0)
        Probability for edges in the ring to be rewired according to the 
        rewiring algorithm for the Watts-Strogatz model with this probability.
        
    directed : bool (default=False)
        If True, return a directed graph. This option is not implemented.
        
    Returns
    -------
    G : networkX Graph
        A 2D ring graph with some rewired edges.
    '''
    if directed:
        print('NotImplementedError in ring_with_fixed_density.')
        return None        
    
    m = int((d*(n-1)*n)//2) # number of edges
    l0 = int(m//n) # number of nearest neighbors that need to connected
    if m % n:
        l1 = int(l0 + 1)
    else:
        l1 = l0
    #print(l0, l1)
    
    # make a ring graph with too few edges
    G = nx.watts_strogatz_graph(int(n), 2*l0, p=0)
    #plt.subplot(2,2,1)
    #plt.imshow(nx.adjacency_matrix(G).todense())
    # make a ring graph with too many edges
    H = nx.watts_strogatz_graph(int(n), 2*l1, p=0)
    #plt.subplot(2,2,2)
    #plt.imshow(nx.adjacency_matrix(H).todense())
    
    # add edges from H to G
    edges_from_H = [e for e in H.edges() if e not in G.edges()]
    extra_edges = m - G.number_of_edges()
    #print('extra_edges', extra_edges)
    G.add_edges_from(random.sample(edges_from_H, extra_edges))
    #plt.subplot(2,2,3)
    #plt.imshow(nx.adjacency_matrix(G).todense())
    
    if p > 0: 
        # rewire edges
        #print('do rewire')
        G = random_rewire(G, p)
        
    #plt.subplot(2,2,4)
    #plt.imshow(nx.adjacency_matrix(G).todense())

    return G


###############################################################################
def fixed_density_ba_graph(n, d, directed=False):
    '''Create a realization of a Barabasi-Albert graph with a fixed network
    density `d`. The graph is generated with `m=d*(n choose 2)/2` new edges per
    new node. If `m` is not an integer, it is replaced by the largest integer 
    smaller than `m`. The resulting Barabasi-Albert graph has a lower density 
    than `d`. To achieve the desired density, this function adds random edges 
    to the Barabasi-Albert graph until its density is `d`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
        
    d : float
        Network density.
        
    directed : bool (default=False)
        If True, return a directed graph. This option is not implemented.
        
    Returns
    -------
    G : networkX Graph
        A Barabasi-Albert graph with additional random edges.
    '''
    
    if directed:
        print('NotImplementedError in fixed_density_ba_graph.')
        return None
    
    num_edges = d*n*(n-1)/2
    m = int(num_edges/n)
    #print('d', d, 'n', n, 'num_edges', num_edges, 'm', m)
    if m > 0:
        G = nx.barabasi_albert_graph(n, m)
    else:
        G = nx.empty_graph(n=n)
    missing_edges = num_edges - G.number_of_edges()
    G = add_random_edges(G, missing_edges)
    
    return G


###############################################################################
def fixed_density_ct_graph(n, d, rho=0, directed=False, **kwargs):
    '''Create a realization of a Cantwell threshold graph with a fixed network
    density `d`. (There seems to be an unresolved bug in this function!)
    
    Parameters
    ----------
    n : int
        Number of nodes.
        
    d : float
        Network density.
        
    directed : bool (default=False)
        If True, return a directed graph. This option is not implemented.
        
    Returns
    -------
    G : networkX Graph
        A Cantwell graph with additional random edges.
    '''
    
    if directed:
        print('NotImplementedError in fixed_density_ct_graph.')
        return None
    
    # find edge weights
    z = np.random.normal(size=n)
    Y = np.random.normal(size=(n,n))
    Z = z[:, None]+z
    EW = np.sqrt(1-2*rho)*Y+np.sqrt(rho)*Z
    np.fill_diagonal(EW, -np.inf)
    
    # find threshold
    num_edges = int(n*(n-1)*d/2) # number of undirected edges
    EW_vals = np.ravel(EW)
    if num_edges==0:
        thr = np.inf
    elif num_edges==int(n*(n-1)/2):
        thr = -1E16
    else:
        thr = np.sort(EW_vals)[len(EW_vals)-2*num_edges]
    EW[EW>=thr] = 1
    EW[EW<thr] = 0
    G = nx.from_numpy_matrix(EW)
    
    return G


###############################################################################
def random_rewire(G, p):
    '''Rewire edges in a graph `G` with probability `p` using the same 
    algorithm as networkx uses for the Watts-Strogatz random-graph model.
    
    Parameters
    ----------
    G : networkX Graph
        A graph to be rewired.

    p : float
        Rewiring probability.
        
    Returns
    -------
    G : a networkX graph
        A graph with rewired edges.
    '''
    
    edges = list(G.edges())
    for e in edges:
        if np.random.uniform()<p:
            # rewire
            new_node = random.choice(
                [n for n in G.nodes if n not in list(G[e[0]])])
            G.add_edge(e[0],new_node)
            G.remove_edge(e[0],e[1])
    return G


###############################################################################
def add_random_edges(G, m, selfedges=False, directed=False):
    '''Add edges uniformly at random to a networkX Graph.
    
    Parameters
    ----------
    G : networkX Graph or networkX DiGraph
        The graph to which edges should be added.
        
    m : float
        Number of edges to be added.
        
    selfedges : bool (default=False)
        If True, allow added edges to be self-edges.
        
    directed : bool (default=False)
        If True, add directed edges to a DiGraph.
        
    Returns
    -------
    G : networkX Graph or networkX DiGraph
        A copy of the input graph with `m` additional edges.
    '''
    n = G.number_of_nodes()
    if directed:
        if selfedges:
            non_edges = [(i,j) for i in range(n) for j in range(n) 
                         if (i,j) not in G.edges()]
        else:
            non_edges = [(i,j) for i in range(n) for j in range(n) 
                         if (i!=j and (i,j) not in G.edges())]
    else:
        if selfedges:
            non_edges = [(i,j) for i in range(n) for j in range(i+1) 
                         if (i,j) not in G.edges()]
        else:
            non_edges = [(i,j) for i in range(n) for j in range(i) 
                         if (i,j) not in G.edges()]
            
    new_edges = random.sample(non_edges, int(m))
    G.add_edges_from(new_edges)
    
    return G
