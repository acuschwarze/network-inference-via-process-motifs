################################################################################
import time
import numpy as np

from simulation import *
from graph_models import *
from inference_methods import *
#from matrix_methods import * # do I need this?
#from plot_utils import * # do I need this?

################################################################################
def siminf(A=None, network_model='ER', n=10, density=0.5, reciprocity=None,
    force_recurrent=True, force_dag=False, normalize_adjacency=True,          
    ctime=1.0, epsilon=0.9, sigma=0.2, eta=0.0, max_lag=1, x0=None, 
    initial_displacement=None, dt=0.5, T=1000,  perturbation_frequency=None, 
    perturbation_magnitude=None, inference_method='LCCF', detrend=False, 
    max_lag_inf=None, verbose=False, **kwargs):
    '''Generate a network, simulate dynamics on the network, infer network from 
    the observed dynamics, and compute accuracy.
    
    Parameters
    ----------
    A : 2D numpy array (default=None)
        Adjacency matrix of a possibly weighted and directed network. If none 
        is given, generate a realization of a (random-)graph model using the
        values of `network_model`, `n`, `density`, `reciprocity`,
        `force_recurrent`, and `force_dag`.
        
    network_model : string (default='ER')
        2-character string indicating a (random-)graph model ('ER': Erdos-Renyi 
        random graph, 'BA': Barabasi-Albert graph, 'RL': regular lattice graph, 
        'RR': regular ring graph, 'SW': Watts-Strogatz small world graph).      

    n : int (default=10)
        Number of nodes.
        
    density : float (default=0.5)
        Edge density.
        
    reciprocity : float (default=1-density)
        Parameter in [0,1] that tunes how many edges (u,v) have a corresponding
        reverse edge (v,u) in the network.
        
    force_recurrent : bool (default=True)
        If True, reject samples from graph model until a recurrent graph (i.e.,
        a graph with cycles) is sampled. Takes precedent over `force_dag`.
        
    force_dag : bool (default=False)
        If `force_dag` is True and `force_recurrent` is False, reject samples 
        from graph models until a directed acyclic graph is sampled. 
        
    normalize_adjacency : bool (default=True)
        If True, normalize adjacency matrix by spectral radius if spectral 
        radius is not 0.
        
    ctime : float (default=1.0)
        Characteristic time of the dynamical system.
        
    epsilon : float (default=0.9)
        Coupling parameter of the dynamical system.
        
    sigma : float (default=0.2)
        Noise strength of the dynamical system.

    eta : float (default=0)
        If `eta`>0, use Gaussian white noise with standard deviation eta to 
        model measurement noise in the dynamical system.
        
    max_lag : int (default=1)
        Maximal transmission time on edges during simulation in units of `dt`.
        This value is used in the simulation.
        
    x0 : 1D numpy array (default=None)
        Initial condition for simulation. If none is given, draw from a normal
        distribution with standard deviation given by `initial_displacement`.
        
    initial_displacement : float (default=None)
        Standard deviation of the initial condition if `x0` is None. If 
        `x0` is None and `initial_displacement` is None, set 
        `initial_displacement` to `sigma*np.sqrt(dt/n)`.
        
    dt : float (default=0.5)
        Duration of a simulation time step in the simulation.
        
    T : int (default=1000)
        Number of simulation steps in the simulation.
        
    perturbation_frequency : float (default=None)
        Frequency (measured in units of 1/`dt`) of external perturbations in 
        simulation of the dynamical system.

    perturbation_magnitude : float (default=None)
        Magnitude of perturbations applied during simulation. If none is 
        specified, use same value as initial displacement.

    detrend : bool (default=False) 
        If True, detrend time-series data by replacing x_i(t) by the difference
        x_i(t)-x_i(t-1).
        
    inference_method : string (default='LRC')
        A string indicating an inference method ('base' for uniformly random
        scores, 'OU' or 'OUI' for Ornstein--Uhlenbeck fit, 'GC' for linear
        Granger causality, 'TE' for transfer entropy, 'CM' for convergent
        crossmapping, 'LC' for lagged correlation, 'LRC' for lagged correlation
        with a correction for reverse causation, and 'LCF' for lagged
        correlation with a correction for confounding factors).
    
    max_lag_inf : integer (default=None)
        Assumed maximal transmission time on edges in units of `dt`. This value
        is used in the inference step. If `max_lag_inf` is None, set 
        use value of `max_lag`.
    
    verbose : bool (default=False)
        If True, print status updates while function is executed.
        
    Returns
    -------
    inference_results (dict)
        A dictionary that includes results from the simulation and network 
        inference.
    '''
    
    inference_results = {} 
    # make adjacency matrix
    if A is None:
        A = make_adjacency(n, density, reciprocity=reciprocity, 
            model=network_model, force_recurrent=force_recurrent, 
            force_dag=force_dag)
        
    # save adjacency matrix to results    
    inference_results['adjacency'] = np.copy(A)

    # simulate time series
    if verbose:
        print('run siminf with', 
              {'A': A, 
               'ctime': ctime, 
               'epsilon': epsilon, 
               'sigma': sigma, 
               'eta' : eta,
               'max_lag' : max_lag,
               'x0' : x0, 
               'initial_displacement': initial_displacement,
               'dt': dt, 
               'T': T, 
               'perturbation_frequency' : perturbation_frequency,
               'perturbation_magnitude' : perturbation_magnitude,
               'detrend' : detrend
               })        
    X = sim(A, normalize_adjacency=normalize_adjacency, 
        ctime=ctime, epsilon=epsilon, sigma=sigma, eta=eta, max_lag=max_lag, 
        x0=x0, initial_displacement=initial_displacement, dt=dt, T=T, 
        perturbation_frequency=perturbation_frequency, 
        perturbation_magnitude=perturbation_magnitude, detrend=detrend)
    
    # save time series data to results
    inference_results['sim'] = np.copy(X)

    # infer network and take time
    if max_lag_inf is not None:
        max_lag = max_lag_inf
        
    t0 = time.time()
    Ai = infer(X, np.sum(np.sign(A)), method=inference_method, 
        max_lag=max_lag, **kwargs)
    
    # save inference time to results
    inference_results['time'] = time.time()-t0
    
    # save inferred network to results
    inference_results['inferred_network'] = np.copy(Ai)

    # save measures of inference quality to results
    for m in ['accuracy', 'tpr', 'fpr', 'tnr', 'fnr']:
        inference_results[m] = get_inference_quality(Ai, A, measure_by=m)
        
    return inference_results


################################################################################
def default_siminf_pars():
    '''Return a dictionary with default parameters for the function `siminf`.
    
    Returns
    -------
    pdict : dictionary
        A dictionary with parameters for `siminf`.
    '''
    
    pdict = {}
    pdict['A'] = None
    pdict['network_model'] = 'ER'
    pdict['n'] = 10
    pdict['density'] = 0.5
    pdict['reciprocity'] = None
    pdict['force_recurrent'] = True
    pdict['force_dag'] = False
    pdict['normalize_adjacency'] = True
    pdict['ctime'] = 1.0
    pdict['epsilon'] = 0.9
    pdict['sigma'] = 0.2
    pdict['eta'] = 0.0
    pdict['max_lag'] = 1
    pdict['x0'] = None
    pdict['initial_displacement'] = None
    pdict['dt'] = 0.5
    pdict['T'] = 1000
    pdict['perturbation_frequency'] = None
    pdict['perturbation_magnitude'] = None
    pdict['detrend'] = False
    pdict['inference_method'] = 'LCCF'
    pdict['max_lag_inf'] = None
    
    return pdict


################################################################################        
def parameters2string(pars,remove=[]):
    '''Turn a dictionary of parameter values into a string.
    
    Parameters
    ----------
    pars : dictionary
        A dictionary with strings as keys and values that can be turned into 
        strings.
        
    remove : list (default=[])
        List of keys in pars that should not be included in the string.
    
    Returns
    -------
    s : string
        A string representation of the dictionary.
    '''
    
    s = ''
    sorted_kws = sorted(pars.keys())
    for kw in sorted_kws:
        if kw not in remove:
            s += kw +str(pars[kw]) + '_'
    s = s[:-1]

    return s 


################################################################################
def is_interactive():
    '''Test if running in python or ipython. From https://stackoverflow.com/ques
    tions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
 
    Returns
    -------
    interactive : bool
        Variable that is True if `is_interactive` is executed in ipython.
    '''
    
    import __main__ as main
    interactive = not hasattr(main, '__file__')

    return interactive

