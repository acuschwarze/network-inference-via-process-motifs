###############################################################################
import sys, os, string, random
import netrd, subprocess
import numpy as np
import scipy.io as sio
from scipy.special import hyp2f1, binom

sys.path.append('../libs/')
from matrix_methods import get_spectral_radius, negate


###############################################################################
def get_accuracy(Ai, A):
    '''Given an inferred adjacency matrix and the true adjacency matrix, 
    computes and returns the accuracy of the inference.
    
    Parameters
    ----------
    A1 : 2D numpy array
        Adjacency matrix of inferred network.
        
    A0 : 2D numpy array
        Adjacency matrix of true network.
        
    Returns
    -------
    acc : float
        Accuracy of inference.
    '''

    acc = get_inference_quality(Ai, A, measure_by='accuracy')

    return acc


###############################################################################
def get_inference_quality(Ai, A, measure_by='accuracy'):
    '''Given an inferred adjacency matrix and the true adjacency matrix, 
    computes and returns the accuracy of the inference.
    
    Parameters
    ----------
    A1 : 2D numpy array
        Adjacency matrix of inferred network.
        
    A0 : 2D numpy array
        Adjacency matrix of true network.

    measure_by : string in [ 'accuracy' | 'tpr' | 'tnr' | 'fpr' | 'fnr' ]
        Quantity to be computed. Choose between accuracy, true-positive rate
        ('tpr'), true-negative rate ('tnr'), false-positive rate ('fpr'),
        false-negative rate ('fnr').
        
    Returns
    -------
    quality : float
        A measure of the inference quality.
    '''
    
    N = A.shape[0]
    I = np.eye(N)
    
    if measure_by=='accuracy':
        max_correct = N ** 2 - N
        correct = (Ai == A)
        
    elif measure_by=='tpr':
        max_correct = np.count_nonzero(A)
        correct = np.multiply(Ai == A, A)
        
    elif measure_by=='tnr':
        max_correct = np.count_nonzero(negate(A)) - N
        correct = np.multiply(Ai == A, negate(A))
        
    elif measure_by=='fpr':
        max_correct = np.count_nonzero(negate(A)) - N
        correct = np.multiply(Ai != A, A)
        
    elif measure_by=='fnr':
        max_correct = np.count_nonzero(A)
        correct = np.multiply(Ai != A, negate(A))
        
    correct_nondiag = np.multiply(correct, negate(I))     
    num_correct = np.count_nonzero(correct_nondiag)
    quality = num_correct/max_correct  
    
    return quality


###############################################################################
def infer(X, m, method='LCCF', **kwargs):
    '''Infer a network with `m` edges from time-series data `X` via a specified
    method.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    m : integer
        Number of edges in inferred network.
        
    method : string (default='LCR')
        A string indicating an inference method ('base' for uniformly random
        scores, 'OU' or 'OUI' for Ornstein--Uhlenbeck fit, 'GC' or 'LGC' for 
        linear Granger causality, 'TE' or 'NTE' for (naive) transfer entropy, 
        'CM' or 'CCM' for convergent crossmapping, 'LC' or 'LCO' for lagged 
        correlation, 'LRC' for lagged correlation with a correction for reverse
        causation, and 'LCF' for lagged correlation with a correction for 
        confounding factors).

    Returns
    -------
    A : 2D numpy array
        An adjacency matrix of the inferred network.

    '''
    
    if method=='base':
        A = inf_baseline(X, m, **kwargs)
    elif method in ['LCRC']: 
        # constant correction factor for reverse causation
        A = inf_via_lrc(X, m, **kwargs)
    elif method in ['LCCF'] : 
        # constant correction factor confounding factors #scc
        A = inf_via_lcf(X, m, **kwargs) 
    elif method in ['LC']:
        A = inf_via_crosscorrelation(X, m, shift=1)
    elif method in ['GC']:
        A = inf_via_granger_causality(X, m, **kwargs)
    elif method in ['TE']:
        A = inf_via_transfer_entropy(X, m, **kwargs)
    elif method in ['CM']:
        A = inf_via_convergent_crossmapping(X, m, **kwargs)
    elif method in ['OUI']:
        A = inf_via_ou_fit(X, m, **kwargs)    
    else:
        print('Error in infer: Unknown inference method "'+str(method)+'".')
        return None

    return A


###############################################################################
def inf_baseline(X, m, **kwargs):
    '''Create an adjacency matrix by placing `m` edges uniformly at random in a
    graph. This function is used to compute baselines for inference accuracy.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    m : integer
        Number of edges in inferred network.
        
    Returns
    -------
    A : 2D numpy array
        An adjacency matrix of the inferred network.
    '''

    n = len(X)
    m = int(m)
    A = np.random.uniform(size=(n,n))
    np.fill_diagonal(A, 0)
    if m==0:
        thr=np.max(A)*2
    else:
        thr = np.sort(np.ravel(A))[-m]
    A[A >= thr] = 1.0
    A[A < thr] = 0.0
    
    return A


###############################################################################
def inf_via_crosscorrelation(X, m, shift=1, apply_thres=True, **kwargs):
    '''Infer a network with `m` edges from time series data `X` using 
    crosscorrelations.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data
        
    m : integer
        Number of edges in inferred network

    shift : integer (default=1)        
        Number of timesteps by which the `X` in the correlation calculation.
        If `shift` is 0, we infer a network from the data's (symmetric) 
        correlation matrix. If `shift` is greater or equal to 1, we infer a 
        network from a lagged correlation matrix.

    apply_thres : bool (default=True)
        If True, return an adjacency matrix with values that are either 0 or 1.
        If False, return score matrix with real-valued elements.
        
    Returns
    -------
    A : 2D numpy array
        An unweighted (if apply_thres is True) or weighted (if apply_thres is 
        False) adjacency matrix of the inferred network.
    '''

    A = crosscorrelation(X, shift=shift)

    if (apply_thres):
        np.fill_diagonal(A, -np.inf)
        A = threshold_values(A, m)

    return A


###############################################################################
def inf_via_granger_causality(X, m, apply_thres=True, max_lag=1, **kwargs):
    '''Infer a network with `m` edges from time series data `X` via (non-
    conditional) linear Granger causality mapping (using implementation in the 
    netrd toolbox).
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    m : integer
        Number of edges in inferred network.
        
    max_lag : int (default=1)
        Maximum lag of signal transmission along an edge.
        
    apply_thres : bool (default=True)
        If True, return an adjacency matrix with values that are either 0 or 1.
        If False, return score matrix with real-valued elements.
        
    Returns
    -------
    A : 2D numpy array
        An unweighted (if apply_thres is True) or weighted (if apply_thres is 
        False) adjacency matrix of the inferred network.
    '''
    # set default values for unspecified input arguments
    if max_lag is None:
        max_lag=1
    
    # compute Granger causality
    recon = netrd.reconstruction.GrangerCausality()
    recon.fit(X, lag=max_lag, cutoffs=[(-1, 1)])
    A = recon.results['weights_matrix']

   # threshold edge scores to get binary adjacency matrix
    if apply_thres:
        np.fill_diagonal(A, -np.inf)
        A = threshold_values(A, m)

    return A


################################################################################
def inf_via_convergent_crossmapping(X, m, apply_thres=True, **kwargs):
    '''Infer a network with `m` edges from time series data `X` using cross-
    convergent mapping (using the implementation in the netrd toolbox).
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    m : integer
        Number of edges in inferred network.
        
    apply_thres : bool (default=True)
        If True, return an adjacency matrix with values that are either 0 or 1.
        If False, return score matrix with real-valued elements.
        
    Returns
    -------
    A : 2D numpy array
        An unweighted (if apply_thres is True) or weighted (if apply_thres is 
        False) adjacency matrix of the inferred network.
    '''

    # compute convergent crossmapping
    recon = netrd.reconstruction.ConvergentCrossMapping()
    try:
        recon.fit(X)
        A = recon.results['pvalues_matrix']
    except:
        # convergent crossmapping can throw an error is data is insufficient
        # in that case return an empty matrix
        A = np.zeros((X.shape[0], X.shape[0]))
    
    # threshold edge scores to get binary adjacency matrix
    if apply_thres:
        np.fill_diagonal(A, np.inf)
        A = threshold_values(A, m, keep_max=False)

    return A


################################################################################
def inf_via_transfer_entropy(X, m, max_lag=1, apply_thres=True, **kwargs):
    '''Infer a network with `m` edges from time series data `X` via a naive
    estimator for transfer entropy (using the implementation in the netrd 
    toolbox).
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    m : integer
        Number of edges in inferred network.

    max_lag : int (default=1)
        Maximum lag of signal transmission along an edge.
                
    apply_thres : bool (default=True)
        If True, return an adjacency matrix with values that are either 0 or 1.
        If False, return score matrix with real-valued elements.
        
    Returns
    -------
    A : 2D numpy array
        An unweighted (if apply_thres is True) or weighted (if apply_thres is 
        False) adjacency matrix of the inferred network.
    '''    

    # set default values for unspecified input arguments
    if max_lag is None:
        max_lag=1
        
    # estimate transfer entropy
    recon = netrd.reconstruction.NaiveTransferEntropy()
    recon.fit(X, delay_max=max_lag, cutoffs=[(-1, 1)]) 
    A = recon.results['weights_matrix']

    # threshold edge scores to get binary adjacency matrix
    if apply_thres:
        np.fill_diagonal(A, -np.inf) 
        A = threshold_values(A, m)

    return A


################################################################################
def inf_via_ou_fit(X, m, apply_thres=True, **kwargs):
    '''Infer a network with m edges from time series data X via Ornstein--
    Uhlenbeck inference (using the implementation in the netrd toolbox).
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    m : integer
        Number of edges in inferred network.
        
    apply_thres : bool (default=True)
        If True, return an adjacency matrix with values that are either 0 or 1.
        If False, return score matrix with real-valued elements.
        
    Returns
    -------
    A : 2D numpy array
        An unweighted (if apply_thres is True) or weighted (if apply_thres is 
        False) adjacency matrix of the inferred network.
    '''    

    # compute OU fit coefficients
    recon = netrd.reconstruction.OUInference()
    recon.fit(X, cutoffs=[(-1, 1)])
    A = recon.results['weights_matrix']

    # threshold edge scores to get binary adjacency matrix
    if apply_thres:
        np.fill_diagonal(A, -np.inf)
        A = threshold_values(A, m)

    return A


################################################################################
def threshold_values(A, m, keep_max=True):
    ''' Given a matrix `A`, converts the `m` highest values (if `keep_max` is 
    True) of `A` to 1 and the remaining values to 0.
    
    Parameters
    ----------
    A : 2D numpy array
        A real-valued square array.
        
    m : integer
        Number of desired non-zero elements.
    
    keep_max : bool (default=True)
        If `keep_max` is True converts the `m` largest values of `A` to 1. If 
        `keep_max` is False converts the `m` smallest values of `A` to 1. 
        
    Returns
    -------
    At : 2D numpy array
        Square array with `m` entries equal to 1 and all other entries equal 
        to 0.
    '''

    m = int(m)
    N = A.size
    At = A.flatten() 
    all_inds = np.arange(N)
    if (keep_max):
        one_inds = np.argsort(At)[N - m:]
    else: # keep_min
        one_inds = np.argsort(At)[:m]
    zero_inds = np.setdiff1d(all_inds, one_inds)
    At[one_inds] = 1
    At[zero_inds] = 0
    At = At.reshape(A.shape)
    
    return At


################################################################################
def crosscorrelation(X, shift=1):
    '''Compute crosscorrelation matrix for time series data X.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    shift : integer (default=1)
        Lag of the correlation matrix. If `shift` is 0, this is equivalent
        to the lag-free correlation matrix.

    Returns
    -------
    C : 2D numpy array
        Lagged sampled correlation matrix for `X`.
    '''

    # set default values for unspecified input arguments
    if shift is None:
        shift=1
        
    if shift < 0:
        # get lagged correlation matrices for a negative lag by transposing
        # a lagged correlation matrix with a positive lag
        C = (crosscorrelation(X, shift=-shift)).T
        return C
    
    # compute lagged correlation matrix
    N = X.shape[0]
    X1 = X[:, :len(X[0])-shift]
    X2 = X[:, shift:]
    C = (np.corrcoef(np.vstack((X1, X2))))[:N, N:]

    return C   


################################################################################
def crosscovariance(X, shift=1):
    '''Compute crosscovariance matrix for time series data X.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    shift : integer (default=1)
        Lag of the covariance matrix. If `shift` is 0, this is equivalent
        to the lag-free covariance matrix.
    
    Returns
    -------
    C : 2D numpy array
        Lagged sample covariance matrix for `X`.
    '''
    
    # set default values for unspecified input arguments
    if shift is None:
        shift=1
        
    if shift < 0:
        # get lagged covariance matrices for a negative lag by transposing
        # a lagged covariance matrix with a positive lag
        C = (crosscovariance(X, shift=-shift)).T
        return C
    
    # compute lagged covariance matrix
    N = X.shape[0]
    X1 = X[:, :len(X[0])-shift]
    X2 = X[:, shift:]
    C = (np.cov(np.vstack((X1, X2))))[:N, N:]

    return C   


################################################################################
def hash_gen(n):
    '''Generate random string of n letters. (Was used to generate unique names
    for temporary files. Probably not in use anymore.)
    
    Parameters
    ----------
    n : integer
        Length of random string.
        
    Returns
    -------
    s : string
        A random string of letters.
    '''

    letters = string.ascii_letters
    s = ''.join(random.choice(letters) for i in range(n))

    return s


###############################################################################
def inf_via_lrc(X, m, max_lag=1, apply_thres=True, correct_order=False, 
    correct_length=False, rescale=False, force_lag=False, fast=True, **kwargs):
    '''Infer a network with `m` edges from time series data `X` using lagged
    correlation with a correction for reverse causation.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    m : integer
        Number of edges in inferred network.

    max_lag : integer (default=1)
        Maximum edge lag that we expect in simulated data. If `max_lag` is 1, 
        we expect that all edges transmit signals within 1 time step.
        
    apply_thres : bool (default=True)
        If True, return an adjacency matrix with values that are either 0 or 1.
        If False, return score matrix with real-valued elements.

    correct_order : bool (default=False)
        Option to include an additional variable correction term for edges
        with transmission times greater than 1 time step. (Included for 
        experimental purposes; not recommended for actual inference runs.)

    correct_length : bool (default=False)
        Option to include an additional variable correction term for length of
        process motifs. (Included for experimental purposes; not recommended
        for actual inference runs.)

    rescale : bool (default=False)
        Option to include an additional correction that depends on parameters
        of the dynamical system. (Included for experimental purposes; not 
        recommended for actual inference runs.)

    force_lag : bool (default=False)
        Option to override value of `max_lag` so that it is always 1. (Included 
        for experimental purposes; not recommended for actual inference runs.)

    fast : bool (default=True)
        If True, compute correction terms using only the function `phi_pqz`. 
        If False, use process-motif contributions.

    Returns
    -------
    A : 2D numpy array
        An unweighted (if apply_thres is True) or weighted (if apply_thres is 
        False) adjacency matrix of the inferred network.
    '''    

    # set default values for unspecified input arguments
    if max_lag is None:
        max_lag=1
        
    # infer parameters of dynamical system from X
    ctime_dt, eps = infer_parameters(X, dt=1)
    dt_tau = 1/ctime_dt

    if force_lag:
        # force max_lag_inf to be 1
        max_lag = 1
    
    # set rescaling function
    if rescale:
        scale = lambda i: (dt_tau*eps)**(i)
    else:
        scale = lambda i: 1.0

    # initialize calculator for correction terms
    if fast:
        correction = lambda dtt, order : lrc_factor_fast(dtt)
    else:
        correction = lambda dtt, order : lrc_factor_slow(dtt, order, 
            epsilon=eps, dt=1.0, n=len(X), correct_order=correct_order, 
            correct_length=correct_length)

    # compute lagged correlation matrices
    Cs = [crosscorrelation(X, shift=i) for i in range(max_lag+1)]

    # compute edge scores for each lag    
    scores = [(Cs[1+i] - correction(dt_tau, 1+i)*Cs[i])/scale(i) 
        for i in range(max_lag)]

    # final score for each edge is maximum over all considered lags
    A = np.max(scores, axis=0)   

    # threshold edge scores to get binary adjacency matrix
    if (apply_thres):
        np.fill_diagonal(A, -np.inf)
        A = threshold_values(A, m)

    return A


###############################################################################
def inf_via_lcf(X, m, max_lag=1, apply_thres=True, correct_order=False, 
    correct_length=False, rescale=False, force_lag=False, fast=True, **kwargs):
    '''Infer a network with `m` edges from time series data `X` using lagged
    correlation with a correction for direct confounding factors.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data
        
    m : integer
        Number of edges in inferred network

    max_lag : integer (default=1)
        Maximum edge lag that we expect in simulated data. If `max_lag` is 1, 
        we expect that all edges transmit signals within 1 time step.
        
    apply_thres : bool (default=True)
        If True, return an adjacency matrix with values that are either 0 or 1.
        If False, return score matrix with real-valued elements.

    correct_order : bool (default=False)
        Option to include an additional variable correction term for edges
        with transmission times greater than 1 time step. (Included for 
        experimental purposes; not recommended for actual inference runs.)

    correct_length : bool (default=False)
        Option to include an additional variable correction term for length of
        process motifs. (Included for experimental purposes; not recommended
        for actual inference runs.)

    rescale : bool (default=False)
        Option to include an additional correction that depends on parameters
        of the dynamical system. (Included for experimental purposes; not 
        recommended for actual inference runs.)

    force_lag : bool (default=False)
        Option to override value of `max_lag` so that it is always 1. (Included
        for experimental purposes; not recommended for actual inference runs.)

    fast : bool (default=True)
        If True, compute correction terms using only the function `phi_pqz`. 
        If False, use process-motif contributions.
        
    Returns
    -------
    A : 2D numpy array
        An unweighted (if apply_thres is True) or weighted (if apply_thres is 
        False) adjacency matrix of the inferred network
    '''    

    # set default values for unspecified input arguments
    if max_lag is None:
        max_lag=1
        
    # infer parameters of dynamical system from X
    ctime_dt, eps = infer_parameters(X, dt=1)
    dt_tau = 1/ctime_dt

    if force_lag:
        # force max_lag_inf to be 1
        max_lag = 1
    
    # set rescaling function
    if rescale:
        scale = lambda i: (dt_tau*eps)**(i)
    else:
        scale = lambda i: 1.0

    # initialize calculator for correction terms
    if fast:
        correction = lambda dtt, order : lcf_factor_fast(dtt)
    else:
        correction = lambda dtt, order : lcf_factor_slow(dtt, order, 
            epsilon=eps, dt=1.0, n=len(X), correct_order=correct_order, 
            correct_length=correct_length)

    # compute lagged correlation matrices
    Cs = [crosscorrelation(X, shift=i) for i in range(max_lag+1)]
    
    # compute edge scores for each lag
    scores = [(Cs[1+i] - correction(dt_tau, i+1)*Cs[i])/scale(i) 
        for i in range(max_lag)]
    
    # final score for each edge is maximum over all considered lags
    A = np.max(scores, axis=0)   

    # threshold edge scores to get binary adjacency matrix
    if (apply_thres):
        np.fill_diagonal(A, -np.inf)
        A = threshold_values(A, m)

    return A


###############################################################################
def infer_parameters(X, dt=1.0):
    '''Infer the parameters tau and epsilon for the SDD model from time-series 
    data `X`.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data.
        
    dt : float (default=1.0)
        Duration of a time step in the time series data.
    
    Returns
    -------
    ctime_inf : float
        Inferred characteristic time.
        
    eps_inf : float
        Inferred coupling parameter.
    '''

    # compute M
    I = np.eye(X.shape[0])
    CC = crosscorrelation(X, 1) # OR crosscovariance(X, time_step)
    C = np.corrcoef(X) # OR np.cov(X)
    M = CC @ np.linalg.pinv(C)

    # compute parameters
    theta_inf = np.median(np.diag((I - M) / (dt)))
    ctime_inf = 1/theta_inf
    eps_inf = get_spectral_radius((M-(1-theta_inf*dt)*I)/(theta_inf*dt))

    return ctime_inf, eps_inf


###############################################################################
def lcf_factor_slow(dtt, lag, epsilon=0.5, dt=1.0, n=10, correct_order=False, 
    correct_length=False):
    '''Compute the correction factor for direct confounding factors using 
    process-motif contributions.
    
    Parameters
    ----------
    dtt : float
        Quotient of inverse sampling rate dt and characteristic time tau of
        the stochastic-difference model.
        
    lag : integer
        Lag of edge to be detected.

    epsilon : float (default=0.5)
        Coupling parameter of the stochastic-difference model.

    dt : float (default=1.0)
        Inverse sampling rate of the stochastic-difference model
        
    n : integer (default=10)
        Number of nodes in the system.
        
    Returns
    -------
    s : float
        Correction factor for direct confounding factors in network inference.
    '''    

    if correct_order:
        o = lag
    else:
        o = 1
    if correct_length:
        L = lag
    else:
        L = 0
        
    # compute contribution of process motif with order=o
    order1 = crosscov_contribution(2+L, 1, order=o, dtt=dtt, epsilon=epsilon,
        dt=dt, n=n)

    # compute contribution of process motif with order=o-1
    order0 = crosscov_contribution(2+L, 1, order=o-1, dtt=dtt, epsilon=epsilon,
        dt=dt, n=n)

    # compute correction term
    s = order1/order0
    
    return s


###############################################################################
def lrc_factor_slow(dtt, lag, epsilon=0.5, dt=1.0, n=10, correct_order=False, 
    correct_length=False):
    '''Compute the correction factor for reverse causation using 
    process-motif contributions.
    
    Parameters
    ----------
    dtt : float
        Quotient of inverse sampling rate dt and characteristic time tau of
        the stochastic-difference model.
        
    lag : integer
        Lag of edge to be detected.

    epsilon : float (default=0.5)
        Coupling parameter of the stochastic-difference model.

    dt : float (default=1.0)
        Inverse sampling rate of the stochastic-difference model

    n : integer (default=10)
        Number of nodes in the system.
        
    Returns
    -------
    s : float
        Correction factor for reverse causation in network inference.
    '''    
        
    if correct_order:
        o = lag
    else:
        o = 1
    if correct_length:
        L = lag
    else:
        L = 0

    # compute contribution of process motif with order=o
    order1 = crosscov_contribution(1+L, 1, order=o, dtt=dtt, epsilon=epsilon,
        dt=dt, n=n)

    # compute contribution of process motif with order=o-1
    order0 = crosscov_contribution(1+L, 1, order=o-1, dtt=dtt, epsilon=epsilon,
        dt=dt, n=n)
    
    # compute correction factor
    s = order1/order0

    return s


###############################################################################
def lcf_factor_fast(dtt):
    '''Compute the correction factor for direct confounding factors using 
    process-motif contributions.
    
    Parameters
    ----------
    dtt : float
        Quotient of inverse sampling rate dt and characteristic time tau of
        the stochastic-difference model.
                
    Returns
    -------
    s : float
        Correction factor for direct confounding factors in network inference.
    '''    

    # set order and length
    #o = 1 # order = lag
    #L = 0 # L = lag

    # compute contribution of process motif with order=o
    #order1 = crosscov_contribution(2+L, 1, order=o, dtt=dtt)

    # compute contribution of process motif with order=o-1
    #order0 = crosscov_contribution(2+L, 1, order=o-1)

    # compute correction term
    #s = order1/order0
    
    s = 2*(1-dtt)/(dtt**2-2*dtt+2)

    return s


###############################################################################
def lrc_factor_fast(dtt):
    '''Compute the correction factor for reverse causation using 
    process-motif contributions.
    
    Parameters
    ----------
    dtt : float
        Quotient of inverse sampling rate dt and characteristic time tau of
        the stochastic-difference model.
        
    Returns
    -------
    s : float
        Correction factor for reverse causation in network inference.
    '''    
        
    # set order and length
    #o = 1 # order = lag
    #L = 0 # L = lag

    # compute contribution of process motif with order=o
    #order1 = crosscov_contribution(1+L, 1, order=o, dtt=dtt)

    # compute contribution of process motif with order=o-1
    #order0 = crosscov_contribution(1+L, 1, order=o-1, dtt=dtt)
    
    # compute correction factor
    #s = order1/order0

    s = 1-dtt

    return s


###############################################################################
def cov_contribution(L, l, dtt=1.0, order=1, epsilon=0.5, sigma=0.2, 
    dt=1.0, n=10):
    '''Compute contribution of a process motif to covariance in the
    stochastic-difference model.

    Parameters
    ----------
    L : integer
        Total length of the process motif.
        
    l : integer
        Length of the left walk in the process motif.
        
    order : integer (default=1)
        Select the order of the crosscovariance matrix. (If 0, return process-
        motif contributions for covariance matrix. If 1, return process-motif
        contributions for 1-crosscovariance, and so on.)

    dtt : float (default=1.0)
        Quotient of inverse sampling rate dt and characteristic time tau of
        the stochastic-difference model.
        
    epsilon : float (default=0.5)
        Coupling parameter of the stochastic-difference model.

    sigma : float (default=0.2)
        Instrinsic noise strength of the stochastic-difference model.

    dt : float (default=1.0)
        Inverse sampling rate of the stochastic-difference model.

    n : integer (default=10)
        Number of nodes in the system.
        
    Returns
    -------
    c : float
        A process-motif contribution for covariance in the stochastic-
        difference model.
    '''

    if L<0 or l>L:
        return 0

    # define shorthands
    mll = np.max([L-l,l])   
    rml = np.abs(L-2*l)
    tau = dt/dtt

    # compute contribution of process motif
    c = (tau*sigma**2/n*((dtt*epsilon)**L)*(1-dtt)**rml
        *binom(mll, rml)*hyp2f1(mll+1,mll+1,rml+1,(1-dtt)**2))

    return c


###############################################################################
def crosscov_contribution(L, l, dtt=1.0, order=1, epsilon=0.5, sigma=0.2, 
    dt=1.0, n=10):
    '''Compute contribution of process motif to k-crosscovariance in the 
    stochastic-difference model.

    Parameters
    ----------
    L : integer
        Total length of the process motif.
        
    l : integer
        Length of the left walk in the process motif.
        
    order : integer (default=1)
        Select the order of the crosscovariance matrix. (If 0, return process-
        motif contributions for covariance matrix. If 1, return process-motif
        contributions for 1-crosscovariance, and so on.)

    dtt : float (default=1.0)
        Quotient of inverse sampling rate dt and characteristic time tau of
        the stochastic-difference model.
        
    epsilon : float (default=0.5)
        Coupling parameter of the stochastic-difference model.

    sigma : float (default=0.2)
        Instrinsic noise strength of the stochastic-difference model.

    dt : float (default=1.0)
        Inverse sampling rate of the stochastic-difference model.

    n : integer (default=10)
        Number of nodes in the system.
                
    Returns
    -------
    c : float
        A process-motif contribution for crosscovariance in the stochastic-
        difference model.
    '''

    if order<0:
        order = -order
        l = L-l
        
    if order==0:
        # order-0 crosscov contributions are cov contributions
        c = cov_contribution(L, l, dtt=dtt, epsilon=epsilon, sigma=sigma, 
            dt=dt, n=n)
    else:
        # compute contribution of process motif w/ length L and order=order-1
        cL = crosscov_contribution(L, l, order=order-1, 
            dtt=dtt, epsilon=epsilon, sigma=sigma, dt=dt, n=n)

        # compute contribution of process motif w/ length L-1 and order=order-1
        cL1 = crosscov_contribution(L-1, l-1, order=order-1, 
            dtt=dtt, epsilon=epsilon, sigma=sigma, dt=dt, n=n)

        # compute crosscov contribution
        c = (1-dtt)*cL + epsilon*dtt*cL1

    return c
