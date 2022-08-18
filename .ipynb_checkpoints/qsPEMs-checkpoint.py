###############################################################################
### This abbreviated library includes all functions that are needed to compute
### the pairwise edge measures LCCF and LCRC.
###
### Copyright (c) Alice C. Schwarze, Sara M. Ichinaga, Bingni W. Brunton (2022)
###############################################################################
import numpy as np

def inf_via_LCRC(X, m, max_lag=1):
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

    Returns
    -------
    A : 2D numpy array
        An unweighted (weighted if `m==None`) adjacency matrix of the inferred
        network.
    '''    

    # infer parameters of dynamical system from X
    dt_tau = infer_dttau(X)
  
    # initialize calculator for correction terms
    correction = lambda dtt: 1.0-dtt

    # compute lagged correlation matrices
    Cs = [crosscorrelation(X, shift=i) for i in range(max_lag+1)]

    # compute edge scores for each lag    
    scores = [(Cs[1+i] - correction(dt_tau)*Cs[i]) for i in range(max_lag)]

    # final score for each edge is maximum over all considered lags
    A = np.max(scores, axis=0)   

    # threshold edge scores to get binary adjacency matrix
    if m is not None:
        np.fill_diagonal(A, -np.inf)
        A = threshold_values(A, m)

    return A


###############################################################################
def inf_via_LCCF(X, m, max_lag=1):
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
        
    Returns
    -------
    A : 2D numpy array
        An unweighted (weighted if `m==None`) adjacency matrix of the inferred 
        network.
    '''    

    # set default values for unspecified input arguments
    if max_lag is None:
        max_lag=1
        
    # infer parameters of dynamical system from X
    dt_tau = infer_dttau(X)
    
    # initialize calculator for correction terms
    correction = lambda dtt: 2*(1-dtt)/(dtt**2-2*dtt+2)

    # compute lagged correlation matrices
    Cs = [crosscorrelation(X, shift=i) for i in range(max_lag+1)]
    
    # compute edge scores for each lag
    scores = [(Cs[1+i] - correction(dt_tau)*Cs[i]) for i in range(max_lag)]
    
    # final score for each edge is maximum over all considered lags
    A = np.max(scores, axis=0)   

    # threshold edge scores to get binary adjacency matrix
    if m is not None:
        np.fill_diagonal(A, -np.inf)
        A = threshold_values(A, m)

    return A


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


###############################################################################
def infer_dttau(X):
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
    theta_inf = np.median(np.diag((I - M)))

    return theta_inf


###############################################################################
def threshold_values(A, m):
    ''' Given a matrix `A`, converts the `m` highest values of `A` to 1 and 
    the remaining values to 0.
    
    Parameters
    ----------
    A : 2D numpy array
        A real-valued square array.
        
    m : integer
        Number of desired non-zero elements.
            
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
    one_inds = np.argsort(At)[N - m:]
    zero_inds = np.setdiff1d(all_inds, one_inds)
    At[one_inds] = 1
    At[zero_inds] = 0
    At = At.reshape(A.shape)
    
    return At