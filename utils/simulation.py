import numpy as np

from matrix_methods import assign_delays

################################################################################
def sim(adj, normalize_adjacency=True, ctime=1.0, epsilon=0.9, sigma=0.2, eta=0, 
    max_lag=1, x0=None, initial_displacement=None, dt=0.5, T=1000, 
    perturbation_frequency=None, perturbation_magnitude=None, detrend=False):
    '''Simulate a vector-valued Markovian or non-Markovian linear stochastic 
    difference equation via Euler-Marayuma.
    
    Parameters
    ----------
    adj : 2D numpy array
        Adjacency matrix of a possibly weighted and directed network.
        
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
        
    Returns
    -------
    X : 2D numpy array
        Time series data of network dynamics.
    '''    
    if detrend:
        T=T+1
        
    if max_lag <= 1:
        X = sim_no_lag(adj, normalize_adjacency=normalize_adjacency, 
            ctime=ctime, epsilon=epsilon, sigma=sigma, eta=eta, 
            x0=x0, initial_displacement=initial_displacement, dt=dt, T=T, 
            perturbation_frequency=perturbation_frequency,
            perturbation_magnitude=perturbation_magnitude)
    else:
        X = sim_lag(adj, normalize_adjacency=normalize_adjacency, 
            ctime=ctime, epsilon=epsilon, sigma=sigma, eta=eta, 
            max_lag=max_lag, x0=x0, initial_displacement=initial_displacement, 
            dt=dt, T=T, perturbation_frequency=perturbation_frequency,
            perturbation_magnitude=perturbation_magnitude)
        
    if detrend:
        X = X[:,1:]-X[:,:-1]
        
    return X
        
    
################################################################################    
def sim_no_lag(adj, normalize_adjacency=True, ctime=1.0, epsilon=0.9, sigma=0.2, 
    eta=0, x0=None, initial_displacement=None, dt=0.5, T=1000, 
    perturbation_frequency=None, perturbation_magnitude=None):
    '''Simulate a vector-valued Markovian linear stochastic difference equation 
    via Euler-Marayuma. For small `dt`, this is a numeric approximation of an
    Ornstein--Uhlenbeck process. For `dt=1`, this is equivalent to simulating 
    an order-1 vector-autoregressive model.
    
    Parameters
    ----------
    adj : 2D numpy array
        Adjacency matrix of a possibly weighted and directed network.
        
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
        
    Returns
    -------
    X : 2D numpy array
        Time series data of network dynamics.
    '''
    # Get number of nodes in graph 
    num_nodes = adj.shape[0]
 
    # set parameters
    noise_scale = _set_sim_noise(sigma, dt, num_nodes)
    initial_displacement = _set_sim_initial_displacement(initial_displacement, 
        noise_scale)
    x0 = _set_sim_x0(x0, initial_displacement, num_nodes)
    perturbation_magnitude = _set_sim_perturbation_magnitude(
        perturbation_magnitude, initial_displacement)

    # create propagator matrix
    I = np.eye(num_nodes)
    if normalize_adjacency: 
        eig_max = np.max(np.abs(np.linalg.eig(adj)[0]))
        if eig_max != 0:
            A = ((epsilon / eig_max) * adj.T) - I
        else:
            # do not normalize nilpotent matrices
            A = epsilon * adj.T - I
    else:
        A = epsilon * adj.T - I

    # Construct time series data for each node 
    X = np.empty((num_nodes, T))
    X[:, 0] = x0
    for i in range(1, T):
        dx = (dt/ctime) * (A @ X[:, i-1])
        noise = noise_scale * np.random.randn(num_nodes)
        X[:, i] = X[:, i-1] + dx + noise

        if (not perturbation_frequency is None):
            # perturb the system 
            if (i % int(1 / perturbation_frequency) == 0):
                X[:, i] += np.random.normal(0, perturbation_magnitude, 
                    size=(num_nodes))
        
    if eta > 0:
        X += np.random.normal(size=X.shape, scale=eta*noise_scale)
        
    return X
 
    
################################################################################    
def sim_lag(adj, normalize_adjacency=True, ctime=1.0, epsilon=0.9, sigma=0.2, 
    eta=0, max_lag=2, x0=None, initial_displacement=None, dt=0.5, T=1000, 
    perturbation_frequency=None, perturbation_magnitude=None):
    '''Simulate a non-Markovianvector-valued linear stochastic difference 
    equation via Euler-Marayuma. For `dt=1`, this is equivalent to simulating 
    vector-autoregressive model.
    
    Parameters
    ----------
    adj : 2D numpy array
        Adjacency matrix of a possibly weighted and directed network.
        
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
        
    max_lag : int (default=2)
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
        
    Returns
    -------
    X : 2D numpy array
        Time series data of network dynamics.
    '''
    # Get number of nodes in graph 
    num_nodes = adj.shape[0]

    # set parameters
    noise_scale = _set_sim_noise(sigma, dt, num_nodes)
    initial_displacement = _set_sim_initial_displacement(initial_displacement, 
        noise_scale)
    x0 = _set_sim_x0(x0, initial_displacement, num_nodes)
    perturbation_magnitude = _set_sim_perturbation_magnitude(
        perturbation_magnitude, initial_displacement)

    # create propagator matrices
    A = np.sign(adj) # do this to reverse normalization

    if normalize_adjacency:
        eig_max = np.max(np.abs(np.linalg.eig(A)[0]))
        if eig_max == 0:
            # do not normalize nilpotent matrices
            eig_max = 1
    else:
        eig_max = 1

    all_lags = np.arange(1, max_lag+1)
    all_adj = assign_delays(A, all_lags)
    all_A = [(epsilon / eig_max) * (all_adj[k]).T for k in range(max_lag)]

    # Construct time series data
    X = np.empty((num_nodes, T))
    X[:, 0] = x0
    for i in range(1, T):
        dx = 0.0
        for k, lag in enumerate(all_lags):
            if (i - lag >= 0):
                dx += all_A[k] @ X[:, i-lag]
        dx -= X[:, i-1]
        dx *= (dt/ctime)
        noise = noise_scale * np.random.randn(num_nodes)
        X[:, i] = X[:, i-1] + dx + noise
        
        # perturb the system 
        if (not perturbation_frequency is None):
            if (i % int(1 / perturbation_frequency) == 0):
                X[:, i] += np.random.normal(0, perturbation_magnitude, 
                    size=(num_nodes))
        
    if eta > 0:
        X += np.random.normal(size=X.shape, scale=eta*noise_scale)

    return X


################################################################################
def _set_sim_noise(sigma, dt, n):
    '''Subfunction of `sim_no_lag` and `sim_lag` that sets a scaling factor
    for the noise amplitude.

    Parameters
    ----------

    sigma : float 
        Noise strength of the dynamical system.
        
    dt : float 
        Duration of a simulation time step in the simulation.

    n : integer
         Number of nodes in the network.

    Returns
    -------
    s : float
        Scaling factor for the noise amplitude.
    '''      
    s = sigma * np.sqrt(dt/n)

    return s


################################################################################
def _set_sim_initial_displacement(initial_displacement, noise_scale):
    '''Subfunction of `sim_no_lag` and `sim_lag` that sets the value of 
    the `initial_displacement` if none is specified.

    Parameters
    ----------

    initial_displacement : float or None
        Standard deviation of the initial condition if `x0` is None. 

    noise_scale : float
        Default value for `initial_displacement`. Usually set via
         `_set_sim_noise`.

    Returns
    -------
    initial_displacement : float
        Standard deviation of the initial condition if `x0` is None. 
    '''  
    
    if initial_displacement is None:
        initial_displacement = noise_scale

    return initial_displacement


################################################################################
def _set_sim_x0(x0, initial_displacement, dim):
    '''Subfunction of `sim_no_lag` and `sim_lag` that sets the value of 
    the initial condition `x0` if none is specified.

    Parameters
    ----------

    x0 : 1D numpy array or None
        Initial condition for simulation. If none is given, draw from a normal
        distribution with standard deviation given by `initial_displacement`.
        
    initial_displacement : float 
        Standard deviation of the initial condition if `x0` is None. 

    dim : integer
        Number of elements in returned x0.

    Returns
    -------
    x0 : 1D numpy array
        Initial condition for simulation. 
    '''    

    if x0 is None:
        x0 = np.random.normal(0, initial_displacement, size=(dim))

    return x0


################################################################################
def _set_sim_perturbation_magnitude(perturbation_magnitude, 
    initial_displacement):
    '''Subfunction of `sim_no_lag` and `sim_lag` that sets the value of 
    `perturbation_magnitude` if none is specified.

    Parameters
    ----------
    perturbation_magnitude : float or None
        Magnitude of perturbations applied during simulation.

    initial_displacement : float
        Standard deviation of the initial condition if `x0` is None. 

    Returns
    -------
    perturbation_magnitude : float
        Magnitude of perturbations applied during simulation.
    '''

    if perturbation_magnitude is None:
        perturbation_magnitude = initial_displacement

    return perturbation_magnitude


################################################################################
def autocorr(X, points=50):
    '''Numerically compute the autocorrelation for time-series data `X`. This
    can take a very long time to run.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data of network dynamics.

    points : int (default=50)
        Number of time lags to consider. Determines length of output array.
        
    Returns
    -------
    atc : 1D numpy array
        An array of length `points`. Values describe autocorrelation function 
        of `X`.
    '''
    atc = np.zeros(points)
    
    for i in range(points):    
        # get cross-covariance matrix
        #C = np.cov(X[:,:len(X[0])-i],X[:,i:])
        #C = C[len(X):,:len(X)]
        C = np.cov(X[:,:len(X[0])-i],X[:,i:])[len(X):,:len(X)]
        
        
        # grab mean diagonal element
        atc[i] = np.mean(np.diag(C))
        
    return atc

################################################################################
def autocorr2(X, points=50):
    '''Numerically compute the autocorrelation for time-series data `X`. This
    can take a very long time to run. This implementation may be faster than
    the `autocorr`.
    
    Parameters
    ----------
    X : 2D numpy array
        Time series data of network dynamics.

    points : int (default=50)
        Number of time lags to consider. Determines length of output array.
        
    Returns
    -------
    atc : 1D numpy array
        An array of length `points`. Values describe autocorrelation function 
        of `X`.
    '''
    atc = np.zeros(points)
    
    for i in range(points):    
        # get cross-covariance matrix
        #C = np.cov(X[:,:len(X[0])-i],X[:,i:])
        #C = C[len(X):,:len(X)]
        
        cors = [ np.cov(X[var,:len(X[0])-i],X[var,i:])[0,1] 
                for var in range(len(X)) ]
                
        # grab mean diagonal element
        atc[i] = np.mean(cors)
        
    return atc