import sys, os, dill, string
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np

from simwrapper import * # can we specify which ones are needed?
sys.path.append('../libs/')
import curvygraph as cg


###############################################################################
class Variable():
    '''Variables are objects that have a name (which is a keyword argument of 
    `siminf()`) and a list of values. We are going to use this class for 
    passing arguments to plotting functions.'''
    
    def __init__(self, name, values=None):
        self.name = name
        self.values = values
        
    def copy(self):       
        new_variable = Variable(self.name, self.values)
        
        return new_variable
    
    def set_name(self, name):
        self.name = name
        
    def set_values(self, values):
        self.values = values
        
    def get_name(self):
        return self.name
    
    def get_values(self):
        return self.values
    
    
###############################################################################
def rcPhysRev(fontsize=9):
    '''Update matplotlib rc parameters to make fonts in figures match fonts in
    Physical Review templates.
    
    Parameters
    ----------
    
    fontsize : int (default=9)
        Default font size for matplotlib figures.
    '''
    
    matplotlib.rcParams['mathtext.fontset'] = 'cm' 
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}', r'\usepackage{amssymb}',
        r'\usepackage{textcomp}', r'\usepackage{wasysym}']
    matplotlib.rcParams['font.family'] = 'STIXGeneral' 
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    matplotlib.rc('text', usetex=True)

    
###############################################################################   
def boldfont():
    '''Create an instance of FontProperty for bold sans-serif font.
    
    Returns
    -------
    font : FontProperty 
        An instance of FontProperty for bold sans-serif font.
    '''
    
    font = FontProperties()
    font.set_weight('bold')
    font.set_family('sans-serif')
    
    return font


################################################################################
def c8(i):
    '''A color from the 8-color pallette from 
    http://jfly.iam.u-tokyo.ac.jp/color/ .
    
    Parameters
    ----------
    i : int
        An integer in `range(8)` to select one of 8 colors.
    
    Returns
    -------
    c : str
        A color hex code.
    '''

    # make list
    c8_list = ['#000000', # black
               '#0072B2', # blue
               '#E69F00', # orange
               '#56B4E9', # sky blue
               '#D55E00', # vermillion
               '#F0E442', # yellow
               '#009E73', # blue green
               '#CC79A7'] # pink

    # select a color
    c = c8_list[i]
    
    return c


################################################################################        
def color63(color, hue):
    '''A color from a 7-color 3-hue color table.
    
    Parameters
    ----------
    color : int
        An integer in `range(7)` to select one of 7 colors.

    hue : int
        An integer in `range(3)` to select one of 3 hues.

    Returns
    -------
    c : str
        A color hex code.
    '''

    # make table
    table = [['#c9a3a3','#a06060','#703030'], # reds
             ['#c3b690','#97885a','#715f2a'], # yellows
             ['#9ab8a4','#60866d','#2d5b3c'], # greens
             ['#86a5b4','#46748b','#235168'], # bluegreens
             ['#ababc9','#717199','#3c3c79'], # blues
             ['#c2a3c2','#8b6d8b','#633063'], # purples
             ['#c0c0c0','#808080','#404040'] # grays
            ]

    # select a color and hue
    c = table[color][hue]

    return c


###############################################################################
def get_labelstring(kw, verbose_label=False):
    '''Get the string to be used for a parameter of the time-series simulation 
    or network inference in figures.
    
    Parameters
    ----------
    kw : A keyword argument for `siminf()`.

    verbose_label : bool (default=False)
        If True, labels include variable names and symbols. If False, labels
        include only symbols.
    
    Returns
    -------
    s : string
        The string label of a parameter for `siminf()`.
    ''' 

    if verbose_label:
        label_dict = {'n' : r'network size $n$',
            'density': r'edge density $d_e$',
            'reciprocity': r'edge reciprocity $r_e$',
            'ctime' : r'characteristic time $\tau$',
            'theta': r'reversion rate $\theta$',
            'epsilon': r'coupling strength $\epsilon$',
            'sigma': r'noise strength $\sigma$',
            'eta' : r'meas. noise strength $\eta$',
            'max_lag' : r'maximum transmission lag $\delta$', 
            'initial_displacement' : 
                r'initial displacement $\langle\vert{\bf x}_0\vert\rangle$', 
            'dt': r'time step $\Delta t$',
            'T' : r'sample number $N$',
            'perturbation_frequency' : r'perturbation frequency $f_{p}$',
            'perturbation_magnitude' : r'perturbation magnitude $m_{p}$',
            'max_lag_inf' : 
                r'anticipated maximum transmission lag $\hat \delta$', 
            'accuracy' : 'accuracy $\Phi$',
            'acc_variance' : 'accuracy variance'}
    else:
        label_dict = {'n' : r'$n$',
            'density': r'$d_e$',
            'reciprocity': r'$r_e$',
            'ctime' : r'$\tau$',
            'theta': r'$\theta$',
            'epsilon': r'$\epsilon$',
            'sigma': r'$\sigma$',
            'eta' : r'$\eta$',
            'max_lag' : r'$\delta$', 
            'initial_displacement' : r'$\langle\vert{\bf x}_0\vert\rangle$', 
            'dt': r'$\Delta t$',
            'T' : r'$N$',
            'perturbation_frequency' : r'$f_{p}$',
            'perturbation_magnitude' : r'$m_{p}$',
            'max_lag_inf' : r'$\hat \delta$', 
            'accuracy' : '$\Phi$',
            'acc_variance' : 'accuracy variance'}
    
    if kw in label_dict.keys():           
        s = label_dict[kw]
    else:
        s = kw
        
    return s


###############################################################################
def draw_process_motif(ax=None, center=(0,0), radius=0.31, node_radius=0.1, 
    standard_node_radius=0.05, label_pad=0.09, head_pad=0.05,     
    node_labels=[r'$j$', r'$i$'], edge_labels=[r'$\ell_i$',r'$\ell_j$'],
    fontsize=9, node_label_fontsize=None, edge_label_fontsize=None, 
    head_width=0.1, head_length=0.1, lim=0.6, set_limits=False):
    '''Draw a process motif for covariance (3 nodes, 2 arrows).
    
    Parameters
    ----------
    ax : matplotlib axes (default=None)
        Matplotlib axes on which the process motif should be drawn. If `ax` is 
        `None`, select current axes via `plt.cga()`.
        
    center : tuple (default=(0,0))
        Tuple or array with 2 elements indicating the center of the process
        motif to be drawn on `ax`.

    radius : float (default=0.31)
        Radius of the process motif to be drawn on `ax`.

    node_radius : float (default=0.1)
        Radius of focal nodes.
        
    standard_node_radius : float (default=0.05)
        Radius of non-focal nodes.
        
    label_pad : float (default=0.09)
        Padding between edges and edge labels.
        
    head_pad : float (default=0.05)
        Padding between arrow tips and nodes.
        
    node_labels : list (default=`[r'$i$', r'$j$']`)
        List with two elements. Each element is a label for an focal node.
        
    edge_labels : list (default=`[r'$\ell_i$',r'$\ell_j$']`)
        List with two elements. Each element is a label for an edge.
        
    fontsize : int (default=9)
        Default value for `node_label_fontsize` and `edge_label_fontsize`.
        
    node_label_fontsize : int (default=None)
        Font size for node labels.
        
    edge_label_fontsize : int (default=None)
        Font size for edge labels.
        
    head_width : float (default=0.1)
        Width of arrow heads for drawing edges.
        
    head_length : float (default=0.1)
        Length of arrow heards for drawing edges.

    lim: float (default=0.6)
        If set_limits is set to `True`, reset the plot range to `[-lim,lim]`
        for both x axis and y axis. Only relevant if set_limits is `True`.
        
    set_limits: bool (default=False)
        If set_limits is `True`, rescale plot range of axes.
    '''    

    # set defaults for fontsize if none are specified via keyword arguments
    if edge_label_fontsize is None:
        edge_label_fontsize = fontsize
    if node_label_fontsize is None:
        node_label_fontsize = fontsize
    
    # select current axes if None is specified via keyword arguments
    if ax is None:
        ax = plt.gca()
        
    # specify what the curvy edges should look like
    DNM_edge = cg.CurvyEdge(curvature=1, color=c8(1), linewidth=1.5,
        heads=[None, cg.CurvyHead(width=head_width, length=head_length, 
            color=c8(1))],
        labels=[cg.CurvyLabel(pad=0.05, color='black')])

    # specify what the focal nodes should look like
    focal_node = cg.CurvyNode(radius=node_radius, facecolor=(1,1,0.35), 
        linecolor="black", linewidth=1)
    
    # specify what the non-focal nodes should look like
    standard_node = cg.CurvyNode(radius=standard_node_radius, 
        facecolor="white", linecolor="black", linewidth=1)
    
    # make a curvy graph of a process motif with the above specifications
    g = cg.makeRingGraph(3, '110', '><_', tilt=-30, center=center, 
        radius=radius, edges_like=DNM_edge, nodes_like=focal_node)
    
    # set node labels and node colors
    g.nodes[0].set_label_text(node_labels[1])
    g.nodes[0].set_label_size(node_label_fontsize)
    g.nodes[0].set_facecolor('orange')
    
    g.nodes[2].set_label_text(node_labels[0])
    g.nodes[2].set_label_size(node_label_fontsize)
    
    g.nodes[1].set_like(standard_node)
    
    # set edge labels
    g.edges[0].set_labels([cg.CurvyLabel(text=edge_labels[1], pad=label_pad, 
        size=edge_label_fontsize)])
    g.edges[1].set_labels([cg.CurvyLabel(text=edge_labels[0], pad=label_pad, 
        size=edge_label_fontsize)])
    
    # set head pad
    g.set_head_pad(head_pad)
    
    # draw curvy graph
    g.draw(ax=ax)

    # set limits of ax
    if set_limits:
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])
    ax.axis('off')

############################################################################### 
def get_colorbar_position(ncols, nrows, full_length=True):
    '''Get position for a good colorbar position for a figure with subplot 
    grid.
    
    Parameters
    ----------
    ncols : integer
        Number of columns in figure.
        
    nrows : integer
        Number of rows in figure.
        
    full_length : bool (default=True)
        If True, the colorbar fills the entire height of the subplots. If 
        False, its height and position are adjusted to fit the top-right 
        subplot.
        
    Returns
    -------
    pos : list
        A list with position information in the form `[x0,y0,dx,dy]`.
    '''

    # dictionary of optimal colorbar positions
    # TODO: We need to adjust these values as we go along with making
    # plots
    pos_dict = { # pos for half-length    # pos for full length
        1 : {1 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             2 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             3 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             4 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             5 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]] },
        2 : {1 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             2 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             3 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             4 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             5 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]] },
        3 : {1 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             2 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             3 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             4 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             5 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]] },
        4 : {1 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             2 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             3 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             4 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             5 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]] },
        5 : {1 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             2 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             3 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             4 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]], 
             5 : [[0.85, 0.10, 0.05, 0.80], [0.85, 0.10, 0.05, 0.80]] }
    }
    
    pos = pos_dict[ncols][nrows][int(full_length)]
    
    return pos


###############################################################################
def subplot_adjustment(ncols, nrows):
    '''Get arguments for `plt.subplots_adjust()` a figure with a grid of 
    subplots.
    
    
    Parameters
    ----------
    ncols : integer
        Number of columns in figure.
        
    nrows : integer
        Number of rows in figure.
        
    full_length : bool (default=True)
        If True, the colorbar fills the entire height of the subplots. If 
        False, its height and position are adjusted to fit the top-right 
        subplot.
        
    Returns
    -------
    pos : list
        A list with arguments for `subplots_adjust()` in the form 
        `[left, bottom, right, top, wspace, hspace]`.
    '''

    # dictionary of optimal colorbar positions
    # TODO: We need to adjust these values as we go along with making
    # plots
    pos_dict = {     
        1 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], # default value
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], # placeholder - in case we want to add a second option
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        2 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        3 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        4 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        5 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] }
    }
    
    pos = pos_dict[ncols][nrows][0]
    
    return pos    
    
    
###############################################################################
def subplot_adjustment_with_colorbar(ncols, nrows, full_length=True):
    '''Get arguments for `plt.subplots_adjust()` a figure with a grid of 
    subplots.
    
    Parameters
    ----------
    ncols : integer
        Number of columns in figure.
        
    nrows : integer
        Number of rows in figure.
        
    full_length : bool (default=True)
        If True, the colorbar fills the entire height of the subplots. If 
        False, its height and position are adjusted to fit the top-right 
        subplot.
        
    Returns
    -------
    pos : list
        A list with arguments for `subplots_adjust()` in the form 
        `[left, bottom, right, top, wspace, hspace]`.
    '''
    # dictionary of optimal colorbar positions
    # TODO: We need to adjust these values as we go along with making
    # plots
    pos_dict = {     
        1 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], # pos for half-length
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], # pos for full length
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        2 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        3 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        4 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] },
        5 : {1 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             2 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             3 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             4 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]], 
             5 : [[0.00, 0.00, 1.00, 1.00, 0.50, 0.50], 
                  [0.00, 0.00, 1.00, 1.00, 0.50, 0.50]] }
    }
    
    pos = pos_dict[ncols][nrows][int(full_length)]
    
    return pos    
    
    
###############################################################################
def default_plotrange(varname):
    '''Get default values for a Variable from its name.
    
    Parameters
    ----------
    varname : string
        Name of a Variable.
        
    Returns
    r : 1D numpy array
        Default values for Variable.
    '''

    rdict = {}
    rdict['n'] = np.arange(5,30,5)
    rdict['density'] = np.arange(0.1,1.,0.1)
    rdict['reciprocity'] = np.arange(0.,1.1,0.1)
    rdict['ctime'] = np.arange(1,11,1) 
    rdict['theta'] = np.arange(0.,1.1,0.1)
    rdict['epsilon'] = np.arange(0.1,1.,0.1)
    rdict['sigma'] = np.arange(0.1,2.2,0.2)
    rdict['eta'] = np.arange(0.,2.1,0.2)
    rdict['max_lag'] = np.arange(1,11,1)
    rdict['initial_displacement'] = np.arange(0.,2.1,0.2)  
    rdict['dt'] = np.arange(0.1,1.1,0.1)
    rdict['T'] = np.arange(50,1050,50)    
    rdict['perturbation_frequency'] = np.arange(1,11)/1000
    rdict['perturbation_magntiude'] = np.arange(0.1,2.2,0.2)
    rdict['max_lag_inf'] = np.arange(1,11,1)
    
    if varname in rdict.keys():
        r = rdict[varname]
    else:
        r = np.arange(0.1, 1.0, 0.1)
        
    return r


############################################################################### 
def copy_lines(target, source):
    '''Copy a line from an axes `source` to an axes `target`.

    Parameters
    ----------
    target : a matplotlib axes
        Axes to which lines should be copied.

    source : a matplotlib axes with line plots
        Axes from which lines should be copied.
    '''
    
    lines = source.lines
    
    for l in lines:
        target.plot(l._x, l._y, 
                    linewidth=l._linewidth,
                    linestyle=l._linestyle,
                    color=l._color,
                    marker=l._marker,
                    markersize=l._markersize)
    target.set_xlim(source.get_xlim())
    target.set_ylim(source.get_ylim())
    target.set_xticks(source.get_xticks())
    target.set_yticks(source.get_yticks())
    target.set_xticklabels(source.get_xticklabels())
    target.set_yticklabels(source.get_yticklabels())    
    target.set_xlabel(source.get_xlabel())
    target.set_ylabel(source.get_ylabel())
    
################################################################################    
def grab_data_from_axes2d(axes):
    '''Collect data from heatmaps in a list of matplotlib axes.
    
    Parameters
    ----------
    axes : list
        A list of matplotlib axes.

    Returns
    -------
    data : list
        A list of 2d numpy arrays containing the data from all heatmaps
        in `axes`.
    '''

    data = []
    for i, ax in enumerate(axes.flat):
        img = [obj for obj in ax.get_children() 
            if isinstance(obj, matplotlib.image.AxesImage)][0]
        data.append(img.get_array())

    return data
        

###############################################################################
def plot1d(ncols=2, nrows=2, x=Variable('epsilon',np.arange(0,1,0.1)), 
           y='accuracy', xvars=None, yvars=None, linevars=None, plotvars=None, 
           linevar_labels = None,
           num_trials=1, default_parameters={}, linecolors=None, 
           linestyles=None, linewidths=None, markers=None, markersizes=None, 
           show_reciprocity=True, add_legend=True, add_subplotlabels=False,
           subplot_titles=True, verbose_titles=False, verbose_labels=False,
           load=True, save=True, sharey=False, fheight=None, 
           verbose_lookup=False, verbose_sim=False, draw=True, 
           verbose_xlabel=True, verbose_ylabel=True, 
           subplotlabelx=None, subplotlabely=None,
           **kwargs):
    '''Generate a figure with several subplots, each of which either shows a 
    parameter plane (heatmap or contour plot with heatmap) or a distribution
    plot (box plot or violin plot) of the values in other subplots.
    
    Parameters
    ----------
    ncols : integer (default=2)
        Number of columns of subplots in figure.
        
    nrows : integer (default=2)
        Number of rows of subplots in figure.
                
    x : string or Variable (default=('epsilon',np.arange(0,1,0.1)))
        Variable or name of variable to be displayed on the x-axis of subplots. 
            
    y : string ['accuracy' | 'time' | 'acc_variance'] (default='accuracy')
        Name of variable to be plotted on the y axis.
        
    xvars : list (default=None)
        If xvars is not None, subplots have different variables on the x axis. 
        The i-th entry of `xvars` is the value to be used for x (i.e., it can 
        be a string or a Variable). Any value passed for the keyword argument 
        `x` is ignored.
        
    linevars : Variable or list (default=None)
        If linevars is not None, plot more than 1 line per subplot. The lines 
        have different parameter settings for `siminf()` (e.g., different 
        inference methods). If a list is passed, it must be a list of variables
        that all have the same number of values.

    yvars : list (default=None)
        If yvars is not None, subplots have different variables on the y axis. 
        The i-th entry of `yvars` is used as the value for `y` (i.e., it is 
        either 'accuracy', 'time', or 'acc_variance'). Any value passed for the
        keyword argument `y` is ignored.
        
    num_trials : integer (default=1)
        Number of simulations run for each data point.
        
    plotvars : Variable or list (default=None)
        If `plotvars` is not None, subplots have different parameters for the
        time-series simulation or the network inference. If a list is passed, 
        it must be a list of variables that all have the same number of values.
    
    default_parameters : dictionary (default={})
        Parameters to be changed from default parameters for `siminf()` for 
        all subplots.
        
    linecolors : list (default=None)
        List of colors for lines in each subplot. Length of `linecolors` should
        be equal to or larger than the length of `linevars`.
    
    linestyles : list (default=None)
        List of styles for lines in each subplot. Length of `linestyles` should
        be equal to or larger than the length of `linevars`.
    
    linestyles : list (default=None)
        List of widths for lines in each subplot. Length of `linewidths` should
        be equal to or larger than the length of `linevars`.
        
    load : bool (default=True)
        If True, check if data from a previous simulation already exists. If 
        so, do not run simulation again.
    
    save : bool (default=True)
        If True, save siminf() results for later use.
    
    
    Returns
    -------
    fig : matplotlib figure
         A figure.
         
    axes : list
         A list of subplot axes in `fig`.
    '''
    
    # TODO add some checks for keywords
    
    # initialize data collection
    num_subplots = ncols * nrows
    data_collection = [0 for _ in range(num_subplots)] 
    
    if add_subplotlabels:
        alphabet = ['('+letter+')' for letter in string.ascii_lowercase]
        if subplotlabelx is None:
            subplotlabelx = 0.9
        if subplotlabely is None:
            subplotlabely = 0.9
        if not hasattr(subplotlabelx, '__iter__'):
            subplotlabelx = [subplotlabelx for _ in range(num_subplots)]
        if not hasattr(subplotlabely, '__iter__'):
            subplotlabely = [subplotlabely for _ in range(num_subplots)]

    # set labels for distribution plots
    labels = []
    if plotvars is not None:
        if isinstance(plotvars, list):
            labels = plotvars[0].values
        else:
            labels = plotvars.values

    elif yvars is not None:
        labels = [y_ for y_ in yvars]
        
    elif xvars is not None:
        for x_ in xvars:
            if isinstance(x_, str):
                # get default plot range for x
                x_ = Variable(x_, default_plotrange(x_))
            labels += [x_.name]
        
    # set titles for subplots
    titles = []
    if subplot_titles==True:
        if plotvars is not None:
            if isinstance(plotvars, list):
                if verbose_titles:
                    titles = [get_labelstring(plotvars[0])+'='+s 
                              for s in plotvars[0].values]
            else:
                if verbose_titles:
                    titles = [get_labelstring(plotvars)+'='+s 
                              for s in plotvars.values]
        else:
            titles = labels

    if not hasattr(num_trials , '__iter__'):
        list_num_trials = [num_trials]*num_subplots
    else:
        list_num_trials = num_trials
    #print(list_num_trials)
        
    if not isinstance(linevars, list):
        linevars = [linevars]
    if linevars[0] is not None:
        num_of_lines = len(linevars[0].values)
    else:
        num_of_lines = 1

    if linecolors is None:
        # set default line colors
        linecolors = [c8(i) for i in range(num_of_lines)]
    if linestyles is None:
        # set default line styles 
        linestyles = ['-' for i in range(num_of_lines)]
    if linewidths is None:
        # set default linewidths
        linewidths = [2 for i in range(num_of_lines)]
    if markers is None:
        # set default markers
        markers = ['' for i in range(num_of_lines)]
    if markersizes is None:
        # set default marker sizes
        markersizes = [3 for i in range(num_of_lines)]
        
    # start data collection
    for i in range(num_subplots):
        
        nt = list_num_trials[i]
        
        # select x, y variables
        if xvars is not None:
            if len(xvars) > i:
                x = xvars[i]
            else:
                break
                
        if yvars is not None:
            if len(yvars) > i:
                y = yvars[i]
            else:
                break
                
                
        # if none are given, set ranges for x
        if isinstance(x, str):
            # get default plot range for x
            x = Variable(x, default_plotrange(x))
            
        # set file and dictionary keys to load the data from
        filename = '../data/plot1d_'+x.name+'-'+y+'.d'
        if y == 'acc_variance':
            filename = '../data/plot1d_'+x.name+'-accuracy.d'
        sx = str(x.values)

        # try to load data from file
        if os.path.exists(filename):
            if verbose_lookup:
                print('Found file...')
            data = dill.load(open(filename, 'rb'))
        else:
            if verbose_lookup:
                print('Found no file. Make a new file...')
            data = {}
            
        # search for entry with the right xrange
        if sx not in data.keys():
            if verbose_lookup:
                print('Did not find key', sx,'. Make a new key...')
            data[sx] = {}
        else:
            if verbose_lookup:
                print('Found key', sx,'.')
                
        # search for entry with the right y value
        if (y == 'accuracy' or y == 'time') and y not in data[sx].keys():
            if verbose_lookup:
                print('Did not find key', y,'. Make a new key...')
            data[sx][y] = {}
        elif y == 'acc_variance' and 'accuracy' not in data[sx].keys():
            if verbose_lookup:
                print('Did not find key accuracy. Make a new key...')
            data[sx]['accuracy'] = {}
        else:
            if verbose_lookup:
                print('Found key', y,'.')
                
        pars = default_siminf_pars()
        for kw in default_parameters.keys():
            pars[kw] = default_parameters[kw]
            
        if plotvars is not None:
            if isinstance(plotvars, list):
                for pv in plotvars:
                    if len(pv.values) > i:
                        varname = pv.name
                        value = pv.values[i]
                        pars[varname] = value
            else:
                if len(plotvars.values) > i:
                    varname = plotvars.name
                    value = plotvars.values[i]
                    pars[varname] = value
        
        # get all the data for one subplot
        data_collection_for_subplot = [0 for _ in range(num_of_lines)] 
        
        for j in range(num_of_lines):
            
            pars_line = dict(pars)
            
            lvs = [None for _ in range(len(linevars))]
            
            for k in range(len(linevars)):            
                if linevars[k] is not None:
                    pars_line[linevars[k].name] = linevars[k].values[j] #TODO: where do I set k? 
                    lvs[k] = Variable(linevars[k].name, [linevars[k].values[j]])
            if len(lvs)==0:
                lvs = [None]
            spars = parameters2string(pars_line,remove=(x.name))
            if verbose_lookup:
                print('Set key:', spars)

            # search for entry with right parameters
            load_failed = True
            if load==True:
                if (y == 'accuracy' or y == 'time') and spars in data[sx][y].keys():
                    if verbose_lookup:
                        print('Found pars string in keys...')
                    arr = data[sx][y][spars]
                    if arr.shape[-1]>=nt:
                        if verbose_lookup:
                            print('Load data')
                        arr = arr[:,:nt]
                        data_collection_for_subplot[j] = [x.copy(), y, lvs[0], np.mean(arr, axis=-1)]
                        load_failed=False 
                    else:
                        if verbose_lookup:
                            print('Did not find enough data. (Needed',nt,
                                  'but found only',arr.shape[-1],'. Make new data ...')
                elif y == 'acc_variance' and spars in data[sx]['accuracy'].keys():
                    arr = data[sx]['accuracy'][spars]
                    if arr.shape[-1]>=nt:
                        data_collection_for_subplot[j] = [x.copy(), y, lvs[0], np.var(arr, axis=-1)]
                        load_failed=False

            if load_failed:
                # make new data
                if verbose_lookup:
                    print('Start to make new data ...')
                accuracy = np.zeros((len(x.values),nt))
                duration = np.zeros((len(x.values),nt))
                for it in range(nt):
                    for ix, xval in enumerate(x.values):
                        pars_line[x.name] = xval
                        res = siminf(verbose=verbose_sim, **pars_line)
                        accuracy[ix,it] = res['accuracy']
                        duration[ix,it] = res['time']
                if y == 'accuracy':
                    data_collection_for_subplot[j] = [x.copy(), y, lvs[0], np.mean(accuracy, axis=-1)]
                elif y == 'time':
                    data_collection_for_subplot[j] = [x.copy(), y, lvs[0], np.mean(duration, axis=-1)]
                elif y == 'acc_variance':
                    data_collection_for_subplot[j] = [x.copy(), y, lvs[0], np.var(accuracy, axis=-1)]
                else:
                    print('Error in plot1d: Unknown input for y.')
                if save: 
                    # save data
                    if 'accuracy' not in data[sx].keys():
                        data[sx]['accuracy'] = {}
                    if 'time' not in data[sx].keys():
                        data[sx]['time'] = {}
                    data[sx]['accuracy'][spars] = accuracy
                    data[sx]['time'][spars] = duration
                    dill.dump(data, open(filename, 'wb'))
                    
        data_collection[i] = data_collection_for_subplot
    
    if draw:
        # set figure size
        if fheight is None:
            if ncols<=2:
                fsize = (4, 4/ncols*nrows+0.4)
            else:
                fsize=(8, 8/ncols*nrows+0.4)
        else:
            if ncols<=2:
                fsize = (4, fheight)
            else:
                fsize=(8, fheight)
    
        # create figure and subplots
        maximum = 0.0
        minimum = 1.0
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fsize, dpi=100)
        for i, ax in enumerate(axes.flat):
            subplot_data = data_collection[i]
            x = subplot_data[0][0]
            y = subplot_data[0][1]
            
            if x.name == 'directedness':
                if show_reciprocity:
                    print('Show reciprocity')
                    x.name = 'edge reciprocity'
                    x.values = 1-x.values
                    print(x.values)

            for j in range(len(subplot_data)):
                if linevar_labels is not None:
                    label = linevar_labels[j]
                elif subplot_data[j][2] is not None:
                    #print(subplot_data[j][2])
                    label = str(subplot_data[j][2].values[0])
                else:
                    label = ''

                minimum = min([minimum, np.nanmin(subplot_data[j][3])])
                maximum = max([maximum, np.nanmax(subplot_data[j][3])])

                if verbose_labels:
                    label = get_labelstring(subplot_data[j][2].name) + '=' + label
                if x.name == 'edge reciprocity':
                    yvals = subplot_data[j][3]#[::-1]
                else:
                    yvals = subplot_data[j][3]
                ax.plot(x.values, yvals, 
                    label=label,
                    color=linecolors[j], ls=linestyles[j], lw=linewidths[j], 
                    marker=markers[j], markersize=markersizes[j])
            ax.set_xlabel(get_labelstring(x.name, verbose_label=verbose_xlabel))
            ax.set_ylabel(get_labelstring(y, verbose_label=verbose_ylabel))
            if add_subplotlabels:
                #print('add subplotlabel', alphabet[i], 'at', 
                #      subplotlabelx[i], subplotlabely[i])
                ax.text(subplotlabelx[i], subplotlabely[i], 
                        alphabet[i], transform=ax.transAxes)
            #else:
            #    #print('add_subplotlabels', add_subplotlabels)
        if add_legend:
            legend = plt.legend()
        else:
            legend = []

        if sharey:    
            minimum = 0.97*minimum
            maximum =1.03*maximum
            #print('yrange', minimum, maximum)
            for i, ax in enumerate(axes.flat):
                ax.set_ylim([minimum,maximum])
            
        plt.subplots_adjust(*subplot_adjustment(
            ncols, nrows))

        return fig, axes, legend

    else:
        
        return 0, 0, 0


###############################################################################
def plot2d(ncols=2, nrows=2, vis='heatmap', 
    x=Variable('epsilon',np.arange(0,1,0.1)), 
    y=Variable('theta',np.arange(0,1,0.1)),
    z='accuracy', xvars = None, yvars=None, zvars = None, plotvars=None, 
    default_parameters={}, num_trials=1, subplot_titles=True, 
    fheight=None, verbose_titles=False, load=True, save=True, 
    #add_subplotlabels=False, subplotlabelx=None, subplotlabely=None, 
    draw=True, **kwargs):
    '''Generate a figure with several subplots, each of which either shows a 
    parameter plane (heatmap or contour plot with heatmap) or a distribution
    plot (box plot or violin plot) of the values in other subplots.
    
    Parameters
    ----------
    ncols : integer (default=2)
        Number of columns of subplots in figure.
        
    nrows : integer (default=2)
        Number of rows of subplots in figure.
        
    vis : string ['heatmap' | 'contour'] (default='heatmap')
        Visualize parameter planes either as heatmaps or as heatmaps with 
        contour lines. (The second option is not implemented yet.)
        
    x : string or Variable (default=Variable('epsilon',np.arange(0,1,0.1)))
        A Variable to be displayed on the y-axis of parameter planes.
    
    y : string or Variable (default=Variable('theta',np.arange(0,1,0.1))))
        A Variable to be displayed on the y-axis of parameter planes. 
        
    z : string ['accuracy' | 'time' | 'acc_variance'] (default='accuracy')
        Name of variable to be plotted in the parameter planes
        
    xvars : list (default=None)
        If `xvars` is not None, parameter planes have different Variables on 
        the x axis. The i-th entry of `xvars` is used as the value for `x` 
        (i.e., it can be a string or a Variable). Any value passed for the 
        keyword argument `x` is ignored.

    yvars : list (default=None)
        If `yvars` is not None, parameter planes have different Variables on 
        the y axis. The i-th entry of `yvars` is used as the value for `y` 
        (i.e., it can be a string or a Variable). Any value passed for the 
        keyword argument `y` is ignored.
        
    zvars : list (default=None)
        If `zvars` is not None, parameter planes have different variables on
        the z axis. The i-th entry of `zvars` is used as the value for `z` 
        (i.e., it is either 'accuracy', 'time', or 'acc_variance'). Any value 
        passed for the keyword argument `z` is ignored.
        
    plotvars : Variable or list (default=None)
        If plotvars is not None, parameter planes have different parameters
        for the time-series simulation or the network inference. It can either
        be a variable or a list of variables. If a list of variables is passed,
        all variables must have the same number of values. Admissable numbers 
        of values are `nrows*ncols` or `nrows*ncols-1`. 
    
    default_parameters : dictionary (default={})
        Parameters to be changed from default parameters for siminf() for all
        subplots.
    
    num_trials : integer (default=1)
        Number of simulations run for each point in the data grid

    fheight : float (default=None)
        If not None, set the height of the figure to `fheight`.
        
    subplot_titles : bool (default=True)
        If True, every parameter plane has its corresponding value of 
        `plotvars[1]` as its title.
    
    verbose_titles : bool (default=False)
        If True, subplot titles include the name of a variable (i.e., 
        'density=0.5' instead of '0.5').
        
    load : bool (default=True)
        If True, check if data from a previous simulation already exists. If
        so, do not run simulation again.
    
    save : bool (default=True)
        If True, save siminf() results for later use.

    draw : bool (default=True)
        If True, draw and return a figure, axes, and colorbar. If False, return
        (0, 0, 0).
    
    Returns
    -------
    fig : matplotlib figure
         A figure.
         
    axes : list
         A list of subplot axes in `fig`.
         
    cbar : matplotlib axes
         The axes with the colorbar for the parameter planes.
    '''
    
    # TODO add some checks for keywords
    
    # initialize data collection
    num_subplots = ncols * nrows
    data_collection = [] #0 for _ in range(num_subplots)] 
    
    # set labels for distribution plots
    # (A distribution plot shows the distribution of values in a 2d array as a
    # box in a boxplot. If a figure includes 5 heatmaps, the distribution plot
    # includes 5 boxes. Boxes are labeled according to the parameters that are
    # different for the different subplots as indicated by `plotvars`, 
    # `yvars`, or `xvars`.)
    labels = []
    if plotvars is not None:
        if isinstance(plotvars, list):
            labels = plotvars[0].values
        else:
            labels = plotvars.values
    elif yvars is not None:
        labels = [y_.name for y_ in yvars]
    elif xvars is not None:
        labels = [x_.name for x_ in xvars]
    
    # set titles for subplots
    # (Subplot titles should reflect what is different between the subplots.
    #  That is why they are indicated by `plotvars`, `yvars`, or `xvars`.)
    titles = []
    if subplot_titles==True:
        if plotvars is not None:
            if isinstance(plotvars, list):
                if verbose_titles:
                    titles = [get_labelstring(plotvars[0].name)+'='+str(s) 
                              for s in plotvars[0].values]
            else:
                if verbose_titles:
                    titles = [get_labelstring(plotvars.name)+'='+str(s)
                              for s in plotvars.values]
        else:
            titles = labels
    
    
    # start data collection
    for i in range(num_subplots):
        # select x, y, and z variables
        if xvars is not None:
            if len(xvars) > i:
                x = xvars[i]
            else:
                break
        if yvars is not None:
            if len(yvars) > i:
                y = yvars[i]
            else:
                break
        if zvars is not None:
            if len(zvars) > i:
                z = zvars[i]
            else:
                break
        
        # complete parameter dictionary
        pars = default_siminf_pars()
        for kw in default_parameters.keys():
            pars[kw] = default_parameters[kw]
            
        #if search_old: 
        #    # complete parameter dictionary (old version)
        #    pars_old = default_siminf_pars_old()
        #    for kw in default_parameters.keys():
        #        pars_old[kw] = default_parameters[kw]
            
        if plotvars is not None:
            if isinstance(plotvars, list):
                for pv in plotvars:
                    if len(pv.values) > i:
                        varname = pv.name
                        value = pv.values[i]
                        pars[varname] = value
                        #if search_old:
                        #    pars_old[varname] = value
                        break_outer = False
                    else:
                        break_outer = True
                        break
                if break_outer:
                    break
            else:
                if len(plotvars.values) > i:
                    varname = plotvars.name
                    value = plotvars.values[i]
                    pars[varname] = value
                    #if search_old:
                    #    pars_old[varname] = value
                else:
                    break
                
        # if none are given, set ranges for x and y
        if isinstance(x, str):
            # get default plot range for x
            x = Variable(x, default_plotrange(x))

        if isinstance(y, str):
            # get default plot range for x
            y = Variable(y, default_plotrange(y))

        # set file and dictionary keys to load the data from
        filename = '../data/plot2d_'+x.name+'-'+y.name+'.d'
        sx = str(x.values)
        sy = str(y.values)
        spars = parameters2string(pars,remove=(x.name,y.name))
        #if search_old:
        #    spars_old = parameters2string(pars_old,remove=(x.name,y.name))

        # try to load data from file
        if os.path.exists(filename):
            data = dill.load(open(filename, 'rb'))
        else:
            data = {}
            
        # search for entry with the right xrange
        if sx not in data.keys():
            data[sx] = {}

        # search for entry with the right yrange
        if sy not in data[sx].keys():
            data[sx][sy] = {}

        # search for entry with the right z value
        if (z == 'accuracy' or z == 'time') and z not in data[sx][sy].keys():
            data[sx][sy][z] = {}
        elif z == 'acc_variance' and 'accuracy' not in data[sx][sy].keys():
            data[sx][sy]['accuracy'] = {}

        # search for entry with right parameters
        load_failed = True
        if load==True:
            if (z == 'accuracy' or z == 'time'): 
                if spars in data[sx][sy][z].keys():
                    data_collection += [[x.copy(), y.copy(), 
                        np.mean(data[sx][sy][z][spars], axis=-1)]]
                    load_failed=False
            elif z == 'acc_variance':
                if spars in data[sx][sy]['accuracy'].keys():
                    data_collection += [[x.copy(), y.copy(), 
                        np.var(data[sx][sy]['accuracy'][spars], axis=-1)]]
                    load_failed=False
            
            #elif search_old:
            #    if (z == 'accuracy' or z == 'time'):
            #        if spars_old in data[sx][sy][z].keys():
            #            # remove rho and max_inf_lag from dict
            #            data_collection = [[x.copy(), y.copy(), 
            #                np.mean(data[sx][sy][z][spars_old], axis=-1)]]
            #            load_failed=False  
            #            if save:
            #                data[sx][sy][z][spars] = np.copy(
            #                    data[sx][sy][z][spars_old])
            #    elif z == 'acc_variance':
            #        if spars_old in data[sx][sy]['accuracy'].keys():
            #            # remove rho and max_inf_lag from dict
            #            data_collection += [[x.copy(), y, 
            #                np.var(data[sx][sy]['accuracy'][spars_old],
            #                    axis=-1)]]
            #
            #            load_failed=False  
            #            if save:
            #                data[sx][sy]['accuracy'][spars] = np.copy(
            #                    data[sx][sy]['accuracy'][spars_old])

        if load_failed:
            # make new data
            accuracy = np.zeros((len(x.values),len(y.values),num_trials))
            duration = np.zeros((len(x.values),len(y.values),num_trials))
            for it in range(num_trials):
                for ix, xval in enumerate(x.values):
                    for iy, yval in enumerate(y.values):
                        pars[x.name] = xval
                        pars[y.name] = yval
                        res = siminf(**pars)
                        accuracy[ix,iy,it] = res['accuracy']
                        duration[ix,iy,it] = res['time']
            if z == 'accuracy':
                data_collection += [[x.copy(), y, np.mean(accuracy, axis=-1)]]
            elif z == 'time':
                data_collection += [[x.copy(), y, np.mean(duration, axis=-1)]]
            elif z == 'acc_variance':
                data_collection += [[x.copy(), y, np.var(accuracy, axis=-1)]]
            else:
                print('Unknown value for z.')
            if save: 
                # save data
                if not 'accuracy' in data[sx][sy]:
                    data[sx][sy]['accuracy'] = {}
                if not 'time' in data[sx][sy]:
                    data[sx][sy]['time'] = {}

                data[sx][sy]['accuracy'][spars] = accuracy
                data[sx][sy]['time'][spars] = duration
                dill.dump(data, open(filename, 'wb'))          
        
    # plot subplots
    if draw==True and vis=='heatmap':
        fig, axes, cbar = plot2d_heatmap(ncols, nrows, data_collection, 
            labels=labels, titles=titles, contour=False, fheight=fheight, 
            **kwargs)
    elif draw==True and vis=='contour':
        fig, axes, cbar = plot2d_heatmap(ncols, nrows, data_collection, 
            labels=labels, titles=titles, contour=True, fheight=fheight,
            **kwargs)
    else:
        fig, axes, cbar = 0, 0, 0

    return fig, axes, cbar

      
###############################################################################    
def plot2d_heatmap(ncols, nrows, data, labels=[], titles=[], dist='box', 
    contour=False, fheight=None, #show_reciprocity=True,
    verbose_xlabel=True, verbose_ylabel=True, add_subplotlabels=False,
    subplotlabelx=0.9, subplotlabely=0.9, **kwargs):
    '''Generate a figure with several subplots, each of which either shows a 
    heatmap or a distribution plot (box plot or violin plot) of the values in 
    other subplots.
    
    Parameters
    ----------
    ncols : integer (default=2)
        Number of columns of subplots in figure.
        
    nrows : integer (default=2)
        Number of rows of subplots in figure.
        
    data : list of lists
        A nested list of data and settings for subplots. Each entry is a list
        of the form [xvariable, yvariable, zdata].
        
    labels : list (default=[])
        A list of strings to be used as xvalues in distribution plots.

    titles : list (default=[])
        A list of strings to be used as subplot titles for heatmaps.
        
    dist : string ['box' | 'violin'] (default='box')
        Select whether distribution plots should be box plots or violin plots.
        (Not implemented yet.)

    contour : bool (default=False)
        If True, use contour plots to visualize 2d data. If False, use heatmaps
        instead.
        
    fheight : float (default=None)
        If not None, set the height of the figure to `fheight`.

    verbose_xlabel : bool (default=True)
        If True, label strings for x axis include variable names and symbols.
        If False, label strings only include symbols.

    verbose_ylabel : bool (default=True)
        If True, label strings for y axis include variable names and symbols.
        If False, label strings only include symbols.

    add_subplotlabels : bool (default=False)
        If True, add labels A, B, C, D , ... to each panel/ subplot of the 
        figure.

    subplotlabelx : float or list (default=0.9)
        Vertical position coordinate for subplot labels.
    subplotlabely : float or list (default=0.9)
        Vertical position coordinate for subplot labels.
        
    Returns
    -------
    fig : matplotlib figure
         A figure.
         
    axes : list
         A list of subplot axes in `fig`.
         
    cbar : matplotlib axes
         The axes with the colorbar for the parameter planes.
    '''
    # set figure size
    if fheight is None:
        if ncols<=2:
            fsize = (4, 4/ncols*nrows+0.4)
        else:
            fsize=(8, 8/ncols*nrows+0.4)
    else:
        if ncols<=2:
            fsize = (4, fheight)
        else:
            fsize=(8, fheight)
    
    # create figure and subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fsize, dpi=100)
    
    # check which of the subplots should show a map and which a distribution
    num_subplots = ncols * nrows
    num_distribution_plots = num_subplots - len(data)
    axes_list = axes.flat

    if add_subplotlabels:
        alphabet = ['('+letter+')' for letter in string.ascii_lowercase]
        if not hasattr(subplotlabelx, '__iter__'):
            subplotlabelx = [subplotlabelx for _ in range(num_subplots)]
        if not hasattr(subplotlabely, '__iter__'):
            subplotlabely = [subplotlabely for _ in range(num_subplots)]
    
    if num_distribution_plots == 0:
        # all subplots are heatmaps
        axes_with_maps = axes_list
        axes_with_histograms = []
        hist2maps = []
        
    elif num_distribution_plots == 1:
        # last subplot is a distribution plot
        axes_with_maps = axes_list[:-1]
        axes_with_histograms = [axes_list[-1]]
        hist2maps = [range(len(axes_with_maps))]
        
    elif num_distribution_plots == ncols:
        # last row are distribution plots
        axes_with_maps = axes_list[:-ncols] 
        axes_with_histograms = axes_list[-ncols:]
        hist2maps = (np.arange(len(axes_with_maps)).reshape(
            len(axes_with_maps)//(ncols),ncols)).T
        
    elif num_distribution_plots == nrows:
        # last column are distribution plots
        axes_with_maps = [ax for axi, ax in enumerate(axes_list) 
                          if (axi+1)%ncols!=0]
        axes_with_histograms = [ax for ax in enumerate(axes_list) 
                                if ax not in axes_with_maps]
        hist2maps = np.arange(len(axes_with_maps)).reshape(
            len(axes_with_maps)//(nrows-1),nrows)
    else:
        print("I don't know how to arrange the subplots.")
        return 0

    #if show_reciprocity:
    #    print('Show reciprocity 2')
    #    for i in range(len(axes_with_maps)):
    #        if data[i][0].name == 'directedness':
    #            data[i][0].name = 'edge reciprocity'
    #            data[i][0].values = (1-data[i][0].values)[::-1]
    #            data[i][2] = np.flip(data[i][2], axis=0)
    #        elif data[i][1].name == 'directedness':
    #            data[i][1].name = 'edge reciprocity'
    #            data[i][1].values = (1-data[i][1].values)[::-1]
    #            data[i][2] = np.flip(data[i][2], axis=1)
    
    # set z range to min-max if none given
    map_data = np.array([data[i][2] for i in range(len(data))])
    if 'vmin' not in kwargs.keys():
        kwargs['vmin'] = np.nanmin(map_data)
    if 'vmax' not in kwargs.keys():
        kwargs['vmax'] = np.nanmax(map_data)
    
    # plot maps   
    for i, ax in enumerate(axes_with_maps):
        im = ax.imshow(data[i][2].T, interpolation="none", origin='lower',
            extent=(data[i][0].values[0],data[i][0].values[-1],
                    data[i][1].values[0],data[i][1].values[-1]), 
            aspect="auto", **kwargs)
        if contour:
            # plot contours over heatmap
            ax.contour(data[i][0].values, data[i][1].values, data[i][2].T, 
                **kwargs)
        ax.set_title(labels[i])
        ax.set_xlabel(get_labelstring(data[i][0].name, 
            verbose_label=verbose_xlabel))
        ax.set_ylabel(get_labelstring(data[i][1].name, 
            verbose_label=verbose_ylabel))
        if i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout() 
    
    # Insert colorbar 
    if num_distribution_plots == 0: 
        cbar = fig.colorbar(im, ax=axes, location='right')
    elif num_distribution_plots == 1 or num_distribution_plots == ncols: 
        cbar = fig.colorbar(im, ax=axes, location='right', 
            shrink=(fsize[0]/nrows)*0.099, anchor=(0, 1))
    else: 
        cbar = fig.colorbar(im, ax=axes, location='left')
        
    # plot distributions
    for i, ax in enumerate(axes_with_histograms):
        # get the right data for distribution plot
        hist_data = [data[j][2] for j in hist2maps[i]]
        # get the right data set labels 
        hist_labels = [labels[j] for j in hist2maps[i]]
        # make boxplots
        sns.boxplot(data=hist_data, ax=ax, width=len(axes_with_maps)*0.09)  
        ax.set_xticks(np.arange(len(hist_labels)))
        ax.set_xticklabels(hist_labels, rotation=90) 
        if num_distribution_plots == 1 or num_distribution_plots == nrows:
            ax.yaxis.tick_right()
            
    if add_subplotlabels:
        for i,ax in enumerate(axes.flat):                
            #print('add subplotlabel', alphabet[i], 'at', 
            #      subplotlabelx[i], subplotlabely[i])
            ax.text(subplotlabelx[i], subplotlabely[i], 
                    alphabet[i], transform=ax.transAxes)
    #else:
    #    #print('add_subplotlabels', add_subplotlabels)

    return fig, axes, cbar