import sys, time
import numpy as np

sys.path.append('../libs/')
from simulation import *
from graph_models import *
from inference_methods import *
from matrix_methods import *
from plotter import *
from simwrapper import *

###############################################################################
def tileList(li, n):
    '''Equivalent to `numpy.tile` but for python lists.
    
    Parameters
    ----------
    li : a list
        A list to be tiled.
        
    n : integer
        Number of tilings.
        
    Returns
    -------
    tl : list
        A list concatenation of `n` instances of the list `li`.'''
    
    tl = [x for i in range(n) for x in li]
    return tl


###############################################################################
def repeatList(li, n):
    '''Equivalent to `numpy.repeat` but for python lists.
    
    Parameters
    ----------
    li : a list
        A list to be tiled.
        
    n : integer
        Number of repetitions of list elements.
        
    Returns
    -------
    rl : list
        A list with `n` repetitions of each element of `li`.'''
    
    rl = [x for x in li for i in range(n)]
    return rl


###############################################################################
def array2string(arr, abbrv=False, max_length=100):
    '''Convert an array to a string without square brackets, linebreaks, or
    spaces (unless these appear in elements of the array).
    
    Parameters
    ----------
    arr : a numpy array
        A numpy array to be converted to a string.
        
    Returns
    -------
    s : string
        A string concatenation of all elements of `arr` that includes hyphens
        as delimiters.
    '''
    s = str(arr)
    s = s.replace(' ', '-')
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('\n', '')
    if abbrv==True:
        if len(s)>max_length:
            s = s[:max_length//2-1] +'----'+ s[len(s)-(max_length//2-1):]
    return s
