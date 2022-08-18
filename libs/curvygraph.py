# Copyright (C) 2020-2021
# Alice Schwarze <schwarze@uw.edu>

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
#  * Neither the name of the software's developers nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, FancyArrow, Patch, FancyBboxPatch
from matplotlib.text import Text
#from matplotlib.font_manager import FontProperties
import numpy as np
from collections.abc import Iterable
import copy
from matplotlib.font_manager import FontProperties
from shapely.geometry.polygon import LinearRing
from shapely.geometry import MultiLineString

################################################################################
#### Helper functions ##########################################################
################################################################################

def plotGrid(x_sep, y_sep, ax=None, color=(0.95, 0.95, 0.95), invert=False,
             zorder=0):
    '''Draw a checkerboard pattern in a matplotlib axis. `x_sep` defines theset
    separating lines along the horizontal axis, `y_sep` defines the separting 
    lines along the vertical axis.'''
    
    if ax is None: ax = plt.gca()
        
    x_sep = sorted(x_sep)
    y_sep = sorted(y_sep)
    
    for i in range(len(x_sep)-1):
        for j in range(len(y_sep)-1):
            if invert:
                if not((i+j) % 2):
                    r = Rectangle((x_sep[i], y_sep[j]), 
                                  x_sep[i+1]-x_sep[i], y_sep[j+1]-y_sep[j], 
                                  color=color, ec=None, zorder=zorder)
                    ax.add_patch(r)
            else:
                if (i+j) % 2:
                    r = Rectangle((x_sep[i], y_sep[j]), 
                                  x_sep[i+1]-x_sep[i], y_sep[j+1]-y_sep[j], 
                                  color=color, ec=None, zorder=zorder)
                    ax.add_patch(r)
                    

def array_as_triples(A):
    '''Represent an array as a list of triples (i,j,value) of all non-zero
    values.'''
    
    # get coordinates
    x,y = A.nonzero()
    
    # get values
    z = A[A!=0]
    
    # get list
    ret = list(zip(x,y,z))
    
    return ret


def edge_directions_from_adjacency(A):
    '''Get a list of triples indicating edges and their directions in the form 
    `(low_node_index, high_node_index, direction)`, where `direction` is either
    1 (if edge start at `low_node_index` and ends at `high_node_index`), -1 (if 
    edge start at `high_node_index` and ends at `low_node_index`), or 2 (if the
    edge is bidirectional).'''
    
    edge_table = np.zeros((len(A),len(A)))
    
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i,j]:
                if i==j:
                    # self edges have are unidirectional as default
                    edge_table[i,i] = 1
                elif A[j,i]:
                    #bidirectional edge between i and j
                    edge_table[max([i,j]),min([i,j])] = 2 
                else:
                    # unidirectional edge from j to i
                    if j < i:
                        # arrow goes from low ID to high ID
                        edge_table[i,j] = 1
                    elif j > i:
                        # arrow goes from high ID to low ID
                        edge_table[j,i] = -1
                            
    edge_list = array_as_triples(edge_table)  
    
    return edge_list


def degSin(x):
    '''Sin for argument in degrees.'''
    return np.sin(x*np.pi/180)


def degCos(x):
    '''Cos for argument in degrees.'''
    return np.cos(x*np.pi/180)


def degArcSin(x):
    '''ArcSin with result in degrees.'''
    return np.arcsin(x)/np.pi*180


def degArcCos(x):
    '''ArcCos with result in degrees.'''
    return np.arccos(x)/np.pi*180


def degArcTan2(x,y):
    '''ArcTan2 with result in degrees.'''
    return np.arctan2(x,y)/np.pi*180


def ellipse_polyline(center, a, b, angle, n=100, max_angle=360, offset=0):
    '''Make an ellipse polyline with `n` line segments.'''
            
    # parameterization of a circle    
    t = np.linspace(offset/180*np.pi, (offset+max_angle)/180*np.pi, n,
                    endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    
    # stretch and tilt to get ellipse
    angle = np.deg2rad(angle)
    sa = np.sin(angle)
    ca = np.cos(angle)
    poly = np.empty((n, 2))
    poly[:, 0] = center[0] + a * ca * ct - b * sa * st
    poly[:, 1] = center[1] + a * sa * ct + b * ca * st
    
    return poly


def polyline_intersection(a, b):
    '''Compute intersection point between two polylines. If there are several
    intersection points, returns only the first one found.'''
    
    # a and b are lists of points defining polylines
    ea = MultiLineString(list(zip(a[:-1],a[1:])))
    eb = MultiLineString(list(zip(b[:-1],b[1:])))
    
    # for bug fixing purposes
    #la, = plt.plot(a[:,0],a[:,1]) #, marker='x')
    #lb, = plt.plot(b[:,0],b[:,1]) #, marker='x')
    
    # get intersections
    mp = ea.intersection(eb)
    
    try:
        ret = np.array([mp.x, mp.y])
        '''If this line runs into an error it is because there was no 
        intersection between the node arc and edge arc. If you were trying to
        draw a self-edge, try increasing (or decreasing) the self-edge radius.''' 
    except:
        # if there are multiple solutions, take the first
        print(mp)
        ret = np.array([mp[0].x, mp[0].y])
      
    # remove the bug fixer plots
    #la.remove()
    #lb.remove()
        
    return ret


def angle_from_point(p, center=(0,0)):
    '''Compute angle of the pointer from center to point `p`.'''
    v = np.array(p)-np.array(center)
    angle = degArcTan2(v[1],v[0]) #TODO: is this the right way round?
    return angle


def point_from_angle(center, hAxis, vAxis, tilt, angle):
    '''Compute point on an ellipse from a given angle.'''
    
    shifted_angle = angle-tilt
    
    # In ellipse-centric coordinates, we parameterize a line passing through
    # (0,0) at the angle `shifted_angle` with a parameter t:
    # line(t) = t*np.array([degCos(shifted_angle), degSin(shifted_angle)])
    # Then, at the intersection between ellipse and line, we have
    t = np.sqrt(1/((degCos(shifted_angle)/hAxis)**2
                   +(degSin(shifted_angle)/vAxis)**2))
    # Do I need to look at cases for the root?
    
    # This gives the desired point in ellipse-centered coordinates
    p = t*np.array([degCos(shifted_angle), degSin(shifted_angle)])
    
    # rotate the point by ellipse' tilt
    p = np.matmul(rotationMatrix(-tilt),p)
    
    # shift point to get point in graph-centric coordinates
    p = p + np.array(center)
    
    return p
    
    
def rotationMatrix(angle):
    '''Generate a 2x2 rotation matrix for a given angle.'''
    
    rad = angle/180*np.pi
    R = np.array([[degCos(angle),degSin(angle)],
                  [-degSin(angle),degCos(angle)]])
    return R


def fillCircle(ax, center=(0,0), radius=0.2, color='white', alpha=1.0,
               zorder=0, face_kwargs={}):
    '''Draw a filled circle in axis `ax`.'''
      
    # define curve to be filled
    theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    # draw filled circle
    ax.fill(x, y, color=color, **face_kwargs, alpha=alpha, zorder=zorder)
    
    return None


################################################################################            
#### CurvyHead class ###########################################################            
################################################################################            
        
class CurvyHead:
    '''Arrow heads for CurvyEdge objects.'''
    
    def __init__(self, pad=0.0, length=0.1, width=0.1, shape='full', 
                 facecolor=None, linecolor=None, color=None, alpha=1.0,
                 zorder=0,kwargs={}):
        
        # get properties from keyword arguments
        self.pad = pad # float
        self.length = length # float
        self.width = width # float
        self.shape = shape # string in ['full', 'left', 'right']
        self.alpha = alpha # float in [0,1]
        self.zorder = zorder # integer
        
        self.kwargs = kwargs # dictionary
        # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.arrow.html
        
        # set head colors
        if facecolor is not None:
            self.facecolor = facecolor #string
        elif color is not None:
            self.facecolor = color #string
        else:
            self.facecolor = 'blue'
        if linecolor is not None:
            self.linecolor = linecolor #string
        elif color is not None:
            self.linecolor = color
        else:
            self.linecolor = 'blue' # blue is the default    
        
        
    def copy(self):
        '''Make a hard copy of a CurvyHead.'''
        
        new = CurvyHead(pad=self.pad, length=self.length, width=self.width, 
                        shape=self.shape, facecolor=self.facecolor, 
                        linecolor=self.linecolor, alpha=self.alpha,
                        zorder=self.zorder, kwargs=self.kwargs)
        return new
    
    
    # set functions
    def set_pad(self, pad):
        self.pad = pad
        
        
    def set_length(self, length):
        self.length = length
        
        
    def set_width(self, width):
        self.width = width
        
        
    def set_shape(self, shape):
        self.shape = shape
        
        
    def set_alpha(self, alpha):
        self.alpha = alpha
        
        
    def set_zorder(self, zorder):
        self.zorder = zorder
        
        
    def set_kwargs(self, kwargs):
        self.kwargs = kwargs
        
        
    def set_facecolor(self, fc):
        self.facecolor = fc
        
        
    def set_linecolor(self, ec):
        self.linecolor = ec
        
        
    def set_color(self, c):
        self.facecolor = c
        self.linecolor = c
        
        
    # def get functions
    def get_pad(self):
        return self.pad

    
    def get_length(self):
        return self.length

    
    def get_width(self):
        return self.width

    
    def get_shape(self):
        return self.shape

    
    def get_alpha(self):
        return self.alpha

    
    def get_zorder(self):
        return self.zorder

    
    def get_kwargs(self):
        return self.kwargs    
    
    
    def get_facecolor(self):
        return self.facecolor

    
    def get_linecolor(self):
        return self.linecolor
    
    
    def get_color(self):
        return (self.facecolor, self.linecolor)

    
################################################################################

def get_head_list(d, heads_like=[CurvyHead()]):
    '''Create a list of with two elements that are either None or a CurvyHead.
    The output can be used for the `heads` keyword for creating a CurvyEdge. If
    `d=1`, the arrow points forward (i.e., from low node index to high node 
    index), if `d=-1` the arrow points backward (i.e., from low node index to 
    high node index). `d=2` indicates a bidirectional arrow.'''
    
    # remove None-types from heads_like
    heads_like = [h for h in heads_like if h is not None]
    if len(heads_like)==0:
        # all heads in heads_like were none-type, so return none-type
        ret = [None, None]
    else:
        # produce new heads list
        if d == 1:
            ret = [None, heads_like[-1].copy()]
        elif d == -1:
            ret = [heads_like[0].copy(), None]
        elif d == 2:
            ret = [heads_like[0].copy(), heads_like[-1].copy()]
        else:
            raise ValueError('Input argument d must be in [-1,1,2]')
        
    return ret


################################################################################   
#### CurvyLabel class ##########################################################            
################################################################################            

class CurvyLabel:
    '''Text labels for CurvyNode and CurvyEdge objects.'''
    
    def __init__(self, text='', pad=0.0, size=12, color='k', alpha=1.0,
                 zorder=0, alignment=['center','center'], position='outside', 
                 shift=[0,0], kwargs={}):
        
        self.text = text # string
        self.pad = pad # float
        self.size = size # integer
        self.color = color # string
        self.alpha = alpha # float in [0,1]
        self.zorder = zorder # integer
        self.alignment = alignment # 2-tuple of alignment values
        self.shift=shift # list of two numbers
        
        self.position = position 
        # string in ['inside', 'outside', 'clockwise', 'counterclockwise']
        
        self.kwargs = kwargs # dictionary
        # compare https://matplotlib.org/api/text_api.html#matplotlib.text.Text
    
    
    def copy(self):
        '''Make a hard copy of a CurvyLabel.'''
        
        new = CurvyLabel(text=self.text, pad=self.pad, size=self.size,  
                         color=self.color, alpha=self.alpha, zorder=self.zorder, 
                         alignment=self.alignment, position=self.position,
                         shift=self.shift,
                         kwargs=self.kwargs)
        return new
    

    # def set functions
    def set_text(self, text):
        self.text = text

        
    def set_pad(self, pad):
        self.pad = pad
        
        
    def set_size(self, size):
        self.size = size
        
        
    def set_color(self, color):
        self.color = color
        
        
    def set_alpha(self, alpha):
        self.alpha = alpha
        
        
    def set_zorder(self, zorder):
        self.zorder = zorder
        
        
    def set_alignment(self, alignment):
        self.alignment = alignment
        
        
    def set_position(self, position):
        self.position = position
        
        
    def set_shift(self, shift):
        self.shift = shift

        
    def set_kwargs(self, kwargs):
        self.kwargs = kwargs
        
        
    # def get functions
    def get_text(self):
        return self.text
    
    
    def get_pad(self):
        return self.pad

    
    def get_size(self):
        return self.size

    
    def get_color(self):
        return self.color

    
    def get_alpha(self):
        return self.alpha

    
    def get_zorder(self):
        return self.zorder
    
    
    def get_alignment(self):
        return self.alignment

    
    def get_position(self):
        return self.position

    
    def get_kwargs(self):
        return self.kwargs    
    
    
################################################################################
#### CurvyGraph class ##########################################################
################################################################################

class CurvyGraph:
    '''CurvyGraph are graphical objects of ring or star graphs (or combinations 
    of ring and star graphs). Edges in CurvyGraphs can be curvy (or straight).
    Nodes are always positioned on the circumference or the center of a ring.'''
    
    def __init__(self, center=(0,0), radius=0.5, tilt=0, nodes={}, edges={},
                 key=0):

        self.center = center # 2-tuple of floats
        self.radius = radius # float
        self.tilt = tilt # integer (is angle in degrees)
        self.nodes = nodes # list of CurvyNode objects
        self.edges = edges # list of CurvyEdge objects
        self.key = key # integer
        
    
    def copy(self, copy_nodes=True, copy_edges=True, copy_labels=True, 
             copy_heads=True):
        '''Make a hard copy of a CurvyGraph.'''
        
        if copy_nodes:
            # make hard copies of nodes
            new_nodes = {i : self.nodes[i].copy() for i in self.nodes.keys()}
        else:
            new_nodes = self.nodes
            
        if copy_edges:
            # make hard copies of edges - while keeping node dependencies
            new_edges = {}
            for i in self.edges.keys():
                ne = self.edges[i].copy(copy_labels=copy_labels, 
                                        copy_heads=copy_heads, 
                                        copy_nodes=False) 
                ne.source = new_nodes[ne.source.key]
                ne.target = new_nodes[ne.target.key]
                new_edges[i] = ne
        else:
            new_edges = self.edges
        
        # make new CurvyGraph
        new = CurvyGraph(nodes=new_nodes, edges=new_edges, center=self.center,
                         radius=self.radius, tilt=self.tilt, ID=self.ID)
        return new
    
    
    # def set functions
    
    def _get_node_list(self, node_keys): #TODO
        
        if node_keys is None:
            list_of_nodes = [self.nodes[i] for i in self.nodes.keys()]
        elif isinstance(node_keys, (int, float, np.integer, np.floating)):
            list_of_nodes = [self.nodes[int(node_keys)]]
        elif isinstance(node_keys, (str)):
            try:
                list_of_nodes = [self.nodes[int(node_keys)]]
            except:
                raise ValueError('Argument node_keys must be a node key (integer or string) or a list of node keys.')
        else:
            try: 
                list_of_nodes = [self.nodes[i] for i in node_keys] 
                # this should work for lists, generators, and arrays
            except:
                raise ValueError('Argument node_keys must be a node key (integer or string) or a list of node keys.')
                
        return list_of_nodes
    
    
    def _get_edge_list(self, edge_keys): #TODO
        
        if edge_keys is None:
            list_of_edges = [self.edges[i] for i in self.edges.keys()]
        elif isinstance(edge_keys, (int, float, np.integer, np.floating)):
            list_of_edges = [self.edges[edge_keys]]
        elif isinstance(edge_keys, (str)):
            try:
                list_of_edges = [self.edges[int(edge_keys)]]
            except:
                raise ValueError('Argument edge_keys must be a edge key (integer or string) or a list of edge keys.')
        else:
            try: 
                list_of_edges = list(nodes) 
                # this should work for lists, generators, and arrays
            except:
                raise ValueError('Argument edge_keys must be a edge key (integer) or a list of edge keys.')
                
        return list_of_edges
    
    
    def set_center(self, center):
        self.center = center
        for k in self.nodes.keys():
            self.nodes[k].graph_center = center
        for k in self.edges.keys():
            self.edges[k].graph_center = center

        
    def set_radius(self,radius):
        self.radius = radius
        for k in self.nodes.keys():
            self.nodes[k].graph_radius = radius
        for k in self.edges.keys():
            self.edges[k].graph_radius = radius
                
        
    def set_tilt(self, tilt):
        self.tilt = tilt
        for k in self.nodes.keys():
            self.nodes[k].graph_tilt = tilt
        for k in self.edges.keys():
            self.edges[k].graph_tilt = tilt

        
    def set_nodes(self, nodes, copy_nodes=True):
        if isinstance(nodes, dict):
            if copy_nodes:
                self.nodes = {i:nodes[i].copy() for i in nodes.keys()}
            else:
                self.nodes = nodes
        elif isinstance(nodes, list):
            if copy_nodes:
                self.nodes = {i:nodes[i].copy() for i in range(len(nodes))}
            else:
                self.nodes = {i:nodes[i] for i in range(len(nodes))}
        else:
            raise ValueError("Unknown type for argument 'nodes'. Must be list or dictionary.")
            
        
    def set_edges(self, edges, copy_edges=True):
        if isinstance(edges, dict):
            if copy_edges:
                self.edges = {i:edges[i].copy() for i in edges.keys()}
            else:
                self.edges = edges
        elif isinstance(edges, list):
            if copy_edges:
                self.edges = {i:edges[i].copy() for i in range(len(edges))}
            else:
                self.edges = {i:edges[i] for i in range(len(edges))}
        else:
            raise ValueError("Unknown type for argument 'edges'. Must be list or dictionary.")
        
        
    def set_color(self, color, nodes=None, edges=None, include_edges=True, include_nodes=True):
        
        if include_nodes:
            for n in self._get_node_list(nodes):
                n.set_color(color)
        
        if include_edges:
            for e in self._get_edge_list(edges):
                e.set_color(color)

                
    def set_linewidth(self, lw, nodes=None, edges=None,
                       include_edges=True, include_nodes=True):
        
        if include_nodes:
            for n in self._get_node_list(nodes):
                n.set_linewidth(lw)
        
        if include_edges:
            for e in self._get_edge_list(edges):
                e.set_linewidth(lw)


    def set_alpha(self, alpha, nodes=None, edges=None, include_edges=True, include_nodes=True):
        
        if include_nodes:
            for n in self._get_node_list(nodes):
                n.set_alpha(alpha)
        
        if include_edges:
            for e in self._get_edge_list(edges):
                e.set_alpha(alpha)


    def set_zorder(self, zorder, nodes=None, edges=None, include_edges=True, include_nodes=True):
        
        if include_nodes:
            for n in self._get_node_list(nodes):
                n.set_zorder(zorder)
        
        if include_edges:
            for e in self._get_edge_list(edges):
                e.set_zorder(zorder)
                
    
    def set_node_label_text(self, text, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(text, (str)):
            for n in list_of_nodes:
                n.set_label_text(text)
        else:
            try:
                texts = list(text)
            except:
                raise ValueError('Unknown type for argument text.')
            if len(text)<len(list_of_nodes):
                raise ValueError('Iterable text must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_label_text(texts[i])


    def set_node_label_pad(self, pad, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(pad, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_label_pad(pad)
        else:
            try:
                pads = list(pad)
            except:
                raise ValueError('Unknown type for argument pad.')
            if len(pad)<len(list_of_nodes):
                raise ValueError('Iterable pad must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_label_pad(pads[i])
                
    def set_node_label_shift(self, shift, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(shift, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_label_shift([shift,shift])
        else:
            try:
                shifts = list(shift)
            except:
                raise ValueError('Unknown type for argument `shift`.')
                
            if isinstance(shifts[0], (int, float, np.integer, np.floating)):
                for n in list_of_nodes:
                    n.set_label_shift(shifts)
            else:    
                if len(shifts)<len(list_of_nodes):
                    raise ValueError('Iterable list of shifts must be same length or longer than list of nodes.')
                for i, n in enumerate(list_of_nodes):
                    n.set_label_shift(shifts[i])                


    def set_node_label_size(self, size, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(size, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_label_size(size)
        else:
            try:
                sizes = list(size)
            except:
                raise ValueError('Unknown type for argument size.')
            if len(size)<len(list_of_nodes):
                raise ValueError('Iterable size must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_label_size(sizes[i])


    def set_node_label_color(self, color, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(color, (str)):
            for n in list_of_nodes:
                n.set_label_color(color)
        else:
            try:
                colors = list(color)
            except:
                raise ValueError('Unknown type for argument color.')
            if len(color)<len(list_of_nodes):
                raise ValueError('Iterable color must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_label_color(colors[i])


    def set_node_label_alpha(self, alpha, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(alpha, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_label_alpha(alpha)
        else:
            try:
                alphas = list(alpha)
            except:
                raise ValueError('Unknown type for argument alpha.')
            if len(alpha)<len(list_of_nodes):
                raise ValueError('Iterable alpha must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_label_alpha(alphas[i])


    def set_node_label_zorder(self, zorder, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(zorder, (int, np.integer)):
            for n in list_of_nodes:
                n.set_label_zorder(zorder)
        else:
            try:
                zorders = list(zorder)
            except:
                raise ValueError('Unknown type for argument zorder.')
            if len(zorder)<len(list_of_nodes):
                raise ValueError('Iterable zorder must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_label_zorder(zorders[i])

            
    def set_node_label_alignment(self, alignment, nodes=None, copy_alignment=True):
        if copy_alignment:
            for n in self._get_node_list(nodes):
                n.label.alignment = [a for a in alignment]
        else:
            for n in self._get_node_list(nodes):
                n.label.alignment = alignment
                
                
    def set_node_label_position(self, position, nodes=None, copy_position=True):
        if copy_position:
            for n in self._get_node_list(nodes):
                n.label.position = [p for p in position]
        else:
            for n in self._get_node_list(nodes):
                n.label.position = position


    def set_node_label_kwargs(self, kwargs, nodes=None, copy_kwargs=True):
        if copy_kwargs:
            for n in self._get_node_list(nodes):
                n.label.kwargs = kwargs.copy()
        else:
            for n in self._get_node_list(nodes):
                n.label.kwargs = kwargs


    def set_edge_label_text(self, text, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(text, (str)):
            for e in list_of_edges:
                e.set_label_text(text)
        else:
            try:
                texts = list(text)
            except:
                raise ValueError('Unknown type for argument text.')
            if len(texts)<len(list_of_edges):
                raise ValueError('Iterable text must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_label_text(texts[i])

    def set_edge_label_shift(self, shift, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(shift[0], (int, float, np.integer, np.floating)): # not a nested list
            for e in list_of_edges:
                e.set_label_shift(shift)
        else:
            try:
                shifts = list(shift)
            except:
                raise ValueError('Unknown type for argument shift.')
            if len(shifts)<len(list_of_edges):
                raise ValueError('Iterable shift must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_label_shift(shifts[i])
                
                
    def set_edge_label_pad(self, pad, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(pad, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_label_pad(pad)
        else:
            try:
                pads = list(pad)
            except:
                raise ValueError('Unknown type for argument pad.')
            if len(pad)<len(list_of_edges):
                raise ValueError('Iterable pad must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_label_pad(pads[i])


    def set_edge_label_size(self, size, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(size, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_label_size(size)
        else:
            try:
                sizes = list(size)
            except:
                raise ValueError('Unknown type for argument size.')
            if len(size)<len(list_of_edges):
                raise ValueError('Iterable size must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_label_size(sizes[i])


    def set_edge_label_color(self, color, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(color, (str)):
            for e in list_of_edges:
                e.set_label_color(color)
        else:
            try:
                colors = list(color)
            except:
                raise ValueError('Unknown type for argument color.')
            if len(color)<len(list_of_edges):
                raise ValueError('Iterable color must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_label_color(colors[i])


    def set_edge_label_alpha(self, alpha, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(alpha, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_label_alpha(alpha)
        else:
            try:
                alphas = list(alpha)
            except:
                raise ValueError('Unknown type for argument alpha.')
            if len(alpha)<len(list_of_edges):
                raise ValueError('Iterable alpha must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_label_alpha(alphas[i])


    def set_edge_label_zorder(self, zorder, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(zorder, (int, np.integer)):
            for e in list_of_edges:
                e.set_label_zorder(zorder)
        else:
            try:
                zorders = list(zorder)
            except:
                raise ValueError('Unknown type for argument zorder.')
            if len(zorder)<len(list_of_edges):
                raise ValueError('Iterable zorder must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_label_zorder(zorders[i])

            
    def set_edge_label_alignment(self, alignment, edges=None, 
                                 copy_alignment=True):
        if copy_alignment:
            for e in self._get_edge_list(edges):
                for l in e.labels:
                    l.alignment = [a for a in alignment]
        else:
            for e in self._get_edge_list(edges):
                for l in e.labels:
                    l.alignment = alignment
                
                
    def set_edge_label_position(self, position, edges=None, copy_position=True):
        if copy_position:
            for e in self._get_edge_list(edges):
                for l in e.labels:
                    l.position = [p for p in position]
        else:
            for e in self._get_edge_list(edges):
                for l in e.labels:
                    l.position = position
                
                
    def set_edge_label_kwargs(self, kwargs, edges=None, copy_kwargs=True):
        if copy_kwargs:
            for e in self._get_edge_list(edges):
                for l in e.labels:
                    l.kwargs = kwargs.copy()
        else:
            for e in self._get_edge_list(edges):
                for l in e.labels:
                    l.kwargs = kwargs

            
    def set_label_size(self, size, include_edges=True, include_nodes=True, 
                       nodes=None, edges=None):
        if include_nodes:
            self.set_node_label_size(size, nodes=nodes)
        if include_edges:
            self.set_edge_label_size(size, edges=edges)


    def set_label_color(self, color, include_edges=True, include_nodes=True, 
                        nodes=None, edges=None):
        if include_nodes:
            self.set_node_label_color(color, nodes=nodes)
        if include_edges:
            self.set_edge_label_color(color, edges=edges)

            
    def set_label_alpha(self, alpha, include_edges=True, include_nodes=True, 
                        nodes=None, edges=None):
        if include_nodes:
            self.set_node_label_alpha(alpha, nodes=nodes)
        if include_edges:
            self.set_edge_label_alpha(alpha, edges=edges)

            
    def set_label_zorder(self, zorder, include_edges=True, include_nodes=True, 
                         nodes=None, edges=None):
        if include_nodes:
            self.set_node_label_zorder(zorder, nodes=nodes)
        if include_edges:
            self.set_edge_label_zorder(zorder, edges=edges)

            
    def set_label_alignment(self, alignment, include_edges=True, 
                            include_nodes=True, nodes=None, edges=None, 
                            copy_alignment=True):
        if include_nodes:
            self.set_node_label_alignment(alignment, nodes=nodes, 
                                          copy_alignment=copy_alignment)
        if include_edges:
            self.set_edge_label_alignment(alignment, edges=edges,
                                          copy_alignment=copy_alignment)

            
    def set_label_position(self, position, include_edges=True,
                           include_nodes=True, nodes=None, edges=None,
                           copy_position=True):
        if include_nodes:
            self.set_node_label_position(position, nodes=nodes,
                                         copy_position=copy_position)
        if include_edges:
            self.set_edge_label_position(position, edges=edges, 
                                         copy_position=copy_position)

            
    def set_label_kwargs(self, kwargs, include_edges=True, include_nodes=True, 
                         nodes=None, edges=None, copy_kwargs=True):
        if include_nodes:
            self.set_node_label_kwargs(kwargs, nodes=nodes, 
                                       copy_kwargs=copy_kwargs)
        if include_edges:
            self.set_edge_label_kwargs(kwargs, edges=edges, 
                                       copy_kwargs=copy_kwargs)

       
    def set_node_label_properties(self, nodes=None, text=None, pad=None, 
                                  size=None, color=None, alpha=None, 
                                  zorder=None, alignment=None, position=None, 
                                  kwargs=None, copy_kwargs=True):
                             
        for n in self._get_node_list(nodes):
            n.set_label_properties(text=text, pad=pad, size=size, 
                                   color=color, alpha=alpha, 
                                   zorder=zorder, alignment=alignment, 
                                   position=position, kwargs=kwargs, 
                                   copy_alignment=copy_alignment,
                                   copy_position=copy_position,
                                   copy_kwargs=copy_kwargs)
        
        
    def set_edge_label_properties(self, edges=None, text=None, pad=None, 
                                  size=None, color=None, alpha=None, 
                                  zorder=None, alignment=None, position=None, 
                                  kwargs=None, copy_kwargs=True):
                             
        for e in self._get_edge_list(edges):
            e.set_label_properties(text=text, pad=pad, size=size, color=color, 
                                   alpha=alpha, zorder=zorder, 
                                   alignment=alignment, position=position, 
                                   kwargs=kwargs, copy_alignment=copy_alignment,
                                   copy_position=copy_position,
                                   copy_kwargs=copy_kwargs)

                
    def set_label_properties(self, nodes=None, edges=None, include_nodes=True, 
                             include_edges=True, text=None, pad=None, size=None,
                             color=None, alpha=None, zorder=None,
                             alignment=None, position=None, kwargs=None,
                             copy_kwargs=True, copy_alignment=True, 
                             copy_position=True):
            if include_nodes:
                self.set_node_label_properties(nodes=nodes, text=text, pad=pad, 
                                               size=size, color=color, 
                                               alpha=alpha, zorder=zorder, 
                                               alignment=alignment, 
                                               position=position, kwargs=kwargs,
                                               copy_alignment=copy_alignment,
                                               copy_position=copy_position,
                                               copy_kwargs=copy_kwargs)
            if include_edges:
                self.set_edge_label_properties(edges=edges, text=text, pad=pad,
                                               size=size, color=color, 
                                               alpha=alpha, zorder=zorder, 
                                               alignment=alignment, 
                                               position=position, kwargs=kwargs,
                                               copy_alignment=copy_alignment,
                                               copy_position=copy_position,
                                               copy_kwargs=copy_kwargs)
    
    
    def set_node_angle(self, angle, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(angle, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_angle(angle)
        else:
            try:
                angles = list(angle)
            except:
                raise ValueError('Unknown type for argument angle.')
            if len(angles)<len(list_of_nodes):
                raise ValueError('Iterable angle must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_angle(angles[i])


    def set_node_radius(self, radius, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(radius, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_radius(radius)
        else:
            try:
                radii = list(radius)
            except:
                raise ValueError('Unknown type for argument radius.')
            if len(radii)<len(list_of_nodes):
                raise ValueError('Iterable radius must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_radius(radii[i])


    def set_node_linewidth(self, linewidth, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(linewidth, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_linewidth(linewidth)
        else:
            try:
                linewidths = list(linewidth)
            except:
                raise ValueError('Unknown type for argument linewidth.')
            if len(linewidths)<len(list_of_nodes):
                raise ValueError('Iterable linewidth must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_linewidth(linewidths[i])


    def set_node_facecolor(self, facecolor, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(facecolor, (str)):
            for n in list_of_nodes:
                n.set_facecolor(facecolor)
        else:
            try:
                facecolors = list(facecolor)
            except:
                raise ValueError('Unknown type for argument facecolor.')
            if len(facecolors)<len(list_of_nodes):
                raise ValueError('Iterable facecolor must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_facecolor(facecolors[i])


    def set_node_linecolor(self, linecolor, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(linecolor, (str)):
            for n in list_of_nodes:
                n.set_linecolor(linecolor)
        else:
            try:
                linecolors = list(linecolor)
            except:
                raise ValueError('Unknown type for argument linecolor.')
            if len(linecolors)<len(list_of_nodes):
                raise ValueError('Iterable linecolor must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_linecolor(linecolors[i])


    def set_node_color(self, color, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(color, (str)):
            for n in list_of_nodes:
                n.set_color(color)
        else:
            try:
                colors = list(color)
            except:
                raise ValueError('Unknown type for argument color.')
            if len(colors)<len(list_of_nodes):
                raise ValueError('Iterable color must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_color(colors[i])


    def set_node_label(self, label, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if hasattr(label, '__iter__'):
            try:
                labels = list(label)
            except:
                raise ValueError('Unknown type for argument label.')
            if len(labels)<len(list_of_nodes):
                raise ValueError('Iterable source must be same length or longer than list of edges.')
            for i, n in enumerate(list_of_nodes):
                n.set_label(labels[i])
        else:
            for n in list_of_edges:
                n.set_label(label)

                                
    def set_node_alpha(self, alpha, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(alpha, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_alpha(alpha)
        else:
            try:
                alphas = list(alpha)
            except:
                raise ValueError('Unknown type for argument alpha.')
            if len(alphas)<len(list_of_nodes):
                raise ValueError('Iterable alpha must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_alpha(alphas[i])


    def set_node_zorder(self, zorder, nodes=None):
        list_of_nodes = self._get_node_list(nodes)
        if isinstance(zorder, (int, float, np.integer, np.floating)):
            for n in list_of_nodes:
                n.set_zorder(zorder)
        else:
            try:
                zorders = list(zorder)
            except:
                raise ValueError('Unknown type for argument zorder.')
            if len(zorders)<len(list_of_nodes):
                raise ValueError('Iterable zorder must be same length or longer than list of nodes.')
            for i, n in enumerate(list_of_nodes):
                n.set_zorder(zorders[i])


    def set_node_kwargs(self, kwargs, nodes=None, copy_kwargs=True):
        list_of_nodes = self._get_node_list(nodes)
        if copy_kwargs:
            for n in list_of_nodes:
                n.kwargs = kwargs.copy()
        else:
            for n in list_of_nodes:
                n.kwargs = kwargs


    def set_node_properties(self, nodes=None, angle=None, radius=None, 
                            linewidth=None, color=None, facecolor=None, 
                            linecolor=None, label=None, alpha=None, zorder=None, 
                            face_kwargs=None, edge_kwargs=None, 
                            copy_labels=True, copy_kwargs=True):
        
        if angle is not None: 
            self.set_node_angle(angle, nodes=nodes)
        if radius is not None: 
            self.set_node_radius(radius, nodes=nodes)
        if linewidth is not None: 
            self.set_node_linewidth(linewidth, nodes=nodes)
        if color is not None: 
            self.set_node_color(color, nodes=nodes)
        if facecolor is not None: 
            self.set_node_facecolor(facecolor, nodes=nodes)
        if linecolor is not None: 
            self.set_node_linecolor(linecolor, nodes=nodes)
        if alpha is not None: 
            self.set_node_alpha(alpha, nodes=nodes)
        if zorder is not None: 
            self.set_node_zorder(zorder, nodes=nodes)
        if label is not None: 
            self.set_node_label(label, nodes=nodes, copy_labels=copy_labels)                     
        if face_kwargs is not None: 
            self.set_node_face_kwargs(face_kwargs, nodes=nodes, 
                                      copy_kwargs=copy_kwargs)
        if edge_kwargs is not None: 
            self.set_node_edge_kwargs(edge_kwargs, nodes=nodes,
                                      copy_kwargs=copy_kwargs)
            
            
    def set_edge_source(self, source, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if hasattr(source, '__iter__'):
            try:
                sources = list(source)
            except:
                raise ValueError('Unknown type for argument source.')
            if len(sources)<len(list_of_edges):
                raise ValueError('Iterable source must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_source(sources[i])
        else:
            for e in list_of_edges:
                e.set_source(source)


    def set_edge_target(self, target, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if hasattr(target, '__iter__'):
            try:
                targets = list(target)
            except:
                raise ValueError('Unknown type for argument target.')
            if len(targets)<len(list_of_edges):
                raise ValueError('Iterable target must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_target(targets[i])
        else:
            for e in list_of_edges:
                e.set_target(target)


    def set_edge_curvature(self, curvature, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(curvature, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_curvature(curvature)
        else:
            try:
                curvatures = list(curvature)
            except:
                raise ValueError('Unknown type for argument curvature.')
            if len(curvatures)<len(list_of_edges):
                raise ValueError('Iterable curvature must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_curvature(curvatures[i])


    def set_edge_doubling(self, doubling, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(doubling, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_doubling(doubling)
        else:
            try:
                doublings = list(doubling)
            except:
                raise ValueError('Unknown type for argument doubling.')
            if len(doublings)<len(list_of_edges):
                raise ValueError('Iterable doubling must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_doubling(doublings[i])


    def set_selfedge_radius(self, selfedge_radius, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(selfedge_radius, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_selfedge_radius(selfedge_radius)
        else:
            try:
                selfedge_radii = list(selfedge_radius)
            except:
                raise ValueError('Unknown type for argument selfedge_radius.')
            if len(colors)<len(list_of_edges):
                raise ValueError('Iterable selfedge_radius must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_selfedge_radius(selfedge_radii[i])


    def set_edge_color(self, color, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(color, (str)):
            for e in list_of_edges:
                e.set_color(color)
        else:
            try:
                colors = list(color)
            except:
                raise ValueError('Unknown type for argument color.')
            if len(colors)<len(list_of_edges):
                raise ValueError('Iterable color must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_color(colors[i])


    def set_edge_linewidth(self, linewidth, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(linewidth, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_linewidth(linewidth)
        else:
            try:
                linewidths = list(linewidth)
            except:
                raise ValueError('Unknown type for argument linewidth.')
            if len(linewidths)<len(list_of_edges):
                raise ValueError('Iterable linewidth must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_linewidth(linewidths[i])


    def set_edge_linestyle(self, linestyle, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(linestyle, (str)):
            for e in list_of_edges:
                e.set_linestyle(linestyle)
        else:
            try:
                linestyles = list(linestyle)
            except:
                raise ValueError('Unknown type for argument linestyle.')
            if len(linestyles)<len(list_of_edges):
                raise ValueError('Iterable linestyle must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_linestyle(linestyles[i])
                
    
    def set_edge_pad(self, pad, edges=None):
        list_of_edges = self._get_edge_list(edges)
        for e in list_of_edges:
            e.set_pad(pad)
            
            
    def set_edge_labels(self, labels, edges=None, copy_labels=True):
        list_of_edges = self._get_edge_list(edges)
        if copy_labels:
            for e in list_of_edges:
                e.labels = [l.copy() for l in labels]
        else:
            for e in list_of_edges:
                e.labels = labels


    def set_edge_heads(self, heads, edges=None, copy_heads=True):
        list_of_edges = self._get_edge_list(edges)
        if copy_heads:
            for e in list_of_edges:
                e.heads = [(h.copy() if h is not None else None) for h in heads]
        else:
            for e in list_of_edges:
                e.heads = heads
            

    def set_edge_alpha(self, alpha, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(alpha, (str)):
            for e in list_of_edges:
                e.set_alpha(alpha)
        else:
            try:
                alphas = list(alpha)
            except:
                raise ValueError('Unknown type for argument alpha.')
            if len(alphas)<len(list_of_edges):
                raise ValueError('Iterable alpha must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_alpha(alphas[i])

        for e in list_of_edges:
            e.alpha = alpha


    def set_edge_zorder(self, zorder, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(zorder, (str)):
            for e in list_of_edges:
                e.set_zorder(zorder)
        else:
            try:
                zorders = list(zorder)
            except:
                raise ValueError('Unknown type for argument zorder.')
            if len(zorders)<len(list_of_edges):
                raise ValueError('Iterable zorder must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_zorder(zorders[i])

        for e in list_of_edges:
            e.zorder = zorder


    def set_edge_kwargs(self, kwargs, edges=None, copy_kwargs=True):
        list_of_edges = self._get_edge_list(edges)
        if copy_kwargs:
            for e in list_of_edges:
                e.kwargs = kwargs.copy()
        else:
            for e in list_of_edges:
                e.kwargs = kwargs


    def set_edge_properties(self, edges=None, source=None, target=None, 
                            curvature=None, selfedge_radius=None, doubling=None, 
                            color=None, linewidth=None, linestyle=None, 
                            labels=None, heads=None, alpha=None, zorder=None, 
                            kwargs=None, copy_labels=True, copy_heads=True, 
                            copy_kwargs=True):
        
        if source is not None: 
            self.set_edge_source(source, edges=edges)
        if target is not None: 
            self.set_edge_target(target, edges=edges)
        if curvature is not None: 
            self.set_edge_curvature(curvature, edges=edges)
        if selfedge_radius is not None: 
            self.set_edge_selfedge_radius(selfedge_radius, edges=edges)
        if doubling is not None: 
            self.set_edge_doubling(doubling, edges=edges)
        if color is not None: 
            self.set_edge_color(color, edges=edges)
        if linewidth is not None: 
            self.set_edge_linewidth(linewidth, edges=edges)
        if linestyle is not None: 
            self.set_edge_linestyle(linestyle, edges=edges)
        if labels is not None: 
            self.set_edge_labels(labels, edges=edges, copy_labels=copy_labels)                    
        if heads is not None: 
            self.set_edge_heads(heads, edges=edges, copy_heads=copy_heads)
        if alpha is not None: 
            self.set_edge_alpha(alpha, edges=edges)
        if zorder is not None: 
            self.set_edge_zorder(zorder, edges=edges)
        if kwargs is not None: 
            self.set_edge_kwargs(kwargs, edges=edges, copy_kwargs=copy_kwargs)
                    
                    
    def set_head_pad(self, pad, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(pad, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_head_pad(pad, head=heads)
        else:
            try:
                pads = list(pad)
            except:
                raise ValueError('Unknown type for argument pad.')
            if len(pad)<len(list_of_edges):
                raise ValueError('Iterable pad must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_pad(pads[i], head=heads)


    def set_head_length(self, length, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(length, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_head_length(length, head=heads)
        else:
            try:
                lengths = list(length)
            except:
                raise ValueError('Unknown type for argument length.')
            if len(length)<len(list_of_edges):
                raise ValueError('Iterable length must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_length(lengths[i], head=heads)


    def set_head_width(self, width, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(width, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_head_width(width, head=heads)
        else:
            try:
                widths = list(width)
            except:
                raise ValueError('Unknown type for argument width.')
            if len(width)<len(list_of_edges):
                raise ValueError('Iterable width must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_width(widths[i], head=heads)


    def set_head_shape(self, shape, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(shape, (str)):
            for e in list_of_edges:
                e.set_head_shape(shape, head=heads)
        else:
            try:
                shapes = list(shape)
            except:
                raise ValueError('Unknown type for argument shape.')
            if len(shape)<len(list_of_edges):
                raise ValueError('Iterable shape must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_shape(shapes[i], head=heads)


    def set_head_facecolor(self, facecolor, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(facecolor, (str)):
            for e in list_of_edges:
                e.set_head_facecolor(facecolor, heads=heads)
        else:
            try:
                facecolors = list(facecolor)
            except:
                raise ValueError('Unknown type for argument facecolor.')
            if len(facecolor)<len(list_of_edges):
                raise ValueError('Iterable facecolor must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_facecolor(facecolors[i], head=heads)


    def set_head_linecolor(self, linecolor, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(linecolor, (str)):
            for e in list_of_edges:
                e.set_head_linecolor(linecolor, head=heads)
        else:
            try:
                linecolors = list(linecolor)
            except:
                raise ValueError('Unknown type for argument linecolor.')
            if len(linecolor)<len(list_of_edges):
                raise ValueError('Iterable linecolor must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_linecolor(linecolors[i], head=heads)


    def set_head_color(self, color, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(color, (str)):
            for e in list_of_edges:
                e.set_head_color(color, head=heads)
        else:
            try:
                colors = list(color)
            except:
                raise ValueError('Unknown type for argument color.')
            if len(color)<len(list_of_edges):
                raise ValueError('Iterable color must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_color(colors[i], head=heads)


    def set_head_alpha(self, alpha, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(alpha, (int, float, np.integer, np.floating)):
            for e in list_of_edges:
                e.set_head_alpha(alpha, head=heads)
        else:
            try:
                alphas = list(alpha)
            except:
                raise ValueError('Unknown type for argument alpha.')
            if len(alpha)<len(list_of_edges):
                raise ValueError('Iterable alpha must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_alpha(alphas[i], head=heads)


    def set_head_zorder(self, zorder, heads=None, edges=None):
        list_of_edges = self._get_edge_list(edges)
        if isinstance(zorder, (int, np.integer)):
            for e in list_of_edges:
                e.set_head_zorder(zorder, head=heads)
        else:
            try:
                zorders = list(zorder)
            except:
                raise ValueError('Unknown type for argument zorder.')
            if len(zorder)<len(list_of_edges):
                raise ValueError('Iterable zorder must be same length or longer than list of edges.')
            for i, e in enumerate(list_of_edges):
                e.set_head_zorder(zorders[i], head=heads)


    def set_head_kwargs(self, kwargs, heads=None, edges=None, copy_kwargs=True):
        list_of_edges = self._get_edge_list(edges)
        for e in list_of_edges:
            e.set_head_kwargs(kwargs, head=heads, copy_kwargs=copy_kwargs)


    def set_head_properties(self, head=None, edges=None, pad=None, length=None,
                            width=None, shape=None, facecolor=None, 
                            linecolor=None, color=None, alpha=None, zorder=None, 
                            kwargs=None,copy_kwargs=True):
        
        list_of_edges = self._get_edge_list(edges)
        
        for e in list_of_edges:
            e.set_head_properties(head=head, pad=pad, length=length, 
                                  width=width, shape=shape, facecolor=facecolor,
                                  linecolor=linecolor, color=color, alpha=alpha, 
                                  zorder=zorder, kwargs=kwargs,
                                  copy_kwargs=copy_kwargs)
            
                
    # get functions
    def get_center(self):
        return self.center
    
    
    def get_radius(self):
        return self.radius
    
    
    def get_tilt(self):
        return self.tilt

    
    def get_nodes(self):
        return self.nodes

    
    def get_edges(self):
        return self.edges
    
    
    def get_id(self):
        return self.ID

    
    # def other functions #TODO
    def add_node_like(self, angle, node):
        '''Add a node to a CurvyGraph at angle 'angle' that looks like 
        'node'.'''
        
        # make a copy of the model node (cannot use __init__ because CurvyNode 
        # class is not defined yet)
        new_node = node.copy()
        
        # copy graph properties
        new_node.graph_center = self.center
        new_node.graph_radius = self.radius
        new_node.graph_tilt = self.tilt
        
        # set angle
        new_node.angle = angle
        
        # get and set key
        key = max(self.nodes.keys())+1
        new_node.key = key
        
        # add new node to graph
        self.nodes[key] = new_node
        
        return key
    
    
    def add_nodes_like(self, angles, node):
        '''Add several nodes to a CurvyGraph at angles 'angles'. All added nodes
        look like 'node'.'''
        
        ret = []
        for a in angles:
            k = self.add_node_like(a, node)
            ret = ret + [k]
            
        return ret

    
    def remove_node(self, node_key):
        self.nodes.pop(node_key)
        
        
    def remove_nodes(self, node_keys):
        for k in node_keys:
            self.nodes.pop(k)
    
    
    def add_edge_like(self, source_key, target_key, edge):
        '''Add an edge between source and target. The added edge looks like 
        'edge'.'''
        
        # make a copy of the model edge (cannot use __init__ because CurvyEdge 
        # class is not defined yet)
        new_edge = edge.copy()
        
        # copy graph properties
        new_edge.graph_center = self.center
        new_edge.graph_radius = self.radius
        new_edge.graph_tilt = self.tilt
        
        # set source and target
        new_edge.source = self.nodes[source_key]
        new_edge.target = self.nodes[target_key]
        
        # get and set key
        if len(self.edges.keys()):
            key = max(self.edges.keys())+1
        else:
            key = 0
        new_edge.key = key
        
        # add new edge to graph
        self.edges[key] = new_edge
        
        return key        
    
    
    def add_edges_like(self, sources, targets, edge):
        '''Add several edges to a CurvyGraph between specified source and target
        nodes. All added edges look like 'edge'.'''
        
        ret = []
        for i in range(min([len(sources), len(targets)])):
            k = self.add_edge_like(sources[i], targets[i], edge)
            ret = ret + [k]
            
        return ret
    
    
    def add_selfedge_like(self, node, edge):
        '''Add a self-selfedge to a CurvyGraph at node 'node'. Newly added edge
        looks like 'edge'.'''
        
        k = self.add_edge_like(node, node, edge)
        return k
    
    
    def add_selfedges_like(self, nodes, edge):
        '''Add several self-edges to a CurvyGraph at specified nodes. All added
        edges look like 'edge'.'''
        
        ret = []
        for i in range(len(nodes)):
            k = self.add_selfedge_like(nodes[i], edge)
            ret = ret + [k]
            
        return ret
        
    
    
    def remove_edge(self, edge_key):
        self.edges.pop(edge_key)
        
        
    def remove_edges(self, edge_keys):
        for k in edge_keys:
            self.edges.pop(k)
        
        
    def remove_edge_between(self, source, target):
        
        # find edges to be removed
        marked_for_removal = []
        for k in self.edges.keys():
            if self.edges[k].source.key == source:
                if self.edges[k].target.key == target:
                    marked_for_removal = marked_for_removal + [k]
                    
        # remove edges                    
        remove_edges(self, marked_for_removal)
    
    
    # def draw functions
    def draw(self, ax=None, include_nodes=True, include_edges=True, 
             labels=True):
        '''Draw a CurvyGraph.'''
        
        if ax is None:
            ax = plt.gca()
            
        if include_nodes:
            self.drawNodes(ax=ax, labels=labels)
        if include_edges: 
            self.drawEdges(ax=ax, labels=labels)
    
    
    def drawNodes(self, nodes=None, ax=None, labels=True):
        '''Draw nodes.'''
        
        for i, node in enumerate(self._get_node_list(nodes)):
            node.draw(ax=ax, labels=labels)
    
    
    def drawEdges(self, edges=None, ax=None, labels=True):
        '''Draw edges.'''
        
        for i, edge in enumerate(self._get_edge_list(edges)):
            edge.draw(ax=ax, labels=labels)
    

################################################################################
#### CurvyNode class ###########################################################
################################################################################

class CurvyNode:
    '''Nodes for CurvyGraph objects.'''
    
    def __init__(self, g=None, angle=0, radius=0.1, linewidth=1, 
                 linecolor=None, facecolor=None, color=None,
                 label=CurvyLabel(), alpha=1.0, zorder=0, 
                 edge_kwargs= {}, face_kwargs = {}, key=None,
                 copy_label=True, copy_kwargs=True, like=None):
        '''Initialize a CurvyNode object. `g` is a CurvyGraph object of which 
        the CurvyNode inherits its `graph_center`, `graph_radius`, and 
        `graph_tilt`.'''
        
        self.graph_center = ((0,0) if g is None else g.center) 
        self.graph_radius = (0.5 if g is None else g.radius)
        self.graph_tilt = (0 if g is None else g.tilt)
        self.angle = angle # integer
        
        if like is None:
            # use default values for node style, color, etc.
            
            self.radius = radius # float
            self.linewidth = linewidth # float
        
            if facecolor is None:
                if color is None:
                    self.facecolor = 'white' # default is white
                else:
                    self.facecolor = color
            else:
                self.facecolor = facecolor
            
            if linecolor is None:
                if color is None:
                    self.linecolor = 'k' # default is black
                else:
                    self.linecolor = color
            else:
                self.linecolor = linecolor
        
            if copy_label:
                self.label = label.copy() # a CurvyLabel object
            else:
                self.label = label
            
            self.alpha = alpha # float in [0,1] 
            self.zorder = zorder # integer
        
            if copy_kwargs:
                self.face_kwargs = face_kwargs.copy() # dict
                self.edge_kwargs = edge_kwargs.copy() # dict
            else:
                self.face_kwargs = face_kwargs # dict
                self.edge_kwargs = edge_kwargs # dict
                
        else:
            # make node like CurvyNode given in 'like' keyword argument
            self.radius = like.radius
            self.linewidth = like.linewidth
            self.facecolor = like.facecolor
            self.linecolor = like.linecolor
            self.label = like.label.copy()
            self.alpha = like.alpha
            self.zorder = like.zorder
            self.face_kwargs = like.face_kwargs.copy()
            self.edge_kwargs = like.edge_kwargs.copy()
            
        if key is None: 
            if g is not None: 
                key = max([-1]+list(g.nodes.keys()))+1 
        self.key = key
        
        if g is not None:
            g.nodes[key] = self
        
    
    def copy(self, copy_label=True, copy_kwargs=True):
        '''Make a hard copy of a CurvyNode.'''
        
        new = CurvyNode(g=CurvyGraph(center=self.graph_center, 
                                     radius=self.graph_radius, 
                                     tilt=self.graph_tilt), 
                        angle=self.angle, radius=self.radius, 
                        linewidth=self.linewidth, 
                        linecolor=self.linecolor,
                        facecolor=self.facecolor, 
                        label=self.label, 
                        alpha=self.alpha,
                        zorder=self.zorder,
                        edge_kwargs=self.edge_kwargs, 
                        face_kwargs=self.face_kwargs, 
                        key=self.key, 
                        copy_label=copy_label, 
                        copy_kwargs=copy_kwargs)
        return new
        
        
    # def set functions
    def set_like(self, like):
        '''Copy properties from another node.'''
        self.radius = like.radius
        self.linewidth = like.linewidth
        self.facecolor = like.facecolor
        self.linecolor = like.linecolor
        self.label = like.label.copy()
        self.alpha = like.alpha
        self.zorder = like.zorder
        self.face_kwargs = like.face_kwargs.copy()
        self.edge_kwargs = like.edge_kwargs.copy()
            
            
    def set_graph(self, g):
        self.graph_center = g.center
        self.graph_radius = g.radius
        self.graph_tilt = g.tilt
        g.nodes[self.key] = self
        
        
    def set_graph_center(self, center):
        self.graph_center = center

        
    def set_graph_radius(self, r):
        self.graph_radius = r
        
        
    def set_graph_tilt(self, tilt):
        self.graph_tilt = tilt
    
    
    def set_angle(self, angle):
        self.angle = angle
        
        
    def set_radius(self, radius):
        self.radius = radius
        
        
    def set_linewidth(self, linewidth):
        self.linewidth = linewidth
        
        
    def set_facecolor(self, fc):
        self.facecolor = fc
        
        
    def set_linecolor(self, ec):
        self.linecolor = ec

        
    def set_color(self, color):
        self.facecolor = color
        self.linecolor = color
       
    
    def set_label(self, label):
        self.label = label
       
    
    def set_alpha(self, alpha, include_label=True):
        self.alpha = alpha
        if include_label: self.label.alpha=alpha
      
    
    def set_zorder(self, zorder, include_label=True):
        self.zorder = zorder
        if include_label: self.label.zorder=zorder
    
    
    def set_face_kwargs(self, kwargs, copy_kwargs=True):
        if copy_kwargs:
            self.face_kwargs = kwargs.copy()
        else:
            self.face_kwargs = kwargs
       
    
    def set_edge_kwargs(self, kwargs, copy_kwargs=True):
        if copy_kwargs:
            self.edge_kwargs = kwargs.copy()
        else:
            self.edge_kwargs = kwargs
       
    
    def set_label_text(self, text):
        self.label.text = text
        
        
    def set_label_pad(self, pad):
        self.label.pad = pad

        
    def set_label_shift(self, shift):
        self.label.shift = shift

        
    def set_label_size(self, size):
        self.label.size = size
        
        
    def set_label_color(self, color):
        self.label.color = color

        
    def set_label_alpha(self, alpha):
        self.label.alpha = alpha
        
        
    def set_label_zorder(self, zorder):
        self.label.zorder = zorder

        
    def set_label_alignment(self, alignment, copy_alignment=True):
        if copy_alignment:
            self.label.alignment = [a for a in alignment]
        else:
            self.label.alignment = alignment
        
        
    def set_label_position(self, position, copy_position=True):
        if copy_position:
            self.label.position = [p for p in position]
        else:
            self.label.position = position
        

    def set_label_kwargs(self, kwargs, copy_kwargs=True):
        if copy_kwargs:
            self.label.kwargs = kwargs.copy()
        else:
            self.label.kwargs = kwargs

        
    def set_label_properties(self, text=None, pad=None, size=None, color=None, 
                             alpha=None, zorder=None, alignment=None, 
                             position=None, kwargs=None, copy_alignment=True, 
                             copy_position=True, copy_kwargs=True):
        
            if text is not None: self.label.text = text
            if pad is not None: self.label.pad = pad
            if size is not None: self.label.size = size
            if color is not None: self.label.color = color
            if alpha is not None: self.label.alpha = alpha
            if zorder is not None: self.label.zorder = zorder
            if alignment is not None: 
                self.set_label_alignment(kwargs, copy_kwargs=copy_alignment)
            if position is not None: 
                self.set_label_position(kwargs, copy_kwargs=copy_position)
            if kwargs is not None: 
                self.set_label_kwargs(kwargs, copy_kwargs=copy_kwargs)

                
    # def get functions
    def get_graph_center(self):
        return self.graph_center
    
    
    def get_graph_radius(self):
        return self.graph_radius
    
    
    def get_graph_tilt(self):
        return self.graph_tilt

    
    def get_angle(self):
        return self.angle

    
    def get_radius(self):
        return self.radius

    
    def get_linewidth(self):
        return self.linewidth

    
    def get_facecolor(self):
        return self.facecolor

    
    def get_linecolor(self):
        return self.linecolor
    
    
    def get_color(self):
        return (self.facecolor, self.linecolor)    

    
    def get_label(self):
        return self.label

    
    def get_alpha(self):
        return self.alpha

    
    def get_zorder(self):
        return self.zorder

    
    def get_face_kwargs(self):
        return self.face_kwargs    

    
    def get_edge_kwargs(self):
        return self.edge_kwargs    

    
    def get_id(self):
        return self.ID
    
    
    def get_label_text(self):
        return self.label.text
        
        
    def get_label_pad(self):
        return self.label.pad

        
    def get_label_size(self):
        return self.label.size
        
        
    def get_label_color(self):
        return self.label.color

        
    def get_label_alpha(self):
        return self.label.alpha
        
        
    def get_label_zorder(self):
        return self.label.zorder

        
    def get_label_alignment(self):
        return self.label.alignment
        
        
    def get_label_position(self):
        return self.label.position
        

    def get_label_kwargs(self):
        return self.label.kwargs
        
        
    # def draw functions
    def get_position(self):
        '''Get position of center of a CurvyNode in axis-centric
        coordinates.'''
        
        if self.angle is None:
            pos = self.graph_center
        else:
            alpha = (self.graph_tilt + self.angle)
            pos = (self.graph_center[0] + self.graph_radius*degCos(alpha),
                   self.graph_center[1] + self.graph_radius*degSin(alpha))
        return np.array(pos)
        
        
    def draw(self, ax=None, labels=True):
        '''Draw a CurvyNode.'''
        
        if ax is None:
            ax = plt.gca()       
            
        # draw edge (i.e., outer lining)
        arc = Arc(self.get_position(), 2*(self.radius), 2*(self.radius), 
                  color=self.linecolor, linewidth=self.linewidth, 
                  alpha=self.alpha, zorder=self.zorder, **self.edge_kwargs)
        
        # draw face
        fillCircle(ax, radius=self.radius, center=self.get_position(), 
                   color=self.facecolor, alpha=self.alpha, zorder=self.zorder,
                   face_kwargs=self.face_kwargs)
        ax.add_patch(arc) 
        
        if labels:
            # draw node label
            self.drawLabel(ax=ax)
    
    
    def drawLabel(self, ax=None):
        '''Draw a label for a CurvyNode.'''
        
        if ax is None:
            ax = plt.gca()
            
        p = self.get_position() + np.array(self.label.shift)
        
        ax.text(p[0]+self.label.shift[0], 
                p[1]+self.label.shift[1], 
                self.label.text,  size=self.label.size, 
                color=self.label.color, alpha=self.label.alpha, 
                zorder=self.label.zorder, verticalalignment="center", 
                horizontalalignment="center", **self.label.kwargs)
        
        
################################################################################
#### CurvyEdge class ###########################################################
################################################################################
    
class CurvyEdge:
    '''Nodes for CurvyGraph objects.'''
    
    def __init__(self, g=None, source=None, target=None, 
                 curvature=0.0, selfedge_radius=None, heads = [None, None], 
                 color='k', pad=0.005, linewidth=1.0, linestyle='-', 
                 doubling=0.025, labels=[CurvyLabel()], alpha=1.0, zorder=0, 
                 kwargs={}, key=None, copy_labels=True, copy_heads=True, 
                 copy_kwargs=True, like=None):
        '''Initialize a CurvyEdge object. `g` is a CurvyGraph object of which 
        the CurvyEdge inherits its `graph_center`, `graph_radius` and 
        `graph_tilt`.'''
        
        self.graph_center = ((0,0) if g is None else g.center)
        self.graph_radius = (0.5 if g is None else g.radius)
        self.graph_tilt = (0 if g is None else g.tilt)
        self.source = (CurvyNode() if source is None else source)
        self.target = (CurvyNode() if target is None else target)
        
        if like is None:
            # make edge using default style or keyword style
            self.curvature = curvature # float
        
            if selfedge_radius is None:
                self.selfedge_radius = self.graph_radius/4.0
            else:
                self.selfedge_radius = selfedge_radius
        
            if copy_heads:
                self.heads = [(h.copy() if h is not None else None) 
                              for h in heads]
            else:
                self.heads = heads # list of two; None's or CurvyHeads
            
            self.color = color # string
            
            if isinstance(pad, list):
                # pad should be a list of two floats
                if len(pad)>1:
                    self.pad = [pad[0], pad[1]]
                else:
                    self.pad = [pad[0], pad[0]]
            else:
                # pad can also be a float
                self.pad = [pad, pad] 
                
            self.linewidth = linewidth # float
            self.linestyle = linestyle # float
            self.doubling = doubling # float
        
            if copy_labels:
                self.labels = [l.copy() for l in labels]
            else:
                self.labels = labels # a CurvyLabel
            
            self.alpha = alpha # float in [0,1]
            self.zorder = zorder # integer
        
            if copy_kwargs:
                self.kwargs = kwargs.copy() # dictionary
            else:
                self.kwargs = kwargs
                
        else:
            # copy style from CurvyEdge given in 'like' keyword argument
            self.curvature = like.curvature
            self.selfedge_radius = like.slefedge_radius
            self.color = like.color
            self.pad = like.pad
            self.linewidth = self.linewidth
            self.linestyle = like.linestyle
            self.doubling = like.doubling
            self.alpha = like.alpha
            self.zorder = like.zorder
            self.heads = [(h.copy() if h is not None else None) 
                          for h in like.heads]
            self.labels = [l.copy() for l in like.labels]
            self.kwargs = like.kwargs.copy()
        
        if key is None: 
            if g is not None:
                key = max([-1]+list(g.edges.keys()))+1 
        self.key = key
        
        if g is not None:
            g.edges[key] = self
        

    def copy(self, copy_nodes=True, copy_labels=True, copy_heads=True,
             copy_kwargs=True):
        '''Make a hard copy of a CurvyEdge.'''
        
        if copy_nodes:
            new_source = self.source.copy(copy_label=copy_labels)
            new_target = self.target.copy(copy_label=copy_labels)
        else:
            new_source = self.source
            new_target = self.target
            
        # make new edge    
        new = CurvyEdge(g=CurvyGraph(center=self.graph_center,
                                     radius=self.graph_radius,
                                     tilt=self.graph_tilt), 
                        source=new_source, target=new_target, 
                        curvature=self.curvature, 
                        selfedge_radius=self.selfedge_radius,
                        heads = self.heads, 
                        color=self.color, 
                        pad=self.pad, 
                        doubling=self.doubling,
                        linewidth = self.linewidth, 
                        linestyle = self.linestyle, 
                        labels = self.labels, 
                        alpha=self.alpha,
                        zorder=self.zorder,
                        kwargs=self.kwargs,
                        key=self.key, copy_labels=copy_labels, 
                        copy_heads=copy_heads, copy_kwargs=copy_kwargs)
        
        return new
        
        
    # set functions
    def set_like(self, like):
        '''Make edge look like another edge.'''
        self.curvature = like.curvature
        self.selfedge_radius = like.slefedge_radius
        self.color = like.color
        self.pad = like.pad
        self.linewidth = self.linewidth
        self.linestyle = like.linestyle
        self.doubling = like.doubling
        self.alpha = like.alpha
        self.zorder = like.zorder
        self.heads = [(h.copy() if h is not None else None) 
                      for h in like.heads]
        self.labels = [l.copy() for l in like.labels]
        self.kwargs = like.kwargs.copy()        
        
        
    def set_graph(self, g):
        self.graph_center = g.center
        self.graph_radius = g.radius
        self.graph_tilt = g.tilt
        g.edges[self.key] = self
        
        
    def set_graph_center(self, center):
        self.graph_center = center

        
    def set_graph_radius(self, r):
        self.graph_radius = r
        
        
    def set_graph_tilt(self, tilt):
        self.graph_tilt = tilt
    
    
    def set_head_pad(self, pad, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].pad = pad
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].pad = pad
        
        
    def set_head_length(self, length, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].length = length
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].length = length

                
    def set_head_width(self, width, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].width = width
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].width = width

                
    def set_head_shape(self, shape, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].shape = shape
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].shape = shape

                
    def set_head_alpha(self, alpha, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].alpha = alpha
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].alpha = alpha

                
    def set_head_zorder(self, zorder, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].zorder = zorder
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].zorder = zorder

                
    def set_head_kwargs(self, kwargs, head=None, copy_kwargs=True):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    if copy_kwargs:
                        self.heads[i].kwargs = kwargs.copy()
                    else:
                        self.heads[i].kwargs = kwargs
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                if copy_kwargs:
                    self.heads[head].kwargs = kwargs.copy()
                else:
                    self.heads[head].kwargs = kwargs
                    
                
    def set_head_facecolor(self, facecolor, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].facecolor = facecolor
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].facecolor = facecolor

                
    def set_head_linecolor(self, linecolor, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].linecolor = linecolor
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].linecolor = linecolor

                
    def set_head_color(self, color, head=None):
        if head is None: # change both heads
            for i in [0,1]:
                if self.heads[i] is not None:
                    self.heads[i].facecolor = color
                    self.heads[i].linecolor = color
        else: # change only head[0] or [1]
            if self.heads[head] is not None:
                self.heads[head].facecolor = color
                self.heads[head].linecolor = color
                
                
    def set_source(self, source):
        self.source = source
        
        
    def set_target(self, target):
        self.target = target
        
        
    def set_curvature(self, curvature):
        self.curvature = curvature
        
        
    def set_selfedge_radius(self, r):
        self.selfedge_radius = r
        
        
    def set_color(self, color, include_labels=False, include_heads=True):
        
        self.color = color
        
        if include_labels:
            for l in self.labels:
                l.color=color
                
        if include_heads:
            self.set_head_color(color)
            
        
    def set_heads(self, heads):
        self.heads = heads

        
    def set_pad(self, pad):
        if isinstance(pad, list):
            # pad should be a list of two floats
            if len(pad)>1:
                self.pad = [pad[0], pad[1]]
            else:
                self.pad = [pad[0], pad[0]]
        else:
            # pad can also be a float
            self.pad = [pad, pad] 
        
        
    def set_linewidth(self, lw):
        self.linewidth = lw            

        
    def set_linestyle(self, ls):
        self.linestyle = ls
        
        
    def set_doubling(self, doubling):
        self.doubling = doubling

        
    def set_labels(self, labels):
        self.labels = labels
        
        
    def set_alpha(self, alpha, include_labels=True, include_heads=True):
        
        self.alpha = alpha
        
        if include_labels:
            for l in self.labels:
                l.alpha=alpha
                
        if include_heads:
            self.set_head_alpha(alpha)

        
        
    def set_zorder(self, zorder, include_labels=True, include_heads=True):
        
        self.zorder = zorder

        if include_labels:
            for l in self.labels:
                l.zorder=zorder
                
        if include_heads:
            self.set_head_zorder(zorder)

        
    def set_kwargs(self, kwargs, copy_kwargs=True):
        if copy_kwargs:
            self.kwargs = kwargs.copy()
        else:
            self.kwargs = kwargs       
            
        
    def set_id(self, ID):
        self.ID = ID
              
            
    def set_head_properties(self, head=None, pad=None, length=None, width=None, 
                            shape=None, facecolor=None, linecolor=None, 
                            color=None, alpha=None, zorder=None, kwargs=None,
                            copy_kwargs=True):
        if pad is not None: self.set_head_pad(pad, head=head)
        if length is not None: self.set_head_length(length, head=head)
        if width is not None: self.set_head_width(width, head=head)
        if shape is not None: self.set_head_shape(shape, head=head)
        if color is not None: self.set_head_color(color, head=head)
        if facecolor is not None: self.set_head_facecolor(facecolor, head=head)    
        if linecolor is not None: self.set_head_linecolor(linecolor, head=head)
        if alpha is not None: self.set_head_alpha(alpha, head=head)
        if zorder is not None: self.set_head_zorder(zorder, head=head)
        if kwargs is not None: 
            self.set_head_kwargs(kwargs, head=head, copy_kwargs=copy_kwargs)
                          
                
    def set_label_text(self, text, label=None):
        if label is None:
            for l in self.labels:
                l.text = text
        else:
            self.labels[label].text = text

    def set_label_shift(self, shift, label=None):
        if label is None:
            for l in self.labels:
                l.shift = shift
        else:
            self.labels[label].shift = shift
            
    def set_label_pad(self, pad, label=None):
        if label is None:
            for l in self.labels:
                l.pad = pad
        else:
            self.labels[label].pad = pad

            
    def set_label_size(self, size, label=None):
        if label is None:
            for l in self.labels:
                l.size = size
        else:
            self.labels[label].size = size

            
    def set_label_color(self, color, label=None):
        if label is None:
            for l in self.labels:
                l.color = color
        else:
            self.labels[label].color = color

            
    def set_label_alpha(self, alpha, label=None):
        if label is None:
            for l in self.labels:
                l.alpha = alpha
        else:
            self.labels[label].alpha = alpha

            
    def set_label_zorder(self, zorder, label=None):
        if label is None:
            for l in self.labels:
                l.zorder = zorder
        else:
            self.labels[label].zorder = zorder

            
    def set_label_alignment(self, alignment, label=None, copy_alignment=True):
        if copy_alignment:
            if label is None:
                for l in self.labels:
                    l.alignment = [a for a in alignment]
            else:
                self.labels[label].alignment = [a for a in alignment]
        else:
            if label is None:
                for l in self.labels:
                    l.alignment = alignment
            else:
                self.labels[label].alignment = alignment

            
    def set_label_position(self, position, label=None, copy_position=True):
        if copy_position:
            if label is None:
                for l in self.labels:
                    l.position = [p for p in position]
            else:
                self.labels[label].position = [p for p in position]
        else:
            if label is None:
                for l in self.labels:
                    l.position = position
            else:
                self.labels[label].position = position

            
    def set_label_kwargs(self, kwargs, label=None, copy_kwargs=True):
        if copy_kwargs:
            if label is None:
                for l in self.labels:
                    l.kwargs = kwargs.copy()
            else:
                self.labels[label].kwargs = kwargs.copy()
        else:
            if label is None:
                for l in self.labels:
                    l.kwargs = kwargs
            else:
                self.labels[label].kwargs = kwargs
            
            
    def set_label_properties(self, label=None, text=None, pad=None, size=None, 
                             color=None, alpha=None, zorder=None,
                             alignment=None, position=None, kwargs=None,
                             copy_alignment=True, copy_position=True, 
                             copy_kwargs=True):
        
        if text is not None: self.set_label_text(text, label=label)
        if pad is not None: self.set_label_pad(pad, label=label)                             
        if size is not None: self.set_label_size(size, label=label)
        if color is not None: self.set_label_color(color, label=label) 
        if alpha is not None: self.set_label_alpha(alpha, label=label)
        if zorder is not None: self.set_label_zorder(zorder, label=label)
        if alignment is not None: 
            self.set_label_alignment(alignment, label=label, 
                                     copy_alignment=copy_alignment) 
        if position is not None: 
            self.set_label_position(position, label=label,
                                    copy_position=copy_position)
        if kwargs is not None: 
            self.set_label_kwargs(kwargs, label=label, copy_kwargs=copy_kwargs)


    # get functions    
    def get_graph_center(self):
        return self.graph_center

    
    def get_graph_radius(self):
        return self.graph_radius

    
    def get_graph_tilt(self):
        return self.graph_tilt


    def get_source(self):
        return self.source


    def get_target(self):
        return self.target


    def get_curvature(self):
        return self.curvature


    def get_selfedge_radius(self):
        return self.selfedge_radius


    def get_heads(self):
        return self.heads

    
    def get_color(self):
        return self.color


    def get_pad(self):
        return self.pad


    def get_labels(self):
        return self.labels


    def get_alpha(self):
        return self.alpha


    def get_zorder(self):
        return self.zorder


    def get_kwargs(self):
        return self.kwargs    


    def get_id(self):
        return self.ID

    
    def get_head_pad(self):
        pads = [(h.pad if h is not None else None) for h in self.heads]
        return pads


    def get_head_length(self):
        lengths = [(h.length if h is not None else None) for h in self.heads]
        return lengths


    def get_head_width(self):
        widths = [(h.width if h is not None else None) for h in self.heads]
        return widths


    def get_head_shape(self):
        shapes = [(h.shape if h is not None else None) for h in self.heads]
        return shapes


    def get_head_alpha(self):
        alphas = [(h.alpha if h is not None else None) for h in self.heads]
        return alphas


    def get_head_zorder(self):
        zorders = [(h.zorder if h is not None else None) for h in self.heads]
        return zorders


    def get_head_kwargs(self):
        kwargs = [(h.kwargs if h is not None else None) for h in self.heads]
        return kwargs    

    
    def get_head_facecolor(self):
        facecolors = [(h.facecolor if h is not None else None) for h in self.heads]
        return facecolors


    def get_head_linecolor(self):
        linecolors = [(h.linecolor if h is not None else None) for h in self.heads]
        return linecolors

    
    def get_head_color(self):
        colors = [(h.get_head_color() if h is not None else None) for h in self.heads]
        return colors


    def get_label_text(self):
        texts = [l.text for l in self.labels]
        return texts


    def get_label_pad(self):
        pads = [l.pad for l in self.labels]
        return pads


    def get_label_size(self):
        sizes = [l.size for l in self.labels]
        return sizes


    def get_label_color(self):
        colors = [l.color for l in self.labels]
        return colors


    def get_label_alpha(self):
        alphas = [l.alpha for l in self.labels]
        return alphas


    def get_label_zorder(self):
        zorders = [l.zorder for l in self.labels]
        return zorders


    def get_label_alignment(self):
        alignments = [l.alignment for l in self.labels]
        return alignments


    def get_label_position(self):
        positions = [l.position for l in self.labels]
        return positions


    def get_label_kwargs(self):
        kwargs = [l.kwargs for l in self.labels]
        return kwargs

    
    # def other functions
    def switch_heads(self):
        heads = self.heads
        self.heads = [heads[1], heads[0]]            
        
    
    # def drawing functions
    def get_ellipse(self):
        '''Get ellipse that is used to create an elliptical arc through the end
        points of a CurvyEdge.'''
        
        r = self.graph_radius
        center = self.graph_center
        
        # compute angular length
        angles = [self.graph_tilt + self.source.angle,
                 self.graph_tilt + self.target.angle]
        d_angle = angles[1]-angles[0]
        
        if d_angle % 360 == 0:
            # selfedge with circular arc
            return(center, r, r, angles[0])

        # Compute ellipse's tilt
        target_angle = (self.target.angle if self.target.angle else 360)
        tilt = (self.graph_tilt + 
                (self.source.angle + target_angle)/2.0 - 90)
        
        # Compute position of right node (in ellipse-centric coordinates)
        point_X = (degCos(90.0-d_angle/2.0)*r, degSin(90.0-d_angle/2.0)*r)        

        # Compute ellipse's axis lengths
        vAxis = point_X[1]+self.curvature*(r-point_X[1])
        hAxis = np.abs(point_X[0])/np.sqrt(1-point_X[1]**2/vAxis**2)
        
        return (center, hAxis, vAxis, tilt) 
    
    
    def get_points_elliptical(self, headcorrection=0):
        '''Get start and end point of the arc of an elliptical edge.'''
        
        r = self.graph_radius
                
        # Step 1: Uncorrected start and end point of arc in polar coordinates
        # this defines an arc that connectes the center points of each node
        angle = [self.graph_tilt + self.source.angle,
                 self.graph_tilt + self.target.angle]
        
        # can use equations for circle to compute positions of the 
        # uncorrected points because they coincide with node positions
        # (which are on a circle)
        point = [r*np.array([degCos(a), degSin(a)]) for a in angle]
        
        # Step 2: Define ellipse
        # Step 2.1: Get ellipse's center, axis lengths, and tilt
        center, hAxis, vAxis, tilt = self.get_ellipse() 

        # Step 2.4: Define polyline object
        E0 = ellipse_polyline(center, hAxis, vAxis, tilt, offset=-90)
        #ax = plt.gca()
        #ax.plot(E0[:,0],E0[:,1], 'b--', zorder=12)
        
        # Step 3: Correct end points for node radii
        for i, node in enumerate([self.source, self.target]):
            
            # define a (half) circle that traces the circumference of the node
            half_ring_angle = angle[i] - i*180
            E1 = ellipse_polyline(point[i]+center,
                                  node.radius + self.pad[i], 
                                  node.radius + self.pad[i], 
                                  half_ring_angle, max_angle=180)   
            
            #ax.plot(E1[:,0],E1[:,1], 'r--', zorder=12)
            
            # get intersection of half ring with ellipse
            point[i] = polyline_intersection(E0, E1)
            
            # compute corresponding ellipse angles
            angle[i] = angle_from_point(point[i], center=self.graph_center)
            
        # Step 4: Correct end points for padding
        for i, head in enumerate(self.heads):
            if head is not None:
                
                if head.pad > 0:    
                    # define a (half) circle
                    half_ring_angle = angle[i] - i*180 
                    E1 = ellipse_polyline(point[i],
                                          head.pad, head.pad, 
                                          half_ring_angle, max_angle=180)
                    
                    # get intersection with ellipse
                    point[i] = polyline_intersection(E0, E1)
                    
                    # compute corresponding ellipse angles
                    angle[i] = angle_from_point(point[i], 
                                                center=self.graph_center)
            
        # Step 5: Correct end points for head length
        if headcorrection:
            for i, head in enumerate(self.heads):
                if head is not None:
                    
                    if head.length > 0:    
                        # define a (half) circle
                        half_ring_angle = angle[i] - i*180 
                        E1 = ellipse_polyline(point[i],
                                              headcorrection*head.length, 
                                              headcorrection*head.length, 
                                              half_ring_angle, max_angle=180)                        
                        
                        # get intersection with ellipse
                        point[i] = polyline_intersection(E0, E1)
                        
                        # compute corresponding ellipse angles
                        angle[i] = angle_from_point(point[i], 
                                                    center=self.graph_center)
        
        return angle
        
    
    def get_points_straight(self, headcorrection=0, ellipse_centric=False):
        '''Get start and end point of the arc of an straight edge.'''
        
        r = self.graph_radius
                
        # Step 1: Uncorrected start and end point of arc in polar coordinates
        # this defines an arc that connectes the center points of each node
        angles = [self.graph_tilt + self.source.angle,
                 self.graph_tilt + self.target.angle]
        d_angle = angles[1] - angles[0]
        
        # can use equations for circle to compute positions of the uncorrected
        # points this is in ellipse-centric coordinates
        points = [r*np.array([degCos(a), degSin(a)]) for a in angles]
        
        # Step 2: Define line
        v = points[1]-points[0]
        if v[0]==0 and v[1]==0:
            raise ValueError('Cannot compute connector between start and end point of edge because start and end point are identical.')
        v1 = v/np.sqrt(np.sum(v**2)) # v normed to length 1
        
        # Step 3: Calculate correction pointers
        # source correction
        if self.heads[0] is None:
            d_source = v1*self.source.radius
        else:
            d_source = v1*(self.source.radius + self.pad[0]
                           + self.heads[0].pad 
                           + headcorrection*self.heads[0].length)
        # target correction
        if self.heads[1] is None:
            d_target = -v1*self.target.radius
        else:
            d_target = -v1*(self.target.radius + self.pad[1]
                            + self.heads[1].pad 
                            + headcorrection*self.heads[1].length)
        
        # Step 4: New points
        points = [points[0]+d_source, points[1]+d_target]
        if not ellipse_centric:
            # rotate
            # shift
            points = [points[0]+self.graph_center,
                     points[1]+self.graph_center]
        
        return points
    
    
    def get_points_straight_with_doubling(self, headcorrection=0):
        '''Get start and end point of the arc of two parallel straight edges.'''
                
        if (self.heads[0] is not None and self.heads[1] is not None
            and self.doubling > 0):
            
            # get points in ellipse-centric coordinates
            points = self.get_points_straight(headcorrection=headcorrection, 
                                              ellipse_centric=True)
            
            # get angle of arrow
            v = points[1]-points[0]
            if v[0]==0 and v[1]==0:
                raise ValueError('Cannot compute connector between start and end point of edge because start and end point are identical.')
            alpha = degArcTan2(v[1],v[0])
            
            # get shift for coordinates
            dpad = np.array([-degSin(alpha), degCos(alpha)])*self.doubling
            
            # shift coordinates
            ret = (points[0]+dpad+self.graph_center, 
                   points[1]+dpad+self.graph_center, 
                   points[0]-dpad+self.graph_center, 
                   points[1]-dpad+self.graph_center)
        
            return ret
            
        else:
            # if there is only one arrow head, no shift is needed
            points = self.get_points_straight(headcorrection=headcorrection, 
                                              ellipse_centric=False)
            return (points[0], points[1], points[0], points[1])
      
    
    def drawHead(self, pointy_end, flat_end, sot, d=0.001, ax=None):
        '''Draw an arrow head for a CurvyEdge. The keyword `sot` sets whether we 
        draw an arrow head at the source end or target end of an edge.'''
        
        if ax is None: ax= plt.gca()
            
        if sot=='source':
            sot_id = 0
        elif sot=='target':
            sot_id = 1
        else:
            raise ValueError('Input sot needs to be string "source" or "target".')
            
        # get pointer
        v = pointy_end-flat_end
        # normalize to length d
        v = d*v/np.sqrt(np.sum(v**2))
        
        # draw arrow head
        ax.arrow(flat_end[0]-v[0], flat_end[1]-v[1], v[0], v[1],
                 shape = self.heads[sot_id].shape,
                 head_width=self.heads[sot_id].width,
                 head_length=self.heads[sot_id].length,
                 facecolor=self.heads[sot_id].facecolor,
                 edgecolor=self.heads[sot_id].linecolor,
                 alpha=self.heads[sot_id].alpha,
                 zorder=self.heads[sot_id].zorder,
                 **self.heads[sot_id].kwargs)

        
    def drawHeads(self, p0, f0, p1, f1, d=0.001, ax=None):
        '''Draw arrow heads for a CurvyEdge.'''
        
        if self.heads[0] is not None:
            self.drawHead(p0, f0, 'source', ax=ax, d=d)
        if self.heads[1] is not None:
            self.drawHead(p1, f1, 'target', ax=ax, d=d)
        
        return None
    
    
    def drawLabel(self, ax=None):
        '''Draw a label for an edge.'''
        
        if ax is None:
            ax = plt.gca()
            
        if self.curvature:
            # get ellipse
            center, hAxis, vAxis, tilt = self.get_ellipse()
            # get angle at the mid point of the arc
            angle = tilt + 90
            if self.source.angle==0 and self.target.angle==360:
                angle = tilt + 180
            
            # make angles positive in [0,360) 
            # and target_angle > source_angle
            #source_angle = self.source.angle #% 360
            #target_angle = self.target.angle #% 360
            #if target_angle < source_angle:
            #    target_angle = target_angle + 360
            #angle = angle % 360
            
            # angle should be in between source and target angle
            #if source_angle <= angle and angle <= target_angle:
            #    pass
            #else:
            #    angle = angle - 180

            # get point in at the mid point of the arc
            ap = point_from_angle(center, hAxis, vAxis, tilt, angle)
            # transfer to ellipse-centric coordinates and normalize
            ap1 = ap-self.graph_center
            ap1 = ap1/np.sqrt(np.sum(ap1**2))
            
            for label in self.labels:
                # should we shift up or down for label position?
                if label.position in ['inside','counterclockwise']:
                    padFactor = -1
                else:
                    padFactor = 1
                # let's do the shift
                apx = ap +ap1*label.pad*padFactor
                # adjust for manual correction 
                apx = apx + np.array(label.shift)
                # draw
                ax.text(apx[0], apx[1], label.text, 
                        size=label.size, color=label.color,
                        alpha=label.alpha, zorder=label.zorder,
                        horizontalalignment=label.alignment[0], 
                        verticalalignment=label.alignment[1],
                        **label.kwargs) 
        else:
            # get line
            points = self.get_points_straight(headcorrection=0)
            vec = points[1]-points[0]
            # get mid point
            ap = points[0]+vec/2
            # transfer to ellipse-centric coordinates and normalize
            ap1 = ap-self.graph_center
            ap1 = ap1/np.sqrt(np.sum(ap**2))

            # get the angle with respect to graph center
            angle = -degArcTan2(vec[0],vec[1])+90
            for label in self.labels:
                # should we shift up or down for label position?
                if label.position in ['inside','counterclockwise']:
                    padFactor = -1
                else:
                    padFactor = 1
                # let's do the shift
                apx = ap +ap1*label.pad*padFactor  
                # adjust for manual correction 
                apx = apx + np.array(label.shift)
                # draw
                ax.text(apx[0], apx[1], label.text,
                        size=label.size, color=label.color,
                        alpha=label.alpha, zorder=label.zorder,
                        horizontalalignment=label.alignment[0], 
                        verticalalignment=label.alignment[1],
                        **label.kwargs) 

            
    def drawElliptical(self, ax=None, labels=True):
        '''Draw CurvyEdge with non-zero curvature.'''
        
        # Elliptical edges are arcs along the partial circumference of an 
        # ellipse. For simplicity, we set the ellipse to have the same center
        # as the CurvyGraph. But its long axis and short axis can have different
        # lengths than the radius of the CurvyGraph. They also can have 
        # different alignments than the horizontal and vertical axis of the 
        # drawing.
        
        # Step 0: set ax
        if ax is None: ax = plt.gca()
        
        # Step 1: compute start and end point of arc
        r = self.graph_radius
        angles = self.get_points_elliptical(headcorrection=0)
        angles05 = self.get_points_elliptical(headcorrection=0.5)
        angles10 = self.get_points_elliptical(headcorrection=1.0)
        
        # Step 2: Define ellipse        
        center, hAxis, vAxis, tilt = self.get_ellipse() 
        
        # Step 3: Draw arc
        arc = Arc(center, hAxis*2, vAxis*2,   
                  theta1=angles05[0]-tilt, theta2=angles05[1]-tilt,  
                  angle=tilt, color=self.color, 
                  linewidth=self.linewidth, linestyle=self.linestyle,
                  alpha=self.alpha, zorder=self.zorder,
                  **self.kwargs)
        ax.add_patch(arc)
        
        # Step 4: Draw arrow heads 
        [p0, f0, p1, f1] = [point_from_angle(center, hAxis, vAxis, tilt, x)
                            for x in [angles[0], angles10[0], angles[1], 
                                      angles10[1]]]
        
        self.drawHeads(p0, f0, p1, f1, ax=ax)
        
        # Step 5: Draw edge label
        if labels:
            self.drawLabel(ax=ax)
    
    
    def drawStraight(self, ax=None, labels=True):
        '''Draw CurvyEdge with no curvature.'''
   
        # Step 1: set ax
        if ax is None: ax = plt.gca()
        
        # Step 2: Draw arrow
        if self.doubling and self.heads[0] is not None and self.heads[1] is not None:
            # draw double arrow instead of single arrow if there are two
            # arrow heads
            a0, b0, c0, d0 = self.get_points_straight_with_doubling(headcorrection=0)
            a1, b1, c1, d1 = self.get_points_straight_with_doubling(headcorrection=0.5)
            a2, b2, c2, d2 = self.get_points_straight_with_doubling(headcorrection=1)
            v_top = b1-a0
            v_bottom = c1-d0
            
            ax.arrow(a0[0], a0[1], v_top[0], v_top[1], 
                     head_width=0, head_length=0, color=self.color,
                     linewidth=self.linewidth, linestyle=self.linestyle,
                     alpha=self.alpha, zorder=self.zorder, **self.kwargs) 
            
            ax.arrow(d0[0], d0[1], v_bottom[0], v_bottom[1], 
                     head_width=0, head_length=0, color=self.color,
                     linewidth=self.linewidth, linestyle=self.linestyle,
                     alpha=self.alpha, zorder=self.zorder, **self.kwargs) 
            
            # points for drawing arrow heads
            p1, f1, p0, f0 = b0, b2, c0, c2
            
        else:
            # draw single arrow
            points = self.get_points_straight(headcorrection=0)
            points05 = self.get_points_straight(headcorrection=0.5)
            points10 = self.get_points_straight(headcorrection=1)
            
            v = points05[1]-points05[0]
            ax.arrow(points05[0][0], points05[0][1], v[0], v[1], 
                     head_width=0, head_length=0, color=self.color,
                     linewidth=self.linewidth, linestyle=self.linestyle,
                     alpha=self.alpha, zorder=self.zorder, **self.kwargs)
            
            # points for drawing arrow heads
            p0, f0, p1, f1 = points[0], points10[0], points[1], points10[1]
                    
        # Step 4: Draw arrow heads
        self.drawHeads(p0, f0, p1, f1, ax=ax)
        
        # Step 4: Draw labels
        if labels:
            self.drawLabel(ax=ax)
            
            
    def draw(self, ax=None, labels=True):
        '''Draw CurvyEdge.'''
        
        if self.source.angle == self.target.angle:
            
            # 1. selfedge is drawn as a ring. get its center
            # 1.1 get node position
            node_pos = self.source.get_position()
            # 1.2 get direction of connector between node position and center
            connector = (node_pos - self.graph_center)
            # 1.3 normalize connector to length selfedge_radius
            connector = connector/np.sqrt(np.sum(connector**2))
            connector = self.selfedge_radius*connector
            # 1.4 new center
            selfedge_center = node_pos+connector
            
            # create selfedge from edge attributes
            selfedge = self.copy(copy_nodes=True)
            selfedge.graph_center = selfedge_center
            selfedge.graph_radius = self.selfedge_radius
            selfedge.graph_tilt = self.graph_tilt + self.source.angle - 180 
            selfedge.source.angle = 0
            selfedge.target.angle = 360
            
            selfedge.drawElliptical(ax=ax, labels=labels)
        
        elif self.source.angle is None or self.target.angle is None:
            
            # make new edge
            new_edge = self.copy(copy_nodes=True)
            
            # new graph base angle and radius (are the same as the old ones)
            new_edge.graph_tilt = self.graph_tilt
            new_edge.graph_radius = self.graph_radius

            # new angles
            if self.source.angle is None:                
                new_edge.source.angle = 60 + self.target.angle
                new_edge.target.angle = 120 + self.target.angle
                new_edge.switch_heads()
                # connector between graph centers
                connector_angle = self.target.angle-60+self.graph_tilt 
            elif self.target.angle is None:
                new_edge.source.angle = 60 + self.source.angle
                new_edge.target.angle = 120 + self.source.angle
                # connector between graph centers
                connector_angle = self.source.angle-60+self.graph_tilt
            else:
                raise LogicalError('This should not have happened.') 
            
            #plt.plot([0,degCos(connector_angle)],[0,degSin(connector_angle)])
            #new graph center
            new_edge.graph_center = [self.graph_center[0] 
                                     + (self.graph_radius
                                        *degCos(connector_angle)), 
                                     self.graph_center[1] 
                                     + (self.graph_radius
                                        *degSin(connector_angle))]
            # draw the new edge             
            if self.curvature != 0:
                # draw elliptical arrow
                new_edge.drawElliptical(ax=ax, labels=labels)
            else:
                # draw straight arrow
                new_edge.drawStraight(ax=ax, labels=labels)  
                
        elif self.source.angle %360 == self.target.angle % 360:
            
            # this is also a selfedge (but we couldnt try to modulo before 
            # ruling out that one value was None)
            
            # make a copy of the edge
            new_edge = self.copy()
            # correct to angles of source and target node
            new_edge.source.angle = self.source.angle % 360
            new_edge.target.angle = self.target.angle % 360
            # this should now trigger the selfedge case above
            new_edge.draw()
        
        else:
        
            if self.curvature != 0:
                # draw elliptical arrow
                self.drawElliptical(ax=ax, labels=labels)
            else:
                # draw straight arrow
                self.drawStraight(ax=ax, labels=labels)
        
                
################################################################################
#### Functions for creating CurvyGraphs ########################################
################################################################################

if True:
                
    def from_adjacency(A, angles=None, node_labels=[], edge_labels=[],
                       center=(0,0), radius=0.5, tilt=0, key=0,
                       nodes_like=CurvyNode(), edges_like=CurvyEdge()):
        '''Create a CurvyGraph from an adjacency matrix.'''
        
        # Step 1: Get number of nodes in graph
        num_nodes = len(A) # A is square numpy array
        
        # Step 2: Get angles for polar coordinates of node positions
        if angles is None:
            # default is equidistant positioning on a ring
            d_angle = 360/len(A)
            angles = [d_angle*i for i in range(num_nodes)]
            
        if len(angles) < num_nodes:
            raise ValueError('Too few node angles specified.')
        
        # Step 3: Make CurvyGraph
        g = CurvyGraph(center=center, radius=radius, tilt=tilt, key=key)
        
        # Step 4: Add nodes
        nodes = {i: CurvyNode(g=g, angle=angles[i], 
                              radius=nodes_like.radius,
                              linewidth=nodes_like.linewidth,
                              linecolor=nodes_like.linecolor,
                              facecolor=nodes_like.facecolor,
                              edge_kwargs=nodes_like.edge_kwargs,
                              face_kwargs=nodes_like.face_kwargs,
                              alpha=nodes_like.alpha,
                              zorder=nodes_like.zorder,
                              label=(node_labels[i] if i < len(node_labels)
                                     else nodes_like.label),
                              key=i) for i in range(num_nodes)}
        g.nodes = nodes
        
        # Step 5: Add edges
        anz = A.nonzero()
        edges = {i: CurvyEdge(g=g, source=g.nodes[e[1]],
                              target=g.nodes[e[0]],
                              curvature=edges_like.curvature,
                              selfedge_radius=edges_like.selfedge_radius,
                              color=edges_like.color,
                              pad=edges_like.pad,
                              doubling=edges_like.doubling,
                              linewidth=edges_like.linewidth,
                              linestyle=edges_like.linestyle,
                              heads=get_head_list(e[2], 
                                                  heads_like=edges_like.heads),
                              labels=(edge_labels[i] if i < len(edge_labels)
                                      else [l.copy() 
                                            for l in edges_like.labels]),
                              alpha=edges_like.alpha,
                              zorder=edges_like.zorder,
                              kwargs=edges_like.kwargs,
                              key=i) 
                 for i, e in enumerate(zip(list(anz[0]), list(anz[1]),
                                           list(A[A!=0])))}
                
        g.edges = edges
        
        return g
    
        
    def makeRingGraph(angles, edges=None, edge_directions=None, node_texts=[], 
                      edge_texts=[], node_labels=None, edge_labels=None, 
                      center=(0,0), radius=0.5, tilt=0, key=0, 
                      nodes_like=CurvyNode(), edges_like=CurvyEdge()):
        '''Create a CurvyGraph with a ring structure.'''
        
        #print(center)
        
        # get angles
        if isinstance(angles, (int, float, np.integer, np.floating)):
            angles = [i*360/int(angles) for i in range(int(angles))]
        
        # get node labels
        if node_labels is None:
            node_labels = []
            for i in range(len(node_texts)):               
                new_label = nodes_like.label.copy()
                new_label.text = node_texts[i]
                node_labels = node_labels + [new_label]
        
        # get edge labels
        if edge_labels is None:
            edge_labels = []
            for i in range(len(edge_texts)):               
                new_label = edges_like.label.copy()
                new_label.text = edge_texts[i]
                edge_labels = edge_labels + [new_label]
                
        # get list of visible edges
        if edges is None:
            edges = '1'*len(angles)
                
        # get edge directions
        if edge_directions is None:
            edge_directions = '<'*len(angles)
                
        # permute edges and edge_directions
        if len(edges) > 1:
            edges = edges[-1]+edges[:-1]
        if len(edge_directions) > 1:
            edge_directions = edge_directions[-1]+edge_directions[:-1]
        
        # get adjacency matrix
        A = np.zeros((len(angles), len(angles)))
        for i in range(len(angles)):
            if edges[i] == '1':
                if edge_directions[i]=='<':
                    A[i,i-1] = 1
                elif edge_directions[i]=='>':
                    A[i,i-1] = -1
                elif edge_directions[i]=='x':
                    A[i,i-1] = 2
        #    if int(edges[i]):
        #        if edge_directions[i]=='>':
        #            if i < len(angles)-1:
        #                A[i,i+1] = 1
        #            else:
        #                A[i,0] = 1
        #        elif edge_directions[i]=='<':
        #            if i < len(angles)-1:
        #                A[i+1,i] = 1
        #            else:
        #                A[0,i] = 1
        #        elif edge_directions[i]=='x':
        #            # bidirectional edge
        #            if i < len(angles)-1:
        #                A[i,i+1] = 1
        #                A[i+1,i] = 1
        #            else:
        #                A[i,0] = 1
        #                A[0,i] = 1
        # make graph
        g = from_adjacency(A, angles=angles, node_labels=node_labels, 
                           edge_labels=node_labels, center=center, 
                           radius=radius, tilt=tilt, key=key,
                           nodes_like=nodes_like, edges_like=edges_like)
        
        return g    
     
        
    def makeStarGraph(angles, edges=None, edge_directions=None, node_texts=[], 
                      edge_texts=[], node_labels=None, edge_labels=None, 
                      center=(0,0), radius=0.5, tilt=0, key=0, 
                      nodes_like=CurvyNode(), edges_like=CurvyEdge()):
        '''Make a CurvyGraph with a star structure.'''
        
        # get angles
        if isinstance(angles, (int, float, np.integer, np.floating)):
            angles = [i*360/int(angles) for i in range(int(angles))]

        # get number of nodes and number of edges
        num_nodes = len(angles)+1
        num_edges = len(angles)
        
        # get node labels
        if node_labels is None:
            node_labels = []
            for i in range(len(node_texts)):               
                new_label = nodes_like.label.copy()
                new_label.text = node_texts[i]
                node_labels = node_labels + [new_label]
        
        # get edge labels
        if edge_labels is None:
            edge_labels = []
            for i in range(len(edge_texts)):               
                new_label = edges_like.label.copy()
                new_label.text = edge_texts[i]
                edge_labels = edge_labels + [new_label]
                
        # get edges
        if edges is None:
            edges = '1'*num_edges
                
        # get edge directions
        if edge_directions is None:
            edge_directions = '<'*num_edges
            
        # get adjacency matrix
        A = np.zeros((num_nodes, num_nodes))
        for i in range(1,num_nodes):
            if int(edges[i-1]):
                if edge_directions[i-1]=='<':
                    A[0,i] = 1
                elif edge_directions[i-1]=='>':
                    A[i,0] = 1
                elif edge_directions[i-1]=='x':
                    # bidirectional edge
                    A[0,i] = 1
                    A[i,0] = 1
                    
        # make graph
        g = from_adjacency(A, angles=[None]+angles, node_labels=node_labels, 
                           edge_labels=node_labels, center=center, 
                           radius=radius, tilt=tilt, key=key,
                           nodes_like=nodes_like, edges_like=edges_like)
        
        return g    

    
    def makeWheelGraph(angles, edges=None, edge_directions=None, node_texts=[], 
                       edge_texts=[], node_labels=None, edge_labels=None, 
                       center=(0,0), radius=0.5, tilt=0, key=0, 
                       nodes_like=CurvyNode(), edges_like=CurvyEdge(),
                       star_edges_like=None, ring_edges_like=None):
        '''Make a CurvyGraph with a wheel structure.'''
        
        # get angles
        if isinstance(angles, int):
            angles = [i*360/angles for i in range(angles)]

        
        # get number of nodes and number of edges
        num_nodes = len(angles)+1
        num_star_edges = len(angles) # same as ring graph
        
        # split edge properties
        if edges is None:
            ring_edges = None
        elif len(edges)<= num_star_edges:
            ring_edges = None
        else:
            ring_edges = edges[num_star_edges:]
            
        if edge_directions is None:
            ring_edge_directions = None
        elif len(edge_directions)<= num_star_edges:
            ring_edge_directions = None
        else:
            ring_edge_directions = edge_directions[num_star_edges:]
            
        if edge_labels is None:
            ring_edge_labels = None
        elif len(edge_labels)<= num_star_edges:
            ring_edge_labels = None
        else:
            ring_edge_labels = edge_labels[num_star_edges:]
            
        if len(edge_texts)<= num_star_edges:
            ring_edge_texts = []
        else:
            ring_edge_texts = edge_texts[num_star_edges:]  
            
        # set edge templates
        if star_edges_like is None:
            star_edges_like = edges_like
        if ring_edges_like is None:
            ring_edges_like = edges_like
        
        # make star graph
        g1 = makeStarGraph(angles, edges=edges, 
                           edge_directions=edge_directions, 
                           node_texts=node_texts, edge_texts=edge_texts, 
                           node_labels=node_labels, edge_labels=edge_labels, 
                           center=center, radius=radius, tilt=tilt, key=key, 
                           nodes_like=nodes_like, edges_like=star_edges_like)
        
        # make ring graph
        g2 = makeRingGraph(angles, edges=ring_edges, 
                           edge_directions=ring_edge_directions, 
                           node_texts=node_texts, edge_texts=ring_edge_texts, 
                           node_labels=node_labels, 
                           edge_labels=ring_edge_labels, 
                           center=center, radius=radius, tilt=tilt, key=key, 
                           nodes_like=nodes_like, edges_like=ring_edges_like)
        print(g2.edges)
        
        # merge graphs by copying edges from ring to star graph #I AM HERE
        for i in g2.edges.keys():
            edge_copy = g2.edges[i].copy()
            edge_copy.key = num_star_edges + i
            edge_copy.source = g1.nodes[edge_copy.source.key+1]
            edge_copy.target = g1.nodes[edge_copy.target.key+1]
            g1.edges[edge_copy.key] = edge_copy
        
        return g1
            