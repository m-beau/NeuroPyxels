# -*- coding: utf-8 -*-
"""
2019-06-13
@author: Maxime Beau, Neural Computations Lab, University College London

WARNING - DO NOT TRY TO ACTUALLY STORE DATA IN THE DATASET OBJECT, THIS WAS THE FLAW OF XTRADATAMANAGER!
INSTEAD, MEMORYMAP EVERYTHING IN THE ROUTINESMEMORY FOLDER
THE DATASET AND SUBSEQUENT CLASSES SHOULD BE CONECPTUALLY NOTHING BUT
1) WAYS TO INITIATE HIGH DATA PROCESSING
2) SYMBOLIC LINKS TO THE PROCESSED DATA STORED ON THE MACHINE

Circuit prophyler: set of classes whose end goal is to represent a Neuropixels Dataset as a network
with nodes being characterized as cell types
and edges as putative connections.

It is exploiting all the basix Neuropixels routines as well as the python module NetworkX.

The way to use it is as follow:

DPs = {'dataset name 1:', path/to/dataset1,...} # dic listing the datasets paths
ds1 = Dataset(DPs[]) # returns an instance of Dataset()

# Find out connections of unit x
ux = ds1.units[x] # returns an instance of Unit()

ux.print_connections()
# if first time, will ask to first generate the connections by calling Dataset.determine_connections()


ux.print_connections(format=dataframe)
# if connections have already been established, returns the list of partners with a significant functional correlation.
# For format=dataframe, a Dataframe with connected_partners as indices and connection_attributes as columns

ux.print_connections(format=graph)
# For format=graph, a networkx style graph where nodes are unit INDICES and edges are connections whose weight is the SIGNED (+ or -) height in standard deviations from baseline.
"""
import os
import os.path as op
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import rtn
import rtn.npix as npix
from rtn.utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, \
                    _as_array, _unique, _index_of
                    
import networkx as nx

class Dataset:
    '''
    >>> dp = path/to/kilosort/output
    >>> ds = Dataset(dp)
    
    Structure of ds: ds generated a networkx graph whose nodes correspond to the dataset units labeled as good.
    Bear in mind that the graph nodes are actual objects, instances of Unit(), making this structure more powerful
    Because the relevant rtn methods can be integrated either in the Dataset class (connections related stuff) or the Units class (individual units related stuff)
    
    If you want to access unit u:
    >>> ds.units[u]
    >>> ds.units[u].trn(): equivalent to rtn.npix.spk_t.trn(dp,u)
    
    The units can also be accessed through the 'unit' attributes of the graph nodes:
    >>> ds.graph.nodes[u]]['unit'].trn() returns the same thing as ds.units[u].trn()
    '''
    
    def __repr__(self):
        return 'Neuropixels dataset at {}.'.format(self.dp)
    
    def __init__(self, datapath):
        self.dp = op.expanduser(datapath)
        
        # Create a networkX graph whose nodes are Units()
        if not op.isdir(op.join(self.dp, 'graph')): os.mkdir(op.join(self.dp, 'graph'))
        self.graph=nx.MultiGraph() # Undirected multigraph - directionality is given by u_src and u_trg. Several peaks -> several edges -> multigraph.
        self.units = {u:Unit(self.dp,u, self.graph) for u in self.get_good_units()} # Units are added to the graph when inititalized
        
    def get_units(self):
        return rtn.npix.gl.get_units(self.dp)
    
    def get_good_units(self):
        return rtn.npix.gl.get_good_units(self.dp)
    
    def print_graph(self):
        print(self.graph.adj)
    
    def correlate_graph(self, cbin=0.2, cwin=80, threshold=2, n_consec_bins=3, rec_section='all', again=False):
        rtn.npix.corr.gen_sfc(self.dp, cbin, cwin, threshold, n_consec_bins, rec_section, graph=self.graph, again=again)
        
        
    

    
    
class Unit:
    '''The object Unit does not store anything itself except from its dataset and index.
    It makes it possible to call its various features.
    '''
    
    def __repr__(self):
        return 'Unit {} from dataset {}.'.format(self.idx, self.dp.split('/')[-1])
    
    def __init__(self, datapath, index, graph):
        self.dp = datapath
        self.idx=index
        self.graph = graph
        self.graph.add_node(self.idx, unit=self) # self refers to the instance not the graph
        
    def trn(self, rec_section='all'):
        return rtn.npix.spk_t.trn(self.dp, self.idx, rec_section=rec_section)
    
    def trnb(self, bin_size, rec_section='all'):
        return rtn.npix.spk_t.trnb(self.dp, self.idx, bin_size, rec_section=rec_section)
    
    def ids(self):
        return rtn.npix.spk_t.ids(self.dp, self.idx)
    
    def isi(self, rec_section='all'):
        return rtn.npix.spk_t.isi(self.dp, self.idx, rec_section=rec_section)
    
    def acg(self, cbin, cwin, normalize='Hertz', rec_section='all'):
        return rtn.npix.corr.acg(self.dp, self.idx, bin_size=cbin, win_size=cwin, normalize=normalize, rec_section=rec_section)
    
    def correlations(self):
        return dict(self.graph[self.idx])
    
    # def wvf(self):
    #     return rtn.npix.spk_t.wvf(self.dp,self.idx)
    #     # TODO make the average waveform lodable by fixing io.py
        

    