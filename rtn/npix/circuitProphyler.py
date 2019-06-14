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
import os, ast
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

def chan_map_3A():
    chan_map_el = npa([[  43.,   20.],
                           [  11.,   20.],
                           [  59.,   40.],
                           [  27.,   40.]])
    vert=npa([[  0,   40.],
              [  0,   40.],
              [  0,   40.],
              [  0,   40.]])
    
    chan_map=chan_map_el.copy()
    for i in range(96-1):
        chan_map = np.vstack((chan_map, chan_map_el+vert*(i+1)))
        
    return chan_map

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
        self.get_peak_channels()

    def get_units(self):
        return rtn.npix.gl.get_units(self.dp)
    
    def get_good_units(self):
        return rtn.npix.gl.get_good_units(self.dp)
    
    def get_peak_channels(self):
        if op.isfile(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv')):
            ft = pd.read_csv(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv'), sep=',', index_col=0)
            bestChs=np.array(ft["WVF-MainChannel"])
            depthIdx = np.argsort(bestChs) # From deep to shallow
            gu=np.array(ft.index, dtype=np.int64)[depthIdx]
            
            self.peak_channels = {gu[i]:bestChs[i] for i in range(len(gu))}
            
        else:
            print('You need to export the features tables using phy first!!')
            return
    
    def connect_graph(self, cbin=0.2, cwin=80, threshold=2, n_consec_bins=3, rec_section='all', again=False):
        rtn.npix.corr.gen_sfc(self.dp, cbin, cwin, threshold, n_consec_bins, rec_section, graph=self.graph, again=again)
    
    def gea(self, at):
        return nx.get_edge_attributes(self.graph, at)
    
    def get_edge_attribute(self, u1, u2, attribute):
        assert attribute in ['u_src','u_trg','amp','t','sign','width','label','criteria']
        al=[]
        for n in range(12): # cf. max 12 peaks by CCG (already too much)...
            try:
                al.append(self.gea(attribute)[(u1,u2,n)])
            except:
                break
        return al
    
    
    def label_nodes(self):
        
        for node in self.graph.nodes:
            # Plot ACG
            rtn.npix.plot.plot_acg(self.dp, node, 0.2, 80)
            
            label=''
            while label=='': # if enter is hit
                label=input("\n\n || Unit {}- label?(<n> to skip):".format(node))
                if label=='n':
                    label=0 # status quo
                    break
            nx.set_node_attributes(self.graph, {node:{'putative_cell_type':label}}) # update graph
            self.units[node].putative_cell_type=label # Update class
            print("Label of edge {} was set to {}.\n".format(node, label))
    
    def label_edges(self):
        
        if self.graph.number_of_edges==0:
            print("No edge detected - connect the graph first by calling Dataset.connect_graph(cbin, cwin, threshold, n_consec_bins, rec_section, again)")
            return
        
        for edge in self.graph.edges:
            u_src=self.gea('u_src')[edge]
            u_trg=self.gea('u_trg')[edge]
            amp=self.gea('amp')[edge]
            t=self.gea('t')[edge]
            width=self.gea('width')[edge]
            criteria=self.gea('criteria')[edge]
            
            rtn.npix.plot.plot_ccg(self.dp, [u_src,u_trg], criteria['cbin'], criteria['cwin'])
            
            label=''
            while label=='': # if enter is hit
                label=input("\n\n || {}->{} sig. corr. of {}s.d., {}ms wide, @{}ms - label?(<n> to skip):".format(u_src, u_trg, amp, width, t))
                if label=='n':
                    label=0 # status quo
                    break
            nx.set_edge_attributes(self.graph, {edge:{'label':label}})
            print("Label of edge {} was set to {}.\n".format(edge, label))
    
    def print_graph(self):
        print(self.graph.adj)
    
    def plot_graph(self, edge_labels=True, node_labels=True):
        chan_pos=chan_map_3A() # REAL peak waveform can be on channels ignored by kilosort
        #chan_map=np.load(op.join(self.dp, 'channel_map.npy')).flatten()
        peak_pos = {u:(chan_pos[c]+npa([-3+6*np.random.rand(),0])).flatten() for u,c in self.peak_channels.items()}
        ec, ew = [], []
        for e in self.graph.edges:
            ec.append('r') if self.gea('sign')[e]==-1 else ec.append('b')
            ew.append(self.gea('amp')[e])
        e_labels={e[0:2]:str(np.round(self.gea('amp')[e], 2))+'@'+str(np.round(self.gea('t')[e], 1))+'ms' for e in self.graph.edges}
        
        fig, ax = plt.subplots(figsize=(6, 16))
        if node_labels:
            nx.draw_networkx(self.graph, pos=peak_pos, node_color='#FFFFFF00', edge_color='white', alpha=1, with_labels=True, font_weight='bold', font_color='#000000FF', font_size=6)
        nx.draw_networkx_nodes(self.graph, pos=peak_pos, node_color='grey', alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos=peak_pos, edge_color=ew, width=4, alpha=0.7, 
                               edge_cmap=plt.cm.RdBu_r, edge_vmin=-5, edge_vmax=5)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos=peak_pos, edge_labels=e_labels,font_color='black', font_size=6, font_weight='bold')
        ax.tick_params(axis='both', reset=True, labelsize=10)
        #ax.invert_yaxis()
        ax.set_ylabel('Depth (um)', fontsize=12)
        ax.set_xlabel('Lat. position (um)', fontsize=12)
        ax.set_ylim([3840,0])
        ax.set_xlim([0,70])
        criteria=self.gea('criteria')[list(self.graph.edges)[0]]
        ax.set_title("Dataset:{}\n Significance criteria:{}".format(self.dp.split('/')[-1], criteria))
        plt.tight_layout()
        
    def export_graph(self, frmt='edgelist'):
        assert frmt in ['edgelist', 'adjlist', 'gexf']
        nx_exp={'edgelist':nx.write_edgelist, 'adjlist':nx.write_adjlist,'gexf':nx.write_gexf}
        nx_exp[frmt](self.graph, op.join(self.dp, 'graph'))

    
    
class Unit:
    '''The object Unit does not store anything itself except from its dataset and index.
    It makes it possible to call its various features.
    '''
    
    def __repr__(self):
        return 'Unit {} from dataset {}.'.format(self.idx, self.dp.split('/')[-1])
    
    def __init__(self, datapath, index, graph):
        self.dp = datapath
        self.idx=index
        self.putative_cell_type=''
        self.classified_cell_type=''
        self.graph = graph
        # self refers to the instance not the class, hehe
        self.graph.add_node(self.idx, unit=self, putative_cell_type=self.putative_cell_type, classified_cell_type=self.classified_cell_type) 
        
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
        

    