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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, ast
import time
import imp
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
        self.name=self.dp.split('/')[-1]
        self.params={}; params=imp.load_source('params', op.join(self.dp,'params.py'))
        for p in dir(params):
            exec("if '__'not in '{}': self.params['{}']=params.{}".format(p, p, p))
        self.fs=self.params['sample_rate']
        self.endTime=int(np.load(op.join(self.dp, 'spike_times.npy'))[-1]*1./self.fs +1)
        
        # Create a networkX graph whose nodes are Units()
        self.dpnet=op.join(self.dp, 'network')
        if not op.isdir(self.dpnet): os.mkdir(self.dpnet)
        self.graph=nx.MultiGraph() # Undirected multigraph - directionality is given by uSrc and uTrg. Several peaks -> several edges -> multigraph.
        self.units = {u:Unit(self, u, self.graph) for u in self.get_good_units()} # Units are added to the graph when inititalized
        self.get_peak_positions()

    def get_units(self):
        return rtn.npix.gl.get_units(self.dp)
    
    def get_good_units(self):
        return rtn.npix.gl.get_good_units(self.dp)
    
    def get_peak_channels(self):
        if op.isfile(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv')):
            ft = pd.read_csv(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv'), sep=',', index_col=0)
            bestChs=np.array(ft["WVF-MainChannel"])
            gu=np.array(ft.index, dtype=np.int64)
            depthIdx = np.argsort(bestChs) # From deep to shallow
            
            self.peak_channels = {gu[depthIdx][i]:bestChs[depthIdx][i] for i in range(len(gu))}
            
        else:
            print('You need to export the features tables using phy first!!')
            return
        
    def get_peak_positions(self):
        if op.isfile(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv')):
            self.get_peak_channels()
            
            # Get peak channel xy positions
            chan_pos=chan_map_3A()[::-1] # REAL peak waveform can be on channels ignored by kilosort so importing channel_map.py does not work
            peak_pos = npa(zeros=(len(self.peak_channels), 3))
            for i, (u,c) in enumerate(self.peak_channels.items()): # find peak positions in x,y
                peak_pos[i,:]=np.append([u], (chan_pos[c]).flatten())
            self.peak_positions_real=peak_pos
            
            # Homogeneously distributes neurons on same channels around mother channel to prevent overlap
            for pp in np.unique(peak_pos[:,1:], axis=0): # space positions if several units per channel
                boolarr=(pp[0]==peak_pos[:,1])&(pp[1]==peak_pos[:,2])
                n1=sum(x > 0 for x in boolarr)
                if n1>1:
                    for i in range(n1):
                        x_spacing=16# x spacing is 32um
                        spacing=x_spacing*1./(n1+1) 
                        boolidx=np.nonzero(boolarr)[0][i]
                        peak_pos[boolidx,1]=peak_pos[boolidx,1]-x_spacing*1./2+(i+1)*spacing # 1 is index of x value
            self.peak_positions={int(pp[0]):pp[1:] for pp in peak_pos}

        else:
            print('You need to export the features tables using phy first!!')
            return
    
    def connect_graph(self, cbin=0.2, cwin=80, threshold=2, n_consec_bins=3, rec_section='all', again=False, againCCG=False, plotsfcdf=False):
        self.graph.remove_edges_from(list(self.graph.edges))
        graphs=[]
        for f in os.listdir(self.dpnet):
            if 'graph' in f:
                graphs.append(f)
        if len(graphs)>0:
            while 1:
                load_choice=input("""Saved graphs found in {}:{}.
Dial a filename index to load it, or <sfc> to build it from the significant functional correlations table:""".format(op.join(self.dp, 'graph'), ["{}:{}".format(gi, g) for gi, g in enumerate(graphs)]))
                try: # works if an int is inputted
                    load_choice=int(ast.literal_eval(load_choice))
                    self.graph=self.import_graph(op.join(self.dpnet, graphs[load_choice]))
                    print("Building Dataset.graph from file {}.".format(graphs[load_choice]))
                    if graphs[load_choice].split('.')[-1]!='gpickle':
                        print("WARNING loaded does not have gpickle format - 'unit' attribute of graph nodes are not saved in this file.")
                    break
                except: # must be a normal or empty string
                    if load_choice=='sfc':
                        print("Building graph connections from significant functional correlations table with cbin={}, cwin={}, threshold={}, n_consec_bins={}".format(cbin, cwin, threshold, n_consec_bins))
                        rtn.npix.corr.gen_sfc(self.dp, cbin, cwin, threshold, n_consec_bins, rec_section, graph=self.graph, again=again, againCCG=againCCG)
                        if plotsfcdf:rtn.npix.plot.plot_sfcdf(self.dp, cbin, cwin, threshold, n_consec_bins, text=False, markers=False, 
                                                     rec_section=rec_section, ticks=False, again = again, saveFig=True, saveDir=self.dpnet)
                        break
                    elif op.isfile(op.join(self.dpnet, load_choice)):
                        self.graph=self.import_graph(op.join(self.dpnet, load_choice))
                        print("Building Dataset.graph from file {}.".format(load_choice))
                        if load_choice.split('.')[-1]!='gpickle':
                            print("WARNING loaded does not have gpickle format - 'unit' attribute of graph nodes are not saved in this file.")
                        break
                    else:
                        print("Filename or 'sfc' misspelled. Try again.")
        else:
            print("Building graph connections from significant functional correlations table with cbin={}, cwin={}, threshold={}, n_consec_bins={}".format(cbin, cwin, threshold, n_consec_bins))
            rtn.npix.corr.gen_sfc(self.dp, cbin, cwin, threshold, n_consec_bins, rec_section, graph=self.graph, again=again, againCCG=againCCG)
            if plotsfcdf: rtn.npix.plot.plot_sfcdf(self.dp, cbin, cwin, threshold, n_consec_bins, text=False, markers=False, 
                                                     rec_section=rec_section, ticks=False, again=again, saveFig=True, saveDir=self.dpnet)

    def get_nodes(self, attributes=False):
        if attributes:
            nodes={}
            for n, na in self.graph.nodes(data=True):
                nodes[n]=na
        else:
            nodes=list(self.graph.nodes())
        
        return nodes
    
    def get_edges(self, attributes=False, keys=True):
        if attributes:
            edges={}
            for e in self.graph.edges(data=True, keys=keys):
                edges[e[0:-1]]=e[-1]
        else:
            edges=list(self.graph.edges(keys=keys))
        return edges
    
    def get_node_attributes(self, n):
        return self.get_nodes(True)[n]
    
    def get_node_attribute(self, n, at):
        assert at in ['unit', 'putativeCellType', 'classifiedCellType']
        return self.get_nodes(True)[n][at]

    def get_edge_attribute(self, e, at):
        assert at in ['uSrc','uTrg','amp','t','sign','width','label','criteria']
        if len(e)==3:
            try: # check that nodes are in the right order - multi directed graph
                return self.get_edges(True)[e][at]
            except:
                return self.get_edges(True)[(e[1],e[0],e[2])][at]
        elif len(e)==2:
            npe=npa(self.get_edges())
            keys=npe[(npe[:,0]==1183)&(npe[:,1]==1187)][:,2]
            edges={}
            for k in keys:
                try: # check that nodes are in the right order - multi directed graph
                    edges[k]=self.get_edges(True)[(e[0],e[1],k)][at]
                except:
                    edges[k]=self.get_edges(True)[(e[1],e[0],k)][at]
            return edges
    
    def get_edge_attributes(self, e):
        if len(e)==3:
            try: # check that nodes are in the right order - multi directed graph
                return self.get_edges(True)[e]
            except:
                return self.get_edges(True)[(e[1],e[0],e[2])]
        elif len(e)==2:
            npe=npa(self.get_edges())
            keys=npe[(npe[:,0]==1183)&(npe[:,1]==1187)][:,2]
            edges={}
            for k in keys:
                try: # check that nodes are in the right order - multi directed graph
                    edges[k]=self.get_edges(True)[(e[0],e[1],k)]
                except:
                    edges[k]=self.get_edges(True)[(e[1],e[0],k)]
            return edges
                
    def get_node_edges(self, u):
        return {unt:len(e_unt) for unt, e_unt in self.graph[u].items()}
    
    def label_nodes(self):
        
        for node in self.graph.nodes:
            # Plot ACG
            #rtn.npix.plot.plot_acg(self.dp, node, 0.2, 80)
            # Plot Waveform
            
            # Pull up features
            
            
            label=''
            while label=='': # if enter is hit
                label=input("\n\n || Unit {}- label?(<n> to skip):".format(node))
                if label=='n':
                    label=0 # status quo
                    break
            nx.set_node_attributes(self.graph, {node:{'putativeCellType':label}}) # update graph
            self.units[node].putativeCellType=label # Update class
            print("Label of node {} was set to {}.\n".format(node, label))
            
        
    
    def label_edges(self):
        
        edges_types=['asym_inh', 'sym_inh', 'asym_exc', 'sym_exc', 'inh_exc', 'PC_CNC', 'CS_SS', 'oscill']
        
        if self.graph.number_of_edges==0:
            print("""No edge detected - connect the graph first by calling
                  Dataset.connect_graph(cbin, cwin, threshold, n_consec_bins, rec_section, again)
                  You will be offered to load a pre-labelled graph if you ever saved one.""")
            return
        
        n_edges_init=self.graph.number_of_edges()
        for ei, edge in enumerate(list(self.graph.edges)):
            ea=self.get_edge_attributes(edge) # u1, u2, i unpacked
            
            rtn.npix.plot.plot_ccg(self.dp, [ea['uSrc'],ea['uTrg']], ea['criteria']['cbin'], ea['criteria']['cwin'])
            
            label=''
            while label=='': # if enter is hit
                print(" \n\n || Edge {}/{} ({} deleted so far)...".format(ei, n_edges_init, n_edges_init-self.graph.number_of_edges()))
                print(" || {0}->{1} (multiedge {2}) sig. corr.: \x1b[1m\x1b[36m{3:.2f}\x1b[0msd high, \x1b[1m\x1b[36m{4:.2f}\x1b[0mms wide, @\x1b[1m\x1b[36m{5:.2f}\x1b[0mms".format(ea['uSrc'], ea['uTrg'], edge[2], ea['amp'], ea['width'], ea['t']))
                print(" || Total edges of source unit: {}".format(self.get_node_edges(ea['uSrc'])))
                label=input(" || Current label: {}. New label? ({},\n || <s> to skip, <del> to delete edge, <done> to exit):".format(self.get_edge_attribute(edge,'label'), ['<{}>:{}'.format(i,v) for i,v in enumerate(edges_types)]))
                try: # Will only work if integer is inputted
                    label=ast.literal_eval(label)
                    label=edges_types[label]
                    nx.set_edge_attributes(self.graph, {edge:{'label':label}})
                    print(" || Label of edge {} was set to {}.\n".format(edge, label))
                except:
                    if label=='del':
                        self.graph.remove_edge(*edge)
                        print(" || Edge {} was deleted.".format(edge))
                        break
                    elif label=='s':
                        nx.set_edge_attributes(self.graph, {edge:{'label':0}}) # status quo
                        break
                    elif label=='':
                        print("Whoops, looks like you hit enter. You cannot leave unnamed edges. Try again.")
                    elif label=="done":
                        print(" || Done - exitting.")
                        break
                    else:
                        nx.set_edge_attributes(self.graph, {edge:{'label':label}})
                        print(" || Label of edge {} was set to {}.\n".format(edge, label))
                        break
            if label=="done":
                break
        
        while 1:
            save=input("\n\n || Do you wish to save your graph with labeled edges? <any>|<enter> to save it, else <n>:")
            if save=='n':
                print(" || Not saving. You can still save the graph later by running ds.export_graph(name, frmt) with frmt in ['edgelist', 'adjlist', 'gexf', 'gml'] (default gpickle).")
                break
            else:
                pass
            name=input(" || Saving graph with newly labelled edges as graph_<name>. Name (<t> for aaaa-mm-dd_hh:mm:ss format):")
            while 1:
                formats=['gpickle', 'gexf', 'edgelist', 'adjlist', 'gml']
                frmt=input(" || In which format? {}".format(['<{}>:{}'.format(i,v) for i,v in enumerate(formats)]))
                try: # Will only work if integer is inputted
                    frmt=ast.literal_eval(frmt)
                    frmt=formats[frmt]
                    break
                except:
                    print(" || Pick an integer between {} and {}!".format(0, len(formats)-1))
                    pass
            
            file='graph_{}.{}'.format(name, frmt)
            if op.isfile(op.join(self.dpnet,file)):
                ow=input(" || Warning, name already taken - overwrite? <y>/<n>:")
                if ow=='y':
                    print(" || Overwriting graph {}.".format(file))
                    self.export_graph(name, frmt, ow=True) # 'graph_' is always appended at the beginning of the file names. It allows to spot presaved graphs.
                    break
                elif ow=='n':
                    print(' || Ok, pick another name.')
                    pass
            else:
                print(" || Saving graph {}.".format(file))
                self.export_graph(name, frmt) # 'graph_' is always appended at the beginning of the file names. It allows to spot presaved graphs.
                break
    
    def plot_graph(self, edge_labels=True, node_labels=True):

        if not op.isfile(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv')):
            print('You need to export the features tables using phy first!!')
            return
        
        ec, ew = [], []
        for e in self.graph.edges:
            ec.append('r') if self.get_edge_attribute(e, 'sign')==-1 else ec.append('b')
            ew.append(self.get_edge_attribute(e, 'amp'))
        e_labels={e[0:2]:str(np.round(self.get_edge_attribute(e, 'amp'), 2))+'@'+str(np.round(self.get_edge_attribute(e, 't'), 1))+'ms' for e in self.graph.edges}
        
        fig, ax = plt.subplots(figsize=(6, 16))
        if node_labels:
            nlabs={}
            for node in list(self.graph.nodes):
                pct=self.get_node_attribute(node, 'putativeCellType')
                cct=self.get_node_attribute(node, 'classifiedCellType')
                l="{}".format(node)
                if pct!='':
                    l+="\nput:{}".format(pct)
                if cct!='':
                    l+="\ncla:{}".format(cct)
                nlabs[node]=l
            nx.draw_networkx_labels(self.graph,self.peak_positions,nlabs, font_weight='bold', font_color='#000000FF', font_size=8)
            #nx.draw_networkx(self.graph, pos=peak_pos, node_color='#FFFFFF00', edge_color='white', alpha=1, with_labels=True, font_weight='bold', font_color='#000000FF', font_size=6)
        nx.draw_networkx_nodes(self.graph, pos=self.peak_positions, node_color='grey', alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos=self.peak_positions, edge_color=ew, width=4, alpha=0.7, 
                               edge_cmap=plt.cm.RdBu_r, edge_vmin=-5, edge_vmax=5)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos=self.peak_positions, edge_labels=e_labels,font_color='black', font_size=8, font_weight='bold')

        ax.set_ylabel('Depth (um)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Lat. position (um)', fontsize=16, fontweight='bold')
        ax.set_ylim([4000,0])
        ax.set_xlim([0,70])
        ax.tick_params(axis='both', reset=True, labelsize=12)
        ax2 = ax.twinx()
        ax2.set_ylabel('Channel #', fontsize=16, fontweight='bold')
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels([int(yt/10 - 16) for yt in ax.get_yticks()], fontsize=12)
        ax2.set_ylim([0,4000])
        try:
            criteria=self.get_edge_attribute(list(self.graph.edges)[0], 'criteria')
            ax.set_title("Dataset:{}\n Significance criteria:{}".format(self.name, criteria))
        except:
            print('Graph not connected! Run ds.connect_graph()')
        plt.tight_layout()
        
        return fig
    
        
    
    def make_graph_directed(self):
        '''
        Should be called once the edges have been manually curated:
        - if several edges remain between pairs of nodes, the one with the biggest standard deviation is kept
        - if the edge a->b has t<-1ms: directed b->a, >1ms: directed a->b, -1ms<t<1ms: a->b AND b->a (use uSrc and uTrg to figure out who's a and b)
        '''
        # A directed graph with the same name, same nodes, 
        # and with each edge (u,v,data) replaced by two directed edges (u,v,data) and (v,u,data).
        self.digraph=self.graph.to_directed()
        
        # - if several edges remain between pairs of nodes, the one with the biggest standard deviation is kept
        # - if the edge a->b has t<-1ms: directed b->a, >1ms: directed a->b, -1ms<t<1ms: a->b AND b->a (use uSrc and uTrg to figure out who's a and b)
        #for 
    
    def export_graph(self, name='', frmt='gpickle', ow=False):
        '''
        name: any srting. If 't': will be graph_aaaa-mm-dd_hh:mm:ss
        frmt: any in ['edgelist', 'adjlist', 'gexf', 'gml'] (default gpickle)'''
        
        assert frmt in ['edgelist', 'adjlist', 'gexf', 'gml', 'gpickle']
        file=op.join(self.dpnet, 'graph_'+name+'_'+self.name+'.'+frmt)
        filePickle=op.join(self.dpnet, 'graph_'+name+'_'+self.name+'.gpickle')
        
        if op.isfile(filePickle) and not ow:
            print("File name {} already taken. PIck another name or run the function with ow=True to overwrite file.")
            return
        
        nx_exp={'edgelist':nx.write_edgelist, 'adjlist':nx.write_adjlist,'gexf':nx.write_gexf, 'gml':nx.write_gml, 'gpickle':nx.write_gpickle}
        if name=='t':
            name=time.strftime("%Y-%m-%d_%H:%M:%S")
        nx_exp['gpickle'](self.graph, filePickle) # Always export in edges list for internal compatibility
        
        if frmt!='gpickle':
            if frmt=='gml' or frmt=='gexf':
                print("GML or GEXF files can only process elements convertable into strings. Getting rid of nodes 'unit' attributes.")
                g=self.graph.copy()
                for n in g.nodes:
                    del g.nodes[n]['unit']
                if frmt=='gml':
                    nx_exp[frmt](g, file, stringizer=str) # np.int not handled... Need to explicitely tell it to use str() for non native python data types...
                elif frmt=='gexf':
                    nx_exp[frmt](g, file)
            else:
                nx_exp[frmt](self.graph, file)
            
    def import_graph(self, name):
        '''
        name: path to file to import, ending in '.edgelist', '.adjlist', '.gexf', '.gml' or '.gpickle'
        '''
        assert op.isfile(name)
        frmt=name.split('.')[-1]
        assert frmt in ['edgelist', 'adjlist', 'gexf', 'gml', 'gpickle']
        nx_exp={'edgelist':nx.read_edgelist, 'adjlist':nx.read_adjlist,'gexf':nx.read_gexf, 'gml':nx.read_gml, 'gpickle':nx.read_gpickle}
        
        return nx_exp[frmt](name)
                
    def export_feat(self, rec_section='all'):
        # TO SET HERE - RECORDING SECTIONS TO CONSIDER TO COMPUTE THE FEATURES TABLE
        endTime = int(np.load(op.join(self.dp, 'spike_times.npy'))[-1]*1./self.fs +1)# above max in seconds
#                t1 = 0 # in seconds
#                t2 = 2000 # in seconds
#                rec_section = [(t1, t2)] # list of tuples (t1, t2) delimiting the recording sections in seconds
        rec_section = 'all' # Whole recording
        print('Recording sections used to perform units features extraction:{}'.format(rec_section))
        # Select the relevant clusters, either the ones selected in phy or the ones sorted as good
        allClusters = np.unique(controller.model.spike_clusters)
        goodClusters = np.array([])
        sortedClusters = np.array([])
        for unt in allClusters:
            quality = controller.supervisor.cluster_meta.get('group', unt)
            if quality=='good':
                goodClusters = np.append(goodClusters, unt)
            if quality=='good' or quality=='mua':
                sortedClusters = np.append(sortedClusters, unt)


        # Create or load FeaturesTable
        # All the Nans will be substituted by the mean of the axis at the end.
        Features = OrderedDict()
        
        Features["ISI-mfr"] = "Mean firing rate (Hz)." 
        Features["ISI-mifr"] = "Mean instantaneous firing rate (Hz)." 
        Features["ISI-med"] = "Median of the interspike intervals distribution."
        Features["ISI-mode"] = "Mode of the interspike intervals distribution."
        Features["ISI-prct5"] = "5th percentile of the interspike intervals distribution - low 5th percentile -> more burstiness."
        Features["ISI-entropy"] = "entropy (bits/ISI), see DugueTihyLena2017 or Dorval 2009 - large entropy -> neuron fires irregularly."
        Features["ISI-cv2"] = "Average coefficient of variation for a sequence of 2 ISIs."
        Features["ISI-cv"] = "Coefficient of variation of the interspike intervals distribution (sd in units of mean)."
        Features["ISI-ir"] = "Instantaneous irregularity - average of the absolute diff between the natural log of consecutive ISIs."
        Features["ISI-lv"] = "Local variation of ISIs."
        Features["ISI-lvr"] = "Revised local variation of ISIs."
        Features["ISI-lcv"] = "Coefficient of variation of the log ISIs."
        Features["ISI-si"] = "Geometric average of the rescaled cross-correlation of ISIs."
        Features["ISI-skew"] = "Skewness of the interspike intervals distribution (ms^3)"
        Features["ISI-ra"] = "Refractory area of the max-normalized ISI distribution, from 0 to the mode (a.u.*ms)."
        Features["ISI-rp"] = "Refractory period of the ISI distribution, from 0 to when it crosses 5% of max (ms)."

        Features["ACG-burst"] = "Presence of burst or not (0 or 1)"
        Features["ACG-mode"] = "Mode of the ACG, only if mode is found meaningful. Else, Nan."
        Features["ACG-do_a"] = "Autocorrelogram dampened oscillation amplitude (AU)"
        Features["ACG-do_t"] = "Autocorrelogram dampened oscillation decay constant (ms)"
        Features["ACG-do_f"] = "Autocorrelogram dampened oscillation frequency (Hz)"
        Features["ACG-Oscillatory"] = "Autocorrelogram considered oscillatory (do_a>=30 and do_t>=2 and do_f>=3) or not (0 or 1)"
        Features["ACG-class"] = "..."
        
        Features["WVF-tpw"] = "Waveform trough to peak width (us)"
        Features["WVF-ptr"] = "Waveform peak to trough ratio (unitless)"
        Features["WVF-es"] = "Waveform end slope (uV/us)"
        Features["WVF-MainChannel"] = "Channel where average waveform has max amplitude (Irrelevant for clustering unless coupled to histology)" # PSDpeaks
        Features["WVF-spat.dis"] = "Waveform spatial distribution (um - range where amplitude undershoots 10% of maximum)"
        
        FeaturesList = Features.keys()
        dp = controller.model.dir_path
        direct = dp+"/FeaturesTable"
        ft = load_featuresTable(direct, sortedClusters, FeaturesList)

        if custom:
            if os.path.isfile(direct+'featuresTable_custom.npy'):
                pass
            
        # Set the features extractors parameters
        # CAREFUL for ACG features: have to be calculated with window=80ms, binsize=0.1ms!!
        # fs = float(controller.model.sample_rate)
        acgw = 80 #ms, by definition to compute the acg features
        acgb = 0.2 #ms, by definition to compute acg features
        # max_waveforms_per_cluster=1e3

        # Fill the table with all the features of relevant clusters
        print("Start computing the features of all %d units."%len(sortedClusters))
        bar = progressbar.ProgressBar(maxval=len(sortedClusters), \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        bar.update(0)
        
        # clustersSubset = sortedClusters[:10]
        for i, clust in enumerate(sortedClusters):
            print('WORKING ON CLUSTER %d...'%clust)
            bar.update(i+1)
            clust = int(clust)

            try:
                # InterSpike Intervals histogram features
                isint = isi(dp, clust, ret=True, sav=True, prnt=False, rec_section=rec_section)
                MFR, MIFR, medISI, modeISI, prct5ISI, entropyD, CV2, CV, IR, Lv, LvR, LcV, SI, SKW, ra, rp = compute_isi_features(isint)
                ft.loc[clust, 'ISI-mfr':'ISI-rp'] = [MFR, MIFR, medISI, modeISI, prct5ISI, entropyD, CV2, CV, IR, Lv, LvR, LcV, SI, SKW, ra, rp]
                #print(MFR, MIFR, medISI, modeISI, prct5ISI, entropyD, CV2, CV, IR, Lv, LvR, LcV, SI, SKW, ra, rp)
            except:
                print("    WARNING ISI Features of cluster %d could not be computed, and are set to NaN."%(clust))
                ft.loc[clust, 'ISI-mfr':'ISI-rp']=[np.nan]*16
            try:
                # Autocorrelogram features
#                        cluster_ids, bin_size, window_size = [clust], acgb*1./1000, acgw*1./1000
#                        spike_ids = controller.selector.select_spikes(cluster_ids,
#                                                                controller.n_spikes_correlograms,
#                                                                subset='random',
#                                                                )
#                        st = controller.model.spike_times[spike_ids]
#                        sc = controller.supervisor.clustering.spike_clusters[spike_ids]
#                        acgr = correlograms(st,
#                                            sc,
#                                            sample_rate=controller.model.sample_rate,
#                                            cluster_ids=cluster_ids,
#                                            bin_size=bin_size,
#                                            window_size=window_size,
#                                            )
                ACG=acg(dp, clust, acgb, acgw, fs=30000, normalize='Hertz', prnt=False, rec_section=rec_section)
                modes, smoothCCG, ACGburst, sm_storage, modeType = find_modes(ACG, acgw*1./1000, acgb*1./1000)#acgr[0,0,:], acgw*1./1000, acgb*1./1000)
                CCG_fitX, CCG_fitY, corParams, Oscillatory = evaluate_oscillation(smoothCCG, modes, acgb*1./1000, initial_guess=[2,2,2,2])
                ACGburst = int(ACGburst)
                ACGmode = max(modes) if (modeType=='Meaningful Mode' or Oscillatory) and max(modes)>4.5 else np.nan
                if Oscillatory:
                    ACGdo_a, ACGdo_t, ACGdo_f = corParams[0], corParams[1], corParams[2]
                else:
                    ACGdo_a, ACGdo_t, ACGdo_f = np.nan, np.nan, np.nan
                Oscillatory=1 if Oscillatory else 0 # Takes into account the None case (no fit possible) and the False case (thoughtfully considered non oscilatory)
                #print([ACGburst, ACGmode, ACGdo_a, ACGdo_t, ACGdo_f, Oscillatory, ACGrp])
                ft.loc[clust, "ACG-burst":"ACG-class"] = [ACGburst, ACGmode, ACGdo_a, ACGdo_t, ACGdo_f, Oscillatory, np.nan]
        
            except:
                print("    WARNING ACG Features of cluster %d could not be computed, and are set to NaN."%(clust))
                ft.loc[clust, "ACG-burst":"ACG-class"]=[np.nan]*7
            
            try:
                # Waveform features
                '''spike_ids = controller.selector.select_spikes([clust],
                                            max_waveforms_per_cluster,
                                            controller.batch_size_waveforms)
                channel_ids=np.arange(controller.model.n_channels_dat) #gets all channels
                wave = controller.model.get_waveforms(spike_ids, channel_ids)'''
                map_nchannels = np.load(op.join(dp, 'channel_map.npy'), mmap_mode='r').squeeze().shape[0]
                bunchs_set = controller._get_waveforms(clust, channel_ids=np.arange(0, map_nchannels, 1)) ## HERE IT SHOULD BE 384 FOR REGULAR NEUROPIXELS RECORDINGS
                wave = bunchs_set.data #
                spl, peak, trough, sp_width, pt_ratio, endslope, ind_endslope, best_channel, wvf_amp = compute_waveforms_features(wave)
                if np.isnan(sp_width):
                    best_channel = int(np.squeeze(best_channel)) # They all remain nan except from best_channel
                else:
                    sp_width, pt_ratio, endslope, best_channel = float(np.squeeze(sp_width)), float(np.squeeze(pt_ratio)), float(np.squeeze(endslope)), int(np.squeeze(best_channel))
                ft.loc[clust, "WVF-tpw":"WVF-spat.dis"] = sp_width, pt_ratio, endslope, best_channel, np.nan
                #print('sp_width, pt_ratio, endslope' , sp_width, pt_ratio, endslope)
            except:
                print("    WARNING WVF Features of cluster %d could not be computed, and are set to NaN."%(clust))
                ft.loc[clust, "WVF-tpw":"WVF-spat.dis"]=[np.nan]*5
        bar.finish()


        # Save the updated features table
        save_featuresTable(direct, ft, goodClusters, Features)                              

    
    
class Unit:
    '''The object Unit does not store anything itself except from its dataset and index.
    It makes it possible to call its various features.
    '''
    
    def __repr__(self):
        return 'Unit {} from dataset {}.'.format(self.idx, self.ds.name)
    
    def __init__(self, dataset, index, graph):
        self.ds=dataset
        self.dp = self.ds.dp
        self.idx=index
        self.putativeCellType=''
        self.classifiedCellType=''
        self.graph = graph
        self.get_peak_position()
        # self refers to the instance not the class, hehe
        self.graph.add_node(self.idx, unit=self, X=self.peak_position_real[0], Y=self.peak_position_real[1], posReal=self.peak_position_real, putativeCellType=self.putativeCellType, classifiedCellType=self.classifiedCellType) 
    
    def get_peak_channel(self):
        if op.isfile(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv')):
            ft = pd.read_csv(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv'), sep=',', index_col=0)
            self.peak_channel=ft.loc[self.idx, "WVF-MainChannel"]
            
        else:
            print('You need to export the features tables using phy first!!')
            return
        
    def get_peak_position(self):
        if op.isfile(op.join(self.dp,'FeaturesTable','FeaturesTable_good.csv')):
            self.get_peak_channel()
            
            # Get peak channel xy positions
            chan_pos=chan_map_3A()[::-1] # REAL peak waveform can be on channels ignored by kilosort so importing channel_map.py does not work
            self.peak_position_real=chan_pos[self.peak_channel].flatten() # find peak positions in x,y

        else:
            print('You need to export the features tables using phy first!!')
            return
                    
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
    
    def connections(self):
        return dict(self.graph[self.idx])
    
    # def wvf(self):
    #     return rtn.npix.spk_t.wvf(self.dp,self.idx)
    #     # TODO make the average waveform lodable by fixing io.py
        

    