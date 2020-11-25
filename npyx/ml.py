# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London

Machine learning tools to cluster/classify cell types.
"""

from pathlib import Path

#%% General import
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from npyx.utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, \
                    _as_array, _unique, _index_of
                    
from npyx.gl import get_good_units
from npyx.corr import acg

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

#%% Machine learning imports
# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Dimensionality reduction
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis, PCA, NMF, LatentDirichletAllocation
from sklearn import manifold
from sklearn.manifold import TSNE
import umap
# Clustering
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
#import sklearn.cluster.optics_ as op
#OPTICS=op.OPTICS
# Classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_cmapsDic(Ncmap=5):
    cmapSpectr = matplotlib.cm.get_cmap('Spectral', Ncmap)
    cmaplist = [(.5,.5,.5, 0.3)]
    for i in range(Ncmap):
        cmaplist.append(cmapSpectr(i)) # unlabeled + 4 cell types
    cmapSpectral=cmapSpectr.from_list('0-grayed Spectral cmap', cmaplist, Ncmap)

    cmapCH = matplotlib.cm.get_cmap('cubehelix', Ncmap+1) #+1 to get rig of white
    cmaplist = [(.5,.5,.5, 0.3)]
    for i in range(Ncmap):
        cmaplist.append(cmapCH(i)) # unlabeled + 4 cell types
    cmapCH=cmapCH.from_list('0-grayed cubehelix cmap', cmaplist, Ncmap)
    
    cmapTer = matplotlib.cm.get_cmap('terrain', Ncmap+1)#+1 to get rig of white
    cmaplist = [(.5,.5,.5, 0.3)]
    for i in range(Ncmap):
        cmaplist.append(cmapTer(i)) # unlabeled + 4 cell types
    cmapTer=cmapTer.from_list('0-grayed cubehelix cmap', cmaplist, Ncmap)
    
    cmapGnt = matplotlib.cm.get_cmap('gist_ncar', Ncmap+1)#+1 to get rig of white
    cmaplist = [(.5,.5,.5, 0.3)]
    for i in range(Ncmap):
        cmaplist.append(cmapGnt(i)) # unlabeled + 4 cell types
    cmapGnt=cmapGnt.from_list('0-grayed cubehelix cmap', cmaplist, Ncmap)
    
    cmapHSV = matplotlib.cm.get_cmap('hsv', Ncmap+1) # avoid redundant red
    cmaplist = [(.5,.5,.5, 0.3)]
    for i in range(Ncmap):
        cmaplist.append(cmapHSV(i)) # unlabeled + 4 cell types
    cmapHSV=cmapHSV.from_list('0-grayed hsv cmap', cmaplist, Ncmap)

#    Ncmap = min(Ncmap, 10) # Only 10 seaborn colors
#    cmaplist = [(.5,.5,.5, 0.5)]
#    for i in range(Ncmap):
#        cmaplist.append(sns.color_palette().as_hex()[i]) # unlabeled + 4 cell types
#    cmapsb = cmapSpectr.from_list('Default seaborn cmap', cmaplist, Ncmap)
    cmapsb5 = ListedColormap(['#E0E0E0', '#4c72b0', '#dd8452', '#55a868', '#c44e52'])
    
    cmap15 = ListedColormap(DistinctColors15[:Ncmap])


    cmaprgb5=cmapSpectr.from_list('RGB cmap 5 colors', [(.5,.5,.5,0.3), (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,0,1)], 5)
    
    cmapsDic={'cmapsb5':cmapsb5, 'cmapSpectral':cmapSpectral, 'cmaprgb5':cmaprgb5, 
              'cmapCH':cmapCH, 'cmapHSV':cmapHSV, 'cmapTer':cmapTer, 'cmapGnt':cmapGnt,
              'cmap15':cmap15}
    
    return cmapsDic

def feat_boxplot(ft, feat, whiskers='std'):
    assert type(feat)==str
    sns.set(style="whitegrid")
    ax1 = sns.boxplot(x=feat, data=ft, orient='h', boxprops=dict(alpha=.7), showfliers=False)
    ax = sns.swarmplot(x=feat, data=ft, color=".25")
    if whiskers=='std':
        pass
    elif whiskers=='extremum':
        pass
    fig = plt.subplot()
    return fig

def feat_violinplot(ft, feat, quality='label', xlabel=None, ylabel=None, title=None, saveDir=None, ylim=None, palette='default'):
    assert type(feat)==str
    assert quality in [None, 'label', 'label_str', 'Histology', 'Histology_str']
    sns.set(style="whitegrid")
    if type(palette)==str:
        if palette=='default':
            palette = [(.5,.5,.5, 1)]
            for i in range(5):
                palette.append((sns.color_palette()[i]))
    fig = plt.figure()
    ax = sns.violinplot(y=feat, x=quality, data=ft, palette=palette)
    if type(ylim)==list or type(ylim)==tuple:
        assert len(ylim)==2
        ax.set_ylim(ylim)
    if type(xlabel)==str:
        ax.set_xlabel(xlabel)
    if type(ylabel)==str:
        ax.set_ylabel(ylabel)
    if type(title)==str:
        ax.set_xlabel(title)
    if type(saveDir)==str:
        fig.savefig(saveDir+'/{}_{}.pdf'.format(feat, quality))
    return fig


def plot_1d_pca_components(ccg, coefficients=None, mean=0, components=None,
                           n_components=8, show_mean=True, cbin=0.1, cwin=10, ccg_title='CCG', explained_variance=None):
    if coefficients is None:
        coefficients = ccg
        
    if components is None:
        components = np.eye(len(coefficients), len(ccg))
        
    mean = np.zeros_like(ccg) + mean
        

    fig = plt.figure(figsize=(1.1 * (5 + n_components), 1.2 * 2))
    g = plt.GridSpec(2, 4 + bool(show_mean) + n_components, hspace=0.3)


    def show(fig, i, j, y, x=None, title=None, ylim=None, x_ticks=False, y_ticks=False, fontsize=12):
        ax = fig.add_subplot(g[i, j])
        if y_ticks=='left':
            ax.yaxis.tick_left()
        elif y_ticks=='right':
            ax.yaxis.tick_right()
        else:
            ax.yaxis.set_ticks([])
        if x_ticks=='bottom':
            ax.xaxis.tick_bottom()
        elif x_ticks=='top':
            ax.xaxis.tick_top()
        else:
            ax.xaxis.set_ticks([])
        x=np.arange(len(y)) if not np.any(x) else x
        ax.plot(x, y)
        if np.any(ylim): ax.set_ylim(ylim)
        if title:
            ax.set_title(title, fontsize=fontsize)

    if (cwin*1./cbin)%2==0: # even
        bins=np.arange(-cwin*1./2, cwin*1./2+cbin, cbin)
    elif (cwin*1./cbin)%2==1: # odd
        bins=np.arange(-cwin*1./2+cbin*1./2, cwin*1./2+cbin*1./2, cbin)
    
    show(fig, slice(2), slice(2), ccg, bins, ccg_title, [0,170], 'bottom', y_ticks='left', fontsize=12)
    
    approx = mean.copy()
    
    counter = 2
    if show_mean:
        show(fig, 0, 2, np.zeros_like(ccg) + mean, bins, r'$\mu$', [0,170], False, False,  fontsize=10)
        show(fig, 1, 2, approx, bins, r'$1 \cdot \mu$', [0,170], 'bottom', False, fontsize=10)
        counter += 1

    for i in range(n_components):
        approx = approx + coefficients[i] * components[i]
        top_ttl = r'$c_{0}$ : {1}'.format(i + 1, round(explained_variance[i], 3)) if np.any(explained_variance) else r'$c_{0}$'.format(i + 1)
        show(fig, 0, i + counter, components[i], bins, top_ttl, [-0.6, 0.6], False, False, fontsize=10)
        bot_ttl=r"+${0:.2f} \cdot c_{1}$".format(coefficients[i], i + 1) if ((i>0) and (coefficients[i]>=0)) else r"${0:.2f} \cdot c_{1}$".format(coefficients[i], i + 1)
        show(fig, 1, i + counter, approx, bins, bot_ttl, [0,170], 'bottom', False, fontsize=10)
#        if show_mean or i > 0:
#            plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom',
#                           transform=plt.gca().transAxes, fontsize=12)

    show(fig, slice(2), slice(-2, None), approx, bins, "= Approximation",[0,170], 'bottom', 'right', fontsize=12)
        
    return fig

def ccg_pca_plot(CCGDF, i, n_comp=8, cbin=0.1, cwin=10, saveDir='~/Desktop'):
    
    sns.set_style('white')
    pca = PCA(n_components=8)
    Xproj = pca.fit_transform(CCGDF)
    
    CCG = CCGDF.loc[CCGDF.index[i], :]
    coefs = Xproj[i]
    mean = pca.mean_
    comps = pca.components_
    exp_var = pca.explained_variance_ratio_
    ccg_title = str(CCGDF.index[i])
    
    fig = plot_1d_pca_components(CCG, coefs, mean, comps,
                           n_components=n_comp, show_mean=True, cbin=cbin, cwin=cwin, ccg_title=ccg_title, explained_variance=exp_var)
    
    if bool(saveDir): fig.savefig(saveDir+'/'+ccg_title)
    
    return fig


def import_features_table(dp, subset='ISI'):
    
    ftdp=dp+'/FeaturesTable/FeaturesTable_good.csv'
    ft = pd.read_csv(ftdp)
    
    if str(subset)=='ISI':
        subset=['ISI-mfr','ISI-rp']
    elif str(subset)=='ACG':
        subset=['ISI-mfr','ISI-rp']
    elif str(subset)=='WVF':
        subset=['ISI-mfr','ISI-rp']
    else:
        subset=['ISI-mfr','WVF-spat.dis']
    
    ft=ft.loc[:, subset[0]:subset[1]]
    
    return ft

def plot_scatter_2d(red_data, method, labels='#1f77b4', cbartitle=None, markers='o', size=15, cmap='cmaprgb5'):
    sns.set(style="whitegrid")
    if type(labels)!=str:
        cmapsDic=get_cmapsDic(len(np.unique(labels)))
        cmap=cmapsDic[cmap]
    else:
        cmap=matplotlib.cm.get_cmap('Spectral')
    mrkrs = ["o", "s", "d", '+', '*', "h", "."]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(method)
    ax.set_xlabel('{}1'.format(method));ax.set_ylabel('{}2'.format(method));
    xs, ys = red_data[:,0], red_data[:,1]
    if type(markers)!=str:
        markers=[mrkrs[i] for i in markers]
        for x_, y_, c_, m_ in zip(xs, ys, labels, markers):
            plot = ax.scatter(x_, y_, c=[c_],  marker=m_, s=size, cmap=cmap)
    else:
        plot = ax.scatter(xs, ys, c=labels, s=size, cmap=cmap)
    if type(labels) != str:
        lb=np.unique(labels)
        cbar = fig.colorbar(plot, ax=ax, ticks=np.arange(lb[0], lb[-1]+1, 1))
        cbar.set_label(str(cbartitle), rotation=270, labelpad=10)
    plt.gca().set_aspect('equal', 'datalim')
    return fig, ax

def plot_scatter_3d(red_data, method, labels='#1f77b4', cbartitle=None, markers='o', size=5, cmap='cmaprgb5'):
    sns.set(style="whitegrid")
    if type(labels)!=str:
        cmapsDic=get_cmapsDic(len(np.unique(labels)))
        cmap=cmapsDic[cmap]
    else:
        cmap=matplotlib.cm.get_cmap('Spectral')
    mrkrs = ["o", "s", "d", '+', '*', "h", "."]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(method)
    ax.set_xlabel('{}1'.format(method));ax.set_ylabel('{}2'.format(method));ax.set_zlabel('{}3'.format(method))
    xs, ys, zs = red_data[:,0], red_data[:,1], red_data[:,2]
    if type(markers)!=str:
        markers=[mrkrs[i] for i in markers]
        for x_, y_, z_, c_, m_ in zip(xs, ys, zs, labels, markers):
            plot = ax.scatter(x_, y_, z_, c=[c_], cmap=cmap, marker=m_, s=size)
    else:
        plot = ax.scatter(xs, ys, zs, c=labels, cmap=cmap, s=size)
    if type(labels) != str:
        lb=np.unique(labels)
        cbar = fig.colorbar(plot, ax=ax, ticks=np.arange(lb[0], lb[-1]+1, 1), aspect=20)
        cbar.set_label(str(cbartitle), rotation=270, labelpad=10)
    plt.gca().set_aspect('equal', 'datalim')
    
#    if method=='UMAP':
#        ax.set_xlim3d([206, 212])
#        ax.set_ylim3d([208, 213])
#        ax.set_zlim3d([207, 213])
    return fig, ax

def red_dim(df1, method='UMAP', Ncomp=2, plot=True, labels='#1f77b4', pairplot=False, **kwargs):
    '''
    df1: data frame containing fetures only
    method: dimensionality reduction method
    Ncomp: target number of dimensions (components)
    plot: boolean, make a plot or not
    labels: provide a list of labels translated as colors on the graph. Does not influence the dimensionality reduction.
    pairplot: boolean, to show or not a matrix of all features plotted against all features on Nfeatures**2 little plots.
    
    Dimensionality reduction methods:
        1] Matrix factorization
        - Principal Component Analysis - PCA +
        - truncated Singular Value Decomposition - tSVD +
        - Factor Analysis - FA +
        - Independant Component Analysis - ICA +
        - Non-negative matrix factorization - NNMF +
        - Linear AutoEncoder - LAE
        - Latent Dirichlet Allocation - LDA +
        - Generalized Low Rank Models - GLRM
        - Word2Vec - W2V
        - GloVe - GlV
        
        2] Neighbour graphs
        - manifold - Mnf +
        - Triplet Networks - TrN
        - Laplacian Eigen Maps - LEM
        - Hessian Eigen Maps - HEM
        - Local Tangent Space Alignment - LTSA
        - JSE
        - Isomap - IsM
        - Locally Linear Embedding - LLE
        - t-SNE +
        - Uniform Manifold Approimation and Pcomponentsrojection - UMAP +
    '''
    # Check inputs
    methodsList=['PCA', 'FA', 'tSVD', 'ICA', 'NNMF', 'LDA', 'Mnf', 't-SNE', 'UMAP']
    if method not in methodsList:
        print('WARNING the selected method is not in {}. Exitting.'.format(methodsList))
        return None, None, None
    
    # Scale data
    df=df1.copy()
    if method=='NNMF' or method=='LDA':
        df.loc[:,:] = MinMaxScaler().fit_transform(df.values)
    else:
        df.loc[:,:] = StandardScaler().fit_transform(df.values)
    
    # Plot scaled data on scatter matrix
    if pairplot: sns.pairplot(df)
    
    # Pick dimensionality reduction method
    random_state = kwargs['random_state'] if 'random_state' in kwargs.keys() else 42
    if method=='PCA':
        cols = ['pc%d'%i for i in range(Ncomp)]
        trans = PCA(n_components=Ncomp, svd_solver='full', random_state=random_state).fit(df.values)
        red_data = trans.transform(df.values)
        print('Explained variance: {}'.format(str([str(cols[i])+'->'+str(trans.explained_variance_ratio_[i]) for i in range(Ncomp)])))
        print('Singular values: {}'.format(str([str(cols[i])+'->'+str(trans.singular_values_[i]) for i in range(Ncomp)])))

    
    elif method=='FA':
        trans = FactorAnalysis(n_components = Ncomp, random_state=random_state).fit(df.values)
        red_data = trans.transform(df.values)
    elif method=='tSVD':
        trans = TruncatedSVD(n_components=Ncomp, random_state=random_state).fit(df.values)
        red_data = trans.transform(df.values)
    elif method=='ICA':
        trans = FastICA(n_components=Ncomp, random_state=random_state).fit(df.values)
        red_data = trans.transform(df.values)
    elif method=='NNMF':
        trans = NMF(n_components=Ncomp, init='random', random_state=random_state).fit(df.values)
        red_data = trans.transform(df.values)
    elif method=='LDA':
        trans = LatentDirichletAllocation(n_components=Ncomp, random_state=random_state).fit(df.values)
        red_data = trans.transform(df.values)
    elif method=='Mnf':
        trans = manifold.Isomap(n_neighbors=5, n_components=Ncomp, n_jobs=-1).fit(df.values)
        red_data = trans.transform(df.values)
    elif method=='t-SNE':
        # High perplexioty -> represents global structure, low perplexity -> represents locol structure
        # If the learning rate is too high, the data may end up looking like a uniform ball.
        perplexity = kwargs['perplexity'] if 'perplexity' in kwargs.keys() else 30
        early_exaggeration = kwargs['early_exaggeration'] if 'early_exaggeration' in kwargs.keys() else 12
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 30
        trans = TSNE(n_components=Ncomp, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, random_state=random_state, n_iter=5000).fit(df.values)
        red_data = trans.embedding_
    elif method=='UMAP':
        n_neighbors = kwargs['n_neighbors'] if 'n_neighbors' in kwargs.keys() else 20
        min_dist = kwargs['min_dist'] if 'min_dist' in kwargs.keys() else 0.3
        trans = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=Ncomp, random_state=random_state).fit(df.values)
        red_data = trans.embedding_
    
    
    # Plot along first dimensions

    # Check kwargs
    cbartitle = kwargs['cbartitle'] if 'cbartitle' in kwargs.keys() else 'color bar'
    markers = kwargs['markers'] if 'markers' in kwargs.keys() else 'o'
    size = kwargs['size'] if 'size' in kwargs.keys() else 15
    cmap = kwargs['cmap'] if 'cmap' in kwargs.keys() else 'cmaprgb5'
    
    # Plot
    if Ncomp>3:
        print('More than 3 components -> 3 first components only are plotted.')
        Ncomp=3
    if Ncomp==2:
        ax = plot_scatter_2d(red_data, method, labels, cbartitle, markers, size, cmap)
    elif Ncomp==3:
        ax = plot_scatter_3d(red_data, method, labels, cbartitle, markers, size, cmap)

    return red_data, trans, ax


def cluster(red_data, method='DBSCAN', plot=True, **kwargs):
    # Check inputs
    methodsList=['DBSCAN', 'hierarchical', 'UPGMA']
    if method not in methodsList:
        print('WARNING the selected method is not in {}. Exitting.'.format(methodsList))
        return
    
    # Find clustering labels
    if method=='DBSCAN':
        eps = kwargs['eps'] if 'eps' in kwargs.keys() else 1
        min_samples =  kwargs['min_samples'] if 'min_samples' in kwargs.keys() else 5
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(red_data)
        labels = clustering.labels_
        
    if method=='OPTICS':
        # Successor of DBSCAN, uses a range of eps instead of a given value.
        min_samples =  kwargs['min_samples'] if 'min_samples' in kwargs.keys() else 9
        rejection_ratio =  kwargs['rejection_ratio'] if 'rejection_ratio' in kwargs.keys() else 0.5
        clustering = OPTICS(min_samples=min_samples, rejection_ratio=rejection_ratio).fit(red_data)
        reachability = clustering.reachability_[clustering.ordering_]
        labels = clustering.labels_[clustering.ordering_]
        
        
    
    
    if plot:
        ax = plot_scatter_2d(red_data, method, labels)
#        color = ['g.', 'r.', 'b.', 'y.', 'c.']
#        for k, c in zip(range(0, 5), color):
#            Xk = X[clust.labels_ == k]
#            ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
#        ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
#        ax2.set_title('Automatic Clustering\nOPTICS')
    return labels, ax


def assess_clusters_purity(ft_, clusters, celltypes, dp):
    assert clusters.ndim==1 # Labels of the clustering
    assert celltypes.ndim==1 # Has to be ft_.loc[:,'label_str']
    
    ft=ft_.copy()
    uclusters = np.unique(clusters)
    utypes=np.unique(celltypes)
    
    df = pd.DataFrame(columns=uclusters)
    df.insert(0, 'Cell type', utypes)

    for i in df.index:
        tp=df.loc[i,'Cell type']
        df.loc[i, 1:]=np.histogram(clusters[np.array(celltypes)==str(tp)], bins=np.arange(uclusters[0]-0.5, uclusters[-1]+1.5, 1))[0]
    
    # Split in tagged and unknown cells
    df_unlabeled = df.filter(df.index[df['Cell type']=='0'], axis=0)
    df_labeled = df.filter(df.index[df['Cell type']!='0'], axis=0)
    # Extract putative cell types conditional probability: P(cell type/Cluster i)
    arr = df_labeled.values[:,1:]
    P_clusti_ct = np.apply_along_axis(lambda x: x*1./np.sum(x) if np.sum(x)!=0 else x, 1, arr)
    P_clusti = np.tile(arr.sum(axis=0)/np.sum(arr), (arr.shape[0],1))
    P_ct = np.array([(arr.sum(axis=1)/np.sum(arr)).tolist(),]*arr.shape[1]).transpose()
    
    # Remove later clusters with no tagged units (does not exist from the perspective of the tagged units)
    # i.e. where P_clusti==0
    is0=(P_clusti==0);
    P_clusti_ct[is0], P_ct[is0], P_clusti[is0]=1,1,1
    P_ct_clusti = (P_clusti_ct*P_ct)/P_clusti
    P_ct_clusti[is0]=0.0
    dfP_ct_clusti = pd.DataFrame(P_ct_clusti, index=df_labeled['Cell type'], columns=df_labeled.columns[1:])
    dfP_ct_clusti.to_csv(dp+'/cellTypesProbability.csv')
    # 1 array of probabilities per cell type of shape df.index.shape[0],1
    P_pc = np.zeros(len(clusters))
    P_mli = np.zeros(len(clusters))
    P_gc = np.zeros(len(clusters))
    P_mf = np.zeros(len(clusters))
    for cl in dfP_ct_clusti.columns:
        idx=(clusters==cl)
        P_pc[idx]=dfP_ct_clusti.loc['PC', cl]
        P_mli[idx]=dfP_ct_clusti.loc['MLI', cl]
        P_gc[idx]=dfP_ct_clusti.loc['GC', cl]
        P_mf[idx]=dfP_ct_clusti.loc['MF', cl]
    
    
    # Change cluster labels fro m-1->12 to 0->13, Convert to percentage
    rename_dic = {old:new for old,new in zip(np.arange(-1,len(uclusters)-1, 1), np.arange(0,len(uclusters),1))}
    df_unlabeled.rename(columns=rename_dic, inplace=True)
    df_labeled.rename(columns=rename_dic, inplace=True)
    df_unlabeled.loc[:,1:]=np.apply_along_axis(lambda x: x*100./np.sum(x) if np.sum(x)!=0 else x, 1, df_unlabeled.values[:,1:])
    df_labeled.loc[:,1:]=np.apply_along_axis(lambda x: x*100./np.sum(x) if np.sum(x)!=0 else x, 1, df_labeled.values[:,1:])
    
    
    # Reshape to suit seaborn factorplot function
    dfl = pd.melt(df_labeled, id_vars='Cell type', var_name="Cluster", value_name="%")
    dful = pd.melt(df_unlabeled, id_vars='Cell type', var_name="Cluster", value_name="%")
    # Plot
    g1=sns.catplot(x='Cluster', y='%', hue='Cell type', data=dfl, kind='bar', hue_order=['PC', 'MLI', 'GC', 'MF'])
    g1.fig.suptitle('Tagged cell types distribution across clusters.')
    g2=sns.catplot(x='Cluster', y='%', hue='Cell type', data=dful, kind='bar')
    g2.fig.suptitle('Unkown cell types distribution across clusters.')
    
    g1.fig.savefig(dp+'/ClustersPurity.pdf')
    g2.fig.savefig(dp+'/ClustersContent.pdf')
    
    ft['cluster']=pd.Series(clusters, index=ft.index)
    ft['Probability PC']=pd.Series(P_pc, index=ft.index)
    ft['Probability MLI']=pd.Series(P_mli, index=ft.index)
    ft['Probability GC']=pd.Series(P_gc, index=ft.index)
    ft['Probability MF']=pd.Series(P_mf, index=ft.index)
    
    ft.to_csv(dp+'/clusteredCelltypedUnits.csv')

    return ft


# def classify(labeled_set, targets, unlabeled_set, method='KNN'):
#     '''train_set should be a dataset labelled with targets, test_set unlabelled data.'''
    
#     methodsList=['SVC', 'KNN']
#     if method not in methodsList:
#         print('WARNING the selected method is not in {}. Exitting.'.format(methodsList))
#         return
    
#     # Reduce dimensionality
#     # rd is the transformed data, trans is the transformation.
#     rd, trans, ax = red_dim(unlabeled_set, method='UMAP', Ncomp=2, n_neighbors=4, min_dist=0.3, size=15, labels=hist, cbartitle='Cerebellar region')

#     # Split the testing set in two bits to evaluate it
#     X_train, X_test, y_train, y_test = train_test_split(labeled_set,
#                                                     targets,
#                                                     stratify=targets,
#                                                     random_state=42)
#     # Train the SVC
#     if method=='SVC':
#         svc = SVC().fit(X_train, y_train)
#         print('SVC score on the test set:{}'.format(svc.score(X_test, y_test)))
#     elif method=='KNN':
#         knn = KNeighborsClassifier().fit(X_train, y_train)
#         print('KNN score on the test set:{}'.format(knn.score(X_test, y_test)))
    
if __name__=='__main__':
    # Load data
    dp_='~/Dropbox/Science/PhD/Data_Analysis/3_CircuitProfiling/SfN2018_allFeatureTableCbCtx.csv'
    ft_=pd.read_csv(dp_, index_col='Unit')
    ft_untagged=ft_.loc[ft_['label']==0]
    df_PCA=ft_.copy().drop(['WVF-MainChannel', 'Histology', 'Histology_str', 'label', 'label_str', 'cluster'], axis=1)
    df_PCA_sc = StandardScaler().fit_transform(df_PCA.values)
    Ncomp=5
    trans = PCA(n_components=Ncomp, svd_solver='full', random_state=42).fit(df_PCA_sc)
    red_data = trans.transform(df_PCA_sc)
    red_data_df = pd.DataFrame(data=red_data, columns=['Component {}'.format(i+1) for i in range(Ncomp)], index=ft_.index)
    red_data_lh=red_data_df.join(ft_.loc[:, 'label'])
    red_data_lh=red_data_lh.join(ft_.loc[:, 'Histology_str']) # transformed features table with 
