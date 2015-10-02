__author__ = 'mfitzp'

import pandas as pd
import numpy as np
from collections import defaultdict
import re
import itertools

import matplotlib.pyplot as plt

"""
Visualization tools for proteomic data, using standard Pandas dataframe structures
from imported data. These functions make some assumptions about the structure of
data, but generally try to accomodate.

Depends on scikit-learn for PCA analysis
"""

from . import analysis



def get_protein_id(s):
    return s.split(';')[0].split(' ')[0].split('_')[0] 

def get_protein_ids(s):
    return [p.split(' ')[0].split('_')[0]  for p in s.split(';') ]

def get_protein_id_list(df):
    protein_list = []
    for s in df.index.get_level_values(0):
        protein_list.extend( get_protein_ids(s) )
 
    return list(set(protein_list))    

def hierarchical_match(d, k, default=None):
    '''
    Match a key against a dict, simplifying element at a time
    '''
    for n, _ in enumerate(k):
        key = k[0:len(k)-n]
        if len(key) == 1:
            key = key[0]
        try:
            d[key]
        except:
            pass
        else:
            return d[key]

    return default

def _pca_scores(scores, pc1=0, pc2=1, fcol=None, ecol=None, marker='o', markersize=30):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    levels = [0,1]    

    for c in set(scores.columns.values):

        try:
            data = scores.loc[:,c].values.reshape(2,-1)
        except:
            continue

        fc = hierarchical_match(fcol, c, 'k')
        ec = hierarchical_match(ecol, c)
        
        if ec is None:
            ec = fc

        if type(markersize) == str:
            # Use as a key vs. index value in this levels
            idx = scores.columns.names.index(markersize)
            s = c[idx]
        elif callable(markersize):
            s = markersize(c)
        else:
            s = markersize

        ax.scatter(data[pc1,:], data[pc2,:], s=s, marker=marker, edgecolors=ec, c=fc)
    
    ax.set_xlabel(scores.index[pc1], fontsize=16)
    ax.set_ylabel(scores.index[pc2], fontsize=16)
    
    return ax


def _pca_weights(weights, pc, threshold=None, label_with=None):
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(weights.iloc[:, pc])
    ylim = np.max( np.abs( weights.values ) ) * 1.1
    ax.set_ylim( -ylim, +ylim  )
    ax.set_xlim(0, weights.shape[0])
    ax.set_aspect(1./ax.get_data_ratio())
    
    wts = weights.iloc[:, pc]
    
    if threshold:
        FILTER_UP = wts.values >= threshold
        FILTER_DOWN = wts.values <= -threshold
        FILTER = FILTER_UP | FILTER_DOWN
        
        wti = np.arange(0, weights.shape[0])
        wti = wti[FILTER]
        if label_with:
            idx = wts.index.names.index(label_with)
            for x in wti:
                y = wts.iloc[x]
                r, ha =  (30, 'left') if y >= 0 else (-30, 'left')
                ax.text(x, y, get_protein_id(wts.index.values[x][idx]), rotation=r, ha=ha, va='baseline', rotation_mode='anchor', bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.4))
                #ax.annotate(get_protein_id(wts.index.values[x][idx]),(x, y) )

        ax.axhline(threshold, 0, 1)
        ax.axhline(-threshold, 0, 1)

    ax.set_ylabel("Weights on Principal Component %d" % (pc+1), fontsize=16)
            
    return ax
    

def pca(df, n_components=2, mean_center=False, fcol=None, ecol=None, marker='o', markersize=None, threshold=None, label_with=None, *args, **kwargs):
    
    scores, weights = analysis.pca(df, n_components=n_components, *args, **kwargs)

    scores_ax = _pca_scores(scores, fcol=fcol, ecol=ecol, marker=marker, markersize=markersize)
    weights_ax = []
    
    for pc in range(0, weights.shape[1]):
        weights_ax.append( _pca_weights(weights, pc, threshold=threshold, label_with=label_with) )
    
        
    return scores_ax, weights_ax

    
def enrichment(df):

    result = analysis.enrichment(df)

    axes = result.plot(kind='pie', subplots=True, figsize=(result.shape[1]*4, 3))
    for n, ax in enumerate(axes):
        #ax.legend().set_visible(False)
        total = result.values[1,n] + result.values[0,n]
        ax.annotate("%.1f%%" % (100 * result.values[0,n]/total), 
                 xy=(0.3, 0.6),  
                 xycoords='axes fraction',
                 color='w',
                 size=22)
        ax.set_xlabel( ax.get_ylabel(), fontsize=22)
        ax.set_ylabel("")
        ax.set_aspect('equal', 'datalim')

    return axes

    