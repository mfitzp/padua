__author__ = 'mfitzp'

import pandas as pd
import numpy as np
import scipy as sp
from collections import defaultdict
import re
import itertools

import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec

from statsmodels.stats.multitest import multipletests
import warnings   


import matplotlib.gridspec as gridspec
from matplotlib.colors import colorConverter
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch
import matplotlib.cm as cm
 
from matplotlib.patches import Ellipse

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

"""
Visualization tools for proteomic data, using standard Pandas dataframe structures
from imported data. These functions make some assumptions about the structure of
data, but generally try to accomodate.

Depends on scikit-learn for PCA analysis
"""

from . import analysis
from . import process


def get_protein_id(s):
    return s.split(';')[0].split(' ')[0].split('_')[0] 


def get_protein_ids(s):
    return [p.split(' ')[0].split('_')[0]  for p in s.split(';') ]

   
def get_protein_id_list(df):
    protein_list = []
    for s in df.index.get_level_values(0):
        protein_list.extend( get_protein_ids(s) )
 
    return list(set(protein_list))    


def get_shortstr(s):
    return s.split(';')[0]


def get_index_list(l, ms):
    if type(ms) != list and type(ms) != tuple:
        ms = [ms]
    return [l.index(s) for s in ms if s in l]


def build_combined_label(sl, idxs):
    return ' '.join([get_shortstr(str(sl[n])) for n in idxs])


def hierarchical_match(d, k, default=None):
    '''
    Match a key against a dict, simplifying element at a time
    '''
    if type(k) == str:
        k = [k]
    for n, _ in enumerate(k):
        key = tuple(k[0:len(k)-n])
        if len(key) == 1:
            key = key[0]
        try:
            d[key]
        except:
            pass
        else:
            return d[key]

    return default


def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return np.array(out)
 
    
def calculate_s0_curve(s0, minpval, maxpval, minratio, maxratio, curve_interval=0.1):

    #maxpval, minpval  = np.nanmin(p), np.nanmax(p)
    #minratio, maxratio = np.nanmin(ratio), np.nanmax(ratio)

    mminpval = -np.log10(minpval)
    mmaxpval = -np.log10(maxpval)
    maxpval_adjust = mmaxpval - mminpval

    ax0 = (s0 + maxpval_adjust * minratio) / maxpval_adjust
    edge_offset = (maxratio-ax0) % curve_interval
    max_x = maxratio-edge_offset
    

    if (max_x > ax0):
        x = np.arange(ax0, max_x, curve_interval)
    else:
        x = np.arange(max_x, ax0, curve_interval)

    fn = lambda x: 10 ** (-s0/(x-minratio) - mminpval)
    y = fn(x)
    
    return x, y, fn    



# Add ellipses for confidence intervals, with thanks to Joe Kington
# http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def plot_point_cov(points, nstd=2, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill=False, **kwargs)

    return ellip



def _pca_scores(scores, pc1=0, pc2=1, fcol=None, ecol=None, marker='o', markersize=30, label_scores=None, show_covariance_ellipse=True, **kwargs):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    levels = [0,1]    

    for c in set(scores.columns.get_level_values('Group')):

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

        if show_covariance_ellipse and data.shape[1] > 2:
            cov = data[[pc1, pc2], :].T
            ellip = plot_point_cov(cov, nstd=2, linestyle='dashed', linewidth=0.5, edgecolor=fc,
                                   alpha=0.8)  #**kwargs for ellipse styling
            ax.add_artist(ellip)

    if label_scores:
        scores_f = scores.iloc[ [pc1, pc2] ]
        idxs = get_index_list( scores_f.columns.names, label_scores )

        for n, (x, y) in enumerate(scores_f.T.values):
            r, ha =  (30, 'left')
            ax.text(x, y, build_combined_label( scores_f.columns.values[n], idxs), rotation=r, ha=ha, va='baseline', rotation_mode='anchor', bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.4))

        
    ax.set_xlabel(scores.index[pc1], fontsize=16)
    ax.set_ylabel(scores.index[pc2], fontsize=16)
    fig.tight_layout()
    return ax


def _pca_weights(weights, pc, threshold=None, label_threshold=None, label_weights=None, **kwargs):
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(weights.iloc[:, pc])
    ylim = np.max( np.abs( weights.values ) ) * 1.1
    ax.set_ylim( -ylim, +ylim  )
    ax.set_xlim(0, weights.shape[0])
    ax.set_aspect(1./ax.get_data_ratio())
    
    wts = weights.iloc[:, pc]

    if threshold:
        if label_threshold is None:
            label_threshold = threshold

        if label_weights:

            FILTER_UP = wts.values >= label_threshold
            FILTER_DOWN = wts.values <= -label_threshold
            FILTER = FILTER_UP | FILTER_DOWN

            wti = np.arange(0, weights.shape[0])
            wti = wti[FILTER]

            idxs = get_index_list( wts.index.names, label_weights )
            for x in wti:
                y = wts.iloc[x]
                r, ha =  (30, 'left') if y >= 0 else (-30, 'left')
                ax.text(x, y, build_combined_label( wts.index.values[x], idxs), rotation=r, ha=ha, va='baseline', rotation_mode='anchor', bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.4))

        ax.axhline(threshold, 0, 1)
        ax.axhline(-threshold, 0, 1)

    ax.set_ylabel("Weights on Principal Component %d" % (pc+1), fontsize=16)
    fig.tight_layout()
    return ax
    

def pca(df, n_components=2, mean_center=False, fcol=None, ecol=None, marker='o', markersize=40, threshold=None, label_threshold=None, label_weights=None, label_scores=None, return_df=False, show_covariance_ellipse=True, *args, **kwargs):
    
    scores, weights = analysis.pca(df, n_components=n_components, *args, **kwargs)

    scores_ax = _pca_scores(scores, fcol=fcol, ecol=ecol, marker=marker, markersize=markersize, label_scores=label_scores, show_covariance_ellipse=show_covariance_ellipse)
    weights_ax = []
    
    for pc in range(0, weights.shape[1]):
        weights_ax.append( _pca_weights(weights, pc, threshold=threshold, label_threshold=label_threshold, label_weights=label_weights) )
    
    if return_df:
        return scores, weights
    else:
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


def volcano(df, a, b, fdr=0.05, labels_from=None, labels_for=None, title=None,  markersize=64, equal_var=True, s0=0.00001, is_log2=False, fillna=None, label_sig_only=True):

    df = df.copy()
    
    if np.any(df.values < 0) and not is_log2:
        warnings.warn("Input data has negative values. If data is log2 transformed, set is_log2=True.")

    if is_log2:
        df = 2 ** df
        
    if fillna:
        df = df.fillna(fillna)
        
    if labels_from is None:
        labels_from = list(df.index.names)

    # Calculate ratio between two groups
    g1, g2 = df[a].values, df[b].values
    
    dr = np.log2(  np.nanmean(g2, axis=1) / np.nanmean(g1, axis=1) )
    
    # Calculate the p value between two groups (t-test)
    t, p = sp.stats.ttest_ind(g1.T, g2.T, equal_var=equal_var) # False = Welch
    
    
    fig = plt.figure()
    fig.set_size_inches(10,10)

    ax = fig.add_subplot(1,1,1)
    
    # There are values below the fdr
    
    s0x, s0y, s0fn = calculate_s0_curve(s0, fdr, np.nanmin(p), 1, np.nanmax(np.abs(dr)), curve_interval=0.001)
    ax.plot(s0x, -np.log10(s0y), 'r', lw=1 )
    ax.plot(-s0x, -np.log10(s0y), 'r', lw=1 )

    # Select data based on s0 curve
    _FILTER_IN = []

    for x, y in zip(dr, p):
        x = np.abs(x)
        spy = s0fn(x)
        if x < np.min(s0x):
            _FILTER_IN.append(False)
            continue

        if y <= spy:
            _FILTER_IN.append(True)
        else:
            _FILTER_IN.append(False)

    _FILTER_IN = np.array(_FILTER_IN)
    _FILTER_OUT = ~ _FILTER_IN
    

    def scatter(ax, f, c, alpha=0.5):
        # lw = [float(l[0][-1])/5 for l in df[ f].index.values]
        
        if type(markersize) == str:
            # Use as a key vs. index value in this levels
            s = df.index.get_level_values(markersize)
        elif callable(markersize):
            s = np.array([markersize(c) for c in df.index.values])
        else:
            s = np.ones((df.shape[0],))*markersize
        
        
        ax.scatter(dr[f], -np.log10(p[f]), c=c, s=s[f], linewidths=0, alpha=0.5)
    

    scatter(ax, _FILTER_OUT, 'grey', alpha=0.3)

    
    if labels_for:
        idxs = get_index_list( df.index.names, labels_from )
        for shown, label, x, y in zip( _FILTER_IN , df.index.values, dr, -np.log10(p)):
            
            if shown or not label_sig_only:
                label = build_combined_label( label, idxs)
                
                if labels_for == True or any([l in label for l in labels_for]):
                    r, ha, ofx, ofy =  (30, 'left', 0.15, 0.02) if x >= 0 else (-30, 'right', -0.15, 0.02)
                    t = ax.text(x+ofx, y+ofy, label , rotation=r, ha=ha, va='baseline', rotation_mode='anchor', bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.4))

    scatter(ax, _FILTER_IN, 'blue')
    
    ax.set_ylabel('-log$_{10}$(p)')
    ax.set_xlabel('log$_2$ ratio')

    
    # Centre the plot horizontally
    xmin, xmax = ax.get_xlim()
    xlim = np.max(np.abs([xmin, xmax]))
    ax.set_xlim((-xlim, xlim))
    _, ymax = ax.get_ylim()
    ax.set_ylim((0, ymax))

    if title:
        ax.set_title(title)
    
    return ax, p, _FILTER_IN
    

def _bartoplabel(ax, name, mx, offset):
# attach some text labels
    for ii,container in enumerate(ax.containers):
        rect = container.patches[0]
        height = rect.get_height()
        rect.set_width( rect.get_width() - 0.05 )
        ax.text(rect.get_x()+rect.get_width()/2., height+150, '%.2f%%'% (name[ii]/mx),
                ha='center', va='bottom')
    
    
def modifiedaminoacids(df, kind='pie'):
    colors =   ['#6baed6','#c6dbef','#bdbdbd']
    total_aas, quants = analysis.modifiedaminoacids(df)
    
    df = pd.DataFrame()
    for a, n in quants.items():
        df[a] = [n]
    df.sort(axis=1, inplace=True)
    
    if kind == 'bar' or kind == 'both':
        ax1 = df.plot(kind='bar', figsize=(7,7), color=colors)
        ax1.set_ylabel('Number of phosphorylated amino acids')
        ax1.set_xlabel('Amino acid')
        ax1.set_xticks([])
        ylim = np.max(df.values)+1000
        ax1.set_ylim(0, ylim )
        _bartoplabel(ax1, 100*df.values[0], total_aas, ylim )

        ax1.set_xlim((-0.3, 0.3))
        return ax
    
    if kind == 'pie' or kind == 'both':

        dfp =df.T
        residues = dfp.index.values
        
        dfp.index = ["%.2f%% (%d)" % (100*df[i].values[0]/total_aas, df[i].values[0]) for i in dfp.index.values ]
        ax2 = dfp.plot(kind='pie', y=0, colors=colors)
        ax2.legend(residues, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax2.set_ylabel('')
        ax2.set_xlabel('')
        ax2.figure.set_size_inches(6,6)

        return ax2

    return ax1, ax2
    
    
def modificationlocalization(df):
    colors =  ["#78c679", "#d9f0a3", "#ffffe5"]

    lp = df['Localization prob'].values
    class_i = np.sum(lp > 0.75)
    class_ii = np.sum( (lp > 0.5) & (lp <= 0.75 ) )
    class_iii = np.sum( (lp > 0.25) & (lp <= 0.5 ) )

    cl = [class_i, class_ii, class_iii]
    total_v = class_i + class_ii + class_iii
    
    df = pd.DataFrame(cl)

    df.index = ["%.2f%% (%d)" % (100*v/total_v, v) for n, v in enumerate(cl) ]
    
    ax = df.plot(kind='pie', y=0, colors=colors)

    ax.legend(['Class I (> 0.75)', 'Class II (> 0.5 ≤ 0.75)', 'Class III (> 0.25, ≤ 0.5)'],
              loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.figure.set_size_inches(6,6)

    return ax
    
    
def box(df, s=None, title_from=None, subplots=False, figsize=(18,6), groups=None, fcol=None, ecol=None, hatch=None, ylabel="", xlabel=""):
    
    df = df.copy()

        
    if type(s) == str:
        s = [s]

    if title_from is None:
        title_from = list(df.index.names)

    # Build the combined name/info string using label_from; replace the index
    title_idxs = get_index_list( df.index.names, title_from )
    df.index = [build_combined_label(r, title_idxs) for r in df.index.values]
    
    if s:
        # Filter the table on the match string (s)
        df = df.iloc[ [all([str(si) in l for si in s]) for l in df.index.values] ]
    
    figures = []
    # Iterate each matching row, building the correct structure dataframe
    for ix in range(df.shape[0]):
        
        dfi = pd.DataFrame(df.iloc[ix]).T
        label = dfi.index.values[0]
        dfi = process.fold_columns_to_rows(dfi, levels_from=len(df.columns.names)-1)

        if subplots:
            gs = gridspec.GridSpec(1, len(subplots), width_ratios=[dfi[sp].shape[1] for sp in subplots])     
        elif isinstance(dfi.columns, pd.MultiIndex) and len(dfi.columns.levels) > 1:
            subplotl = dfi.columns.levels[0]
            gs = gridspec.GridSpec(1, len(subplotl), width_ratios=[dfi[sp].shape[1] for sp in subplots])
        else:
            # Subplots
            subplotl = [None]
            gs =  gridspec.GridSpec(1, 1) 


        first_ax = None

        fig = plt.figure(figsize=figsize)

        for n, sp in enumerate(subplotl):

            if sp is None:
                dfp = dfi
            else:
                dfp = dfi[sp]

            ax = fig.add_subplot(gs[n], sharey=first_ax)

            medians = dfp.median(axis=1, level=0).reset_index().set_index('Replicate') #.dropna(axis=1)

            if groups:
                medians = medians[ groups ]

            ax, dic = medians.plot(
                kind='box', 
                return_type = 'both',
                patch_artist=True,
                ax = ax,
            )

            ax.set_xlabel('')

            for n, c in enumerate(medians.columns.values):
                if sp is None:
                    hier = []
                else:
                    hier = [sp]  
                if type(c) == tuple:
                    hier.extend(c)
                else:
                    hier.append(c)
                    
                if fcol:
                    color = hierarchical_match(fcol, hier, None)
                    if color:
                        dic['boxes'][n].set_color( color )
                if ecol:
                    color = hierarchical_match(ecol, hier, None)
                    if color:
                        dic['boxes'][n].set_edgecolor(  )
                if hatch:
                    dic['boxes'][n].set_hatch( hierarchical_match(hatch, hier, '') )

            ax.set_xlabel(xlabel)
            ax.tick_params(axis='both', which='major', labelsize=12)

            if first_ax is None:
                first_ax = ax
            else:
                for yl in ax.get_yticklabels():
                    yl.set_visible(False)

        first_ax.set_ylabel(ylabel, fontsize=14)
        fig.subplots_adjust(wspace=0.05)

        fig.suptitle(label)
        
        figures.append(fig)
        
    return figures

    
def column_correlations(df, cmap=cm.Reds):
    df = df.copy()
    df = df.astype(float)
    
    dfc = analysis.column_correlations(df)
    dfc = dfc.values

    # Plot the distributions
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = fig.add_subplot(1,1,1)

    #vmax = np.max(dfc)
    #vmin = np.min(dfc)

    i = ax.imshow(dfc, cmap=cmap, vmin=0.5, vmax=1.0, interpolation='none')
    ax.figure.colorbar(i)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax
    
    
def _process_ix(i, idx):
    if idx is None:
        return set(i)
        
    if len(idx) > 1:
        return set( zip([i.get_level_values(l) for l in idx]) )
    else:
        return set( i.get_level_values(idx[0]) )
        
    
def venn(df1, df2, df3=None, labels=None, ix1=None, ix2=None, ix3=None, return_intersection=False):
    try:
        import matplotlib_venn as mplv
    except:
        ImportError("To plot venn diagrams, install matplotlib-venn package: pip install matplotlib-venn")
    
    if labels is None:
        labels = ["A", "B", "C"]
        
    s1 = _process_ix(df1.index, ix1)
    s2 = _process_ix(df2.index, ix2)
    if df3 is not None:
        s3 = _process_ix(df3.index, ix3)
        
        
    if df3 is not None:
        ax = mplv.venn3([s1,s2,s3], set_labels=labels)
        intersection = s1 & s2 & s3
    else:
        ax = mplv.venn2([s1,s2], set_labels=labels)
        intersection = s1 & s2

    if return_intersection:
        return ax, intersection
    else:
        return ax

        
def sitespeptidesproteins(df, labels=None, colors=None, site_localization_probability=0.75):
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(1,1,1)

    shift = 0.5
    values = analysis.sitespeptidesproteins(df, site_localization_probability)
    if labels is None:
        labels = ['Sites (Class I)', 'Peptides', 'Proteins']
    if colors is None:
        colors = ['#756bb1', '#bcbddc', '#dadaeb']

    for n, (c, l, v) in enumerate(zip(colors, labels, values)):
        ax.fill_between([0,1,2], np.array([shift,0,shift]) + n, np.array([1+shift,1,1+shift]) + n, color=c, alpha=0.5 )

        ax.text(1, 0.5 + n, "{}\n{:,}".format(l, v), ha='center', color='k', fontsize=16 )
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    
    return ax
    
    
def find_nearest_idx(array,value):
    array = array.copy()
    array[np.isnan(array)] = 1
    idx = (np.abs(array-value)).argmin()
    return idx


def rankintensity(df, colors=None, ids_from = None, number_of_annotations=5, show_go_enrichment=True, go_segments=5, go_enrichment='function', go_max_labels=8, go_fdr=0.05):
    if colors is None:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#bcbd22",
            "#17becf",
        ] * 3  # Duplicate

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)

    labels = df['Protein names'].values

    if ids_from:
        ids = df[ids_from]
    else:
        if 'Proteins' in df.columns.values:
            ids = df['Protein']
        elif 'Protein IDs' in df.columns.values:
            ids = df['Protein IDs']
        else:
            ids = None

    y = np.log10(df['Intensity'].values)

    filt = ~np.array( np.isnan(y) | np.isinf(y) )
    y = y[filt]
    labels = labels[filt]
    if ids is not None:
        ids = ids[filt]

    sort = np.argsort(y)
    y = y[sort]
    labels = labels[sort]
    if ids is not None:
        ids = ids[sort]

    x_max = y.shape[0]
    x = np.arange(x_max)

    ax.scatter( x, y, s=25, c='k', lw=0, zorder=100)
    ax.set_ylabel("Peptide intensity ($log_{10}$)")
    ax.set_xlabel("Peptide rank")

    # Defines ranges over which text can be slotted in (avoid y overlapping)
    slot_size = 0.03 # FIXME: We should calculate this
    text_y_slots = np.arange(0.1, 0.95, slot_size)

    # Build set of standard x offsets at defined y data points
    # For each y slot, find the (nearest) data y, therefore the x
    text_x_slots = []
    inv =  ax.transLimits.inverted()
    for ys in text_y_slots:
        _, yd = inv.transform( (0, ys) )
        text_x_slots.append( ax.transLimits.transform( (x[find_nearest_idx(y, yd)], 0 ) )[0] )
    text_x_slots = np.array(text_x_slots)

    text_x_slots[text_y_slots < 0.5] += 0.15
    text_x_slots[text_y_slots > 0.5] -= 0.15


    def annotate_obj(ax, n, labels, xs, ys, idx, yd, ha):
        ni = 1
        previous = []
        for l, xi, yi in zip(labels, xs, ys):
            if type(l) == str and l not in previous:
                if text_y_slots[idx]:
                    axf = text_x_slots[idx]
                    ayf = text_y_slots[idx]
                    text_x_slots[idx] = np.nan
                    text_y_slots[idx] = np.nan

                    ax.text(axf, ayf, get_shortstr(l), transform=ax.transAxes,  ha=ha, va='center', color='k')
                    ax.annotate("", xy=(xi, yi), xycoords='data',  xytext=(axf, ayf), textcoords='axes fraction',
                                va='center', color='k',
                                arrowprops=dict(arrowstyle='-', connectionstyle="arc3", ec='k', lw=1), zorder=100)
                idx += yd
                ni += 1
                previous.append(l)

            if ni > n:
                break

    _n = np.min([labels.shape[0], 20])

    annotate_obj(ax, number_of_annotations, labels[-1:-_n:-1], x[-1:-_n:-1], y[-1:-_n:-1], -1, -1, 'right')
    annotate_obj(ax, number_of_annotations, labels[:_n], x[:_n], y[:_n], 0, +1, 'left')

    if go_enrichment and ids is not None:

        shrink = x_max / 30

        for n, c in enumerate(chunks(x, go_segments)):

            mask = np.zeros((len(ids),), dtype=bool)
            mask[c] = True
            # Comparison relative to background
            gids = list(set(ids[mask]) - set(ids[~mask]))
            go = analysis.go_enrichment(gids, enrichment=go_enrichment, fdr=go_fdr)

            labels = [gi[1] for gi in go.index]

            # Filter out less specific GO terms where specific terms exist (simple text matching)
            labels_f = []
            for l in labels:
                for ll in labels:
                    if l in ll and l != ll:
                        break
                else:
                    labels_f.append(l)
            labels = labels_f[:go_max_labels]

            # Get the xrange of values for this og
            if n+1 < go_segments:
                c = c[:-shrink]
            if n > 0:
                c = c[shrink:]

            yr = ax.transLimits.transform( (0, y[c[0]]) )[1], ax.transLimits.transform( (0, y[c[-1]]) )[1]

            # find axis label point for both start and end
            if yr[0] < 0.5:
                yr = yr[0]-slot_size, yr[1]
            else:
                yr = yr[0]+slot_size, yr[1]

            yr = find_nearest_idx(text_y_slots, yr[0]), find_nearest_idx(text_y_slots, yr[1])

            yrange = list(range(yr[0], yr[1]))

            # Center ish
            if len(yrange) > len(labels):
                crop = (len(yrange) - len(labels)) // 2
                if crop > 1:
                    yrange = yrange[crop:-crop]

            # display ranked top to bottom
            for idx, l in zip(yrange, labels):
                axf = text_x_slots[idx]
                ayf = text_y_slots[idx]
                text_x_slots[idx] = np.nan
                text_y_slots[idx] = np.nan

                if ayf > 0.5:
                    ha = 'right'
                else:
                    ha = 'left'
                ax.text(axf, ayf, l, transform=ax.transAxes, ha=ha, color=colors[n])

            # Calculate GO enrichment terms for each region?
            ax.plot(c, y[c], lw=25, alpha=0.5, solid_capstyle='round', zorder=99, color=colors[n])

    return ax

    
    
def hierarchical(df, cluster_cols=True, cluster_rows=False, n_col_clusters=False, n_row_clusters=False, fcol=None, z_score=True, method='ward'):

    # helper for cleaning up axes by removing ticks, tick labels, frame, etc.
    def clean_axis(ax):
        """Remove ticks, tick labels, and frame from axis"""
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_axis_bgcolor('#ffffff')
        for sp in ax.spines.values():
            sp.set_visible(False)

    def optimize_clusters(clusters, denD, target_n):
        target_n = target_n - 1 # We return edges; not regions
        threshold = np.max(clusters)
        max_iterations = threshold

        i = 0
        while i < max_iterations:
            cc = sch.fcluster(clusters, threshold, 'distance')
            cco = cc[ denD['leaves'] ]
            edges = [n for n in range(cco.shape[0]-1) if cco[n] != cco[n+1]  ]
            n_clusters = len(edges)
            
            if n_clusters == target_n:
                break

            if n_clusters < target_n:
                threshold = threshold // 2

            elif n_clusters > target_n:
                threshold = int( threshold * 1.5 )

            i += 1

        return edges

    if z_score:
        dfc = (df - df.median(axis=0)) / df.std(axis=0)            
    else:
        dfc = df.copy()
        
    #dfc.dropna(axis=0, how='any', inplace=True)

    # make norm
    vmin = dfc.min().min()
    vmax = dfc.max().max()
    vmax = max([vmax, abs(vmin)])  # choose larger of vmin and vmax
    vmin = vmax * -1
    my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    df[np.isnan(df)] = 0
    df[np.isinf(df)] = 0

    # dendrogram single color
    sch.set_link_color_palette(['black'])

    # cluster
    if cluster_rows:
        row_pairwise_dists = distance.squareform(distance.pdist(dfc))
        row_clusters = sch.linkage(row_pairwise_dists, method=method)

    if cluster_cols:
        col_pairwise_dists = distance.squareform(distance.pdist(dfc.T))
        col_clusters = sch.linkage(col_pairwise_dists, method=method)

    # heatmap with row names
    fig = plt.figure(figsize=(12, 12))
    heatmapGS = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0, width_ratios=[0.25, 1], height_ratios=[0.25, 1])

    if cluster_cols:
        # col dendrogram
        col_denAX = fig.add_subplot(heatmapGS[0, 1])
        col_denD = sch.dendrogram(col_clusters, color_threshold=np.inf)
        clean_axis(col_denAX)

    rowGSSS = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=heatmapGS[1, 0], wspace=0.0, hspace=0.0, width_ratios=[1, 0.05])

    if cluster_rows:
        # row dendrogram
        row_denAX = fig.add_subplot(rowGSSS[0, 0])
        row_denD = sch.dendrogram(row_clusters, color_threshold=np.inf, orientation='right')
        clean_axis(row_denAX)

    row_denD = {
        'leaves':range(0, dfc.shape[0])
    }

    # row colorbar
    if fcol and 'Group' in dfc.index.names:
        class_idx = dfc.index.names.index('Group')

        classcol = [fcol[x] for x in dfc.index.get_level_values(0)[row_denD['leaves']]]
        classrgb = np.array([colorConverter.to_rgb(c) for c in classcol]).reshape(-1, 1, 3)
        row_cbAX = fig.add_subplot(rowGSSS[0, 1])
        row_axi = row_cbAX.imshow(classrgb, interpolation='nearest', aspect='auto', origin='lower')
        clean_axis(row_cbAX)

    # heatmap
    heatmapAX = fig.add_subplot(heatmapGS[1, 1])

    axi = heatmapAX.imshow(dfc.iloc[row_denD['leaves'], col_denD['leaves']], interpolation='nearest', aspect='auto', origin='lower'
                           , norm=my_norm, cmap=cm.PuOr_r)
    clean_axis(heatmapAX)

    # row labels
    if dfc.shape[0] <= 100:
        heatmapAX.set_yticks(range(dfc.shape[0]))
        heatmapAX.yaxis.set_ticks_position('right')
        ylabels = [" ".join([str(t) for t in i]) if type(i) == tuple else str(i) for i in dfc.index[row_denD['leaves']]]
        heatmapAX.set_yticklabels(ylabels)

    # col labels
    if dfc.shape[1] <= 100:
        heatmapAX.set_xticks(range(dfc.shape[1]))
        xlabels = [" ".join([str(t) for t in i]) if type(i) == tuple else str(i) for i in dfc.columns[col_denD['leaves']]]
        xlabelsL = heatmapAX.set_xticklabels(xlabels)
        # rotate labels 90 degrees
        for label in xlabelsL:
            label.set_rotation(90)

    # remove the tick lines
    for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines():
        l.set_markersize(0)

    heatmapAX.grid('off')

    if cluster_cols and n_col_clusters:
        edges = optimize_clusters(col_clusters, col_denD, n_col_clusters)
        for edge in edges:
            heatmapAX.axvline(edge +0.5, color='k', lw=3)

    if cluster_rows and n_row_clusters:
        edges = optimize_clusters(row_clusters, row_denD, n_row_clusters)
        for edge in edges:
            heatmapAX.axhline(edge +0.5, color='k', lw=3)



    return fig


def correlation(df, cm=cm.Reds, vmin=None, vmax=None):
    data = analysis.correlation(df).values

    # Plot the distributions
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(1,1,1)

    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)

    i = ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax, interpolation='none')
    fig.colorbar(i)

    return fig
