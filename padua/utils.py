import numpy as np
import scipy as sp
import scipy.interpolate


def qvalues(pv, m = None, verbose = False, lowmem = False, pi0 = None):
    """
    Copyright (c) 2012, Nicolo Fusi, University of Sheffield
    All rights reserved.

    Estimates q-values from p-values

    Args
    =====

    m: number of tests. If not specified m = pv.size
    verbose: print verbose messages? (default False)
    lowmem: use memory-efficient in-place algorithm
    pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
         For most GWAS this is not necessary, since pi0 is extremely likely to be
         1

    :param pv:
    :param m:
    :param verbose:
    :param lowmem:
    :param pi0:
    :return:
    """

    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel() # flattens the array in place, more efficient than flatten() 

    if m == None:
        m = float(len(pv))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 == None:
        pi0 = 1.0
    elif pi0 != None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = sp.arange(0, 0.90, 0.01)
        counts = sp.array([(pv > i).sum() for i in sp.arange(0, 0.9, 0.01)])
        
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = sp.array(pi0)

        # fit natural cubic spline
        tck = sp.interpolate.splrep(lam, pi0, k = 3)
        pi0 = sp.interpolate.splev(lam[-1], tck)
        
        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0


    if lowmem:
        # low memory version, only uses 1 pv and 1 qv matrices
        qv = sp.zeros((len(pv),))
        last_pv = pv.argmax()
        qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
        pv[last_pv] = -sp.inf
        prev_qv = last_pv
        for i in range(int(len(pv))-2, -1, -1):
            cur_max = pv.argmax()
            qv_i = (pi0*m*pv[cur_max]/float(i+1))
            pv[cur_max] = -sp.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = sp.argsort(pv)    
        pv = pv[p_ordered]
        qv = pi0 * m/len(pv) * pv
        qv[-1] = min(qv[-1],1.0)

        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])
        
        # reorder qvalues
        qv_temp = qv.copy()
        qv = sp.zeros_like(qv)
        qv[p_ordered] = qv_temp

        # reshape qvalues
        qv = qv.reshape(original_shape)
        
    return qv


def get_protein_id(s):
    """
    Return a shortened string, split on spaces, underlines and semicolons.

    Extract the first, highest-ranked protein ID from a string containing
    protein IDs in MaxQuant output format: e.g. P07830;P63267;Q54A44;P63268

    Long names (containing species information) are eliminated (split on ' ') and
    isoforms are removed (split on '_').

    :param s:  protein IDs in MaxQuant format
    :type s: str or unicode
    :return: string
    """
    return str(s).split(';')[0].split(' ')[0].split('_')[0]


def get_protein_ids(s):
    """
    Return a list of shortform protein IDs.

    Extract all protein IDs from a string containing
    protein IDs in MaxQuant output format: e.g. P07830;P63267;Q54A44;P63268

    Long names (containing species information) are eliminated (split on ' ') and
    isoforms are removed (split on '_').

    :param s:  protein IDs in MaxQuant format
    :type s: str or unicode
    :return: list of string ids
    """
    return [p.split(' ')[0].split('_')[0]  for p in s.split(';') ]


def get_protein_id_list(df, level=0):
    """
    Return a complete list of shortform IDs from a DataFrame

    Extract all protein IDs from a dataframe from multiple rows containing
    protein IDs in MaxQuant output format: e.g. P07830;P63267;Q54A44;P63268

    Long names (containing species information) are eliminated (split on ' ') and
    isoforms are removed (split on '_').

    :param df: DataFrame
    :type df: pandas.DataFrame
    :param level: Level of DataFrame index to extract IDs from
    :type level: int or str
    :return: list of string ids
    """
    protein_list = []
    for s in df.index.get_level_values(level):
        protein_list.extend( get_protein_ids(s) )

    return list(set(protein_list))


def get_shortstr(s):
    """
    Return the first part of a string before a semicolon.

    Extract the first, highest-ranked protein ID from a string containing
    protein IDs in MaxQuant output format: e.g. P07830;P63267;Q54A44;P63268

    :param s:  protein IDs in MaxQuant format
    :type s: str or unicode
    :return: string
    """
    return str(s).split(';')[0]


def get_index_list(l, ms):
    """

    :param l:
    :param ms:
    :return:
    """
    if type(ms) != list and type(ms) != tuple:
        ms = [ms]
    return [l.index(s) for s in ms if s in l]


def build_combined_label(sl, idxs, sep=' ', label_format=None):
    """
    Generate a combined label from a list of indexes
    into sl, by joining them with `sep` (str).

    :param sl: Strings to combine
    :type sl: dict of str
    :param idxs: Indexes into sl
    :type idxs: list of sl keys
    :param sep:

    :return: `str` of combined label
    """
    if label_format:
        return label_format % tuple([get_shortstr(str(sl[n])) for n in idxs])
    else:
        return sep.join([get_shortstr(str(sl[n])) for n in idxs])


def hierarchical_match(d, k, default=None):
    """
    Match a key against a dict, simplifying element at a time


    :param df: DataFrame
    :type df: pandas.DataFrame
    :param level: Level of DataFrame index to extract IDs from
    :type level: int or str
    :return: hiearchically matched value or default
    """
    if d is None:
        return default

    if type(k) != list and type(k) != tuple:
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
    """
    Separate `seq` (`np.array`) into `num` series of as-near-as possible equal
    length values.

    :param seq: Sequence to split
    :type seq: np.array
    :param num: Number of parts to split sequence into
    :type num: int
    :return: np.array of split parts
    """

    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return np.array(out)


def calculate_s0_curve(s0, minpval, maxpval, minratio, maxratio, curve_interval=0.1):
    """
    Calculate s0 curve for volcano plot.

    Taking an min and max p value, and a min and max ratio, calculate an smooth
    curve starting from parameter `s0` in each direction.

    The `curve_interval` parameter defines the smoothness of the resulting curve.

    :param s0: `float` offset of curve from interset
    :param minpval: `float` minimum p value
    :param maxpval: `float` maximum p value
    :param minratio: `float` minimum ratio
    :param maxratio: `float` maximum ratio
    :param curve_interval: `float` stepsize (smoothness) of curve generator
    :return: x, y, fn  x,y points of curve, and fn generator
    """

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


def find_nearest_idx(array,value):
    """

    :param array:
    :param value:
    :return:
    """
    array = array.copy()
    array[np.isnan(array)] = 1
    idx = (np.abs(array-value)).argmin()
    return idx
