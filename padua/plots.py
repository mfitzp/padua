
import numpy as np

import matplotlib.pyplot as plt

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mplimg
import matplotlib.cm as cm

from matplotlib.patches import Ellipse
from matplotlib.colors import colorConverter
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .utils import qvalues, get_protein_id, get_protein_ids, get_protein_id_list, get_shortstr, get_index_list, build_combined_label, \
                   hierarchical_match, chunks, calculate_s0_curve, find_nearest_idx

from . import styles

def _plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
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

# Add ellipses for confidence intervals, with thanks to Joe Kington
# http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def _plot_point_cov( points, nstd=2, **kwargs):
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
    return _plot_cov_ellipse(cov, pos, nstd, **kwargs)





def scatter(df, a=0, b=1, style=None, labels_from=None, show_covariance_ellipse=True, **kwargs):
    """
    Plot a scatter plot for two principal components as AxB scatter plot.

    Returns the plotted axis.

    :param df: DataFrame containing scores
    :param a: Column indexer into scores for X
    :param b: Column indexer into scores for Y
    :param style: Style object or name
    :param labels_from: Index level to label markers with
    :param show_covariance_ellipse: Plot covariance (2*std) ellipse around each grouping
    :param kwargs:
    :return: Generated axes
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    levels = [0,1]
    pc_n = df.shape[0]

    # Convert to style object, or get default style
    style = styles.get(style)

    for c in set(df.columns.get_level_values('Group')):

        data = df[c].values.reshape(pc_n, -1)

        fc, ec, marker, markersize = style.get_values(c, ['fc','ec','marker','markersize'])
        ax.scatter(data[a,:], data[b,:], s=markersize, marker=marker, edgecolors=ec, c=fc)

        if show_covariance_ellipse and data.shape[1] > 2:
            cov = data[[a, b], :].T
            ellip = _plot_point_cov(cov, nstd=2, linestyle='dashed', linewidth=0.5, edgecolor=ec or fc,
                                   alpha=0.8)  #**kwargs for ellipse styling
            ax.add_artist(ellip)

    if labels_from:
        scores_f = df.iloc[ [a, b] ]
        idxs = get_index_list( scores_f.columns.names, labels_from )

        for n, (x, y) in enumerate(scores_f.T.values):
            r, ha = (30, 'left')
            ax.text(x, y, build_combined_label( scores_f.columns.values[n], idxs, ', ' ),
                            rotation=r, ha=ha, va='baseline', rotation_mode='anchor',
                            bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.6)
                    )

    ax.set_xlabel(df.index[a], fontsize=16)
    ax.set_ylabel(df.index[b], fontsize=16)
    fig.tight_layout()
    return ax


def box(df, s=None, title_from=None, subplots=False, figsize=(18,6), groups=None, fcol=None, ecol=None, hatch=None, ylabel="", xlabel=""):
    """
    Generate a box plot from pandas DataFrame with sample grouping.

    Plot group mean, median and deviations for specific values (proteins) in the dataset. Plotting is controlled via
    the `s` param, which is used as a search string along the y-axis. All matching values will be returned and plotted.
    Multiple search values can be provided as a `list` of `str` and these will be searched as an `and` query.

    Box fill and edge colors can be controlled on a full-index basis by passing a `dict` of indexer:color to
    `fcol` and `ecol` respectively. Box hatching can be controlled by passing a `dict` of indexer:hatch to `hatch`.


    :param df: Pandas `DataFrame`
    :param s: `str` search y-axis for matching values (case-insensitive)
    :param title_from: `list` of `str` of index levels to generate title from
    :param subplots: `bool` use subplots to separate plot groups
    :param figsize: `tuple` of `int` size of resulting figure
    :param groups:
    :param fcol: `dict` of `str` indexer:color where color is hex value or matplotlib color code
    :param ecol: `dict` of `str` indexer:color where color is hex value or matplotlib color code
    :param hatch: `dict` of `str` indexer:hatch where hatch is matplotlib hatch descriptor
    :param ylabel: `str` ylabel for boxplot
    :param xlabel: `str` xlabel for boxplot
    :return: `list` of `Figure`
    """

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
        df = df.iloc[ [all([str(si).lower() in l.lower() for si in s]) for l in df.index.values] ]

    figures = []
    # Iterate each matching row, building the correct structure dataframe
    for ix in range(df.shape[0]):

        dfi = pd.DataFrame(df.iloc[ix]).T
        label = dfi.index.values[0]
        dfi = process.fold_columns_to_rows(dfi, levels_from=len(df.columns.names)-1)

        if subplots:
            gs = gridspec.GridSpec(1, len(subplots), width_ratios=[dfi[sp].shape[1] for sp in subplots])
            subplotl = subplots
        elif isinstance(dfi.columns, pd.MultiIndex) and len(dfi.columns.levels) > 1:
            subplotl = dfi.columns.levels[0]
            gs = gridspec.GridSpec(1, len(subplotl), width_ratios=[dfi[sp].shape[1] for sp in subplotl])
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

            if groups and all([g in medians.columns.get_level_values(0)]):
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
                        dic['boxes'][n].set_edgecolor( color )
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

def _areahist(ax, v, c, bins=100, by=None, alpha=1):
    """
    Plot the histogram distribution but as an area plot
    """
    y, x = np.histogram(v[~np.isnan(v)], bins)
    x = x[:-1]

    if by is None:
        by = np.zeros( (bins, ) )

    ax.fill_between(x, y, by, facecolor=c, alpha=alpha)

