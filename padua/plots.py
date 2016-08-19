
import numpy as np

import matplotlib.pyplot as plt


from matplotlib.patches import Ellipse

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


def _areahist(ax, v, c, bins=100, by=None, alpha=1):
    """
    Plot the histogram distribution but as an area plot
    """
    y, x = np.histogram(v[~np.isnan(v)], bins)
    x = x[:-1]

    if by is None:
        by = np.zeros( (bins, ) )

    ax.fill_between(x, y, by, facecolor=c, alpha=alpha)

