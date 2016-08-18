import pandas as pd
import numpy as np
import collections
import jinja2

try:
    import sklearn
except ImportError:
    sklearn = False
else:
    from sklearn import decomposition, cross_decomposition


import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mplimg
import matplotlib.cm as cm

from matplotlib.patches import Ellipse
from matplotlib.colors import colorConverter
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ..utils import qvalues, get_protein_id, get_protein_ids, get_protein_id_list, get_shortstr, get_index_list, build_combined_label, \
                   hierarchical_match, chunks, calculate_s0_curve, find_nearest_idx

from .. import styles
from .. import plots
from . import SklearnModel



class DecompositionSklearnModel(SklearnModel):

    data = None
    groups = None

    scores = None
    weights = None


    def _plot_weights(self, pc, threshold=None, label_threshold=None, labels_from=None, **kwargs):
        """

        :param weights:
        :param pc:
        :param threshold:
        :param label_threshold:
        :param label_weights:
        :param kwargs:
        :return:
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.weights.iloc[:, pc].values)
        ylim = np.max( np.abs( self.weights.values ) ) * 1.1
        ax.set_ylim( -ylim, +ylim  )
        ax.set_xlim(0, self.weights.shape[0])
        ax.set_aspect(1./ax.get_data_ratio())

        wts = self.weights.iloc[:, pc]

        if threshold:
            if label_threshold is None:
                label_threshold = threshold

            if labels_from:

                FILTER_UP = wts.values >= label_threshold
                FILTER_DOWN = wts.values <= -label_threshold
                FILTER = FILTER_UP | FILTER_DOWN

                wti = np.arange(0, self.weights.shape[0])
                wti = wti[FILTER]

                idxs = get_index_list( wts.index.names, labels_from )
                for x in wti:
                    y = wts.iloc[x]
                    r, ha =  (30, 'left') if y >= 0 else (-30, 'left')
                    ax.text(x, y, build_combined_label( wts.index.values[x], idxs), rotation=r, ha=ha, va='baseline', rotation_mode='anchor', bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.4))

            ax.axhline(threshold, 0, 1)
            ax.axhline(-threshold, 0, 1)

        ax.set_ylabel("Weights on Principal Component %d" % (pc+1), fontsize=16)
        fig.tight_layout()
        return ax

    def plot_scores(self, pc1=0, pc2=1, style=None, labels_from=None, show_covariance_ellipse=True, *args, **kwargs):
        """
        Plot Scores
        :param fcol:
        :param ecol:
        :param marker:
        :param markersize:
        :param label_scores:
        :param show_covariance_ellipse:
        :param args:
        :param kwargs:
        :return:
        """
        # FIXME: Generate scores x/y plot? or provide list of matches
        scores_ax = plots.scatter(self.scores, a=pc1, b=pc2, style=None, labels_from=labels_from, show_covariance_ellipse=show_covariance_ellipse)
        return scores_ax.figure

    def plot_weights(self, threshold=None, label_threshold=None, labels_from=None, *args, **kwargs):
        #FIXME: This should generate a compound figure?
        weights_ax = []
        for pc in range(0, self.weights.shape[1]):
            weights_ax.append( self._plot_weights(pc, threshold=threshold, label_threshold=label_threshold, labels_from=labels_from) )

        return weights_ax



class PCA(DecompositionSklearnModel):
    """
    Perform Principal Component Analysis (PCA) from input DataFrame and generate scores and weights plots.

    Principal Component Analysis is a technique for identifying the largest source of variation in a dataset. This
    function uses the implementation available in scikit-learn. The PCA is calculated via `analysis.pca` and will
    therefore give identical results.

    Resulting scores and weights plots are generated showing the distribution of samples within the resulting
    PCA space. Sample color and marker size can be controlled by label, lookup and calculation (lambda) to
    generate complex plots highlighting sample separation.

    For further information see the examples included in the documentation.

    :param df: Pandas `DataFrame`
    :param n_components: `int` number of Principal components to return
    :param mean_center: `bool` mean center the data before performing PCA
    :param fcol: `dict` of indexers:colors, where colors are hex colors or matplotlib color names
    :param ecol: `dict` of indexers:colors, where colors are hex colors or matplotlib color names
    :param marker: `str` matplotlib marker name (default "o")
    :param markersize: `int` or `callable` which returns an `int` for a given indexer
    :param threshold: `float` weight threshold for plot (horizontal line)
    :param label_threshold: `float` weight threshold over which to draw labels
    :param label_weights: `list` of `str`
    :param label_scores: `list` of `str`
    :param return_df: `bool` return the resulting scores, weights as pandas DataFrames
    :param show_covariance_ellipse: `bool` show the covariance ellipse around each group
    :param args: additional arguments passed to analysis.pca
    :param kwargs: additional arguments passed to analysis.pca
    :return:
    """

    name = "Principal Component Analysis (PCA)"
    shortname = "PCA"
    help_url = "http://padua.readthedocs.io/en/latest/analysis.html#padua.analysis.pca"

    attributes = ['scores','weights','model','data']
    available_plots = ['scores','weights','loadings','3dscores']
    default_plots = ['scores','weights']

    def __init__(self, df, n_components=2, mean_center=False, *args, **kwargs):
        super(PCA, self).__init__(**kwargs)

        if not sklearn:
            assert('This library depends on scikit-learn (sklearn) to perform PCA analysis')

        df = df.copy()

        # We have to zero fill, nan errors in PCA
        df[ np.isnan(df) ] = 0

        if mean_center:
            mean = np.mean(df.values, axis=0)
            df = df - mean

        pca = decomposition.PCA(n_components=n_components, **kwargs)
        pca.fit(df.values.T)

        self.scores = pd.DataFrame(pca.transform(df.values.T)).T
        self.scores.index = ['Principal Component %d (%.2f%%)' % ( (n+1), pca.explained_variance_ratio_[n]*100 ) for n in range(0, self.scores.shape[0])]
        self.scores.columns = df.columns

        self.weights = pd.DataFrame(pca.components_).T
        self.weights.index = df.index
        self.weights.columns = ['Weights on Principal Component %d' % (n+1) for n in range(0, self.weights.shape[1])]

        self.model = pca
        self.data = df

        # Store the parameters of the analysis
        self.parameters.extend([
            ('Number of components', n_components),
            ('Number of samples', df.shape[1]),
            ('Mean centered?', mean_center),
            ('Dimensions (features x samples)', ' x '.join(str(d) for d in self.data.shape) if self.data is not None else (None, None)),
        ])


class PLSDA(DecompositionSklearnModel):
    """
    Partial Least Squares Discriminant Analysis, based on `sklearn.cross_decomposition.PLSRegression`

    Performs a binary group partial least squares discriminant analysis (PLS-DA) on the supplied
    dataframe, selecting the first ``n_components``.

    Sample groups are defined by the selectors ``a`` and ``b`` which are used to select columns
    from the supplied dataframe. The result model is applied to the entire dataset,
    projecting non-selected samples into the same space.

    For more information on PLS regression and the algorithm used, see the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html>`_.

    :param df: Pandas ``DataFrame`` to perform the analysis on
    :param a: Column selector for group a
    :param b: Column selector for group b
    :param n_components: ``int`` number of components to select
    :param mean_center: ``bool`` mean center the data before performing PLS regression
    :param kwargs: additional keyword arguments to `sklearn.cross_decomposition.PLSRegression`
    :return: scores ``DataFrame`` of PLSDA scores n_components x n_samples
             weights ``DataFrame`` of PLSDA weights n_variables x n_components
    """

    name = "Partial Least Squares Discriminant Analysis (PLS-DA)"
    shortname = "PLS-DA"

    loadings = None

    attributes = ['scores','weights','loadings','model','data']
    available_plots = ['scores','weights','loadings','3dscores']
    default_plots = ['scores','weights']

    def __init__(self, df, a, b, n_components=2, mean_center=False, scale=True, **kwargs):
        super(PLSDA, self).__init__(**kwargs)


        if not sklearn:
            assert('This library depends on scikit-learn (sklearn) to perform PLS-DA')



        df = df.copy()
        df[ np.isnan(df) ] = 0

        if mean_center:
            mean = np.mean(df.values, axis=0)
            df = df - mean

        sxa, _ = df.columns.get_loc_level(a)
        sxb, _ = df.columns.get_loc_level(b)

        dfa = df.iloc[:, sxa]
        dfb = df.iloc[:, sxb]

        dff = pd.concat([dfa, dfb], axis=1)
        y = np.ones(dff.shape[1])
        y[np.arange(dfa.shape[1])] = 0

        plsr = cross_decomposition.PLSRegression(n_components=n_components, scale=scale, **kwargs)
        plsr.fit(dff.values.T, y)

        # Apply the generated model to the original data
        x_scores = plsr.transform(df.values.T)

        self.scores = pd.DataFrame(x_scores.T)
        self.scores.index = ['Latent Variable %d' % (n+1) for n in range(0, self.scores.shape[0])]
        self.scores.columns = df.columns

        self.weights = pd.DataFrame(plsr.x_weights_)
        self.weights.index = df.index
        self.weights.columns = ['Weights on Latent Variable %d' % (n+1) for n in range(0, self.weights.shape[1])]

        self.loadings = pd.DataFrame(plsr.x_loadings_)
        self.loadings.index = df.index
        self.loadings.columns = ['Loadings on Latent Variable %d' % (n+1) for n in range(0, self.loadings.shape[1])]

        self.model = plsr
        self.data = df
        self.groups = [a,b]

        # Store the parameters of the analysis
        self.parameters.extend([
            ('Comparison', " x ".join(str(g) for g in self.groups) if self.groups else (None, None)),
            ('Number of samples', " x ".join(str(df[g].shape[1]) for g in self.groups) if self.groups else (None, None)),
            ('Number of components', n_components),
            ('Mean centered?', mean_center),
            ('Dimensions (features x samples)', ' x '.join(str(d) for d in self.data.shape) if self.data is not None else (None, None)),
        ])
