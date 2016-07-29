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

from .utils import qvalues, get_protein_id, get_protein_ids, get_protein_id_list, get_shortstr, get_index_list, build_combined_label, \
                   hierarchical_match, chunks, calculate_s0_curve, find_nearest_idx

class PlotManager(object):
    """
    Plot interface for analysis models. To generate plots, use Pandas-like syntax:

    model.plot('plottype', **kwargs)
    model.plot.plottype(**kwargs)

    The list of available plots is provided for the model (view the model in Jupyter, or access the list of
    plots at `model.available_plots`).

    For information about arguments for specific plots, request help:

    ?model.plot.plottype

    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, kind=None, *args, **kwargs):
        if kind is None:
            output = []
            for kind in self.obj.default_plots:
                output.append( getattr(self.obj, 'plot_' + kind)(*args, **kwargs) )
            return output
        else:
            return getattr(self.obj, 'plot_' + kind)(*args, **kwargs)

    def __getattr__(self, kind, *args, **kwargs):
        return getattr(self.obj, 'plot_' + kind) #(*args, **kwargs)


class Analysis(object):
    parameters = {}
    available_plots = []
    default_plots = []

    def __init__(self):
        self.plot = PlotManager(self)

    def _repr_html_(self, rows=None):
        if rows is None:
            rows = []

        html = []
        html.append('<tr style="background-color:#000; color:#fff;"><th colspan="2">%s</th></tr>' % (self.name))
        if self.parameters:
            html.append('<tr><th colspan="2">Parameters</th></tr>')

        for l, val in rows:
            if l is not None:
                html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (l, val))
        s = self.name


        if self.attributes:
            html.append('<tr><th colspan="2">Attributes</th></tr>')
            for a in self.attributes:
                html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (a, type(getattr(self,a)).__name__ ))

        if self.available_plots:
            suggested_plots = ['<strong>%s</strong>' % s if s in self.default_plots else s for s in self.available_plots]
            html.append('<tr style="font-style:italic; background-color:#eee;"><th>Suggested .plot()s</th><td>%s</td></tr>' % ', '.join(suggested_plots))

        return '<table>' + ''.join(html) + '</table>'

    def set_style(self, fcol=None, ecol=None):
        """
        Apply a default set of styles to this model (overrides global default styles).
        To unset custom styles, use `set_style` to `None`.


        :param fcol:
        :param ecol:
        :return:
        """

class SklearnModel(Analysis):
    """
    Base type for working with sklearn-based models. The raw model is stored internally
    as .model, to be used for additional cv, etc.
    """
    model = None


class MultivariateSklearnModel(SklearnModel):

    data = None
    groups = None

    scores = None
    weights = None

    def _repr_html_(self, rows=None):
        if rows is None:
            rows = []

        rows.extend([
            ('Comparison', " x ".join(str(g) for g in self.groups)) if self.groups else (None, None),
            ('Number of components', self.parameters.get('n_components', None)),
            ('Mean centered?', self.parameters.get('mean_center', None)),
            ('Dimensions (features x samples)', ' x '.join(str(d) for d in self.data.shape)) if self.data is not None else (None, None),

        ])

        return super(MultivariateSklearnModel, self)._repr_html_(rows)

    # Add ellipses for confidence intervals, with thanks to Joe Kington
    # http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    def _plot_point_cov(self, points, nstd=2, **kwargs):
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
        return self._plot_cov_ellipse(cov, pos, nstd, **kwargs)


    def _plot_cov_ellipse(self, cov, pos, nstd=2, **kwargs):
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



    def _plot_scores(self, pc1=0, pc2=1, fcol=None, ecol=None, marker='o', markersize=30, label_scores=None, show_covariance_ellipse=True, **kwargs):
        """
        Plot a scores plot for two principal components as AxB scatter plot.

        Returns the plotted axis.

        :param scores: DataFrame containing scores
        :param pc1: Column indexer into scores for PC1
        :param pc2: Column indexer into scores for PC2
        :param fcol: Face (fill) color definition
        :param ecol: Edge color definition
        :param marker: Marker style (matplotlib; default 'o')
        :param markersize: int Size of the marker
        :param label_scores: Index level to label markers with
        :param show_covariance_ellipse: Plot covariance (2*std) ellipse around each grouping
        :param kwargs:
        :return: Generated axes
        """

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1,1,1)
        levels = [0,1]
        pc_n = self.scores.shape[0]

        for c in set(self.scores.columns.get_level_values('Group')):

            data = self.scores[c].values.reshape(pc_n, -1)

            fc = hierarchical_match(fcol, c, 'k')
            ec = hierarchical_match(ecol, c)

            if ec is None:
                ec = fc

            if type(markersize) == str:
                # Use as a key vs. index value in this levels
                idx = self.scores.columns.names.index(markersize)
                s = c[idx]
            elif callable(markersize):
                s = markersize(c)
            else:
                s = markersize

            ax.scatter(data[pc1,:], data[pc2,:], s=s, marker=marker, edgecolors=ec, c=fc)

            if show_covariance_ellipse and data.shape[1] > 2:
                cov = data[[pc1, pc2], :].T
                ellip = self._plot_point_cov(cov, nstd=2, linestyle='dashed', linewidth=0.5, edgecolor=ec or fc,
                                       alpha=0.8)  #**kwargs for ellipse styling
                ax.add_artist(ellip)

        if label_scores:
            scores_f = self.scores.iloc[ [pc1, pc2] ]
            idxs = get_index_list( scores_f.columns.names, label_scores )

            for n, (x, y) in enumerate(scores_f.T.values):
                r, ha = (30, 'left')
                ax.text(x, y, build_combined_label( scores_f.columns.values[n], idxs, ', '), rotation=r, ha=ha, va='baseline', rotation_mode='anchor', bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.6))

        ax.set_xlabel(self.scores.index[pc1], fontsize=16)
        ax.set_ylabel(self.scores.index[pc2], fontsize=16)
        fig.tight_layout()
        return ax


    def _plot_weights(self, pc, threshold=None, label_threshold=None, label_weights=None, **kwargs):
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

            if label_weights:

                FILTER_UP = wts.values >= label_threshold
                FILTER_DOWN = wts.values <= -label_threshold
                FILTER = FILTER_UP | FILTER_DOWN

                wti = np.arange(0, self.weights.shape[0])
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

    def plot_scores(self, pc1=0, pc2=1, fcol=None, ecol=None, marker='o', markersize=40, label_scores=None, show_covariance_ellipse=True, *args, **kwargs):
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
        scores_ax = self._plot_scores(pc1=pc1, pc2=pc2,fcol=fcol, ecol=ecol, marker=marker, markersize=markersize, label_scores=label_scores, show_covariance_ellipse=show_covariance_ellipse)
        return scores_ax.figure

    def plot_weights(self, threshold=None, label_threshold=None, label_weights=None, *args, **kwargs):
        #FIXME: This should generate a compound figure?
        weights_ax = []
        for pc in range(0, self.weights.shape[1]):
            weights_ax.append( self._plot_weights(pc, threshold=threshold, label_threshold=label_threshold, label_weights=label_weights) )

        return weights_ax



class PCA(MultivariateSklearnModel):
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

        # Store the result of the analysis in the model object
        self.parameters.update({
            'n_components': n_components,
            'mean_center': mean_center,
        })



class PLSDA(MultivariateSklearnModel):
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

        # Store the result of the analysis in the model object
        self.parameters.update({
            'n_components': n_components,
            'mean_center': mean_center,
        })

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

