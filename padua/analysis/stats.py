import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats

import warnings

from . import Analysis

from ..utils import qvalues, calculate_s0_curve

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mplimg
import matplotlib.cm as cm

from matplotlib.patches import Ellipse
from matplotlib.colors import colorConverter
from matplotlib.backends.backend_agg import FigureCanvasAgg


#ttest_1samp(a, popmean[, axis, nan_policy])	Calculates the T-test for the mean of ONE group of scores.
#ttest_ind(a, b[, axis, equal_var, nan_policy])	Calculates the T-test for the means of two independent samples of scores.
#ttest_rel(a, b[, axis, nan_policy])	Calculates the T-test on TWO RELATED samples of scores, a and b.

class TtestBase(Analysis):

    def plot_volcano(self, yp='p', fdr=0.05, threshold=2,
                     labels_from=None, labels_for=None, label_format=None, label_sig_only=True,
                     markersize=64, s0=0.00001, draw_fdr=True, ax=None,
                     xlim=None, ylim=None,
                     is_log2=False,
                     figsize=(8,10), show_numbers=True, title=None,
                     fc='grey', fc_sig='blue', fc_sigr='red'
                     ):
        """
        Volcano plot of two sample groups showing t-test p value vs. log2(fc).

        Generates a volcano plot for two sample groups, selected from `df` using `a` and `b` indexers. The mean of
        each group is calculated along the y-axis (per protein) and used to generate a log2 ratio. If a log2-transformed
        dataset is supplied set `islog2=True` (a warning will be given when negative values are present).

        A two-sample independent t-test is performed between each group. If `minimum_sample_n` is supplied, any values (proteins)
        without this number of samples will be dropped from the analysis.

        Individual data points can be labelled in the resulting plot by passing `labels_from` with a index name, and `labels_for`
        with a list of matching values for which to plot labels.

        :param fdr: `float` false discovery rate cut-off
        :param threshold: `float` log2(fc) ratio cut -off
        :param labels_from: `str` or `int` index level to get labels from
        :param labels_for: `list` of `str` matching labels to show
        :param title: `str` title for plot
        :param markersize: `int` size of markers
        :param s0: `float` smoothing factor between fdr/fc cutoff
        :param draw_fdr: `bool` draw the fdr/fc curve
        :param is_log2: `bool` is the data log2 transformed already?
        :param fillna: `float` fill NaN values with value (default: 0)
        :param label_sig_only: `bool` only label significant values
        :param ax: matplotlib `axis` on which to draw
        :param fc: `str` hex or matplotlib color code, default color of points
        :return:
        """
        if labels_from is None:
            labels_from = list(self.data.index.names)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)

        ab = self.ab

        if len(ab) == 1:
            dr, = ab

        else:
            a, b = self.ab
            if (np.any(a < 0) or np.any(b < 0) ) and not is_log2:
                warnings.warn("Input data has negative values. If data is log2 transformed, set is_log2=True.")

            if is_log2:
                dr = a - b
            else:
                dr = np.log2(b / a)

        p = self.result[yp].values

        # There are values below the fdr
        s0x, s0y, s0fn = calculate_s0_curve(s0, fdr, min(fdr/2, np.nanmin(p)), np.log2(threshold), np.nanmax(np.abs(dr)), curve_interval=0.001)

        if draw_fdr is True:
            ax.plot(s0x, -np.log10(s0y), fc_sigr, lw=1 )
            ax.plot(-s0x, -np.log10(s0y), fc_sigr, lw=1 )

        # Select data based on s0 curve
        _FILTER_IN = []

        for x, y in zip(dr, p):
            x = np.abs(x)
            spy = s0fn(x)

            if len(s0x) == 0 or x < np.min(s0x):
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
                s = self.data.index.get_level_values(markersize)
            elif callable(markersize):
                s = np.array([markersize(c) for c in self.data.index.values])
            else:
                s = np.ones((self.data.shape[0],))*markersize


            ax.scatter(dr[f], -np.log10(p[f]), c=c, s=s[f], linewidths=0, alpha=0.5)

        _FILTER_OUT1 = _FILTER_OUT & ~np.isnan(p) & (np.array(p) > fdr)
        scatter(ax, _FILTER_OUT1, fc, alpha=0.3)

        _FILTER_OUT2 = _FILTER_OUT & (np.array(p) <= fdr)
        scatter(ax, _FILTER_OUT2, fc_sig, alpha=0.3)

        if labels_for:
            idxs = get_index_list( df.index.names, labels_from )
            for shown, label, x, y in zip( _FILTER_IN , df.index.values, dr, -np.log10(p)):

                if shown or not label_sig_only:
                    label = build_combined_label( label, idxs, label_format=label_format)

                    if labels_for == True or any([l in label for l in labels_for]):
                        r, ha, ofx, ofy =  (30, 'left', 0.15, 0.02) if x >= 0 else (-30, 'right', -0.15, 0.02)
                        t = ax.text(x+ofx, y+ofy, label , rotation=r, ha=ha, va='baseline', rotation_mode='anchor', bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='none', alpha=0.4))

        scatter(ax, _FILTER_IN, fc_sigr)

        ax.set_xlabel('log$_2$ ratio')
        ax.set_ylabel('-log$_{10}$(%s)' % yp)

        # Centre the plot horizontally
        if xlim is None:
            xmin, xmax = ax.get_xlim()
            xlim = np.max(np.abs([xmin, xmax]))
        ax.set_xlim((-xlim, xlim))

        if ylim is None:
            _, ylim = ax.get_ylim()
        ax.set_ylim((0, ylim))

        if title:
            ax.set_title(title)

        if show_numbers:
            # Annotate axes with the numbers of points in each category (up & down):
            # filtered (red), Sig-filtered (blue), nonsig (grey)
            dr_up, dr_down = dr > 0, dr < 0
            grey_up, blue_up, red_up = np.sum(_FILTER_OUT1 & dr_up), np.sum(_FILTER_OUT2 & dr_up), np.sum(_FILTER_IN & dr_up)
            grey_do, blue_do, red_do = np.sum(_FILTER_OUT1 & dr_down), np.sum(_FILTER_OUT2 & dr_down), np.sum(_FILTER_IN & dr_down)

            ax.text(0.95, 0.95, '%d' % red_up, horizontalalignment='right', transform=ax.transAxes, color=fc_sigr, fontsize=14)
            ax.text(0.95, 0.90, '%d' % blue_up, horizontalalignment='right', transform=ax.transAxes, color=fc_sig, fontsize=14)
            ax.text(0.95, 0.05, '%d' % grey_up, horizontalalignment='right', transform=ax.transAxes, color=fc, fontsize=14)

            ax.text(0.05, 0.95, '%d' % red_do, horizontalalignment='left', transform=ax.transAxes, color=fc_sigr, fontsize=14)
            ax.text(0.05, 0.90, '%d' % blue_do, horizontalalignment='left', transform=ax.transAxes, color=fc_sig, fontsize=14)
            ax.text(0.05, 0.05, '%d' % grey_do, horizontalalignment='left', transform=ax.transAxes, color=fc, fontsize=14)

        return ax


    def filter_by_p(self, p):
        return self.data.iloc[ self.p >= p, :]

    def filter_by_q(self, q):
        return self.data.iloc[ self.q >= q, :]

    def _store_result(self, df, t, p, q):
        self.t = t
        self.p = p
        self.q = q

        columnv = [t,p]
        columns = ['t Statistic', 'p']

        if q is not None:
            columnv.append(q)
            columns.append('Q')

        self.result = pd.DataFrame(np.array(columnv).T, columns=columns, index=df.index)

        self.data = df

    def _generate_observations(self):
        self.observations.extend([
            ('P range', "%.3f…%.3f" % (np.nanmin(self.p), np.nanmax(self.p)) ),
            ('P ≤ 0.05', "%d" % np.sum(self.p<=0.05) ),
            ('P ≤ 0.01', "%d" % np.sum(self.p<=0.01) ),

            ('Q range', "%.3f…%.3f" % (np.nanmin(self.q), np.nanmax(self.q)) if self.q is not None else None),
            ('Q ≤ 0.05', "%d" % np.sum(self.q<=0.05) if self.q is not None else None),
            ('Q ≤ 0.01', "%d" % np.sum(self.q<=0.01) if self.q is not None else None),

            ('Number valid', np.sum(~np.isnan(self.p)) ),
            ('Number of NaN P|Q', np.sum(np.isnan(self.p)) ),
        ])



class Ttest1Sample(TtestBase):

    name = "1 Sample T-test"
    shortname = "ttest_1samp"

    attributes = ['p','t','q','result']
    available_plots = ['volcano']
    default_plots = ['volcano']

    def __init__(self, df, a, popmean, nan_policy='omit', estimate_qvalues=True, *args, **kwargs):
        super(Ttest1Sample, self).__init__(**kwargs)

        df = df.copy()
        t, p = sp.stats.ttest_1samp(df[a].values, popmean, axis=1, nan_policy=nan_policy)
        q = qvalues(p) if estimate_qvalues else None

        self._store_result(df, t, p, q)
        self.ab = df[a].mean(axis=1).values,

        # Store the result of the analysis in the model object
        self.parameters.extend([
            ('Group', a),
            ('Number of samples', df[a].shape[1] ),
            ('Population mean (μ)', popmean),
            ('NaN policy', nan_policy),
        ])

        self._generate_observations()



class TtestIndependent(TtestBase):

    name = "Independent Samples T-test"
    shortname = "ttest_ind"

    attributes = ['p','t','result']
    available_plots = ['volcano']
    default_plots = ['volcano']

    def __init__(self, df, a, b, equal_var=True, is_log2=False, nan_policy='omit', estimate_qvalues=True, *args, **kwargs):
        super(TtestIndependent, self).__init__(**kwargs)

        df = df.copy()
        t, p = sp.stats.ttest_ind(df[a].values, df[b].values, equal_var=equal_var, axis=1, nan_policy=nan_policy)
        q = qvalues(p) if estimate_qvalues else None

        self._store_result(df, t, p, q)
        self.ab = df[a].mean(axis=1).values, df[b].mean(axis=1).values

        # Store the result of the analysis in the model object
        self.parameters.extend([
            ('Test', "%s x %s" % (a, b) ),
            ('Number of samples', "%s x %s" % (df[a].shape[1], df[b].shape[1]) ),
            ('Equal variance?', equal_var),
            ('NaN policy', nan_policy),
        ])

        self._generate_observations()



class TtestRelated(TtestBase):

    name = "Related Samples T-test"
    shortname = "ttest_rel"

    attributes = ['p','t','result']
    available_plots = ['volcano']
    default_plots = ['volcano']

    def __init__(self, df, a, b, nan_policy='omit', is_log2=False, estimate_qvalues=True, *args, **kwargs):
        super(TtestRelated, self).__init__(**kwargs)

        df = df.copy()
        t, p = sp.stats.ttest_rel(df[a].values, df[b].values, axis=1, nan_policy=nan_policy)
        q = qvalues(p) if estimate_qvalues else None

        self._store_result(df, t, p, q)
        self.ab = df[a].mean(axis=1).values, df[b].mean(axis=1).values

        if np.any(df.values < 0) and not is_log2:
            warnings.warn("Input data has negative values. If data is log2 transformed, set is_log2=True.")


        self._generate_observations()

        self.parameters.extend([
            ('Group', "%s x %s" % (a, b) ),
            ('Number of samples', "%s x %s" % (df[a].shape[1], df[b].shape[1]) ),
            ('NaN policy', nan_policy),
        ])


