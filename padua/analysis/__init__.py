import numpy as np
import pandas as pd

TYPE_REPR = {
    str: "%s",

    float: "%.4f",
    np.float16: "%.4f",
    np.float32: "%.4f",
    np.float64: "%.4f",

    int: "%d",
    np.int: "%d",
    np.int8: "%d",
    np.int16: "%d",
    np.int32: "%d",
    np.int64: "%d",

    np.uint: "%d",
    np.uint8: "%d",
    np.uint16: "%d",
    np.uint32: "%d",
    np.uint64: "%d",

    pd.DataFrame: lambda v: "DataFrame (%s)" % ("x".join(str(s) for s in v.shape)),
    np.ma.MaskedArray: lambda v: "MaskedArray (%s)" % ("x".join(str(s) for s in v.shape)),
}

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
    available_plots = []
    default_plots = []

    help_url = None
    demo_url = None
    attribute_fmt = None
    attribute_labels = None

    def __init__(self):

        self.parameters = []
        self.observations = []
        self.plot = PlotManager(self)

    def _repr_html_(self):
        # FIXME: Change this to use Jinja2 templating?
        html = []
        help_url = '<a href="%s" target="_blank" title="Go to help"><i class="fa-question-circle fa"></i></a>' % self.help_url if self.help_url else ""
        demo_url = '<a href="%s" target="_blank" title="See a demo"><i class="fa-lightbulb-o fa"></i></a>' % self.demo_url if self.demo_url else ""

        html.append('<tr style="background-color:#000; color:#fff;"><th colspan="2">%s %s %s</th></tr>' % (self.name, help_url, demo_url))
        if self.parameters:
            html.append('<tr><th colspan="2">Parameters</th></tr>')

            for l, v in self.parameters:
                if v is not None:
                    html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (l, v))

        if self.attributes:
            html.append('<tr><th colspan="2">Attributes</th></tr>')
            for a in self.attributes:
                v = getattr(self,a)
                if self.attribute_fmt is not None and a in self.attribute_fmt:
                    attribute_fmt = self.attribute_fmt.get(a)
                else:
                    attribute_fmt = TYPE_REPR.get(type(v), lambda v: type(v).__name__)

                if callable(attribute_fmt):
                    display = attribute_fmt(v)
                else:
                    display = attribute_fmt % v

                if self.attribute_labels is not None and a in self.attribute_labels:
                    a = "%s (%s)" % (self.attribute_labels[a], a)

                html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (a, display))

        if self.observations:
            html.append('<tr><th colspan="2">Observations</th></tr>')

            for l, v in self.observations:
                if v is not None:
                    html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (l, v))

        if self.available_plots:
            available_plots = ['<strong>%s</strong>' % s if s in self.default_plots else s for s in self.available_plots]
            html.append('<tr style="font-style:italic; background-color:#eee;"><th>Available .plot()s</th><td>%s</td></tr>' % ', '.join(available_plots))

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