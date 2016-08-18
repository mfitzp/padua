
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

    def __init__(self):

        self.parameters = []
        self.observations = []
        self.plot = PlotManager(self)

    def _repr_html_(self):
        # FIXME: Change this to use Jinja2 templating?
        html = []
        help_url = ""
        if self.help_url:
            help_url = '<a href="%s" target="_blank"><i class="fa-question-circle fa"></i></a>' % self.help_url

        html.append('<tr style="background-color:#000; color:#fff;"><th colspan="2">%s %s</th></tr>' % (self.name, help_url))
        if self.parameters:
            html.append('<tr><th colspan="2">Parameters</th></tr>')

            for l, v in self.parameters:
                if v is not None:
                    html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (l, v))

        if self.attributes:
            html.append('<tr><th colspan="2">Attributes</th></tr>')
            for a in self.attributes:
                html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (a, type(getattr(self,a)).__name__ ))

        if self.observations:
            html.append('<tr><th colspan="2">Observations</th></tr>')

            for l, v in self.observations:
                if v is not None:
                    html.append('<tr><th style="font-weight:normal; text-align:right">%s</th><td>%s</td></tr>' % (l, v))

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