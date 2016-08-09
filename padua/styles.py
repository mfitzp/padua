"""
Styles manager for PaDuA plotting functions.

This defines a number of default styles for plotting PaDuA plots, mapping via
ordinal class names to specific colors. Styles consist of fill and edge colors
(fill colors are used for edge colours if not defined) and hashes.

The matplotlib project cycler is used to auto-generate colors for a given group/class
which has not been previously defined. Direct mappings can be provided where the colors
for a given class/mapping have already been decided.

"""

from cycler import cycler
from collections import defaultdict, Iterable
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io
import urllib
import base64

# Current default-in-use style
_current = 'brewer9set1'

# Color lists for default sets

# Color Brewer color schemes are Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State University.
# under Apache v2 license (see LICENSE file for full license)
brewer8accent = [ '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666']
brewer8dark2 = [ '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666' ]
brewer8paired = [ '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00' ]
brewer8pastel1 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec']
brewer8pastel2 = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc']
brewer8set1 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
brewer8set2 = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
brewer8set3 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']

brewer9paired = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6']
brewer9pastel1 = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2']
brewer9set1 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
brewer9set3 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']

brewer12paired = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
                  '#cab2d6', '#6a3d9a', '#ffff99', '#b15928' ]
brewer12set3 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
                '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f' ]
# End Color Brewer schemes

defaults = {
    'fc': '#000000',
    'ec': '#000000',
    'hatch': '',
    'marker': 'o',
    'markersize': 50,
    'lw': 1.5,
    'ls':'-',
}

_scatter_filter = {'fc':'facecolors', 'ec':'edgecolors', 'marker':'marker', 'markersize':'s'}
_patch_filter = {'fc':'fc', 'ec':'ec', 'hatch':'hatch'}
_line_filter = {'fc':'c', 'lw':'lw'}


def filter_dict(d, f):
    return {f[k]:v for k,v in d.items() if k in f}

def apply_defaults(d):
    global defaults
    return {k: d[k] if k in d else v for k, v in defaults.items() }

def _normalize_key(k):

    if isinstance(k, Iterable) and not isinstance(k, str):
        if len(k) == 1:
            # Single elements converted to bare strings/items
            return k[0]
        else:
            # Sequences converted to (hashable) tuples
            return tuple([i for i in k])
    else:
        return k

def base64mplfigure(fig):
    img = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(img, format='png', bbox_inches='tight')
    img.seek(0)  # rewind the data
    return 'data:image/png;base64,' + urllib.parse.quote(base64.b64encode(img.getbuffer()))


class Style(object):

    cycler = None
    _dict = None

    def __repr__(self):
        return "<Style '%s'> %s" % ( self.name, list( zip(self._dict.keys(), self._dict.values()) ) )
    def _repr_html_(self):

        html = []
        html.append('<tr style="background-color:#000; color:#fff;"><th colspan="100">Style \'%s\'</th></tr>' % self.name)
        #if self.description:
            #html.append('<tr><td colspan="100">%s</td></tr>' % self.description )

        n_styles = len(self._cycler)

        fig = plt.Figure(figsize=(n_styles,2))
        fig.set_canvas(plt.gcf().canvas)
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_axis_off()

        wobble = [0.1,0.25]

        # Build example map of all available styles defined on this cycler
        for n, s in enumerate(iter( self._cycler )):
            s = apply_defaults(s)

            ax.add_patch(patches.Rectangle((n/n_styles, 0.5), 1/n_styles, 1,  **filter_dict(s, _patch_filter)))
            ax.scatter((n+0.5)/n_styles, -0.25, **filter_dict(s, _scatter_filter) )
            ax.plot([n/n_styles, (n+1)/n_styles], wobble, **filter_dict(s, _line_filter) )

            wobble = wobble[::-1]

        html.append('<tr><td colspan="100" cellpadding=0 style="text-align:center; padding:0;"><img src="%s"></td></tr>' % base64mplfigure(fig))

        """
        Output the key values in the data (columns) and a column table of the data set of each value?
        """
        def get_all_keys(d):
            keys = []
            for k, v in d.items():
                keys.extend(v.keys())
            return sorted(list(set(keys)))

        all_keys = get_all_keys(self._dict)
        if all_keys:
            html.append('<tr><th>In use</th>%s</tr>' % ''.join(['<th>%s</th>' % s for s in all_keys]))
            for k, v in self._dict.items():
                htmla = []
                for ak in all_keys:
                    if ak in v:
                        if v[ak].startswith('#'): #FIXME: attribute lookup
                            htmla.append('<td style="background-color:%s">&nbsp;</td>' % v[ak])

                        else:
                            htmla.append('<td>%s</td>' % v[ak])
                    else:
                        htmla.append('<td></td>')


                html.append('<tr><td>%s</td>%s</tr>' % (k, ''.join(htmla)))

        print(all_keys)


        return '<table>' + ''.join(html) + '</table>'

        #return '<img src="%s">' % base64mplfigure(fig)


    def __init__(self, name='unnamed', description=None, levels=1, **kwargs):

        self.name = name
        self.description = description
        self._levels = levels

        if 'fc' in kwargs and 'ec' not in kwargs:
            kwargs['ec'] = kwargs['fc']

        self._cycler = cycler(**{k:v for k,v in kwargs.items() if v is not None})
        self.reset()


    def get(self, k):
        """
        Get the style information for a specific requested class pattern
        :param k:
        :return:
        """

        k = _normalize_key(k)

        # If specific match is in the dictionary, return hit
        try:
            return apply_defaults(self._dict[k])
        except (KeyError, TypeError):
            pass

        # If passed an iterable, and levels is set (normally 1), truncate the match index
        if isinstance(k, Iterable) and self._levels is not None:
            k = k[:self._levels]

        return apply_defaults(self._dict[k])

    def get_values(self, k, values):
        s = self.get(k)
        return (s[i] if i in s else defaults[i] for i in values)

    def set(self, k, **kwargs):
        """
        Set a defined color for a specific matching class pattern
        :param k:
        :param fc:
        :param ec:
        :param hatch:
        :return:
        """
        k = _normalize_key(k)

        #d = {'fc':fc, 'ec':ec, 'hatch':hatch}
        self._dict[k] = {k:v for k,v in kwargs.items() if v is not None}

    def reset(self, initial=None):
        # Define persistent (looping) cycler as per http://matplotlib.org/cycler/#persistent-cycles
        self.cycler = iter( self._cycler() )
        self._dict = defaultdict(lambda: next(self.cycler))

        if initial:
            """
            Pull values in order of initial mapping, force subsequent ordering
            """
            for k in initial:
                _ = self._dict[k]



# Contains all currently available styles
available = {
    # 8 step colors
    'brewer8accent': Style('brewer8accent', fc=brewer8accent),
    'brewer8dark2': Style('brewer8dark2', fc=brewer8dark2),
    'brewer8paired': Style('brewer8paired', fc=brewer8paired),
    'brewer8pastel1': Style('brewer8pastel1', fc=brewer8pastel1),
    'brewer8pastel2': Style('brewer8pastel2', fc=brewer8pastel2),
    'brewer8set1': Style('brewer8set1', fc=brewer8set1),
    'brewer8set2': Style('brewer8set2', fc=brewer8set2),
    'brewer8set3': Style('brewer8set3', fc=brewer8set3),
    # 9 step colors
    'brewer9paired': Style('brewer9paired', fc=brewer8accent),
    'brewer9pastel1': Style('brewer9pastel1', fc=brewer9pastel1),
    'brewer9set1': Style('brewer9set1', description="Qualitative color map for up to 9 classes. © Cynthia Brewer, Mark Harrower and The Pennsylvania State University", fc=brewer9set1),
    'brewer9set3': Style('brewer9set3', fc=brewer9set3),
    # 12 step colors
    'brewer12paired': Style('brewer12paired', fc=brewer12paired),
    'brewer12set3': Style('brewer12set3', fc=brewer12set3),
}

def define(name, **kwargs):
    global available

    available[name] = Style(name, **kwargs)

def use(style):
    global available, _current

    if style in available:
        _current = style
    else:
        raise Exception("Requested style not available")

def get(style=None):
    global available, _current

    if style is None:
        style = _current

    if style in available:
        return available[style]

    if isinstance(style, Style):
        return style

    raise Exception("Requested style not available")


