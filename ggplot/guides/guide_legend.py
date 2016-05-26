from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import hashlib
from itertools import islice
from collections import OrderedDict

import six
from six.moves import zip, range
import numpy as np
import pandas as pd
from matplotlib.cbook import Bunch
from matplotlib.offsetbox import (TextArea, HPacker, VPacker)

from ..scales.scale import scale_continuous
from ..utils import gg_import, ColoredDrawingArea, suppress
from ..utils.exceptions import gg_warn, GgplotError
from ..geoms import geom_text
from .guide import guide

# See guides.py for terminology


class guide_legend(guide):
    # general
    nrow = None
    ncol = None
    byrow = False

    keyseparation = None

    # parameter
    available_aes = 'any'

    def train(self, scale):
        """
        Create the key for the guide

        The key is a dataframe with two columns:
            - scale name : values
            - label : labels for each value

        scale name is one of the aesthetics
        ['x', 'y', 'color', 'fill', 'size', 'shape', 'alpha']

        Returns this guide if training is successful and None
        if it fails
        """
        breaks = scale.get_breaks()
        if isinstance(breaks, OrderedDict):
            if all([np.isnan(x) for x in breaks.values()]):
                return None
        elif not len(breaks) or all(np.isnan(breaks)):
            return None

        with suppress(AttributeError):
            breaks = list(breaks.keys())

        key = pd.DataFrame({
            scale.aesthetics[0]: scale.map(breaks),
            'label': scale.get_labels(breaks)})

        # Drop out-of-range values for continuous scale
        # (should use scale$oob?)

        # Currently, numpy does not deal with NA (Not available)
        # When they are introduced, the block below should be
        # adjusted likewise, see ggplot2, guide-lengend.r
        if isinstance(scale, scale_continuous):
            limits = scale.limits
            b = np.asarray(breaks)
            noob = np.logical_and(limits[0] <= b,
                                  b <= limits[1])
            key = key[noob]

        if len(key) == 0:
            return None

        self.key = key

        # create a hash of the important information in the guide
        labels = ' '.join(six.text_type(x) for x in self.key['label'])
        info = '\n'.join([self.title, labels, str(self.direction),
                          self.__class__.__name__])
        self.hash = hashlib.md5(info.encode('utf-8')).hexdigest()
        return self

    def merge(self, other):
        """
        Merge overlapped guides
        e.g
        >>> from ggplot import *
        >>> gg = ggplot(aes(x='cut', fill='cut', color='cut'), data=diamonds)
        >>> gg = gg + stat_bin()
        >>> gg

        This would create similar guides for fill and color where only
        a single guide would do
        """
        self.key = pd.merge(self.key, other.key)
        duplicated = set(self.override_aes) & set(other.override_aes)
        if duplicated:
            gg_warn("Duplicated override_aes is ignored.")
        self.override_aes.update(other.override_aes)
        for ae in duplicated:
            self.override_aes.pop(ae)
        return self

    def create_geoms(self, plot):
        """
        Make information needed to draw a legend for each of the layers.

        For each layer, that information is a dictionary with the geom
        to draw the guide together with the data and the parameters that
        will be used in the call to geom.
        """
        # A layer either contributes to the guide, or it does not. The
        # guide entries may be ploted in the layers
        self.glayers = []
        for l in plot.layers:
            all_ae = (six.viewkeys(l.mapping) |
                      plot.mapping if l.inherit_aes else set() |
                      six.viewkeys(l.stat.DEFAULT_AES))
            geom_ae = l.geom.REQUIRED_AES | six.viewkeys(l.geom.DEFAULT_AES)
            matched = all_ae & geom_ae & set(self.key.columns)
            matched = matched - set(l.geom.aes_params)

            if len(matched):
                # This layer contributes to the legend
                if l.show_legend is None or l.show_legend:
                    # Default is to include it
                    tmp = self.key[list(matched)].copy()
                    data = l.use_defaults(tmp)
                else:
                    continue
            else:
                # This layer does not contribute to the legend
                if l.show_legend is None or not l.show_legend:
                    continue
                else:
                    zeros = [0] * len(self.key)
                    data = l.use_defaults(pd.DataFrame())[zeros]

            # override.aes in guide_legend manually changes the geom
            for ae in set(self.override_aes) & set(data.columns):
                data[ae] = self.override_aes[ae]

            if hasattr(l.geom, 'draw_legend'):
                geom = l.geom.__class__
            else:
                geom = gg_import('geom_{}'.format(l.geom.legend_geom))
            self.glayers.append(Bunch(geom=geom, data=data, layer=l))

        if not self.glayers:
            return None
        return self

    def _set_defaults(self, theme):
        guide._set_defaults(self, theme)
        nbreak = len(self.key)

        # rows and columns
        if self.nrow is not None and self.ncol is not None:
            if guide.nrow * guide.ncol < nbreak:
                raise GgplotError(
                    "nrow x ncol need to be larger",
                    "than the number of breaks")

        if self.nrow is None:
            if self.ncol is not None:
                self.nrow = int(np.ceil(nbreak/self.ncol))
            elif self.direction == 'horizontal':
                self.nrow = 1
            else:
                self.nrow = nbreak

        if self.ncol is None:
            if self.nrow is not None:
                self.ncol = int(np.ceil(nbreak/self.nrow))
            elif self.direction == 'horizontal':
                self.ncol = nbreak
            else:
                self.ncol = 1

        if self.keyseparation is None:
            self.keyseparation = 2

        # key width and key height for each legend entry
        #
        # Take a peak into data['size'] to make sure the
        # legend dimensions are big enough
        """
        >>> gg = ggplot(diamonds, aes(x='cut', y='clarity'))
        >>> gg = gg + stat_sum(aes(group='cut'))
        >>> gg + scale_size(range=(3, 25))

        Note the different height sizes for the entries
        """
        default_size = 20
        default_pad = 14

        def determine_side_length():
            size = np.ones(nbreak) * default_size
            for i in range(nbreak):
                for gl in self.glayers:
                    _size = 0
                    pad = default_pad
                    # Full size of object to appear in the
                    # legend key
                    if 'size' in gl.data:
                        _size = gl.data.iloc[i]['size']
                        if 'stroke' in gl.data:
                            _size += 2*gl.data.iloc[i]['stroke']

                    # special case, color does not apply to
                    # border/linewidth
                    if issubclass(gl.geom, geom_text):
                        pad = 0
                        if _size < default_size:
                            continue

                    try:
                        # color(edgecolor) affects size(linewidth)
                        # When the edge is not visible, we should
                        # not expand the size of the keys
                        if gl.data.iloc[i]['color'] is not None:
                            size[i] = np.max([_size+pad, size[i]])
                    except KeyError:
                        break

            return size

        if self.keywidth is None:
            width = determine_side_length()
            if self.direction == 'vertical':
                width[:] = width.max()
            self._keywidth = width
        else:
            self._keywidth = [self.keywidth]*nbreak

        if self.keyheight is None:
            height = determine_side_length()
            if self.direction == 'horizontal':
                height[:] = height.max()
            self._keyheight = height
        else:
            self._keyheight = [self.keyheight]*nbreak

    def draw(self, theme):
        """
        Draw guide

        Parameters
        ----------
        theme : theme

        Returns
        -------
        out : matplotlib.offsetbox.Offsetbox
            A drawing of this legend
        """
        obverse = slice(0, None)
        reverse = slice(None, None, -1)
        nbreak = len(self.key)
        sep = 5  # gap between the legends
        themeable = theme.figure._themeable

        # title
        title_box = TextArea(
            self.title, textprops=dict(color='k', weight='bold'))
        themeable['legend_title'] = title_box

        # labels
        labels = []
        for item in self.key['label']:
            if isinstance(item, np.float) and np.float.is_integer(item):
                item = np.int(item)  # 1.0 to 1
            va = 'center' if self.label_position == 'top' else 'baseline'
            ta = TextArea(item, textprops=dict(color='k', va=va))
            labels.append(ta)
        themeable['legend_text'] = labels

        # Drawings
        drawings = []
        for i in range(nbreak):
            da = ColoredDrawingArea(self._keywidth[i],
                                    self._keyheight[i],
                                    0, 0, color='#E5E5E5')
            # overlay geoms
            for gl in self.glayers:
                data = gl.data.iloc[i]
                data.is_copy = None
                da = gl.geom.draw_legend(data, da, gl.layer)
            drawings.append(da)
        themeable['legend_key'] = drawings

        # Match Drawings with labels to create the entries
        # TODO: theme me
        lookup = {
            'right': (HPacker, reverse),
            'left': (HPacker, obverse),
            'bottom': (VPacker, reverse),
            'top': (VPacker, obverse)}
        packer, slc = lookup[self.label_position]
        entries = []
        label_align = 'center' if packer == HPacker else 'left'
        for d, l in zip(drawings, labels):
            e = packer(children=[l, d][slc],
                       align=label_align,
                       pad=0, sep=sep)
            entries.append(e)

        # Put the entries together in rows or columns
        # A chunk is either a row or a column of entries
        # for a single legend
        if self.byrow:
            chunk_size, packers = self.ncol, [HPacker, VPacker]
        else:
            chunk_size, packers = self.nrow, [VPacker, HPacker]

        if self.reverse:
            entries = entries[::-1]
        chunks = []
        for i in range(len(entries)):
            start = i*chunk_size
            stop = start + chunk_size
            s = islice(entries, start, stop)
            chunks.append(list(s))
            if stop >= len(entries):
                break

        chunk_boxes = []
        for chunk in chunks:
            d1 = packers[0](children=chunk,
                            align='left', pad=0,
                            sep=self.keyseparation)
            chunk_boxes.append(d1)

        # Put all the entries (row & columns) together
        entries_box = packers[1](children=chunk_boxes,
                                 align='baseline',
                                 pad=0,
                                 sep=sep)
        # TODO: theme me
        # Put the title and entries together
        packer, slc = lookup[self.title_position]
        children = [title_box, entries_box][slc]
        box = packer(children=children, align=self._title_align,
                     pad=0, sep=sep)
        return box
