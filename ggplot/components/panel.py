from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..scales.scales import Scales
from ..utils import match, xy_panel_scales, suppress
from ..utils.exceptions import GgplotError


class Panel(object):
    # n = no. of visual panels
    layout = None     # table n rows, grid information

    # whether to shrink scales to fit
    # output of statistics, not raw data
    shrink = False

    ranges = None     # list of n dicts.
    x_scales = None   # scale object(s). 1 or n of them
    y_scales = None   # scale object(s). 1 or n of them
    axs = None        # MPL axes

    def train_layout(self, facet, layer_data, plot_data):
        self.layout = facet.train_layout([plot_data] + layer_data)
        self.shrink = facet.shrink

    def map_layout(self, facet, layer_data, plot_data):
        new_data = []
        for data in layer_data:
            if data is None:
                data = plot_data.copy()
            new_data.append(facet.map_layout(data, self.layout))
        return new_data

    def train_position(self, data, x_scale, y_scale):
        """
        Create all the required x_scales and y_scales
        and set the ranges for each scale according
        to the data
        """
        layout = self.layout
        # Initialise scales if needed, and possible.
        if not self.x_scales and x_scale:
            n = layout['SCALE_X'].max()
            self.x_scales = Scales([x_scale.clone() for i in range(n)])
        if not self.y_scales and y_scale:
            n = layout['SCALE_Y'].max()
            self.y_scales = Scales([y_scale.clone() for i in range(n)])

        # loop over each layer, training x and y scales in turn
        for layer_data in data:
            match_id = match(layer_data['PANEL'], layout['PANEL'])
            if x_scale:
                x_vars = list(set(x_scale.aesthetics) &
                              set(layer_data.columns))
                # the scale index for each data point
                SCALE_X = layout['SCALE_X'].iloc[match_id].tolist()
                self.x_scales.train(layer_data, x_vars, SCALE_X)

            if y_scale:
                y_vars = list(set(y_scale.aesthetics) &
                              set(layer_data.columns))
                # the scale index for each data point
                SCALE_Y = layout['SCALE_Y'].iloc[match_id].tolist()
                self.y_scales.train(layer_data, y_vars, SCALE_Y)
        return self

    def map_position(self, data, x_scale, y_scale):
        layout = self.layout

        for layer_data in data:
            match_id = match(layer_data['PANEL'], layout['PANEL'])
            if x_scale:
                x_vars = list(set(x_scale.aesthetics) &
                              set(layer_data.columns))
                SCALE_X = layout['SCALE_X'].iloc[match_id].tolist()
                self.x_scales.map(layer_data, x_vars, SCALE_X)

            if y_scale:
                y_vars = list(set(y_scale.aesthetics) &
                              set(layer_data.columns))
                SCALE_Y = layout['SCALE_Y'].iloc[match_id].tolist()
                self.y_scales.map(layer_data, y_vars, SCALE_Y)

        return data

    def panel_scales(self, i):
        """
        Return the x scale and y scale for panel i
        if they exist.
        """
        # wrapping with np.asarray prevents an exception
        # on some datasets
        bool_idx = (np.asarray(self.layout['PANEL']) == i)
        xsc = None
        ysc = None

        if self.x_scales:
            idx = self.layout.loc[bool_idx, 'SCALE_X'].values[0]
            xsc = self.x_scales[idx-1]

        if self.y_scales:
            idx = self.layout.loc[bool_idx, 'SCALE_Y'].values[0]
            ysc = self.y_scales[idx-1]

        return xy_panel_scales(x=xsc, y=ysc)

    def reset_scales(self):
        """
        Reset x and y scales
        """
        if not self.shrink:
            return

        with suppress(AttributeError):
            self.x_scales.reset()

        with suppress(AttributeError):
            self.y_scales.reset()

    def train_ranges(self, coord):
        """
        Calculate the x & y ranges for each panel
        """
        if not self.x_scales:
            raise GgplotError('Missing an x scale')

        if not self.y_scales:
            raise GgplotError('Missing a y scale')

        # ranges
        self.ranges = []
        cols = ['SCALE_X', 'SCALE_Y']
        for i, j in self.layout[cols].itertuples(index=False):
            i, j = i-1, j-1
            xinfo = coord.train(self.x_scales[i])
            yinfo = coord.train(self.y_scales[j])
            xinfo.update(yinfo)
            self.ranges.append(xinfo)
