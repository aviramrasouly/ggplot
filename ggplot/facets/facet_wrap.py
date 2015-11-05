from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy

from .layouts import layout_wrap
from .locate import locate_wrap


class facet_wrap(object):

    def __init__(self, facets=None, nrow=None, ncol=None, scales='fixed',
                 shrink=True, as_table=True, drop=True):
        self.vars = (facets,)
        self.nrow = nrow
        self.ncol = ncol
        self.shrink = shrink
        self.as_table = as_table
        self.drop = drop
        self.free = {'x': scales in ('free_x', 'free'),
                     'y': scales in ('free_y', 'free')}

    def __radd__(self, gg):
        gg = deepcopy(gg)
        gg.facet = self
        return gg

    def train_layout(self, data):
        layout = layout_wrap(data, vars=self.vars, nrow=self.nrow,
                             ncol=self.ncol, as_table=self.as_table,
                             drop=self.drop)
        n = layout.shape[0]
        nrow = layout['ROW'].max()

        # Add scale identification
        layout['SCALE_X'] = range(1, n+1) if self.free['x'] else 1
        layout['SCALE_Y'] = range(1, n+1) if self.free['y'] else 1

        # Figure out where axes should go
        layout['AXIS_X'] = True if self.free['x'] else layout['ROW'] == nrow
        layout['AXIS_Y'] = True if self.free['y'] else layout['COL'] == 1

        self.nrow = nrow
        self.ncol = layout['COL'].max()
        return layout

    def map_layout(self, data, layout):
        """
        Assign a data points to panels

        Parameters
        ----------
        data : DataFrame
            dataframe for a layer
        layout : DataFrame
            As returned by self.train_layout
        """
        return locate_wrap(data, layout, self.vars)
