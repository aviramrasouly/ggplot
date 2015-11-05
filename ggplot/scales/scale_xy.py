from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from types import MethodType

import numpy as np
import pandas as pd

from ..utils import identity, match, is_waive
from ..utils import DISCRETE_KINDS, CONTINUOUS_KINDS
from ..utils.exceptions import GgplotError
from .utils import expand_range
from .utils import identity_trans
from .scale import scale_discrete, scale_continuous


# positions scales have a couple of differences (quirks) that
# make necessary to override some of the scale_discrete and
# scale_continuous methods
#
# scale_position_discrete and scale_position_continuous
# are intermediate base classes where the required overiding
# is done
class scale_position_discrete(scale_discrete):
    """
    Base class for discrete position scales
    """
    # All positions have no guide
    guide = None

    # After transformations all position values map
    # to themselves
    palette = staticmethod(identity)

    # Keeps two ranges, range and range_c
    range_c = None

    def reset(self):
        # Can't reset discrete scale because
        # no way to recover values
        self.range_c = None

    def is_empty(self):
        return (self.range is None and
                self.limits is None and
                self.range_c is None)

    def train(self, series):
        # The discrete position scale is capable of doing
        # training for continuous data.
        # This complicates training and mapping, but makes it
        # possible to place objects at non-integer positions,
        # as is necessary for jittering etc.
        if series.dtype.kind in CONTINUOUS_KINDS:
            # trick the training method into training
            # range_c by temporarily renaming it to range
            backup, self.range = self.range, self.range_c
            self.train_continuous(series)
            # restore
            self.range_c, self.range = self.range, backup
        else:
            # super() does not work well with reloads
            scale_discrete.train(self, series)

    def map(self, series, limits=None):
        # Discrete values are converted into integers starting
        # at 1
        if not limits:
            limits = self.limits
        if series.dtype.kind in DISCRETE_KINDS:
            seq = np.arange(1, len(limits)+1)
            return seq[match(series, limits)]
        return series

    @property
    def limits(self):
        if self.is_empty():
            return (0, 1)

        if self._limits:
            return self._limits
        elif self.range:
            # discrete range
            return self.range
        elif self.range_c:
            # discrete limits for a continuous range
            mn = int(np.floor(np.min(self.range_c)))
            mx = int(np.ceil(np.max(self.range_c)))
            return list(range(mn, mx+1))
        else:
            raise GgplotError(
                'Lost, do not know what the limits are.')

    def coord_range(self):
        """
        Return the range for the coordinate axis
        """
        if self._limits:
            rng = self._limits
        elif is_waive(self._expand):
            rng = self.dimension((0, 0.6))
        else:
            rng = self.dimension(self._expand)

        return rng

    def dimension(self, expand=(0, 0)):
        """
        The phyical size of the scale, if a position scale
        Unlike limits, this always returns a numeric vector of length 2
        """
        # This is special e.g x scale for a categorical bar plot
        # calculate a dimension acc. to the discrete items(limits)
        # and a dimension acc. to the continuous range (range_c)
        # pick the (min, max)
        disc_range = (1, len(self.limits))
        disc = expand_range(disc_range, 0, expand[1], 1)
        cont = expand_range(self.range_c, expand[0], 0, expand[1])
        a = np.array([x for x in [disc, cont] if x is not None])
        return (a.min(), a.max())


# Discrete position scales should be able to make use of the train
# method bound to continuous scales
scale_position_discrete.train_continuous = scale_continuous.__dict__['train']


class scale_position_continuous(scale_continuous):
    """
    Base class for continuous position scales
    """
    # All positions have no guide
    guide = None

    # After transformations all position values map
    # to themselves
    palette = staticmethod(identity)

    def map(self, series, limits=None):
        # Position aesthetics don't map, because the coordinate
        # system takes care of it.
        # But the continuous scale has to deal with out of bound points
        if not len(series):
            return series
        if not limits:
            limits = self.limits
        scaled = self.oob(series, limits)
        scaled[pd.isnull(scaled)] = self.na_value
        return scaled


class scale_x_discrete(scale_position_discrete):
    aesthetics = ['x', 'xmin', 'xmax', 'xend']


class scale_y_discrete(scale_position_discrete):
    aesthetics = ['y', 'ymin', 'ymax', 'yend']


class scale_x_continuous(scale_position_continuous):
    aesthetics = ['x', 'xmin', 'xmax', 'xend', 'xintercept']


class scale_y_continuous(scale_position_continuous):
    aesthetics = ['y', 'ymin', 'ymax', 'yend', 'yintercept',
                  'ymin_final', 'ymax_final',
                  'lower', 'middle', 'upper']


# Transformed scales
class scale_x_datetime(scale_position_continuous):
    trans = 'datetime'
    aesthetics = ['x', 'xmin', 'xmax', 'xend']


class scale_y_datetime(scale_position_continuous):
    trans = 'datetime'
    aesthetics = ['y', 'ymin', 'ymay', 'yend']


scale_x_date = scale_x_datetime
scale_y_date = scale_y_datetime


class scale_x_timedelta(scale_position_continuous):
    trans = 'timedelta'
    aesthetics = ['x', 'xmin', 'xmax', 'xend']


class scale_y_timedelta(scale_position_continuous):
    trans = 'timedelta'
    aesthetics = ['y', 'ymin', 'ymay', 'yend']


class scale_x_sqrt(scale_x_continuous):
    trans = 'sqrt'


class scale_y_sqrt(scale_y_continuous):
    trans = 'sqrt'


class scale_x_log10(scale_x_continuous):
    trans = 'log10'


class scale_y_log10(scale_y_continuous):
    trans = 'log10'


# For the trans object of reverse scales
def _modify_axis(self, ax):
    attr = 'invert_{}axis'.format(self.aesthetic)
    getattr(ax, attr)()


class scale_x_reverse(scale_x_continuous):
    trans = identity_trans()
    trans.modify_axis = MethodType(_modify_axis, trans)


class scale_y_reverse(scale_y_continuous):
    trans = identity_trans()
    trans.modify_axis = MethodType(_modify_axis, trans)
