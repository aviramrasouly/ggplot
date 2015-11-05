from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import types
from copy import deepcopy
from collections import OrderedDict
from six.moves import zip

import numpy as np
import pandas as pd
import pandas.core.common as com
import matplotlib.cbook as cbook
from matplotlib.ticker import Locator, Formatter, FuncFormatter

from ..utils import waiver, is_waive
from ..utils import match, is_sequence_of_strings
from ..utils import round_any, suppress, CONTINUOUS_KINDS
from ..utils.exceptions import gg_warn, GgplotError
from .utils import rescale, censor, expand_range, zero_range
from .utils import identity_trans, gettrans
from ..components.aes import is_position_aes


class scale(object):
    """
    Base class for all scales
    """
    aesthetics = []     # aesthetics affected by this scale
    range = None        # range of aesthetic
    na_value = np.NaN   # What to do with the NA values
    expand = waiver()   # multiplicative and additive expansion constants.
    name = None         # used as the axis label or legend title
    breaks = waiver()   # major breaks
    labels = waiver()   # labels at the breaks
    guide = 'legend'    # legend or any other guide
    _limits = None      # (min, max) - set by user

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif hasattr(self, '_'+k):
                setattr(self, '_'+k, v)
            else:
                msg = '{} could not recognise parameter `{}`'
                gg_warn(msg.format(self.__class__.__name__, k))

        if cbook.iterable(self.breaks) and cbook.iterable(self.labels):
            if len(self.breaks) != len(self.labels):
                raise GgplotError(
                    "Breaks and labels have unequal lengths")

        if (self.breaks is None and
                not is_position_aes(self.aesthetics) and
                self.guide is not None):
            self.guide = None

    def __radd__(self, gg):
        """
        Add this scales to the list of scales for the
        ggplot object
        """
        gg = deepcopy(gg)
        gg.scales.append(self)
        return gg

    @staticmethod
    def palette(x):
        """
        Aesthetic mapping function
        """
        raise NotImplementedError('Not Implemented')

    def map(self, x, limits=None):
        """
        Map every element of x

        The palette should do the real work, this should
        make sure that sensible values are sent and
        return from the palette.
        """
        raise NotImplementedError('Not Implemented')

    def train(self, x):
        """
        Train scale

        Parameters
        ----------
        x: pd.series | np.array
            a column of data to train over
        """
        raise NotImplementedError('Not Implemented')

    def dimension(self, expand=None):
        """
        The phyical size of the scale.
        """
        raise NotImplementedError('Not Implemented')

    def transform_df(self, df):
        """
        Transform dataframe
        """
        raise NotImplementedError('Not Implemented')

    def transform(self, x):
        """
        Transform array|series x
        """
        raise NotImplementedError('Not Implemented')

    def clone(self):
        return deepcopy(self)

    def reset(self):
        """
        Set the range of the scale to None.

        i.e Forget all the training
        """
        self.range = None

    def is_empty(self):
        return self.range is None and self._limits is None

    @property
    def limits(self):
        if self.is_empty():
            return (0, 1)

        # Fall back to the range if the limits
        # are not set or if any is NaN
        if self._limits is not None:
            if not any(map(pd.isnull, self._limits)):
                return self._limits
        return self.range

    @limits.setter
    def limits(self, value):
        self._limits = value

    def train_df(self, df):
        """
        Train scale from a dataframe
        """
        aesthetics = set(self.aesthetics) & set(df.columns)
        for ae in aesthetics:
            self.train(df[ae])

    def map_df(self, df):
        """
        Map df
        """
        if len(df) == 0:
            return

        aesthetics = set(self.aesthetics) & set(df.columns)
        for ae in aesthetics:
            df[ae] = self.map(df[ae])

        return df


class scale_discrete(scale):
    """
    Base class for all discrete scales
    """
    drop = True        # drop unused factor levels from the scale

    def train(self, x, drop=None):
        """
        Train scale

        Parameters
        ----------
        x: pd.series| np.array
            a column of data to train over

        A discrete range is stored in a list
        """
        if drop is None:
            drop = self.drop

        if self.range is None:
            self.range = []

        # new range values
        if com.is_categorical_dtype(x):
            rng = list(x.cat.categories)
            if drop:
                present = set(x.drop_duplicates())
                rng = [i for i in rng if i in present]
        elif x.dtype.kind in CONTINUOUS_KINDS:
            msg = "Continuous value supplied to discrete scale"
            raise GgplotError(msg)
        else:
            rng = list(x.drop_duplicates().sort(inplace=False))

        # update range
        old_range = set(self.range)
        self.range += [i for i in rng if (i not in old_range)]

    def dimension(self, expand=(0, 0)):
        """
        The phyical size of the scale, if a position scale
        Unlike limits, this always returns a numeric vector of length 2
        """
        return expand_range(len(self.limits), expand[0], expand[1])

    def map(self, x, limits=None):
        """
        Return an array-like of x mapped to values
        from the scales palette
        """
        if limits is None:
            limits = self.limits

        n = sum(~pd.isnull(limits))
        pal = self.palette(n)
        if isinstance(pal, dict):
            # manual palette with specific assignments
            pal_match = [pal[val] for val in x]
        else:
            pal = np.asarray(pal)
            pal_match = pal[match(x, limits)]
            pal_match[pd.isnull(pal_match)] = self.na_value
        return pal_match

    def break_info(self, range=None):
        if range is None:
            range = self.dimension()
        # for discrete, limits != range
        limits = self.limits
        major = self.scale_breaks(limits)
        if major is None:
            labels = None
        else:
            labels = self.scale_labels(major)
            major = pd.Categorical(major.keys())
            major = self.map(major)
        return {'range': range,
                'labels': labels,
                'major': major,
                'minor': None}

    def scale_breaks(self, limits=None, can_waive=False):
        """
        Returns a ordered dictionary of the form {break: position}

        The form is suitable for use by the guides

        e.g.
        {'fair': 1, 'good': 2, 'very good': 3,
        'premium': 4, 'ideal': 5}
        """
        if self.is_empty():
            return []

        if limits is None:
            limits = self.limits

        if self.breaks is None:
            return None
        elif is_waive(self.breaks):
            breaks = limits
        elif callable(self.breaks):
            breaks = self.breaks(limits)
        else:
            breaks = self.breaks

        # Breaks can only occur only on values in domain
        in_domain = list(set(breaks) & set(self.limits))
        pos = match(in_domain, breaks)
        tups = zip(in_domain, pos)
        return OrderedDict(sorted(tups, key=lambda t: t[1]))

    def scale_labels(self, breaks=None, can_waive=False):
        """
        Generate labels for the legend/guide breaks
        """
        if self.is_empty():
            return []

        if breaks is None:
            breaks = self.scale_breaks(can_waive=can_waive)

        # The labels depend on the breaks if the breaks are None
        # or are waived, it is likewise for the labels
        if breaks is None or self.labels is None:
            return None
        elif is_waive(breaks):
            return waiver()
        elif is_waive(self.labels):
            # if breaks is a dict (ordered by value)
            #   {'I': 2, 'G': 1, 'P': 3, 'V': 4, 'F': 0}
            # The keys are the labels
            # i.e ['F', 'G', 'I', 'P', 'V']
            try:
                return list(breaks.keys())
            except AttributeError:
                return breaks
        elif callable(self.labels):
            return self.labels(breaks)
        # if a dict is used to rename some labels
        elif isinstance(self.labels, dict):
            labels = breaks
            lookup = list(self.labels.items())
            mp = match(lookup, labels, nomatch=-1)
            for idx in mp:
                if idx != -1:
                    labels[idx] = lookup[idx]
            return labels
        else:
            # TODO: see ggplot2
            # Need to ensure that if breaks were dropped,
            # corresponding labels are too
            return self.labels

    def transform_df(self, df):
        """
        Transform dataframe
        """
        # Discrete scales do not do transformations
        return df

    def transform(self, x):
        """
        Transform array|series x
        """
        # Discrete scales do not do transformations
        return x


class scale_continuous(scale):
    """
    Base class for all continuous scales
    """
    rescaler = staticmethod(rescale)  # Used by diverging & n colour gradients
    oob = staticmethod(censor)     # what to do with out of bounds data points
    minor_breaks = waiver()
    trans = 'identity'             # transform class

    def __init__(self, **kwargs):
        # Make sure we have a transform and it
        # should know the main aesthetic,
        # in case it has to manipulate the axis
        trans = kwargs.get('trans', self.trans)
        with suppress(KeyError):
            del kwargs['trans']
        trans = gettrans(trans)
        trans.aesthetic = self.aesthetics[0]

        # The limits are given in original dataspace
        # but they leave in transformed space since all
        # computations happen on transformed data. The
        # labeling of the plot axis and the guides are in
        # the original dataspace.
        if 'limits' in kwargs:
            kwargs['limits'] = trans.transform(kwargs['limits'])

        # We can set the breaks to user defined values or
        # have matplotlib calculate them using the default locator
        # function. In case of transform, special locator and
        # formatter functions are created for mpl to use.

        # When both breaks and transformation are specified,
        # the trans object should not modify the axis. The
        # trans object will still transform the data
        if 'breaks' in kwargs:
            # locator wins
            if (callable(kwargs['breaks']) and
                    isinstance(kwargs['breaks'](), Locator)):
                trans.breaks_locator = kwargs.pop('breaks')
            # trust the user breaks
            elif trans != identity_trans:
                # kill the locator but not the and the formatter.
                trans.breaks_locator = waiver()

        if 'minor_breaks' in kwargs:
            # locator wins
            if (callable(kwargs['minor_breaks']) and
                    isinstance(kwargs['minor_breaks'](), Locator)):
                trans.minor_breaks_locator = kwargs.pop('minor_breaks')
            # trust the user breaks
            elif trans != identity_trans:
                trans.minor_breaks_locator = waiver()

        if 'labels' in kwargs:
            # Accept an MPL Formatter, a function or a list-like
            if (callable(kwargs['labels']) and
                    isinstance(kwargs['labels'](), Formatter)):
                trans.labels_formatter = kwargs.pop('labels')
            elif isinstance(kwargs['labels'], types.FunctionType):
                trans.labels_formatter = FuncFormatter(
                    kwargs.pop('labels'))
            elif is_sequence_of_strings(kwargs['labels']):
                trans.labels_formatter = waiver()
            elif not is_sequence_of_strings(kwargs['labels']):
                msg = 'labels should be function or a sequence of strings'
                raise GgplotError(msg)

        self.trans = trans
        scale.__init__(self, **kwargs)

    def train(self, x):
        """
        Train scale

        Parameters
        ----------
        x: pd.series | np.array
            a column of data to train over

        """
        if not len(x):
            return

        mn = x.min()
        mx = x.max()
        if not (self.range is None):
            _mn, _mx = self.range
            mn = np.min([mn, _mn])
            mx = np.max([mx, _mx])

        self.range = [mn, mx]

    def transform_df(self, df):
        """
        Transform dataframe
        """
        if len(df) == 0:
            return

        if self.trans.name == 'identity':
            return df

        aesthetics = set(self.aesthetics) & set(df.columns)
        for ae in aesthetics:
            with suppress(TypeError):
                df[ae] = self.transform(df[ae])

        return df

    def transform(self, x):
        """
        Transform array|series x
        """
        try:
            return self.trans.transform(x)
        except TypeError:
            return [self.trans.transform(val) for val in x]

    def dimension(self, expand=(0, 0)):
        """
        The phyical size of the scale, if a position scale
        Unlike limits, this always returns a numeric vector of length 2
        """
        return expand_range(self.limits, expand[0], expand[1])

    def map(self, x, limits=None):
        if limits is None:
            limits = self.limits

        x = self.oob(self.rescaler(x, from_=limits))

        # Points are rounded to the nearest 500th, to reduce the
        # amount of work that the scale palette must do - this is
        # particularly important for colour scales which are rather
        # slow.  This shouldn't have any perceptual impacts.
        x = round_any(x, 1 / 500)
        uniq = np.unique(x)
        pal = np.asarray(self.palette(uniq))
        scaled = pal[match(x, uniq)]
        scaled[pd.isnull(scaled)] = self.na_value
        return scaled

    def break_info(self, range=None):
        if range is None:
            range = self.dimension()

        major = self.scale_breaks(range)
        with suppress(TypeError):
            major = [x for x in self.scale_breaks(range) if not np.isnan(x)]

        minor = self.minor_breaks
        with suppress(TypeError):
            minor = [x for x in minor if range[0] <= x <= range[1]]

        labels = self.scale_labels(major)

        return {'range': range,
                'labels': labels,
                'major': major,
                'minor': minor}

    def scale_breaks(self, limits=None, can_waive=True):
        """
        Generate breaks for the legend/guide

        Parameters
        ----------
        limits : list-like
        can_waive : bool
            Whether the method can return a waiver object.
            When the guides request breaks they really need
            them and cannot rely on Matplotlib. This option
            is for them.
        """
        if self.is_empty():
            return []

        if limits is None:
            limits = self.limits
        # Limits in transformed space need to be
        # converted back to data space
        limits = self.trans.inverse(limits)

        if not self.breaks:  # None, False, []
            return []
        elif zero_range(limits):
            breaks = [limits[0]]
        elif can_waive and is_waive(self.breaks):
            # The MPL Locator will handle them
            return self.breaks
        elif is_waive(self.breaks):
            breaks = self.trans.breaks_locator(4).tick_values(*limits)
        elif callable(self.breaks):
            breaks = self.breaks(limits)
        else:
            breaks = self.breaks

        # Breaks in data space need to be converted back to
        # transformed space And any breaks outside the
        # dimensions need to be flagged as missing
        breaks = censor(self.transform(breaks),
                        self.transform(limits))
        return breaks

    def scale_labels(self, breaks=None, can_waive=False):
        """
        Generate labels for the legend/guide breaks
        """
        if breaks is None:
            breaks = self.scale_breaks(can_waive=can_waive)

        # The labels depend on the breaks if the breaks are None
        # or are waived, it is likewise for the labels
        if breaks is None or self.labels is None:
            return None
        elif is_waive(breaks):
            return waiver()
        elif can_waive and is_waive(self.labels):
            # The MPL Formatter will handle them
            return self.labels
        elif is_waive(self.labels):
            # Instantiate a formatter and "prep"
            # it for use
            breaks = np.asarray(breaks)
            locs = breaks[~np.isnan(breaks)]
            if not len(locs):
                locs = [0, 1]
            formatter = self.trans.labels_formatter()
            formatter.create_dummy_axis()
            formatter.set_locs(locs)
            # This is what really matters
            labels = [formatter(b) for b in breaks]
        elif callable(self.labels):
            labels = self.labels(breaks)
        else:
            labels = self.labels

        if len(labels) != len(breaks):
            raise GgplotError(
                "Breaks and labels are different lengths")

        return labels
