from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import re
from copy import deepcopy
import six
from patsy.eval import EvalEnvironment


class aes(dict):
    """
    Creates a dictionary that is used to evaluate
    things you're plotting. Most typically, this will
    be a column in a pandas DataFrame.

    Parameters
    -----------
    x : x-axis value
        Can be used for continuous (point, line) charts and for
        discrete (bar, histogram) charts.
    y : y-axis value
        Can be used for continuous charts only
    color (colour) : color of a layer
        Can be continuous or discrete. If continuous, this will be
        given a color gradient between 2 colors.
    shape : shape of a point
        Can be used only with geom_point
    size : size of a point or line
        Used to give a relative size for a continuous value
    alpha : transparency level of a point
        Number between 0 and 1. Only supported for hard coded values.
    ymin : min value for a vertical line or a range of points
        See geom_area, geom_ribbon, geom_vline
    ymax : max value for a vertical line or a range of points
        See geom_area, geom_ribbon, geom_vline
    xmin : min value for a horizonal line
        Specific to geom_hline
    xmax : max value for a horizonal line
        Specific to geom_hline
    slope : slope of an abline
        Specific to geom_abline
    intercept : intercept of an abline
        Specific to geom_abline

    Examples
    --------
    >>> aes(x='x', y='y')
    >>> aes('x', 'y')
    >>> aes(x='weight', y='height', color='salary')
    """

    DEFAULT_ARGS = ['x', 'y', 'color']

    def __init__(self, *args, **kwargs):
        if args:
            dict.__init__(self, zip(self.DEFAULT_ARGS, args))
        if kwargs:
            self.update(kwargs)
        if 'colour' in self:
            self['color'] = self.pop('colour')

        self.aes_env = EvalEnvironment.capture(1)

    def __deepcopy__(self, memo):
        '''deepcopy support for ggplot'''
        # Just copy the keys and point to the env
        result = aes()
        for key, item in self.items():
            try:
                result[key] = deepcopy(self[key], memo)
            except:
                raise

        result.aes_env = self.aes_env
        return result


def is_calculated_aes(aesthetics):
    """
    Return a list of the aesthetics that are calculated
    """
    pattern = "^\.\.([a-zA-Z._]+)\.\.$"
    calculated_aesthetics = []
    for k, v in aesthetics.items():
        if not isinstance(v, six.string_types):
            continue
        if re.match(pattern, v):
            calculated_aesthetics.append(k)
    return calculated_aesthetics


def strip_dots(ae):
    return ae.strip('..')


def aes_to_scale(var):
    """
    Look up the scale that should be used for a given aesthetic
    """
    if var in {'x', 'xmin', 'xmax', 'xend', 'xintercept'}:
        var = 'x'
    elif var in {'y', 'ymin', 'ymax', 'yend', 'yintercept'}:
        var = 'y'
    return var
