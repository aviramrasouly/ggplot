from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .geom import geom


class geom_blank(geom):
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity'}

    def draw_panel(self, data, panel_scales, coord, ax, **params):
        pass
