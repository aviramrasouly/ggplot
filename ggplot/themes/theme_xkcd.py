from copy import copy, deepcopy

from .theme import theme
import matplotlib.pyplot as plt
import matplotlib as mpl


class theme_xkcd(theme):
    """
    xkcd theme

    The theme internaly uses the settings from pyplot.xkcd().
    """
    def __init__(self, scale=1, length=100, randomness=2):
        theme.__init__(self, complete=True)
        with plt.xkcd(scale=scale, length=length, randomness=randomness):
            _xkcd = mpl.rcParams.copy()

        # no need to a get a deprecate warning for nothing...
        for key in mpl._deprecated_map:
            if key in _xkcd:
                del _xkcd[key]

        if 'tk.pythoninspect' in _xkcd:
            del _xkcd['tk.pythoninspect']

        self._rcParams.update(_xkcd)

        d = {
             'figure.figsize': '11, 8',
             'figure.subplot.hspace': '0.5'}
        self._rcParams.update(d)

    def __deepcopy__(self, memo):
        """
        Deep copy support for theme_xkcd
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for key, item in self.__dict__.items():
            if key == '_rcParams':
                continue

            result.__dict__[key] = deepcopy(self.__dict__[key], memo)

        result._rcParams = {}
        for k, v in self._rcParams.items():
            try:
                result._rcParams[k] = deepcopy(v, memo)
            except NotImplementedError:
                # deepcopy raises an error for objects that are drived from or
                # composed of matplotlib.transform.TransformNode.
                # Not desirable, but probably requires upstream fix.
                # In particular, XKCD uses matplotlib.patheffects.withStrok
                # -gdowding
                result._rcParams[k] = copy(v)

        return result

    def apply_more(self, ax):
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markeredgewidth(2)
