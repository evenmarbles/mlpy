from __future__ import division, print_function, absolute_import

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Annotate3D(object):
    def __init__(self, fig, axes, data, labels):
        self._fig = fig
        self._ax = axes
        self._data = data
        self._labels = labels

        self._annotation = []

        x2, y2, _ = proj3d.proj_transform(self._data[0][0], self._data[1][0], self._data[2][0], self._ax.get_proj())
        self._label = self._ax.annotate(
            str(labels[0]),
            xy=(x2, y2), xytext=(x2+0.5, y2+0.5)
        )

        # for x, y, z, txt in zip(self._data[0], self._data[1], self._data[2], self._labels):
        #     x2, y2, _ = proj3d.proj_transform(x, y, z, self._ax.get_proj())
        #     self._annotation.append(self._ax.annotate(
        #         str(txt),
        #         xy=(x2, y2), xytext=(x2+0.5, y2+0.5)
        #     ))

        self._fig.canvas.mpl_connect('motion_notify_event', self.draw)

    # noinspection PyUnusedLocal
    def draw(self, e):
        x2, y2, _ = proj3d.proj_transform(self._data[0][0], self._data[1][0], self._data[2][0], self._ax.get_proj())
        self._label.xy = x2, y2
        self._label.update_positions(self._fig.canvas.renderer)
        self._fig.canvas.draw()

        # for (x, y, z), a in zip(self._data[0], self._data[1], self._data[2], self._annotation):
        #     x2, y2, _ = proj3d.proj_transform(x, y, z, self._ax.get_proj())
        #     a.xy = x2, y2
        #     a.update_positions(self._fig.canvas.renderer)
        #     self._fig.canvas.draw()


class Arrow3D(FancyArrowPatch):
    """3d-arrow class.

    Draws a 3d-arrow on specified axis from (x, y, z) to (x + dx, y + dy, z + dz)
    in a 3d Matplotlib figure. Uses :class:`matplotlib.patches.FancyArrowPatch`
    to construct the arrow.

    Parameters
    ----------
    xs : tuple(float, float)
        The x component of the directional vector, specifying the start
        and end position: (x, x + dx)
    ys : tuple(float, float)
        The y component of the directional vector, specifying the start
        and end position: (y, y + dy)
    zs : tuple(float, float)
        The z component of the directional vector, specifying the start
        and end position: (z, z + dz)
    args : tuple
        Positional arguments controlling arrow construction and properties.
        For valid arguments see :func:`matplotlib.pyplot.arrow`.
    kwargs : dict
        Non-positional arguments controlling arrow construction and properties.
        For valid arguments see :func:`matplotlib.pyplot.arrow`.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>>
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1, projection='3d')
    >>>
    >>> a = Arrow3D((0, 1), (0, 1), (0, 1), mutation_scale=10, lw=1, arrowstyle="-|>", color='r')
    >>> ax.add_artist(a)
    >>> fig.show()

    .. image:: ../_static/arrow3d.png
       :width: 300pt

    .. attention::
        The Arrow3D class requires the `matplotlib <http://matplotlib.org/>`_ library.

    .. note::
        | Project: Code from `StackOverflow <http://stackoverflow.com/a/11156353>`_.
        | Code author: `HYRY <http://stackoverflow.com/users/772649/hyry>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
