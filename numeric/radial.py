"""
Module containing specific classes to build numerical steady and transient state models for simulating 2D radial
groundwater flow.
"""
import numpy as np
from ._base import Grid as BaseGrid
from ._base import HorizontalFlowParameters as BaseHorizontalFlowParameters
from ._base import StressPeriod as BaseStressPeriod
from ._base import Model as BaseModel


class Grid(BaseGrid):
    """
    Class defining a two-dimensional axi-symmetric grid to build a finite-difference model for simulating radial flow.

    Parameters
    ----------
    rb : array_like
       One-dimensional array with radial distance [L] of grid cell boundaries.
       The length of `rb` is `nr + 1`, with `nr` the number of columns (rings) in the grid.
    D : array_like
      One-dimensional array with height (thickness) [L] of grid layers.
      The length of `D` is `nl`, with `nl` the number of grid layers.
    constant : array_like, default: None
             Two-dimensional boolean array that indicates which cells have a constant head.
             The shape of `constant` is `(nl, nr)`, but it is broadcast if dimensions are missing.
             By default, there are no constant-head cells.
    inactive : array_like, default: None
             Two-dimensional boolean array that indicates which cells are inactive.
             The shape of `inactive` is `(nl, nr)`, but it is broadcast if dimensions are missing.
             By default, there are no inactive cells.
    connected : array_like, default: None
              Two-dimensional integer array that indicates which cells are connected.
              The shape of `connected` is `(nl, nr)`, but it is broadcast if dimensions are missing.
              By default, there are no connected grid cells.

    Attributes
    ----------
    nr : int
       Number of columns/rings (the `r` in `nr` refers to the radial distances represented by the cells).
    nl : int
       Number of grid layers.
    n : int
      Number of cells in the grid, i.e. ``nr * nl``.
    r : ndarray
      One-dimensional array with the radial distance [L] of the center of the grid cells (i.e. the radii of the cells).
      The length of `r` is `nr`.
    hs : ndarray
       One-dimensional array representing the horizontal surface [L²] of the grid cells.
       The length of `hs` is `nr`.
    vol : ndarray
        Two-dimensional array representing the volume [L³] of the grid cells.
        The shape of `vol` is `(nl, nr)`.
    variable : ndarray
             Two-dimensional boolean array that indicates which cells have a variable head.
             By default, all cells have a variable head.
    """

    def __init__(self, rb, D, constant=None, inactive=None, connected=None):
        BaseGrid.__init__(self, rb=rb, D=D, constant=constant, inactive=inactive, connected=connected)

    def _set_r_hs_vol(self):
        """
        Set attributes `r`, `hs`, and `vol`.

        Returns
        -------
        None
        """
        self.r = np.sqrt(self.rb[:-1] * self.rb[1:])  # (nr, )
        rb2 = self.rb ** 2  # (nr+1, )
        self.hs = np.pi * (rb2[1:] - rb2[:-1])  # (nr, )
        self.vol = np.outer(self.D, self.hs)  # (nl, nr)


class HorizontalFlowParameters(BaseHorizontalFlowParameters):
    """
    Class defining the radial flow parameters.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric two-dimensional grid.
    k : array_like, default: None
        Two-dimensional array with radial conductivities [L/T] of the grid cells.
        The shape of `k` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    c : array_like, default: None
        Two-dimensional array with radial resistances [T] between the grid cells.
        The shape of `c` is `(nl, nr - 1)`, but it is broadcast if dimensions are missing.

    Attributes
    ----------
    qc : ndarray
       Two-dimensional array with radial conductances [L²/T] between the grid cells.
       The shape of `qc` is `(nl, nr + 1)`, which means the zero conductances of the model boundaries are included.
    """

    def __init__(self, grid, k=None, c=None):
        BaseHorizontalFlowParameters.__init__(self, grid, k, c)

    def _calculate_qc(self):
        """
        Calculate radial conductances.

        Returns
        -------
        qc : ndarray
           Two-dimensional array with radial conductances [L²/T] between the grid cells.
           The shape of `qc` is `(nl, nr - 1)`, which means the model boundaries are not included.
        """
        if self.k is None:
            qc = 2 * np.pi * np.outer(self.D, self.rb[1:-1]) / self.c  # (nl, nr-1)
        else:
            rbc = np.log(self.rb[1:] / self.rb[:-1]) / self.k / 2  # (nl, nr)
            qc = 2 * np.pi * self.D[:, np.newaxis] / (rbc[:, :-1] + rbc[:, 1:])  # (nl, nr-1)
            if self.c is not None:
                irow, icol = (~np.isnan(self.c)).nonzero()
                qc[irow, icol] = 2 * np.pi * self.D[irow] * self.rb[icol + 1] / self.c[irow, icol]
        return qc  # (nl, nr-1)


class StressPeriod(BaseStressPeriod):

    def __init__(self, grid, timesteps=None, previous=None):
        BaseStressPeriod.__init__(self, grid, timesteps, previous)
        self._horizontal_flow_constructor = HorizontalFlowParameters


class Model(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self._grid_constructor = Grid
        self._period_constructor = StressPeriod
