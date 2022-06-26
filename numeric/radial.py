"""
Module containing specific classes to build finite-difference models for simulating two-dimensional radial
groundwater flow.
"""
import numpy as np
from ._discretization import Grid as BaseGrid
from ._parameters import HorizontalFlowParameters as BaseHorizontalFlowParameters
from ._model import StressPeriod as BaseStressPeriod
from ._model import Model as BaseModel


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

    Notes
    -----
    If input parameter `c` is not given, then conductances are colculated using input parameters `k`, and
    if input parameter `k` is not given, then conductances are colculated using input parameters `c`.
    If both input parameters `k` and `c` are given, then conductances are calculated using `k`, after which the
    calculated conductances are replaced by conductances calculated using elements of `c` which are not `nan`.

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
    """
    Class defining a stress period.

    A stress period is a period during which boundary conditions do not change with time.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric two-dimensional grid.
    timesteps : TimeSteps object, default: None
              Contains the time steps if transient state; is `None` if steady state.
    previous : TimeSteps object, default: None
             Contains the time steps of previous stress period.
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    add_kh(kh, ch) :
                  Add radial flow parameters.
    add_kv(kv, cv) :
                   Add vertical flow parameters.
    add_ss(ss, sy) :
                   Add storage parameters.
    add_q(q) :
             Add discharges.
    add_h0(h0, add) :
                    Add initial heads.
    add_hc(hc) :
               Add constant heads.
    add_hdep(dependent, cdep, hdep) :
                                    Add head-dependent flux boundary conditions.
    solve() :
            Solve the system of finite-difference equations to obtain the head in each grid cell, and for each time step
            if transient state.

    """

    def __init__(self, grid, timesteps=None, previous=None, no_warnings=True):
        BaseStressPeriod.__init__(self, grid, timesteps, previous, no_warnings)
        self._horizontal_flow_constructor = HorizontalFlowParameters


class Model(BaseModel):
    """
    Class to build finite-difference models for simulating steady and transient two-dimensional axi-symmetric flow.

    Parameters
    ----------
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Attributes
    ----------
    grid : Grid object
         Two-dimensional axi-symmetric finite-difference grid.
    timesteps : list of TimeSteps objects
              Contains the subsequent time steps.
    periods : list of StressPeriod objects
            Contains the subsequent stress periods.

    Methods
    -------
    add_grid(rb, D, constant, inactive, connected) :
                                                   Define the two-dimensional finite-difference grid.
    add_period(t) :
                  Define and add a new stress period.
    solve() :
            Solve the matrix system of finite-difference equations.
    """

    def __init__(self, no_warnings=True):
        BaseModel.__init__(self, no_warnings)
        self._grid_constructor = Grid
        self._period_constructor = StressPeriod
