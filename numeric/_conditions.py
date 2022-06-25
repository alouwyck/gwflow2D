"""
Base module containing base classes to define initial and boundary conditions for two-dimensional finite-difference
groundwater flow models.
"""
from ._discretization import GridDependent
from ._parameters import HydraulicParameters


class Discharges(GridDependent):
    """
    Class defining discharges.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    q : array_like
      Two-dimensional array with discharge [L³/T or L³/T /L] in each grid cell.
      The shape of `q` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    """

    def __init__(self, grid, q):
        GridDependent.__init__(self, grid)
        self.q = self._broadcast(q)  # (nl, nr)


class ConstantHeads(GridDependent):
    """
    Class defining constant heads.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    h : array_like
      Two-dimensional array with constant head [L] in each grid cell.
      The shape of `h` is `(nl, nr)`, but it is broadcast if dimensions are missing.

    Notes
    -----
    The defined constant head in a cell is taken into account only if the cell is defined as having a constant head.
    This is indicated by `grid.constant`. By default, the head in a constant-head cell is zero.
    This means defining constant heads using this class is required only if some of these heads are nonzero.
    """

    def __init__(self, grid, h):
        GridDependent.__init__(self, grid)
        self.h = self._broadcast(h)  # (nl, nr)


class InitialHeads(ConstantHeads):
    """
    Class defining the initial heads of a stress period.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    h : array_like
      Two-dimensional array with initial head [L] in each grid cell.
      The shape of `h` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    add : bool, default: False
        Indicates whether the given initial heads should be added to the heads simulated for the last time step of the
        previous stress period or not.
        # TODO: checken of dit klopt!!

    Notes
    -----
    The defined initial head in a cell is taken into account only if the cell is defined as having a variable head.
    This is indicated by `grid.variable`. By default, the initial head in a variable-head cell is zero.
    This means defining initial heads using this class is required only if some of these heads are nonzero.
    # TODO: checken of dit klopt!!
    """

    def __init__(self, grid, h, add=False):
        ConstantHeads.__init__(self, grid, h)
        self.add = add


class HeadDependentFluxes(HydraulicParameters, ConstantHeads):
    """
    Class defining head-dependent flux boundary conditions.

    If a cell contains a head-dependent flux boundary condition, the vertical flow is determined by the difference
    in head between the cell and the given constant head and by the given resistance.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    dependent : array_like
              Two-dimensional boolean array that indicates which cells have a head-dependent flux boundary condition.
              The shape of `dependent` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    c : array_like
      Two-dimensional array with the vertical resistances [T].
      The shape of `c` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    h : array_like, default: None
      Two-dimensional array with the constant heads [L].
      The shape of `h` is `(nl, nr)`, but it is broadcast if dimensions are missing.
      By default, array `h` contains zeros only.

    Attributes
    ----------
    idx : ndarray
        One-dimensional array with the linear indices of the cells in which a head-dependent flux boundary condition is
        defined.
    qc : ndarray
       Two-dimensional array with the vertical conductances [L²/T] of the head-dependent boundary conditions.
       As the flow is vertical, the conductance is defined as the horizontal surface `hs` of the cell divided by the
       vertical resistance `c`. The shape of `qc` is `(nl, nr)`.
    """

    def __init__(self, grid, dependent, c, h=None):
        HydraulicParameters.__init__(self, grid)
        ConstantHeads.__init__(self, grid, h)
        self.dependent = self._broadcast(dependent, dtype=bool)  # (nl, nr)
        self.c = self._broadcast(c)  # (nl, nr)
        if self.h is None:
            self.h = self._zeros()  # (nl, nr)
        self._set_qc()

    def _set_qc(self):
        """
        Calculate vertical conductances and assign them to attribute `qc`.

        Returns
        -------
        None
        """
        irow, icol = self.dependent.nonzero()
        self.idx = irow * self.nr + icol
        self.qc = self.hs[icol] / self.c.flatten()[self.idx]


