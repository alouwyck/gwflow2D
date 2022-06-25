"""
Base module containing base classes to define hydraulic parameters for two-dimensional finite-difference groundwater
flow models.
"""
from abc import ABC, abstractmethod
import numpy as np
from ._discretization import GridDependent


class HydraulicParameters(GridDependent, ABC):
    """
    Abstract base class for classes that define hydraulic parameters.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.

    Attributes
    ----------
    qc : ndarray
       Two-dimensional array with conductances.

    Notes
    -----
    Subclasses implement protected abstract method `_set_qc` to set attribute `qc`.
    """

    def __init__(self, grid):
        GridDependent.__init__(self, grid)
        self.qc = None

    @abstractmethod
    def _set_qc(self):
        """
        Calculate conductances and assign them to attribute `qc`.

        Abstract method.

        Returns
        -------
        None
        """
        pass


class FlowParameters(HydraulicParameters):
    """
    Abstract base class for classes defining flow parameters.

    Parameters
    ----------
    grid: Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    k : array_like, default: None
        Two-dimensional array with hydraulic conductivities [L/T].
        The shape of `k` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    c : array_like, default: None
        Two-dimensional array with hydraulic resistances [T].

    Attributes
    ----------
    qc : ndarray
       Two-dimensional array with conductances.

    Notes
    -----
    Subclasses implement protected abstract method `_check_c` to check parameter `c` and assign it to attribute `c`.
    """

    def __init__(self, grid, k=None, c=None):
        HydraulicParameters.__init__(self, grid)
        self.k = self._broadcast(k)
        self.c = None
        self._check_c(c)
        self._set_qc()

    @abstractmethod
    def _check_c(self, c):
        """
        Check input parameter `c` and assign it to attribute `c`.

        Abstract method.

        Parameters
        ----------
        c : array_like
          Array with hydraulic resistances [T].

        Returns
        -------
        None
        """
        pass


class HorizontalFlowParameters(FlowParameters):
    """
    Abstract base class for classes defining the radial or horizontal flow parameters.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    k : array_like, default: None
        Two-dimensional array with radial or horizontal conductivities [L/T] of the grid cells.
        The shape of `k` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    c : array_like, default: None
        Two-dimensional array with radial or horizontal resistances [T] between the grid cells.
        The shape of `c` is `(nl, nr - 1)`, but it is broadcast if dimensions are missing.

    Attributes
    ----------
    qc : ndarray
       Two-dimensional array with radial or horizontal conductances [L²/T] between the grid cells.
       The shape of `qc` is `(nl, nr + 1)`, which means the zero conductances of the model boundaries are included.

    Notes
    -----
    Subclasses implement protected abstract method `_calculate_qc` to set attribute `qc`.

    If input parameter `c` is not given, then conductances are colculated using input parameters `k`, and
    if input parameter `k` is not given, then conductances are colculated using input parameters `c`.
    If both input parameters `k` and `c` are given, then conductances are calculated using `k`, after which the
    calculated conductances are replaced by conductances calculated using elements of `c` which are not `nan`.

    """

    def __init__(self, grid, k=None, c=None):
        FlowParameters.__init__(self, grid, k, c)

    def _check_c(self, c):
        """
        Check input parameter `c` and assign it to attribute `c`.
        The input array is broadcast if dimensions are missing.

        Parameters
        ----------
        c : array_like
          Array with radial or horizontal resistances [T] between the grid cells.

        Returns
        -------
        None
        """
        self.c = self._broadcast(c, ncol=self.nr - 1)

    def _set_qc(self):
        """
        Calculate radial or horizontal conductances and assign them to attribute `qc`.

        Calls method `_calculate_qc`.

        Returns
        -------
        None
        """
        self.qc = self._zeros(ncol=self.nr + 1)  # (nl, nr+1)
        qc = self._calculate_qc()
        b = self.inactive[:, :-1] | self.inactive[:, 1:]  # (nl, nr-1)
        qc[b] = 0.0
        self.qc[:, 1:-1] = qc

    @abstractmethod
    def _calculate_qc(self):
        """
        Calculate radial or horizontal conductances.

        Returns
        -------
        qc : ndarray
           Two-dimensional array with radial or horizontal conductances [L²/T] between the grid cells.
           The shape of `qc` is `(nl, nr - 1)`, which means the model boundaries are not included.
        """
        pass


class VerticalFlowParameters(FlowParameters):
    """
    Class defining the vertical flow parameters.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    k : array_like, default: None
        Two-dimensional array with vertical conductivities [L/T] of the grid cells.
        The shape of `k` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    c : array_like, default: None
        Two-dimensional array with vertical resistances [T] between the grid layers.
        The shape of `c` is `(nl - 1, nr)`, but it is broadcast if dimensions are missing.

    Attributes
    ----------
    qc : ndarray
       Two-dimensional array with vertical conductances [L²/T] between the grid layers.
       The shape of `qc` is `(nl + 1, nr)`, which means the zero conductances of the model boundaries are included.

    Notes
    -----
    If input parameter `c` is not given, then conductances are colculated using input parameters `k`, and
    if input parameter `k` is not given, then conductances are colculated using input parameters `c`.
    If both input parameters `k` and `c` are given, then conductances are calculated using `k`, after which the
    calculated conductances are replaced by conductances calculated using elements of `c` which are not `nan`.

    """

    def __init__(self, grid, k=None, c=None):
        FlowParameters.__init__(self, grid, k, c)

    def _check_c(self, c):
        """
        Check input parameter `c` and assign it to attribute `c`.
        The input array is broadcast if dimensions are missing.

        Parameters
        ----------
        c : array_like
          Array with vertical resistances [T] between the grid layers.

        Returns
        -------
        None
        """
        self.c = self._broadcast(c, nrow=self.nl - 1)

    def _set_qc(self):
        """
        Calculate vertical conductances and assign them to attribute `qc`.

        Returns
        -------
        None
        """
        self.qc = self._zeros(nrow=self.nl + 1)  # (nl+1, nr)
        if self.k is None:
            c = self.c  # (nl-1, nr)
        else:
            c = self.D[:, np.newaxis] / self.k  # (nl, nr)
            c = (c[:-1, :] + c[1:, :]) / 2.0  # (nl-1, nr)
            if self.c is not None:
                b = ~np.isnan(self.c)  # (nl-1, nr)
                c[b] = self.c
        b = self.inactive[:-1, :] | self.inactive[1:, :]  # (nl-1, nr)
        c[b] = np.inf
        self.qc[1:-1, :] = self.hs / c


class StorageParameters(HydraulicParameters):
    """
    Class defining the storage parameters.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    ss : array_like
        Two-dimensional array with specific storage [1/L] of the grid cells.
        The shape of `ss` is `(nl, nr)`, but it is broadcast if dimensions are missing.
    sy : array_like
        One-dimensional array with specific yield [-] of the grid cells in the top layer.
        The length of `sy` is `nr`, but it is broadcast if only one value is given.

    Attributes
    ----------
    qc : ndarray
       Two-dimensional array with constant terms to calculate storage change in each cell.
       In case of `ss`, the constant term of a cell is calculated as the volume of the cell multiplied by `ss`.
       If `sy` is given, then the horizontal surface of the cell multiplied by `sy` is added to the constant term.
       The shape of `qc` is `(nl, nr)`.
    """

    def __init__(self, grid, ss, sy=None):
        HydraulicParameters.__init__(self, grid)
        self.ss = self._broadcast(ss)  # (nl, nr)
        self.sy = None  # (nr, )
        self._check_sy(sy)
        self._set_qc()

    def _check_sy(self, sy):
        """
        Check input parameter `sy` and assign it to attribute `sy`.
        The input array is broadcast if only one value is given.

        Parameters
        ----------
        sy : array_like
           One-dimensional array with specific yield [-] of the grid cells in the top layer.

        Returns
        -------
        None
        """
        if sy is not None:
            self.sy = np.array(sy)
            if self.sy.ndim == 0:
                self.sy = self.sy * np.ones(self.nr)

    def _set_qc(self):
        """
        Calculate constant storage change terms and assign them to attribute `qc`.

        Returns
        -------
        None
        """
        self.qc = self._zeros()  # (nl, nr)
        b = self.variable  # (nl, nr)
        self.qc[b] = self.vol[b] * self.ss[b]
        if self.sy is not None:  # take into account specific yield of top layer
            self.qc[0, :] += self.hs * self.sy

