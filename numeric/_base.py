"""
Base module containing classes for building steady and transient state 2D finite-difference groundwater flow models.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve
import warnings
from scipy.linalg import LinAlgWarning


class Object:
    """
    Base class from which all other numerical classes inherit.

    Contains methods for initialization and broadcasting of arrays.
    """

    def _broadcast(self, arr, nrow, ncol, dtype=float):
        """
        Convert array into numpy array and broadcast it.

        This method is not static as some subclasses overriding it make use of object `self`.

        Parameters
        ----------
        arr : array_like
            Input array.
        nrow : int
             Required number of rows.
        ncol : int
             Required number of cols.
        dtype : data-type, default: float
              Required datatype. Any object that can be interpreted as a numpy data type.

        Returns
        -------
        arr : ndarray
            Shape of `arr` is `(nrow, ncol)`; datatype is `dtype`.
        """
        if arr is None:
            return None
        arr = np.array(arr, dtype=dtype)
        if arr.ndim == 0:
            return arr * np.ones((nrow, ncol))
        if arr.ndim == 1:
            if len(arr) == nrow:
                return np.repeat(arr[:, np.newaxis], ncol, axis=1)
            else:
                return np.repeat(arr[np.newaxis, :], nrow, axis=0)
        if arr.shape[0] == 1:  # ndim == 2
            arr = np.repeat(arr, nrow, axis=0)
        if arr.shape[1] == 1:  # ndim == 2
            arr = np.repeat(arr, ncol, axis=1)
        return arr

    def _zeros(self, nrow, ncol, dtype=float):
        """
        Create numpy array of given shape and type, filled with zeros. Makes use of numpy function `zeros`.

        This method is not static as some subclasses overriding it make use of object `self`.

        Parameters
        ----------
        nrow : int
             Required number of rows.
        ncol : int
             Required number of cols.
        dtype : data-type, default: float
              Required datatype. Any object that can be interpreted as a numpy data type.

        Returns
        -------
        arr : ndarray
            Array of zeros.
            Shape of `arr` is `(nrow, ncol)`; datatype is `dtype`.
        """
        return np.zeros((nrow, ncol), dtype=dtype)

    def _ones(self, nrow, ncol, dtype=float):
        """
        Create numpy array of given shape and type, filled with ones. Makes use of numpy function `ones`.

        This method is not static as some subclasses overriding it make use of object `self`.

        Parameters
        ----------
        nrow : int
             Required number of rows.
        ncol : int
             Required number of cols.
        dtype : data-type, default: float
              Required datatype. Any object that can be interpreted as a numpy data type.

        Returns
        -------
        arr : ndarray
            Array of ones.
            Shape of `arr` is `(nrow, ncol)`; datatype is `dtype`.
        """
        return np.ones((nrow, ncol), dtype=dtype)


class Grid(Object, ABC):
    """
    Abstract base class for classes that implement a two-dimensional axi-symmetric or a rectilinear grid.

    Parameters
    ----------
    rb : array_like
       One-dimensional array with radial or horizontal distance [L] of grid cell boundaries.
       The length of `rb` is `nr + 1`, with `nr` the number of columns in the grid.
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
       Number of grid columns (the `r` in `nr` refers to the radial or horizontal distances represented by the cells).
    nl : int
       Number of grid layers.
    n : int
      Number of cells in the grid, i.e. ``nr * nl``.
    r : ndarray
      One-dimensional array representing the radial or horizontal distance [L] of the center of the grid cells.
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

    Notes
    -----
    Subclasses implement protected abstract method `_set_r_hs_vol`, which calculates the horizontal or radial
    distance of the centers of the grid cells, and the horizontal surface and the volume of the cells.

    The class also contains protected method `_is_connected`.

    """

    def __init__(self, rb, D, constant=None, inactive=None, connected=None):
        Object.__init__(self)
        self.rb = np.array(rb)  # (nr+1, )
        self.D = np.array(D)  # (nl, )
        if self.D.ndim == 0:  # if D is a scalar
            self.D = self.D[np.newaxis]
        self.nr = len(self.rb) - 1
        self.nl = len(self.D)
        self.n = self.nr * self.nl
        if constant is None:
            self.constant = self._zeros(self.nl, self.nr, bool)  # (nl, nr)
        else:
            self.constant = self._broadcast(constant, self.nl, self.nr, bool)  # (nl, nr)
        if inactive is None:
            self.inactive = self._zeros(self.nl, self.nr, bool)  # (nl, nr)
        else:
            self.inactive = self._broadcast(inactive, self.nl, self.nr, bool)  # (nl, nr)
        if connected is None:
            self.connected = self._zeros(self.nl, self.nr, int)  # (nl, nr)
        else:
            self.connected = self._broadcast(connected, self.nl, self.nr, int)  # (nl, nr)
        self.variable = ~(self.constant | self.inactive | self._is_connected())  # (nl, nr)
        self.r = None  # (nr, )
        self.hs = None  # (nr, )
        self.vol = None  # (nl, nr)
        self._set_r_hs_vol()

    @abstractmethod
    def _set_r_hs_vol(self):
        """
        Set attributes `r`, `hs`, and `vol`.

        Abstract method.

        Returns
        -------
        None
        """
        pass

    def _is_connected(self):
        """
        Check which grid cells are connected.

        Returns
        -------
        is_connected : ndarray
                     Two-dimensional array indicating which grid cells are connected.
                     The shape of `is_connected` is `(nl, nr)`.

        Notes
        -----
        Sets protected attribute `_connected_ids` with linear indices of cells that are connected to each other.
        `_connected_ids` is a list of tuples `(first, remaining)` with `first` the index of the first cell and
        `remaining` the indices of the cells connected to the first.

        """
        is_connected = self.connected.astype(bool)  # (nl, nr)
        self._connected_ids = []
        for i in range(1, np.max(self.connected) + 1):
            row, col = (self.connected == i).nonzero()
            row0, col0 = row[0], col[0]
            row, col = row[1:], col[1:]
            first, remaining = row0 * self.nr + col0, row * self.nr + col
            is_connected[row0, col0] = False
            self._connected_ids.append((first, remaining))
        return is_connected  # (nl, nr)


class GridDependent(Object):
    """
    Base class for classes that implement grid-dependent parameters and boundary conditions.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    """

    def __init__(self, grid):
        Object.__init__(self)
        self.grid = grid

    @property
    def nr(self):
        """
        Number of grid columns.

        Returns
        -------
        nr: int
          Number of grid columns.
        """
        return self.grid.nr

    @property
    def nl(self):
        """
        Number of grid layers.

        Returns
        -------
        nl: int
          Number of grid layers.
        """
        return self.grid.nl

    @property
    def n(self):
        """
        Number of grid cells.

        Returns
        -------
        n: int
          Number of grid cells.
        """
        return self.grid.n  # n = nl x nr

    @property
    def constant(self):
        """
        Boolean array that indicates which cells have a constant head.

        Returns
        -------
        constant : ndarray
                 Two-dimensional boolean array that indicates which cells have a constant head.
                 The shape of `constant` is `(nl, nr)`.
        """
        return self.grid.constant  # (nl, nr) boolean

    @property
    def inactive(self):
        """
        Boolean array that indicates which cells have are inactive.

        Returns
        -------
        inactive : ndarray
                 Two-dimensional boolean array that indicates which cells are inactive.
                 The shape of `inactive` is `(nl, nr)`.
        """
        return self.grid.inactive  # (nl, nr) boolean

    @property
    def connected(self):
        """
        Integer array that indicates which cells are connected.

        Returns
        -------
        connected : ndarray
                  Two-dimensional integer array that indicates which cells are connected.
                  The shape of `connected` is `(nl, nr)`.
        """
        return self.grid.connected  # (nl, nr) boolean

    @property
    def variable(self):
        """
        Boolean array that indicates which cells have a variable head.

        Returns
        -------
        variable : ndarray
                 Two-dimensional boolean array that indicates which cells have a variable head.
                 The shape of `variable` is `(nl, nr)`.
        """
        return self.grid.variable  # (nl, nr) boolean

    @property
    def rb(self):
        """
        One-dimensional array with radial or horizontal distance of grid cell boundaries.

        Returns
        -------
        rb : ndarray
           One-dimensional array with radial or horizontal distance [L] of grid cell boundaries.
           The length of `rb` is `nr + 1`, with `nr` the number of columns in the grid.
        """
        return self.grid.rb  # (nr+1, )

    @property
    def r(self):
        """
        One-dimensional array with radial or horizontal distances represented by the grid cells.

        Returns
        -------
        r : ndarray
          One-dimensional array representing the radial or horizontal distance [L] of the center of the grid cells.
          The length of `r` is `nr`.
        """
        return self.grid.r  # (nr, )

    @property
    def hs(self):
        """
        One-dimensional array representing the horizontal surface of the grid cells.

        Returns
        -------
        hs : ndarray
           One-dimensional array representing the horizontal surface [L²] of the grid cells.
           The length of `hs` is `nr`.
        """
        return self.grid.hs  # (nr, )

    @property
    def D(self):
        """
        One-dimensional array with the thickness of the grid layers.

        Returns
        -------
        D : array_like
          One-dimensional array with height (thickness) [L] of grid layers.
          The length of `D` is `nl`, with `nl` the number of grid layers.
        """
        return self.grid.D  # (nl, )

    @property
    def vol(self):
        """
        Two-dimensional array representing the volume of the grid cells.

        Returns
        -------
        vol : ndarray
            Two-dimensional array representing the volume [L³] of the grid cells.
            The shape of `vol` is `(nl, nr)`.
        """
        return self.grid.vol  # (nl, nr)

    @property
    def _connected_ids(self):
        """
        Linear indices of connected cells.

        Returns
        -------
        _connected_ids : list
                       List of tuples `(first, remaining)` with `first` the index of the first cell and
                       `remaining` the indices of the cells connected to the first.
        """
        return self.grid._connected_ids

    def _broadcast(self, arr, nrow=None, ncol=None, dtype=float):
        """
        Convert array into numpy array and broadcast it.

        Overrides `Object._broadcast`.

        Parameters
        ----------
        arr : array_like
            Input array.
        nrow : int, default: None
             Required number of rows. If `None`, the number of grid layers `nl` is considered.
        ncol : int, default: None
             Required number of cols. If `None`, the number of grid rows `nr` is considered.
        dtype : data-type, default: float
              Required datatype. Any object that can be interpreted as a numpy data type.

        Returns
        -------
        arr : ndarray
            Shape of `arr` is `(nrow, ncol)`; datatype is `dtype`.
        """
        if nrow is None:
            nrow = self.nl
        if ncol is None:
            ncol = self.nr
        return Object._broadcast(self, arr, nrow, ncol, dtype)

    def _zeros(self, nrow=None, ncol=None, dtype=float):
        """
        Create numpy array of given shape and type, filled with zeros.

        Overrides `Object._zeros`.

        Parameters
        ----------
        nrow : int, default: None
             Required number of rows. If `None`, the number of grid layers `nl` is considered.
        ncol : int, default: None
             Required number of cols. If `None`, the number of grid rows `nr` is considered.
        dtype : data-type, default: float
              Required datatype. Any object that can be interpreted as a numpy data type.

        Returns
        -------
        arr : ndarray
            Array of zeros.
            Shape of `arr` is `(nrow, ncol)`; datatype is `dtype`.
        """
        if nrow is None:
            nrow = self.nl
        if ncol is None:
            ncol = self.nr
        return Object._zeros(self, nrow, ncol, dtype)

    def _ones(self, nrow=None, ncol=None, dtype=float):
        """
        Create numpy array of given shape and type, filled with ones.

        Overrides `Object._ones`.

        Parameters
        ----------
        nrow : int, default: None
             Required number of rows. If `None`, the number of grid layers `nl` is considered.
        ncol : int, default: None
             Required number of cols. If `None`, the number of grid rows `nr` is considered.
        dtype : data-type, default: float
              Required datatype. Any object that can be interpreted as a numpy data type.

        Returns
        -------
        arr : ndarray
            Array of ones.
            Shape of `arr` is `(nrow, ncol)`; datatype is `dtype`.
        """
        if nrow is None:
            nrow = self.nl
        if ncol is None:
            ncol = self.nr
        return Object._ones(self, nrow, ncol, dtype)


class TimeSteps:
    """
    Class defining the time steps in the model.

    Parameters
    ----------
    t : array_like
      One-dimensional array with simulation times [T]. Length of `t` is `nt`.
      If the first element of `t` is not zero, 0.0 is inserted in the front. This first element represents time t = 0
      for which the initial conditions are defined.

    Attributes
    ----------
    dt : ndarray
       One-dimensional array with the duration [T] of time steps. Length of `dt` is `nt - 1`.
    nt : int
       Number of simulation times.
    """

    def __init__(self, t):
        # t is (nt, ) floating point array (required)
        # t0 = 0 is not given
        self.t = np.array(t, dtype=float)  # (nt-1, )
        if t[0] > 0.0:  # add t0 = 0 if t doesn't contain it
            self.t = np.insert(self.t, 0, 0)  # (nt, )
        self.dt = self.t[1:] - self.t[:-1]  # (nt-1, )
        self.nt = len(self.t)


class TimeDependent:
    """
    Base class for classes that depend on time steps.

    Parameters
    ----------
    timesteps : TimeSteps object or list of TimeSteps objects, default: None
              Contains the time steps if transient state; is `None` if steady state.
    """

    def __init__(self, timesteps=None):
        self.timesteps = timesteps

    @property
    def steady(self):
        """
        Indicates whether the simulation is steady state or transient state.

        Returns
        -------
        steady : bool
               Is `True` if steady state.
        """
        return self.timesteps is None

    @property
    def nt(self):
        """
        Number of simulation times.

        Returns
        -------
        nt : int
           Number of simulation times. In case of steady state, `nt` equals one.
        """
        if self.steady:
            return 1
        else:
            return self.timesteps.nt

    @property
    def t(self):
        """
        One-dimensional array with simulation times.

        Returns
        -------
        t : ndarray
          One-dimensional array with simulation times [T]. Length of `t` is `nt`.
          In case of steady state, an array only holding 0.0 is returned.
        """
        if self.steady:
            return np.array([0.0])
        else:
            return self.timesteps.t  # (nt, )

    @property
    def dt(self):
        """
        One-dimensional array with the duration of time steps.

        Returns
        -------
        dt : ndarray
           One-dimensional array with the duration [T] of time steps. Length of `dt` is `nt - 1`.
           In case of steady state, `None` is returned.
        """
        if self.steady:
            return None
        else:
            return self.timesteps.dt  # (nt-1, )


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
        Two-dimensional array with horizontal conductivities [L/T] of the grid cells.
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


class SolveHeads(GridDependent, TimeDependent, ABC):
    """
    Abstract base class for classes that simulate hydraulic heads.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    timesteps : TimeSteps object or list of TimeSteps objects, default: None
              Contains the time steps if transient state; is `None` if steady state.

    Methods
    -------
    solve() : abstract
            Solve the system of finite-difference equations to obtain the head in each grid cell, and for each time step
            if transient state. This method is implemented in the subclasses.
    """
    def __init__(self, grid, timesteps=None):
        GridDependent.__init__(self, grid)
        TimeDependent.__init__(self, timesteps)
        self._h = None

    @property
    def h(self):
        """
        Hydraulic heads.

        Returns
        -------
        h : ndarray
          Three-dimensional array with simulated head [L] in each grid cell and for each time step.
          The shape of `h` is `(nl, nr, nt)`. Note that `nt` equals 1 in case of steady state simulation.
        """
        return self._h

    @abstractmethod
    def solve(self):
        """
        Solve the system of finite-difference equations to obtain the head in each grid cell, and for each time step
        if transient state.

        Abstract method.

        Returns
        -------
        None
        """
        pass


class StressPeriod(SolveHeads):

    def __init__(self, grid, timesteps=None, previous=None):
        SolveHeads.__init__(self, grid, timesteps)
        self.previous = previous
        self._horizontal_flow_parameters = None
        self._vertical_flow_parameters = None
        self._storage_parameters = None
        self._initial_heads = None
        self._constant_heads = None
        self._discharges = None
        self._head_dependent_fluxes = None
        self._new_A0 = False
        self._new_b0 = False
        self._new_dS0 = False
        self._horizontal_flow_constructor = None  # SUBCLASSING: assign HorizontalFlowParameters class!

    def add_kh(self, kh=None, ch=None):
        self._horizontal_flow_parameters = self._horizontal_flow_constructor(self.grid, kh, ch)
        self._new_A0 = True
        self._new_b0 = True

    @property
    def kh(self):  # (nl, nr)
        if self.nr == 1:
            return None
        elif self._horizontal_flow_parameters is not None:
            return self._horizontal_flow_parameters.k
        else:
            return self.previous.kh

    @property
    def ch(self):  # (nl, nr-1)
        if self.nr == 1:
            return None
        elif self._horizontal_flow_parameters is not None:
            return self._horizontal_flow_parameters.c
        else:
            return self.previous.ch

    @property
    def qhc(self):  # (nl, nr+1)
        if self.nr == 1:
            return np.zeros((self.nl, 2))
        elif self._horizontal_flow_parameters is not None:
            return self._horizontal_flow_parameters.qc
        else:
            return self.previous.qhc

    def add_kv(self, kv=None, cv=None):
        self._vertical_flow_parameters = VerticalFlowParameters(self.grid, kv, cv)
        self._new_A0 = True
        self._new_b0 = True

    @property
    def kv(self):  # (nl, nr)
        if self.nl == 1:
            return None
        elif self._vertical_flow_parameters is not None:
            return self._vertical_flow_parameters.k
        else:
            return self.previous.kv

    @property
    def cv(self):  # (nl-1, nr)
        if self.nl == 1:
            return None
        elif self._vertical_flow_parameters is not None:
            return self._vertical_flow_parameters.c
        else:
            return self.previous.cv

    @property
    def qvc(self):  # (nl+1, nr)
        if self.nl == 1:
            return np.zeros((2, self.nr))
        elif self._vertical_flow_parameters is not None:
            return self._vertical_flow_parameters.qc
        else:
            return self.previous.qvc

    def add_ss(self, ss):
        self._storage_parameters = StorageParameters(self.grid, ss)
        self._new_dS0 = True

    @property
    def ss(self):  # (nl, nr)
        if self.steady:
            return None
        elif self._storage_parameters is not None:
            return self._storage_parameters.ss
        else:
            return self.previous.ss

    @property
    def qsc(self):  # (nl, nr)
        if self.steady:
            return None
        elif self._storage_parameters is not None:
            return self._storage_parameters.qc
        else:
            return self.previous.qsc

    def add_q(self, q):
        self._discharges = Discharges(self.grid, q)
        self._new_b0 = True

    @property
    def q(self):  # (nl, nr)
        if self._discharges is not None:
            return self._discharges.q
        elif self.previous is not None:
            return self.previous.q
        else:
            return np.zeros((self.nl, self.nr))

    def add_h0(self, h0, add=False):
        self._initial_heads = InitialHeads(self.grid, h0, add)

    @property
    def h0(self):  # (nl, nr)
        if self.previous is not None:
            if self._initial_heads.add:
                return self._initial_heads.h + (self.previous.h if self.previous.steady else self.previous.h[:, :, -1])
            else:
                return self.previous.h if self.previous.steady else self.previous.h[:, :, -1]
        elif self._initial_heads is not None:
            return self._initial_heads.h
        else:
            return np.zeros((self.nl, self.nr))

    def add_hc(self, hc):
        self._constant_heads = ConstantHeads(self.grid, hc)
        self._new_b0 = True

    @property
    def hc(self):  # (nl, nr)
        if self._constant_heads is not None:
            return self._constant_heads.h
        elif self.previous is not None:
            return self.previous.hc
        else:
            return np.zeros((self.nl, self.nr))

    def add_hdep(self, dependent, cdep, hdep=None):
        self._head_dependent_fluxes = HeadDependentFluxes(self.grid, dependent, cdep, hdep)
        self._new_A0 = True
        self._new_b0 = True

    @property
    def dependent(self):  # (nl, nr)
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.dependent
        elif self.previous is not None:
            return self.previous.dependent
        else:
            return None

    @property
    def cdep(self):  # (nl, nr)
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.c
        elif self.previous is not None:
            return self.previous.cdep
        else:
            return None

    @property
    def hdep(self):  # (nl, nr)
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.h
        elif self.previous is not None:
            return self.previous.hdep
        else:
            return None

    @property
    def qdc(self):  # tuple (idx, qdc)
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.idx, self._head_dependent_fluxes.qc
        elif self.previous is not None:
            return self.previous.qdc
        else:
            return None, None

    def _initialize_h(self):
        # initial heads
        self._h = self.h0.flatten()
        # inactive cells
        self._h[self.inactive.flatten()] = np.nan
        # constant heads
        self._h[self.constant.flatten()] = self.hc[self.constant]
        # add time dimension to h
        self._h = np.repeat(self._h[:, np.newaxis], self.nt, axis=1)

    def _calculate_A0(self):
        if self._new_A0:
            # indices
            nr = self.nr
            self._idx = np.diag_indices(self.n)
            irow, icol = self._idx
            # initialize A0
            self._A0 = np.zeros((self.n, self.n))  # (n, n)
            # set A0 diagonals
            qhc, qvc = self.qhc, self.qvc
            self._A0[irow[:-1] + 1, icol[:-1]] = self._A0[irow[:-1], icol[:-1] + 1] = qhc[:, 1:].flatten()[:-1]
            self._A0[irow[:-nr] + nr, icol[:-nr]] = self._A0[irow[:-nr], icol[:-nr] + nr] = qvc[1:, :].flatten()[:-nr]
            self._A0[irow, icol] = -(qhc[:, :-1] + qhc[:, 1:] + qvc[:-1, :] + qvc[1:, :]).flatten()
            # head dependent cells
            i, qdc = self.qdc
            if i is not None:
                self._A0[i, i] -= qdc
        else:
            self._A0 = self.previous._A0

    def _calculate_b0(self):
        if self._new_b0:
            # discharges
            self._b0 = -self.q.flatten()  # (n, )
            # constant heads: first calculate A0 and initialize h!!
            is_constant = self.constant.flatten()
            self._b0 -= np.dot(self._A0[:, is_constant], self._h[is_constant, 0])
            # head dependent cells
            i, qdc = self.qdc
            if i is not None:
                self._b0[i] -= qdc * self.hdep.flatten()[i]
        else:
            self._b0 = self.previous._b0

    def _get_A0_b0(self):
        self._calculate_A0()
        self._calculate_b0()
        A0, b0 = self._A0.copy(), self._b0.copy()
        # connected cells
        for first, remaining in self._connected_ids:
            A0[first, :] += A0[remaining, :].sum(axis=0)
            A0[:, first] += A0[:, remaining].sum(axis=1)
            b0[first] += b0[remaining].sum()
        # remove rows and columns corresponding to cells with no variable head
        self._is_variable = self.variable.flatten()
        A0 = A0[self._is_variable, :][:, self._is_variable]
        b0 = b0[self._is_variable]
        return A0, b0

    def _get_dS0(self):
        if self._new_dS0:
            self._dS0 = self.qsc.flatten()  # (n, )
        else:
            self._dS0 = self.previous._dS0
        dS0 = self._dS0.copy()
        for first, remaining in self._connected_ids:
            dS0[first] += dS0[remaining].sum()
        return dS0[self._is_variable]

    def _initialize(self):
        # indices
        nr = self.nr
        self._idx = np.diag_indices(self.n)
        irow, icol = self._idx
        # initialize b0 and A0
        self._b0 = -self.q.flatten()  # (n, )
        self._A0 = np.zeros((self.n, self.n))  # (n, n)
        # set A0 diagonals
        qhc, qvc = self.qhc, self.qvc
        self._A0[irow[:-1] + 1, icol[:-1]] = self._A0[irow[:-1], icol[:-1] + 1] = qhc[:, 1:].flatten()[:-1]
        self._A0[irow[:-nr] + nr, icol[:-nr]] = self._A0[irow[:-nr], icol[:-nr] + nr] = qvc[1:, :].flatten()[:-nr]
        self._A0[irow, icol] = -(qhc[:, :-1] + qhc[:, 1:] + qvc[:-1, :] + qvc[1:, :]).flatten()
        # initialize h
        self._h = self.h0.flatten()
        # inactive cells
        self._h[self.inactive.flatten()] = np.nan
        # constant heads
        is_constant = self.constant.flatten()
        self._h[is_constant] = self.hc[self.constant]
        self._b0 -= np.dot(self._A0[:, is_constant], self._h[is_constant])
        # connected cells
        for first, remaining in self._connected_ids:
            self._b0[first] += self._b0[remaining].sum()
            self._A0[first, :] += self._A0[remaining, :].sum(axis=0)
            self._A0[:, first] += self._A0[:, remaining].sum(axis=1)
        # head dependent cells
        i, qdc = self.qdc
        if i is not None:
            self._b0[i] -= qdc * self.hdep.flatten()[i]
            self._A0[i, i] -= qdc
        # remove cells with no variable head from h0, b0 and A0
        self._is_variable = self.variable.flatten()
        self._b0 = self._b0[self._is_variable]
        self._A0 = self._A0[self._is_variable, :][:, self._is_variable]
        # add time dimension to h
        self._h = np.repeat(self._h[:, np.newaxis], self.nt, axis=1)
        # transient: set dS0
        if not self.steady:
            self._dS0 = self.qsc.flatten()  # (n, )
            for first, remaining in self._connected_ids:  # connected cells
                self._dS0[first] += self._dS0[remaining].sum()
            self._dS0 = self._dS0[self._is_variable]  # keep variable head cells only

    def solve(self):
        self._initialize()
        if self.steady:
            self._h[self._is_variable, 0] = solve(self._A0, self._b0)
        else:  # transient
            h = self._h[self._is_variable, 0]
            i = np.diag_indices(self._A0.shape[0])
            for k in range(self.nt - 1):
                dS = self._dS0 / self.dt[k]
                b = self._b0 - dS * h
                A = self._A0.copy()
                A[i] -= dS
                h = solve(A, b)
                self._h[self._is_variable, k + 1] = h
        for first, remaining in self._connected_ids:
            self._h[remaining, :] = self._h[first, :]
        self._h = np.reshape(self._h, (self.nl, self.nr, self.nt))


class Model(SolveHeads):

    def __init__(self):
        SolveHeads.__init__(self, None, [])
        self.periods = []
        self.no_warnings = True
        self._grid_constructor = None  # SUBCLASSING: assign Grid class!
        self._period_constructor = None  # SUBCLASSING: assign StressPeriod class!

    def add_grid(self, rb, D, constant=None, inactive=None, connected=None):
        self.grid = self._grid_constructor(rb=rb, D=D, constant=constant, inactive=inactive, connected=connected)
        return self.grid

    def add_period(self, t=None):
        timesteps = None if t is None else TimeSteps(t)
        self.timesteps.append(timesteps)
        previous = None if len(self.periods) == 0 else self.periods[-1]
        period = self._period_constructor(grid=self.grid, timesteps=timesteps, previous=previous)
        self.periods.append(period)
        return period

    @property
    def steady(self):
        return np.array([period.steady for period in self.periods])

    @property
    def nt(self):
        return sum([period.nt for period in self.periods if not period.steady])

    @property
    def t(self):
        return np.hstack([period.t + (period.previous.t[-1] if period.previous else 0.0) for period in self.periods])

    @property
    def dt(self):
        t = self.t
        return t[1:] - t[:-1]

    def solve(self):
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            for period in self.periods:
                period.solve()

    @property
    def h(self):
        return np.concatenate([period.h for period in self.periods], axis=2)

