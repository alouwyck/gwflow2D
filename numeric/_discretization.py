"""
Base module containing base classes to define grid and time steps for two-dimensional finite-difference groundwater flow
models.
"""
from abc import ABC, abstractmethod
import numpy as np


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
    Abstract base class for classes that implement a two-dimensional axi-symmetric or a rectilinear finite-difference
    grid.

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

