"""
Base module containing base classes for building two-dimensional finite-difference groundwater flow models.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve
import warnings
from scipy.linalg import LinAlgWarning
from ._discretization import GridDependent, TimeDependent, TimeSteps
from ._parameters import VerticalFlowParameters, StorageParameters
from ._conditions import Discharges, ConstantHeads, InitialHeads, HeadDependentFluxes


class SolveHeads(GridDependent, TimeDependent, ABC):
    """
    Abstract base class for classes that simulate hydraulic heads.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    timesteps : TimeSteps object or list of TimeSteps objects, default: None
              Contains the time steps if transient state; is `None` if steady state.
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    solve() : abstract
            Solve the system of finite-difference equations to obtain the head in each grid cell, and for each time step
            if transient state. This method is implemented in the subclasses.
    """
    def __init__(self, grid, timesteps=None, no_warnings=True):
        GridDependent.__init__(self, grid)
        TimeDependent.__init__(self, timesteps)
        self.no_warnings = no_warnings
        self._h = None

    @property
    def h(self):
        """
        Hydraulic heads.

        Returns
        -------
        h : ndarray
          Three-dimensional array with simulated head [L] in each grid cell and for each time.
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
    """
    Base class for classes defining a stress period.

    A stress period is a period during which boundary conditions do not change with time.

    Parameters
    ----------
    grid : Grid object
         Axi-symmetric or rectilinear two-dimensional grid.
    timesteps : TimeSteps object, default: None
              Contains the time steps if transient state; is `None` if steady state.
    previous : TimeSteps object, default: None
             Contains the time steps of previous stress period.
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    add_kh(kh, ch) :
                  Add radial or horizontal flow parameters.
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

    Notes
    -----
    Subclasses assign the appropriate `HorizontalFLowParameters` class to protected attribute
    `_horizontal_flow_constructor` during instantiation.

    """

    def __init__(self, grid, timesteps=None, previous=None, no_warnings=True):
        SolveHeads.__init__(self, grid, timesteps, no_warnings)
        self.previous = previous
        self._horizontal_flow_parameters = None
        self._vertical_flow_parameters = None
        self._storage_parameters = None
        self._initial_heads = None
        self._constant_heads = None
        self._discharges = None
        self._head_dependent_fluxes = None
        self._horizontal_flow_constructor = None  # SUBCLASSING: assign HorizontalFlowParameters class!

    def add_kh(self, kh=None, ch=None):
        """
        Add radial or horizontal flow parameters.

        Parameters
        ----------
        kh : array_like, default: None
             Two-dimensional array with radial or horizontal conductivities [L/T] of the grid cells.
             The shape of `kh` is `(nl, nr)`, but it is broadcast if dimensions are missing.
        ch : array_like, default: None
             Two-dimensional array with radial or horizontal resistances [T] between the grid cells.
             The shape of `ch` is `(nl, nr - 1)`, but it is broadcast if dimensions are missing.

        Returns
        -------
        par : HorizontalFlowParameters object

        Notes
        -----
        If input parameter `ch` is not given, then conductances are colculated using input parameters `kh`, and
        if input parameter `kh` is not given, then conductances are colculated using input parameters `ch`.
        If both input parameters `kh` and `ch` are given, then conductances are calculated using `kh`, after which the
        calculated conductances are replaced by conductances calculated using elements of `ch` which are not `nan`.
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._horizontal_flow_parameters = self._horizontal_flow_constructor(self.grid, kh, ch)
            return self._horizontal_flow_parameters

    @property
    def kh(self):
        """
        Radial or horizontal conductivities.

        Returns
        -------
        kh : ndarray
           Two-dimensional array with radial or horizontal conductivity [L/T] in each cell of the grid.
           The shape of `kh` is `(nl, nr)`.
        """
        if self.nr == 1:
            return None
        elif self._horizontal_flow_parameters is not None:
            return self._horizontal_flow_parameters.k  # (nl, nr)
        else:
            return self.previous.kh  # (nl, nr)

    @property
    def ch(self):
        """
        Radial or horizontal resistances.

        Returns
        -------
        ch : ndarray
           Two-dimensional array with radial or horizontal resistances [T] between the grid cells.
           The shape of `ch` is `(nl, nr - 1)`, which means the infinitely large resistances of the model boundaries are
           not included.
        """
        if self.nr == 1:
            return None
        elif self._horizontal_flow_parameters is not None:
            return self._horizontal_flow_parameters.c  # (nl, nr-1)
        else:
            return self.previous.ch  # (nl, nr-1)

    @property
    def qhc(self):
        """
        Radial or horizontal conductances.

        Returns
        -------
        qhc : ndarray
            Two-dimensional array with radial or horizontal conductances [L²/T] between the grid cells.
            The shape of `qhc` is `(nl, nr + 1)`, which means the zero conductances of the model boundaries are included.
        """
        if self.nr == 1:
            return np.zeros((self.nl, 2))
        elif self._horizontal_flow_parameters is not None:
            return self._horizontal_flow_parameters.qc  # (nl, nr+1)
        else:
            return self.previous.qhc  # (nl, nr+1)

    def add_kv(self, kv=None, cv=None):
        """
        Add vertical flow parameters.

        Parameters
        ----------
        kv : array_like, default: None
             Two-dimensional array with vertical conductivities [L/T] of the grid cells.
             The shape of `kv` is `(nl, nr)`, but it is broadcast if dimensions are missing.
        cv : array_like, default: None
             Two-dimensional array with vertical resistances [T] between the grid cells.
             The shape of `cv` is `(nl, nr - 1)`, but it is broadcast if dimensions are missing.

        Returns
        -------
        par : VerticalFlowParameters object

        Notes
        -----
        If input parameter `cv` is not given, then conductances are colculated using input parameters `kv`, and
        if input parameter `kv` is not given, then conductances are colculated using input parameters `cv`.
        If both input parameters `kv` and `cv` are given, then conductances are calculated using `kv`, after which the
        calculated conductances are replaced by conductances calculated using elements of `cv` which are not `nan`.
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._vertical_flow_parameters = VerticalFlowParameters(self.grid, kv, cv)
            return self._vertical_flow_parameters

    @property
    def kv(self):
        """
        Vertical conductivities.

        Returns
        -------
        kv : ndarray
           Two-dimensional array with vertical conductivity [L/T] in each cell of the grid.
           The shape of `kv` is `(nl, nr)`.
        """
        if self.nl == 1:
            return None
        elif self._vertical_flow_parameters is not None:
            return self._vertical_flow_parameters.k  # (nl, nr)
        else:
            return self.previous.kv  # (nl, nr)

    @property
    def cv(self):
        """
        Vertical resistances.

        Returns
        -------
        cv : ndarray
           Two-dimensional array with vertical resistances [T] between the grid cells.
           The shape of `cv` is `(nl - 1, nr)`, which means the infinitely large resistances of the model boundaries are
           not included.
        """
        if self.nl == 1:
            return None
        elif self._vertical_flow_parameters is not None:
            return self._vertical_flow_parameters.c  # (nl-1, nr)
        else:
            return self.previous.cv  # (nl-1, nr)

    @property
    def qvc(self):
        """
        Vertical conductances.

        Returns
        -------
        qvc : ndarray
            Two-dimensional array with vertical conductances [L²/T] between the grid cells.
            The shape of `qvc` is `(nl + 1, nr)`, which means the zero conductances of the model boundaries are included.
        """
        if self.nl == 1:
            return np.zeros((2, self.nr))
        elif self._vertical_flow_parameters is not None:
            return self._vertical_flow_parameters.qc  # (nl+1, nr)
        else:
            return self.previous.qvc  # (nl+1, nr)

    def add_ss(self, ss, sy=None):
        """
        Add storage parameters.

        Parameters
        ----------
        ss : array_like
            Two-dimensional array with specific storage [1/L] of the grid cells.
            The shape of `ss` is `(nl, nr)`, but it is broadcast if dimensions are missing.
        sy : array_like
            One-dimensional array with specific yield [-] of the grid cells in the top layer.
            The length of `sy` is `nr`, but it is broadcast if only one value is given.

        Returns
        -------
        par : StorageParameters object
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._storage_parameters = StorageParameters(self.grid, ss, sy)
        return self._storage_parameters

    @property
    def ss(self):
        """
        Specific elastic storages.

        Returns
        -------
        ss : ndarray
           Two-dimensional array with specific elastic storage [1/L] in each cell of the grid.
           The shape of `ss` is `(nl, nr)`.
        """
        if self.steady:
            return None
        elif self._storage_parameters is not None:
            return self._storage_parameters.ss  # (nl, nr)
        else:
            return self.previous.ss  # (nl, nr)

    @property
    def sy(self):
        """
        Specific yields in the top layer.

        Returns
        -------
        sy : ndarray
           One-dimensional array with specific yield [-] of the grid cells in the top layer.
           The length of `sy` is `nr`.
        """
        if self.steady:
            return None
        elif self._storage_parameters is not None:
            return self._storage_parameters.sy  # (nr, )
        else:
            return self.previous.sy  # (nr, )

    @property
    def qsc(self):
        """
        Constant terms to calculate storage change in each cell.

        Returns
        -------
        qsc : ndarray
            Two-dimensional array with constant terms to calculate storage change in each cell.
            In case of `ss`, the constant term of a cell is calculated as the volume of the cell multiplied by `ss`.
            If `sy` is given, then the horizontal surface of the cell multiplied by `sy` is added to the constant term.
            The shape of `qsc` is `(nl, nr)`.
        """
        if self.steady:
            return None
        elif self._storage_parameters is not None:
            return self._storage_parameters.qc  # (nl, nr)
        else:
            return self.previous.qsc  # (nl, nr)

    def add_q(self, q):
        """
        Add discharges.

        Parameters
        ----------
        q : array_like
          Two-dimensional array with discharge [L³/T or L³/T /L] in each grid cell.
          The shape of `q` is `(nl, nr)`, but it is broadcast if dimensions are missing.

        Returns
        -------
        cond : Discharges object
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._discharges = Discharges(self.grid, q)
            return self._discharges


    @property
    def q(self):
        """
        Discharges.

        Returns
        -------
        q : ndarray
          Two-dimensional array with discharge [L³/T or L³/T /L] in each grid cell.
          The shape of `q` is `(nl, nr)`.
        """
        if self._discharges is not None:
            return self._discharges.q  # (nl, nr)
        elif self.previous is not None:
            return self.previous.q  # (nl, nr)
        else:
            return np.zeros((self.nl, self.nr))  # (nl, nr)

    def add_h0(self, h0, add=False):
        """
        Add initial heads.

        Parameters
        ----------
        h0 : array_like
           Two-dimensional array with initial head [L] in each grid cell.
           The shape of `h0` is `(nl, nr)`, but it is broadcast if dimensions are missing.
        add : bool, default: False
            Indicates whether the given initial heads should be added to the heads simulated for the last time step of
            the previous stress period or not.

        Returns
        -------
        cond : InitialHeads object

        Notes
        -----
        The defined initial head in a cell is taken into account only if the cell is defined as having a variable head.
        This is indicated by `grid.variable`. By default, the initial head in a variable-head cell is zero.
        This means defining initial heads using this class is required only if some of these heads are nonzero.
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._initial_heads = InitialHeads(self.grid, h0, add)
            return self._initial_heads

    @property
    def h0(self):
        """
        Initial heads.

        Returns
        -------
        h0 : ndarray
           Two-dimensional array with initial head [L] in each grid cell.
           The shape of `h0` is `(nl, nr)`.
        """
        if self.previous is not None:
            if self._initial_heads is None:
                return self.previous.h if self.previous.steady else self.previous.h[:, :, -1]  # (nl, nr)
            elif self._initial_heads.add:
                return self._initial_heads.h + (self.previous.h if self.previous.steady else self.previous.h[:, :, -1])  # (nl, nr)
            else:
                return self._initial_heads.h  # (nl, nr)
        elif self._initial_heads is not None:
            return self._initial_heads.h  # (nl, nr)
        else:
            return np.zeros((self.nl, self.nr))  # (nl, nr)

    def add_hc(self, hc):
        """
        Add constant heads.

        Parameters
        ----------
        hc : array_like
           Two-dimensional array with constant head [L] in each grid cell.
           The shape of `h0` is `(nl, nr)`, but it is broadcast if dimensions are missing.

        Returns
        -------
        cond : ConstantHeads object

        Notes
        -----
        The defined constant head in a cell is taken into account only if the cell is defined as having a constant head.
        This is indicated by `grid.constant`. By default, the head in a constant-head cell is zero.
        This means defining constant heads using this class is required only if some of these heads are nonzero.
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._constant_heads = ConstantHeads(self.grid, hc)
            return self._constant_heads

    @property
    def hc(self):
        """
        Constant heads.

        Returns
        -------
        hc : ndarray
           Two-dimensional array with constant head [L] in each grid cell.
           The shape of `hc` is `(nl, nr)`.
        """
        if self._constant_heads is not None:
            return self._constant_heads.h  # (nl, nr)
        elif self.previous is not None:
            return self.previous.hc  # (nl, nr)
        else:
            return np.zeros((self.nl, self.nr))  # (nl, nr)

    def add_hdep(self, dependent, cdep, hdep=None):
        """
        Add head-dependent flux boundary conditions.

        Parameters
        ----------
        dependent : array_like
                  Two-dimensional boolean array that indicates which cells have a head-dependent flux boundary condition.
                  The shape of `dependent` is `(nl, nr)`, but it is broadcast if dimensions are missing.
        cdep : array_like
             Two-dimensional array with the vertical resistances [T].
             The shape of `c` is `(nl, nr)`, but it is broadcast if dimensions are missing.
        hdep : array_like, default: None
             Two-dimensional array with the constant heads [L].
             The shape of `h` is `(nl, nr)`, but it is broadcast if dimensions are missing.
             By default, array `h` contains zeros only.

        Returns
        -------
        cond : HeadDependentFluxes object
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._head_dependent_fluxes = HeadDependentFluxes(self.grid, dependent, cdep, hdep)
            return self._head_dependent_fluxes

    @property
    def dependent(self):
        """
        Indicates which cells have a head-dependent flux boundary condition.

        Returns
        -------
        dependent : ndarray
                  Two-dimensional boolean array that indicates which cells have a head-dependent flux boundary condition.
                  The shape of `dependent` is `(nl, nr)`.
        """
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.dependent  # (nl, nr)
        elif self.previous is not None:
            return self.previous.dependent  # (nl, nr)
        else:
            return None

    @property
    def cdep(self):
        """
        Vertical resistances of the head-dependent flux boundary conditions.

        Returns
        -------
        cdep : ndarray
             Two-dimensional array with the vertical resistances [T].
             The shape of `c` is `(nl, nr)`.
        """
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.c  # (nl, nr)
        elif self.previous is not None:
            return self.previous.cdep  # (nl, nr)
        else:
            return None

    @property
    def hdep(self):
        """
        Constant heads of the head-dependent flux boundary conditions.

        Returns
        -------
        hdep : ndarray
             Two-dimensional array with the constant heads [L].
             The shape of `h` is `(nl, nr)`.
        """
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.h  # (nl, nr)
        elif self.previous is not None:
            return self.previous.hdep  # (nl, nr)
        else:
            return None

    @property
    def qdc(self):
        """
        Vertical conductances of the head-dependent flux boundary conditions.

        Returns
        -------
        qdc : tuple
            First element is `idx`, second is `qdc`.
        idx : ndarray
            One-dimensional array with the linear indices of the cells in which a head-dependent flux boundary condition
            is defined.
        qdc : ndarray
            Two-dimensional array with the vertical conductances [L²/T] of the head-dependent boundary conditions.
            As the flow is vertical, the conductance is defined as the horizontal surface `hs` of the cell divided by
            the vertical resistance `c`. The shape of `qdc` is `(nl, nr)`.

        # TODO: checken of die idx echt nodig is!
        """
        if self._head_dependent_fluxes is not None:
            return self._head_dependent_fluxes.idx, self._head_dependent_fluxes.qc  # tuple (idx, qdc)
        elif self.previous is not None:
            return self.previous.qdc  # tuple (idx, qdc)
        else:
            return None, None

    def _initialize(self):
        """
        Initialize the matrix system.

        Returns
        -------
        None
        """
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
        """
        Solve the matrix system of finite-difference equations.

        Returns
        -------
        None
        """
        self._initialize()
        if self.steady:
            self._h[self._is_variable, 0] = solve(self._A0, self._b0)
        else:  # transient
            h = self._h[self._is_variable, 0]
            for k in range(self.nt - 1):
                dS = self._dS0 / self.dt[k]
                b = self._b0 - dS * h
                A = self._A0.copy()
                A[self._idx] -= dS
                h = solve(A, b)
                self._h[self._is_variable, k + 1] = h
        for first, remaining in self._connected_ids:
            self._h[remaining, :] = self._h[first, :]
        self._h = np.reshape(self._h, (self.nl, self.nr, self.nt))


class Model(SolveHeads):
    """
    Base model for classes implementing steady and transient state 2D finite-difference groundwater flow models.

    Parameters
    ----------
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Attributes
    ----------
    grid : Grid object
         Two-dimensional axi-symmetric or a rectilinear finite-difference grid.
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

    Notes
    -----
    During instantiation, subclasses assign the appropriate `Grid` class to protected attribute `_grid_constructor`,
    and the appropriate `StressPeriod` class to protected attribute `_period_constructor`.
    """

    def __init__(self, no_warnings=True):
        SolveHeads.__init__(self, None, [], no_warnings)
        self.periods = []
        self._grid_constructor = None  # SUBCLASSING: assign Grid class!
        self._period_constructor = None  # SUBCLASSING: assign StressPeriod class!

    def add_grid(self, rb, D, constant=None, inactive=None, connected=None):
        """
        Define the two-dimensional axi-symmetric or rectilinear finite-difference grid.

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

        Returns
        -------
        grid : Grid object
             Two-dimensional axi-symmetric or rectilinear finite-difference grid.
        """
        self.grid = self._grid_constructor(rb=rb, D=D, constant=constant, inactive=inactive, connected=connected)
        return self.grid

    def add_period(self, t=None):
        """
        Define and add new stress period.

        A stress period is a period during which boundary conditions do not change with time.

        Parameters
        ----------
        t : array_like, default: None
          One-dimensional array with simulation times [T]. Length of `t` is `nt`.
          If `t` is `None`, then the stress period is steady state.
          If the first element of `t` is not zero, 0.0 is inserted in the front. This first element represents time
          t = 0 for which the initial conditions are defined.

        Returns
        -------
        period : StressPeriod object
               Stress period with time steps.
        """
        timesteps = None if t is None else TimeSteps(t)
        self.timesteps.append(timesteps)
        previous = None if len(self.periods) == 0 else self.periods[-1]
        period = self._period_constructor(grid=self.grid, timesteps=timesteps, previous=previous,
                                          no_warnings=self.no_warnings)
        self.periods.append(period)
        return period

    @property
    def steady(self):
        """
        Indicates whether the stress periods are steady or transient state.

        Returns
        -------
        steady : ndarray
               One-dimensional boolean array indicating whether the stress periods are steady or transient state.
               The length of `steady` is equal to the number of stress periods.
        """
        return np.array([period.steady for period in self.periods])

    @property
    def nt(self):
        """
        Number of simulation times.

        Returns
        -------
        nt : int
           Total number of simulation times.
        """
        return sum([period.nt for period in self.periods if not period.steady])

    @property
    def t(self):
        """
        Simulation times.

        Returns
        -------
        t : ndarray
          One-dimensional array with simulation times [T].
        """
        t = self.periods[0].t
        for period in self.periods[1:]:
            t = np.hstack((t, t[-1] + period.t))
        return t

    @property
    def dt(self):
        """
        Time steps.

        Returns
        -------
        dt : ndarray
           One-dimensional array with duration [T] of time steps.
        """
        t = self.t
        return t[1:] - t[:-1]

    def solve(self):
        """
        Solve the matrix system of finite-difference equations.

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            for period in self.periods:
                period.solve()

    @property
    def h(self):
        """
        Hydraulic heads.

        Returns
        -------
        h : ndarray
          Three-dimensional array with simulated head [L] in each grid cell and for each time.
          The shape of `h` is `(nl, nr, nt)`. Note that `nt` equals 1 in case of steady state simulation.
        """
        return np.concatenate([period.h for period in self.periods], axis=2)

