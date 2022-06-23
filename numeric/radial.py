from abc import ABC, abstractmethod
import numpy as np
import warnings
from scipy.linalg import LinAlgWarning


class Object:

    def _broadcast(self, arr, nrow, ncol, dtype=float):
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
        return np.zeros((nrow, ncol), dtype=dtype)

    def _ones(self, nrow, ncol, dtype=float):
        return np.ones((nrow, ncol), dtype=dtype)


class Grid(Object):

    def __init__(self, rb, D, constant=None, inactive=None, connected=None):
        # rb is (nr+1, ) floating point array (required)
        # D is (nl, ) floating point array (required)
        # constant is (nl, nr) boolean array (optional)
        # inactive is (nl, nr) boolean array (optional)
        # connected is (nl, nr) integer array (optional)
        Object.__init__(self)
        self.rb = np.array(rb)  # (nr+1, )
        self.r = np.sqrt(self.rb[:-1] * self.rb[1:])  # (nr, )
        rb2 = self.rb ** 2  # (nr+1, )
        self.hs = np.pi * (rb2[1:] - rb2[:-1])  # (nr, )
        self.D = np.array(D)  # (nl, )
        if self.D.ndim == 0:  # D is scalar
            self.D = self.D[np.newaxis]
        self.vol = np.outer(self.D, self.hs)  # (nl, nr)
        self.nr = len(self.r)
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

    def _is_connected(self):
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

    def __init__(self, grid):
        Object.__init__(self)
        self.grid = grid

    @property
    def nr(self):
        return self.grid.nr

    @property
    def nl(self):
        return self.grid.nl

    @property
    def n(self):
        return self.grid.n  # n = nl x nr

    @property
    def constant(self):
        return self.grid.constant  # (nl, nr) boolean

    @property
    def inactive(self):
        return self.grid.inactive  # (nl, nr) boolean

    @property
    def connected(self):
        return self.grid.connected  # (nl, nr) boolean

    @property
    def variable(self):
        return self.grid.variable  # (nl, nr) boolean

    @property
    def rb(self):
        return self.grid.rb  # (nr+1, )

    @property
    def r(self):
        return self.grid.r  # (nr, )

    @property
    def hs(self):
        return self.grid.hs  # (nr, )

    @property
    def D(self):
        return self.grid.D  # (nl, )

    @property
    def vol(self):
        return self.grid.vol  # (nl, nr)

    @property
    def _connected_ids(self):
        return self.grid._connected_ids

    def _broadcast(self, arr, nrow=None, ncol=None, dtype=float):
        if nrow is None:
            nrow = self.nl
        if ncol is None:
            ncol = self.nr
        return Object._broadcast(self, arr, nrow, ncol, dtype)

    def _zeros(self, nrow=None, ncol=None, dtype=float):
        if nrow is None:
            nrow = self.nl
        if ncol is None:
            ncol = self.nr
        return Object._zeros(self, nrow, ncol, dtype)

    def _ones(self, nrow=None, ncol=None, dtype=float):
        if nrow is None:
            nrow = self.nl
        if ncol is None:
            ncol = self.nr
        return Object._ones(self, nrow, ncol, dtype)


class TimeSteps:

    def __init__(self, t):
        # t is (nt, ) floating point array (required)
        # t0 = 0 is not given
        self.t = np.array(t, dtype=float)  # (nt-1, )
        if t[0] > 0.0:  # add t0 = 0 if t doesn't contain it
            self.t = np.insert(self.t, 0, 0)  # (nt, )
        self.dt = self.t[1:] - self.t[:-1]  # (nt-1, )
        self.nt = len(self.t)


class TimeDependent:

    def __init__(self, timesteps=None):
        self.timesteps = timesteps

    @property
    def steady(self):
        return self.timesteps is None

    @property
    def nt(self):
        if self.steady:
            return 1
        else:
            return self.timesteps.nt

    @property
    def t(self):
        if self.steady:
            return 0
        else:
            return self.timesteps.t  # (nt, )

    @property
    def dt(self):
        if self.steady:
            return None
        else:
            return self.timesteps.dt  # (nt-1, )


class HydraulicParameters(GridDependent, ABC):

    def __init__(self, grid):
        GridDependent.__init__(self, grid)

    @abstractmethod
    def _set_qc(self):
        pass


class FlowParameters(HydraulicParameters):

    def __init__(self, grid, k=None, c=None):
        HydraulicParameters.__init__(self, grid)
        self.k = self._broadcast(k)
        self._check_c(c)
        self._set_qc()

    @abstractmethod
    def _check_c(self, c):
        pass


class HorizontalFlowParameters(FlowParameters):

    def __init__(self, grid, k=None, c=None):
        FlowParameters.__init__(self, grid, k, c)

    def _check_c(self, c):
        self.c = self._broadcast(c, ncol=self.nr-1)

    def _set_qc(self):
        self.qc = self._zeros(ncol=self.nr + 1)  # (nl, nr+1)
        if self.k is None:
            qc = 2 * np.pi * np.outer(self.D, self.rb[1:-1]) / self.c  # (nl, nr-1)
        else:
            rbc = np.log(self.rb[1:] / self.rb[:-1]) / self.k / 2  # (nl, nr)
            qc = 2 * np.pi * self.D[:, np.newaxis] / (rbc[:, :-1] + rbc[:, 1:])  # (nl, nr-1)
            if self.c is not None:
                irow, icol = (~np.isnan(self.c)).nonzero()
                qc[irow, icol] = 2 * np.pi * self.D[irow] * self.rb[icol + 1] / self.c[irow, icol]
        b = self.inactive[:, :-1] | self.inactive[:, 1:]  # (nl, nr-1)
        qc[b] = 0.0
        self.qc[:, 1:-1] = qc


class VerticalFlowParameters(FlowParameters):

    def __init__(self, grid, k=None, c=None):
        FlowParameters.__init__(self, grid, k, c)

    def _check_c(self, c):
        self.c = self._broadcast(c, nrow=self.nl - 1)

    def _set_qc(self):
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

    def __init__(self, grid, ss):
        HydraulicParameters.__init__(self, grid)
        self.ss = self._broadcast(ss)  # (nl, nr)
        self._set_qc()

    def _set_qc(self):
        self.qc = self._zeros()  # (nl, nr)
        b = self.variable  # (nl, nr)
        self.qc[b] = self.vol[b] * self.ss[b]


class Discharges(GridDependent):

    def __init__(self, grid, q):
        GridDependent.__init__(self, grid)
        self.q = self._broadcast(q)  # (nl, nr)


class ConstantHeads(GridDependent):

    def __init__(self, grid, h):
        GridDependent.__init__(self, grid)
        self.h = self._broadcast(h)  # (nl, nr)


class InitialHeads(ConstantHeads):

    def __init__(self, grid, h, add=False):
        ConstantHeads.__init__(self, grid, h)
        self.add = add


class HeadDependentFluxes(HydraulicParameters, ConstantHeads):

    def __init__(self, grid, dependent, c, h=None):
        HydraulicParameters.__init__(self, grid)
        ConstantHeads.__init__(self, grid, h)
        self.dependent = self._broadcast(dependent, dtype=bool)  # (nl, nr)
        self.c = self._broadcast(c)  # (nl, nr)
        if self.h is None:
            self.h = self._zeros()  # (nl, nr)
        self._set_qc()

    def _set_qc(self):
        irow, icol = self.dependent.nonzero()
        self.idx = irow * self.nr + icol
        self.qc = self.hs[icol] / self.c.flatten()[self.idx]


class SolveHeads(GridDependent, TimeDependent, ABC):

    def __init__(self, grid, timesteps=None):
        GridDependent.__init__(self, grid)
        TimeDependent.__init__(self, timesteps)
        self._h = None
        self.gpu = False

    @property
    def h(self):
        return self._h

    @abstractmethod
    def solve(self):
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

    def add_kh(self, kh=None, ch=None):
        self._horizontal_flow_parameters = HorizontalFlowParameters(self.grid, kh, ch)
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
        if self.gpu:
            from scipy.linalg import solve
        else:
            from scipy.linalg import solve
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

    def solve_new(self):
        # import cupy functions if gpu
        if self.gpu:
            from cupy import array
            from cupy.linalg import solve
        # import scipy functions if no gpu
        else:
            from scipy.linalg import solve
        # initialize h, A, and b
        self._initialize_h()
        A0, b0 = self._get_A0_b0()
        # steady state
        if self.steady:
            A0, b0 = array(A0), array(b0)
            h = solve(A0, b0)
            self._h[self._is_variable, 0] = h.get() if self.gpu else h
        # transient state
        else:
            # initialize dS and h
            dS0 = self._get_dS0()
            h = self._h[self._is_variable, 0]
            i = np.diag_indices(A0.shape[0])
            # convert to cupy arrays if gpu
            if self.gpu:
                A0, b0, dS0, h = array(A0), array(b0), array(dS0), array(h)
            # loop through time steps
            for k in range(self.nt - 1):
                dS = dS0 / self.dt[k]
                b = b0 - dS * h
                A = A0.copy()
                A[i] -= dS
                h = solve(A, b)
                self._h[self._is_variable, k + 1] = h.get() if self.gpu else h
        for first, remaining in self._connected_ids:
            self._h[remaining, :] = self._h[first, :]
        self._h = np.reshape(self._h, (self.nl, self.nr, self.nt))


class Model(SolveHeads):

    def __init__(self):
        SolveHeads.__init__(self, None, [])
        self.periods = []
        self.no_warnings = True

    def add_grid(self, rb, D, constant=None, inactive=None, connected=None):
        self.grid = Grid(rb=rb, D=D, constant=constant, inactive=inactive, connected=connected)
        return self.grid

    def add_period(self, t=None):
        timesteps = None if t is None else TimeSteps(t)
        self.timesteps.append(timesteps)
        previous = None if len(self.periods) == 0 else self.periods[-1]
        period = StressPeriod(grid=self.grid, timesteps=timesteps, previous=previous)
        period.gpu = self.gpu
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

    def solve_new(self):
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            for period in self.periods:
                period.solve_new()

    @property
    def h(self):
        return np.concatenate([period.h for period in self.periods], axis=2)


class TransientNumerical:

    def __init__(self, rb, t, T, S, c=None, constant=None, h_const=None, Q=None, inactive=None,
                 connected=None, dependent=None, h_dep=None, c_dep=None):
        # rb is (nr+1, ) floating point array (required)
        # t is (nt, ) floating point array (required)
        # T is (nl, nr) floating point array (required)
        # S is (nl, nr) floating point array (required)
        # c is (nl-1, nr) floating point array (required if nl > 1)
        # constant is (nl, nr) boolean array (optional)
        # h_const is (nl, nr) floating point array (optional)
        # Q is (nl, nr) floating point array (optional)
        # inactive is (nl, nr) boolean array (optional)
        # connected is (nl, nr) integer array (optional)
        # dependent is (nl, nr) boolean array (optional)
        # h_dep is (nl, nr) floating point array (optional)
        # c_dep is (nl, nr) floating point array (optional)
        self.rb = rb
        self.r = np.sqrt(self.rb[:-1] * self.rb[1:])
        self.nr = len(self.r)
        self.t = np.insert(t, 0, 0)
        self.dt = self.t[1:] - self.t[:-1]
        self.nt = len(self.t)
        self.T = T
        self.nl = self.T.shape[0]
        self.S = S
        if c is None:
            self.c = np.zeros((0, self.nr))
        else:
            self.c = c
        if constant is None:
            self.constant = np.zeros((self.nl, self.nr), dtype=bool)
        else:
            self.constant = constant.astype(bool)
        if h_const is None:
            self.h_const = np.zeros((self.nl, self.nr))
        else:
            self.h_const = h_const
        if Q is None:
            self.Q = np.zeros((self.nl, self.nr))
        else:
            self.Q = Q
        if inactive is None:
            self.inactive = np.zeros((self.nl, self.nr), dtype=bool)
        else:
            self.inactive = inactive.astype(bool)
        if connected is None:
            self.connected = np.zeros((self.nl, self.nr), dtype=int)
        else:
            self.connected = connected.astype(int)
        if dependent is None:
            self.dependent = np.zeros((self.nl, self.nr), dtype=bool)
            self.c_dep = None
            self.h_dep = None
        else:
            self.dependent = dependent.astype(bool)
            self.c_dep = c_dep
            if h_dep is None:
                self.h_dep = np.zeros((self.nl, self.nr))
            else:
                self.h_dep = h_dep
        self.gpu = False
        self.no_warnings = True

    def _initialize(self):
        nr = self.nr
        self._n = nr * self.nl
        self._idx = np.diag_indices(self._n)
        irow, icol = self._idx
        rb2 = self.rb ** 2  # (nr+1, )
        self._hs = np.pi * (rb2[1:] - rb2[:-1])  # (nr, )
        self._b0 = -self.Q.flatten()  # (n, )
        self._A0 = np.zeros((self._n, self._n))  # (n, n)
        self._dS0 = (self._hs * self.S).flatten()  # (n, )
        qrc, qzc = self._qrc(), self._qzc()
        self._A0[irow[:-1] + 1, icol[:-1]] = self._A0[irow[:-1], icol[:-1] + 1] = qrc[:, 1:].flatten()[:-1]
        self._A0[irow[:-nr] + nr, icol[:-nr]] = self._A0[irow[:-nr], icol[:-nr] + nr] = qzc[1:, :].flatten()[:-nr]
        self._A0[irow, icol] = -(qrc[:, :-1] + qrc[:, 1:] + qzc[:-1, :] + qzc[1:, :]).flatten()
        is_inactive = self.inactive.flatten()
        is_constant = self._check_constant_heads()
        is_connected = self._check_connected()
        self._check_head_dependent()
        self._is_variable = ~(is_constant | is_inactive | is_connected)
        self._b0 = self._b0[self._is_variable]
        self._A0 = self._A0[self._is_variable, :][:, self._is_variable]
        self._dS0 = self._dS0[self._is_variable]
        self._h0 = self.h[self._is_variable]
        self.h = np.repeat(self.h[:, np.newaxis], self.nt, axis=1)
        self.h[is_inactive, :] = np.nan

    def _qrc(self):
        X = np.log(self.rb[1:] / self.rb[:-1]) / self.T / 2  # (nl, nr)
        qrc = 2 * np.pi / (X[:, :-1] + X[:, 1:])  # (nl, nr-1)
        b = self.inactive[:, :-1] | self.inactive[:, 1:]  # (nl, nr-1)
        qrc[b] = 0.0
        qrc_full = np.zeros((self.nl, self.nr + 1))  # (nl, nr+1)
        qrc_full[:, 1:-1] = qrc
        return qrc_full

    def _qzc(self):
        c = self.c.copy()  # (nl-1, nr)
        b = self.inactive[:-1, :] | self.inactive[1:, :]  # (nl-1, nr)
        c[b] = np.inf
        c_full = np.inf * np.ones((self.nl + 1, self.nr))  # (nl+1, nr)
        c_full[1:-1, :] = c
        return self._hs / c_full  # (nl+1, nr)

    def _check_constant_heads(self):
        is_constant = self.constant.flatten()
        self.h = self.h_const.flatten()
        self._b0 -= np.dot(self._A0[:, is_constant], self.h[is_constant])
        return is_constant

    def _check_connected(self):
        is_connected = self.connected.astype(bool)
        self._ic = []
        for i in range(1, np.max(self.connected) + 1):
            row, col = (self.connected == i).nonzero()
            row0, col0 = row[0], col[0]
            row, col = row[1:], col[1:]
            first, remaining = row0 * self.nr + col0, row * self.nr + col
            is_connected[row0, col0] = False
            self._ic.append((first, remaining))
            self._b0[first] += self._b0[remaining].sum()
            self._dS0[first] += self._dS0[remaining].sum()
            self._A0[first, :] += self._A0[remaining, :].sum(axis=0)
            self._A0[:, first] += self._A0[:, remaining].sum(axis=1)
        return is_connected.flatten()

    def _check_head_dependent(self):
        if np.any(self.dependent):
            irow, icol = self.dependent.nonzero()
            i = irow * self.nr + icol
            qc = self._hs[icol] / self.c_dep.flatten()[i]
            self._b0[i] -= qc * self.h_dep.flatten()[i]
            i = irow * self.nr + icol
            self._A0[i, i] -= qc

    def _solve(self):
        from scipy.linalg import solve
        h = self._h0
        idx = np.diag_indices(len(self._h0))
        for k in range(self.nt - 1):
            self._dS = self._dS0 / self.dt[k]
            self._b = self._b0 - self._dS * h
            self._A = self._A0.copy()
            self._A[idx] -= self._dS
            h = solve(self._A, self._b)
            self.h[self._is_variable, k + 1] = h
        for first, remaining in self._ic:
            self.h[remaining, :] = self.h[first, :]

    def solve(self):
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._initialize()
            self._solve()
        self.h = np.reshape(self.h, (self.nl, self.nr, self.nt))


class SteadyNumerical:

    def __init__(self, rb, T, c=None, constant=None, h_const=None, Q=None, inactive=None, connected=None,
                 dependent=None, h_dep=None, c_dep=None):
        # rb is (nr+1, ) floating point array
        # T is (nl, nr) floating point array
        # c is (nl-1, nr) floating point array
        # constant is (nl, nr) boolean array
        # h_const is (nl, nr) floating point array
        # Q is (nl, nr) floating point array
        # inactive is (nl, nr) boolean array
        # connected is (nl, nr) integer array
        # dependent is (nl, nr) boolean array
        # h_dep is (nl, nr) floating point array
        # c_dep is (nl, nr) floating point array
        self.rb = rb
        self.r = np.sqrt(self.rb[:-1] * self.rb[1:])
        self.nr = len(self.r)
        self.T = T
        self.nl = self.T.shape[0]
        if c is None:
            self.c = np.array([[]])
        else:
            self.c = c
        if constant is None:
            self.constant = np.zeros((self.nl, self.nr), dtype=bool)
        else:
            self.constant = constant.astype(bool)
        if h_const is None:
            self.h_const = np.zeros((self.nl, self.nr))
        else:
            self.h_const = h_const
        if Q is None:
            self.Q = np.zeros((self.nl, self.nr))
        else:
            self.Q = Q
        if inactive is None:
            self.inactive = np.zeros((self.nl, self.nr), dtype=bool)
        else:
            self.inactive = inactive.astype(bool)
        if connected is None:
            self.connected = np.zeros((self.nl, self.nr), dtype=int)
        else:
            self.connected = connected.astype(int)
        if dependent is None:
            self.dependent = np.zeros((self.nl, self.nr), dtype=bool)
            self.c_dep = None
            self.h_dep = None
        else:
            self.dependent = dependent.astype(bool)
            self.c_dep = c_dep
            if h_dep is None:
                self.h_dep = np.zeros((self.nl, self.nr))
            else:
                self.h_dep = h_dep
        self.gpu = False
        self.no_warnings = True

    def _initialize(self):
        nr = self.nr
        self._n = nr * self.nl
        self._idx = np.diag_indices(self._n)
        irow, icol = self._idx
        self._b0 = -self.Q.flatten()
        self._A0 = np.zeros((self._n, self._n))
        qrc, qzc = self._qrc(), self._qzc()
        self._A0[irow[:-1] + 1, icol[:-1]] = self._A0[irow[:-1], icol[:-1] + 1] = qrc[:, 1:].flatten()[:-1]
        self._A0[irow[:-nr] + nr, icol[:-nr]] = self._A0[irow[:-nr], icol[:-nr] + nr] = qzc[1:, :].flatten()[:-nr]
        self._A0[irow, icol] = -(qrc[:, :-1] + qrc[:, 1:] + qzc[:-1, :] + qzc[1:, :]).flatten()
        is_inactive = self.inactive.flatten()
        is_constant = self._check_constant_heads()
        is_connected = self._check_connected()
        self._check_head_dependent()
        self._is_variable = ~(is_constant | is_inactive | is_connected)
        self._b0 = self._b0[self._is_variable]
        self._A0 = self._A0[self._is_variable, :][:, self._is_variable]

    def _qrc(self):
        X = np.log(self.rb[1:] / self.rb[:-1]) / self.T / 2  # (nl, nr)
        qrc = 2 * np.pi / (X[:, :-1] + X[:, 1:])  # (nl, nr-1)
        b = self.inactive[:, :-1] | self.inactive[:, 1:]  # (nl, nr-1)
        qrc[b] = 0.0
        qrc_full = np.zeros((self.nl, self.nr + 1))  # (nl, nr+1)
        qrc_full[:, 1:-1] = qrc
        return qrc_full

    def _qzc(self):
        rb2 = self.rb ** 2  # (nr+1, )
        self._hs = np.pi * (rb2[1:] - rb2[:-1])  # (nr, )
        c = self.c.copy()  # (nl-1, nr)
        b = self.inactive[:-1, :] | self.inactive[1:, :]  # (nl-1, nr)
        c[b] = np.inf
        c_full = np.inf * np.ones((self.nl + 1, self.nr))  # (nl+1, nr)
        c_full[1:-1, :] = c
        return self._hs / c_full  # (nl+1, nr)

    def _check_constant_heads(self):
        is_constant = self.constant.flatten()
        self.h = self.h_const.flatten()
        self._b0 -= np.dot(self._A0[:, is_constant], self.h[is_constant])
        return is_constant

    def _check_connected(self):
        is_connected = self.connected.astype(bool)
        self._ic = []
        for i in range(1, np.max(self.connected) + 1):
            row, col = (self.connected == i).nonzero()
            row0, col0 = row[0], col[0]
            row, col = row[1:], col[1:]
            first, remaining = row0 * self.nr + col0, row * self.nr + col
            is_connected[row0, col0] = False
            self._ic.append((first, remaining))
            self._b0[first] += self._b0[remaining].sum()
            self._A0[first, :] += self._A0[remaining, :].sum(axis=0)
            self._A0[:, first] += self._A0[:, remaining].sum(axis=1)
        return is_connected.flatten()

    def _check_head_dependent(self):
        if np.any(self.dependent):
            irow, icol = self.dependent.nonzero()
            i = irow * self.nr + icol
            qc = self._hs[icol] / self.c_dep.flatten()[i]
            self._b0[i] -= qc * self.h_dep.flatten()[i]
            i = irow * self.nr + icol
            self._A0[i, i] -= qc

    def _solve(self):
        if self.gpu:
            from cupy import array, zeros, dot
            from cupy.linalg import solve
            self._b0 = array(self._b0)
            self._A0 = array(self._A0)
        else:
            from scipy.linalg import solve
            from numpy import zeros, dot
        h = solve(self._A0, self._b0)
        if self.gpu:
            h = h.get()
        self.h[self._is_variable] = h
        self.h[self.inactive.flatten()] = np.nan
        for first, remaining in self._ic:
            self.h[remaining] = self.h[first]

    def solve(self):
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            self._initialize()
            self._solve()
        self.h = np.reshape(self.h, (self.nl, self.nr))


