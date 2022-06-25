"""
Module containing classes to build steady and transient state multi-layer groundwater flow models
applying the principle of superposition.
"""
from abc import ABC, abstractmethod
import numpy as np
from .radial import SteadyQ, TransientQ


class VariableQH:
    """
    Base class for classes that apply the principle of superposition to implement
    transient state 2D groundwater flow models with time-dependent stresses at the inner boundary.

    Parameters
    ----------
    model : _base.Transient object
          2D transient state model with constant inner boundary condition.
    t : array_like
      Times [L] at which the stresses at the inner boundary change.
    QH : array_like
       Time dependent stresses at the inner boundary corresponding to times `t`.
       The shape of `QH` is `(nl, nt)` with `nl` the number of layers and `nt` the length of `t`.
    QHstr : str
          String ``'Q'`` if variable discharge or string ``'H'`` if variable head.

    Methods
    -------
    h(r, t) :
            Calculate hydraulic head h at given distances r and given times t.
    """

    def __init__(self, model, t, QH, QHstr):
        self.model = model
        self.t = t
        self._QH = QH
        self._QHstr = QHstr
        self._t_to_array()
        self._QH_to_array()

    def _t_to_array(self):
        """
        Convert input argument t into 1D numpy array.

        Returns
        -------
        None
        """
        self.t = np.array(self.t, dtype=float)
        if self.t.ndim == 0:
            self.t = self.t[np.newaxis]

    def _QH_to_array(self):
        """
        Convert input array QH into 2D numpy array.

        Returns
        -------
        None
        """
        self._QH = np.array(self._QH)
        if self._QH.ndim == 0:
            self._QH = self._QH[np.newaxis, np.newaxis]
        elif self._QH.ndim == 1:
            if self.model.nl == 1:
                self._QH = self._QH[np.newaxis, :]
            else:
                self._QH = self._QH[:, np.newaxis]

    def h(self, r, t):
        """
        Calculate hydraulic head h in each layer at given distances r and times t.

        Parameters
        ----------
        r : array_like
          Radial or horizontal distances [L].
        t : array_like
          Times [T].

        Returns
        -------
        h : ndarray
          Hydraulic heads [L] at given distances `r` and times `t`.
          The shape of `h` is  `(nl, nr, nt)`, where `nl` is the number of layers, `nr` the length of `r`,
          and `nt` the length of `t`.
        """
        t = np.array(t)
        if t.ndim == 0:
            t = t[np.newaxis]
        h = self.model.h(r, t)
        QH0 = QH = getattr(self.model, self._QHstr)
        for i in range(len(self.t)):
            setattr(self.model, self._QHstr, self._QH[:, i] - QH)
            QH = self._QH[:, i]
            b = t > self.t[i]
            dt = t[b] - self.t[i]
            h[:, :, b] += self.model.h(r, dt)
        setattr(self.model, self._QHstr, QH0)
        return h

    @property
    def nl(self):
        """
        Number of layers.

        Returns
        -------
        nl: int
          Number of layers.
        """
        return self.model.nl

    @property
    def T(self):
        """
        Layer transmissivities [L²/T].

        Returns
        -------
        T : ndarray
          Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
        """
        return self.model.T

    @property
    def S(self):
        """
        Layer storativities [-].

        Returns
        -------
        S : ndarray
          Layer storativities [-]. The length of `S` is equal to the number of layers.
        """
        return self.model.S

    @property
    def c(self):
        """
        Vertical resistances [T] between layers.

        Returns
        -------
        c : ndarray
          Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
        """
        return self.model.c

    @property
    def c_top(self):
        """
        Vertical resistance [T] of the lower boundary of the aquifer system.

        Returns
        -------
        c_top : float
              Vertical resistance [T] of the lower boundary of the aquifer system.
        """
        return self.model.c_top

    @property
    def c_bot(self):
        """
        Vertical resistance [T] of the upper boundary of the aquifer system.

        Returns
        -------
        c_bot : float
              Vertical resistance [T] of the upper boundary of the aquifer system.
        """
        return self.model.c_bot

    @property
    def r_in(self):
        """
        Radial or horizontal distance [L] of the inner model boundary.

        Returns
        -------
        r_in : float
             Radial or horizontal distance [L] of the inner model boundary.
        """
        return self.model.r_in

    @property
    def r_out(self):
        """
        Radial or horizontal distance [L] of the outer model boundary.

        Returns
        -------
        r_out : float
              Radial or horizontal distance [L] of the outer model boundary.
        """
        return self.model.r_out


class VariableQ(VariableQH):
    """
    Class to build transient state 2D groundwater flow models with time-dependent discharges at the inner boundary
    applying the superposition principle.

    Parameters
    ----------
    model : radial.TransientQ or parallel.TransientQ object
          2D transient state model with constant discharge at the inner boundary.
    t : array_like
      Times [L] at which the discharges at the inner boundary change.
    Q : array_like
      Time dependent discharges [L³/T or L³/T/L] at the inner boundary corresponding to times `t`.
      The shape of `Q` is `(nl, nt)` with `nl` the number of layers and `nt` the length of `t`.

    Methods
    -------
    h(r, t) :
            Calculate hydraulic head h at given distances r and given times t.
    """

    def __init__(self, model, t, Q):
        super().__init__(model, t, Q, 'Q')

    @property
    def Q(self):
        """
        Get time dependent discharges at the inner model boundary.

        Returns
        -------
        Q : ndarray
          Layer discharges [L³/T or L³/T/L] at the inner model boundary.
          Shape of `Q` is `(nl, nt + 1)` with `nl` the number of layers and `nt` the number of changes in time.
        """
        return np.hstack((self.model.Q[:, np.newaxis], self._QH))

    @Q.setter
    def Q(self, Q):
        """
        Set time dependent discharges at the inner model boundary.

        Parameters
        ----------
        Q : array_like
          Layer discharges [L³/T or L³/T/L] at the inner model boundary.
          Shape of `Q` is `(nl, nt + 1)` with `nl` the number of layers and `nt` the number of changes in time.

        Returns
        -------
        None
        """
        self._QH = Q
        self._QH_to_array()


class VariableH(VariableQH):
    """
    Class to build transient state 2D groundwater flow models with time-dependent heads at the inner boundary
    applying the superposition principle.

    Parameters
    ----------
    model : radial.TransientH or parallel.TransientH object
          2D transient state model with constant head at the inner boundary.
    t : array_like
      Times [L] at which the heads at the inner boundary change.
    h_in : array_like
         Time dependent specified heads [L] at the inner boundary corresponding to times `t`.
         The shape of `h_in` is `(nl, nt)` with `nl` the number of layers and `nt` the length of `t`.

    Methods
    -------
    h(r, t) :
            Calculate hydraulic head h at given distances r and given times t.
    """

    def __init__(self, model, t, h_in):
        super().__init__(model, t, h_in, 'h_in')

    @property
    def h_in(self):
        """
        Get time dependent heads at the inner model boundary.

        Returns
        -------
        h_in : ndarray
             Specified heads [L] at the inner model boundary.
             Shape of `h_in` is `(nl, nt + 1)` with `nl` the number of layers and `nt` the number of changes in time.
        """
        return np.hstack((self.model.h_in[:, np.newaxis], self._QH))

    @h_in.setter
    def h_in(self, h_in):
        """
        Set time dependent heads at the inner model boundary.

        Parameters
        ----------
        h_in : array_like
             Specified heads [L] at the inner model boundary.
             Shape of `h_in` is `(nl, nt + 1)` with `nl` the number of layers and `nt` the number of changes in time.

        Returns
        -------
        None
        """
        self._QH = h_in
        self._QH_to_array()


class Well:
    """
    Class to define pumping well.

    Parameters
    ----------
    x : float
      X coordinate [L] of well.
    y : float
      Y coordinate [L] of well.
    model : radial.SteadyQ, radial.TransientQ, or superposition.VariableQ object
          2D axi-symmetric model with constant discharge at the inner boundary.

    Methods
    -------
    h(x, y, t) :
               Calculate heads at given coordinates `x` and `y`, and at given times `t` if `model` is transient.
    """

    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self._model = model

    @property
    def rw(self):
        """
        Get the well radius, which corresponds to the radial distance of the inner model boundary.

        Returns
        -------
        rw : float
           Pumping well radius [L].
        """
        return self._model.r_in

    @property
    def Q(self):
        """
        Get the pumping rates of the well, which correspond to the radial discharges at the inner model boundary.

        Returns
        -------
        Q : ndarray
          Pumping rates [L³/T] of the well in each layer.
          Shape of `Q` is `(nl, )` if rates are constant, with `nl` the number of layers.
          Shape of `Q` is `(nl, nt + 1)` if rates are variable, with `nt` the number of changes in time.

        """
        return self._model.Q

    def h(self, x, y, *t):
        """
        Calculate heads at given coordinates x and y, and at given times t if model is transient.

        Parameters
        ----------
        x : ndarray
          X coordinates [L], two-dimensional array, output from numpy function `meshgrid`.
        y : ndarray
          Y coordinates [L], two-dimensional array, output from numpy function `meshgrid`.
        t : ndarray
          Times [T], one-dimensional array.

        Returns
        -------
        h : ndarray
          Calculated heads in each layer at coordinates `x` and `y`, and at times `t` if `model` is transient.
          Shape of `h` is `(nl, ny, nx)` if `model` is steady, and `(nl, ny, nx, nt)` if `model` is transient,
          with `nl` the number of layers, `(ny, nx)` the shape of `x` and `y`, and `nt` the length of `t`.
        """
        r = np.sqrt(np.square(x - self.x) + np.square(y - self.y))
        r[r < self._model.r_in] = self._model.r_in
        shape = (self._model.nl, ) + x.shape
        if len(t) > 0:
            shape = shape + (len(t[0]), )
        return np.reshape(self._model.h(r.flatten(), *t), shape)


class MultipleWells(ABC):
    """
    Abstract base class for classes implementing superposition models with multiple pumping wells.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    c : array_like
      Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
    c_top : float, default: inf
          Vertical resistance [T] of the upper boundary of the aquifer system.
          By default, the upper boundary is impervious.
    c_bot : float, default: inf
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.

    Attributes
    ----------
    nl : int
       Number of layers.
    wells : list of Well objects
          Contains all pumping wells added to the model.

    Methods
    -------
    add(*args) : abstract method
               Add pumping well.
    h(*args) : abstract method
             Calculate heads at given coordinates x and y, and at given times t if model is transient.

    Notes
    -----
    Subclasses implement abstract methods `add` and `h`.
    """
    def __init__(self, T, c=None, c_top=np.inf, c_bot=np.inf):
        self.T = T
        try:
            self.nl = len(T)
        except:
            self.nl = 1
        self.c = c
        self.c_top = c_top
        self.c_bot = c_bot
        self.wells = []

    @abstractmethod
    def add(self, *args):
        """
        Add pumping well.

        Parameters
        ----------
        *args

        Returns
        -------
        well : Well object
             Added pumping well.
        """
        pass

    @abstractmethod
    def h(self, *args):
        """
        Calculate heads at given coordinates x and y, and at given times t if model is transient.

        Parameters
        ----------
        *args

        Returns
        -------
        h : ndarray
          Hydraulic heads in each layer at given coordinates x and y, and at given times t if model is transient.
        """
        pass


class SteadyWells(MultipleWells):
    """
    Class to build steady state superposition models with multiple pumping wells.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    c : array_like
      Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
    c_top : float, default: inf
          Vertical resistance [T] of the upper boundary of the aquifer system.
          By default, the upper boundary is impervious.
    c_bot : float, default: inf
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.

    Attributes
    ----------
    nl : int
       Number of layers.
    wells : list of Well objects
          Contains all pumping wells added to the model.

    Methods
    -------
    add(x, y, rw, Q) :
                     Add pumping well with radius rw and rate Q, at position with coordinates x and y.
    h(x, y) :
            Calculate heads at given coordinates x and y.
    """

    def __init__(self, T, c=None, c_top=np.inf, c_bot=np.inf):
        super().__init__(T=T, c=c, c_top=c_top, c_bot=c_bot)

    def add(self, x, y, rw, Q):
        """
        Add pumping well with radius rw and rate Q, at position with coordinates x and y.

        Parameters
        ----------
        x : float
          X coordinate [L] of well.
        y : float
          Y coordinate [L] of well.
        rw : float
           Pumping well radius [L].
        Q : array_like
          Pumping rates [L³/T] in each layer. The length of `Q` is equal to the number of layers.

        Returns
        -------
        well : Well object
             Added pumping well.
        """
        model = SteadyQ(T=self.T, c=self.c, c_top=self.c_top, c_bot=self.c_bot, r_in=rw, Q=Q)
        well = Well(x, y, model)
        self.wells.append(well)
        return well

    def h(self, x, y):
        """
        Calculate heads at given grid coordinates x and y.

        Parameters
        ----------
        x : array_like
          One-dimensional array representing the x coordinates [L] of the grid.
        y : array_like
          One-dimensional array representing the y coordinates [L] of the grid.

        Returns
        -------
        h : ndarray
          Hydraulic heads in each layer at given coordinates `x` and `y`.
          Shape of `h` is `(nl, ny, nx)`, with `nl` the number of layers, `nx` the length of `x`,
          and `ny` the length of `y`.

        Notes
        -----
        `x` and `y` are input to numpy function `meshgrid`.
        """
        x, y = np.meshgrid(x, y)
        h = np.zeros((self.nl, ) + x.shape)
        for well in self.wells:
            h += well.h(x, y)
        return h


class TransientWells(MultipleWells):
    """
    Class to build transient state superposition models with multiple pumping wells.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    S : array_like
      Layer storativities [-]. The length of `S` is equal to the number of layers.
    c : array_like
      Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
    c_top : float, default: inf
          Vertical resistance [T] of the upper boundary of the aquifer system.
          By default, the upper boundary is impervious.
    c_bot : float, default: inf
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.
    nstehfest: int, default: 16
             Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
             Must be a positive, even integer.

    Attributes
    ----------
    nl : int
       Number of layers.
    wells : list of Well objects
          Contains all pumping wells added to the model.

    Methods
    -------
    add(x, y, rw, Qstart, t, Q) :
                                Add pumping well with radius rw and variable rates at position with coordinates x and y.
    h(x, y, t) :
            Calculate heads at given coordinates x and y, and at given time t.
    """

    def __init__(self, T, S, c=None, c_top=np.inf, c_bot=np.inf, nstehfest=16):
        super().__init__(T=T, c=c, c_top=c_top, c_bot=c_bot)
        self.S = S
        self.nstehfest = nstehfest

    def add(self, x, y, rw, Qstart, t=None, Q=None):
        """
        Add pumping well with radius rw and variables rates, at position with coordinates x and y.

        Parameters
        ----------
        x : float
          X coordinate [L] of well.
        y : float
          Y coordinate [L] of well.
        rw : float
           Pumping well radius [L].
        Qstart : array_like
               Pumping rates [L³/T] in each layer at t = 0. The length of `Q` is equal to the number of layers.
        t : array_like
          Times [L] at which the pumping rate change.
        Q : array_like
          Pumping rates [L³/T] corresponding to times `t`.
          The shape of `Q` is `(nl, nt)` with `nl` the number of layers and `nt` the length of `t`.

        Returns
        -------
        well : Well object
             Added pumping well.
        """
        model = TransientQ(T=self.T, S=self.S, c=self.c, c_top=self.c_top, c_bot=self.c_bot, r_in=rw, Q=Qstart,
                           nstehfest=self.nstehfest)
        if t is not None:
            model = VariableQ(model, t, Q)
        well = Well(x, y, model)
        self.wells.append(well)
        return well

    def h(self, x, y, t):
        """
        Calculate heads at given grid coordinates x and y, and at given times t.

        Parameters
        ----------
        x : array_like
          One-dimensional array representing the x coordinates [L] of the grid.
        y : array_like
          One-dimensional array representing the y coordinates [L] of the grid.
        t : array_like
          One-dimensional array with simulation times [T].

        Returns
        -------
        h : ndarray
          Hydraulic heads in each layer at given coordinates `x` and `y`, and at given times `t`.
          Shape of `h` is `(nl, ny, nx, nt)`, with `nl` the number of layers, `nx` the length of `x`,
          `ny` the length of `y`, and `nt` the length of `t`.

        Notes
        -----
        `x` and `y` are input to numpy function `meshgrid`.
        """
        x, y = np.meshgrid(x, y)
        t = np.array(t)
        if t.ndim == 0:
            t = t[np.newaxis]
        h = np.zeros((self.nl, ) + x.shape + (len(t), ))
        for well in self.wells:
            h += well.h(x, y, t)
        return h
