"""
Base module containing base classes for analytical steady and transient state 2D groundwater flow models.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eig, inv
from math import factorial, log
import warnings
from scipy.linalg import LinAlgWarning


class Model(ABC):
    """
    Abstract super class from which all classes inherit that implement a 2D groundwater flow model.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    c : array_like
      Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
    c_top : float, default: inf
          Vertical resistance [T] of the upper boundary of the aquifer system.
          By default, the upper boundary is impervious.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.
    c_bot : float, default: inf
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.
    h_bot : float, default: 0.0
          Constant head [L] of the lower boundary condition.
    r_in : float, default: 0.0
         Radial or horizontal distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial or horizontal distance [L] of the outer model boundary.
    h_out : array_like, default: None
          Constant head [L] at the outer model boundary for each layer.
          The length of `h_out` is equal to the number of layers.
          By default, the constant heads at the outer boundary are zero.
    N : array_like, default: None
      Recharge flux [L/T] for each layer.
      The length of `N` is equal to the number of layers.
      By default, the recharge in each layer is zero.

    Attributes
    ----------
    nl : int
       Number of layers
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    h(r, t) :
            Calculate hydraulic head h at given distances r, and at given times t.
    qh(r, t) :
             Calculate radial or horizontal discharge Qh at given distances r, and at given times t.

    Notes
    -----
    Subclasses implement protected abstract methods `_h_`, `_qh_`, `_ini_`, `_Ab_`, `_eig_`, and `_bc_`.
    Subclasses also override and extend protected method `_out_`.

    The class contains protected static method `_to_array` to convert input arrays into numpy arrays.
    Protected method `_check_array` also converts input arrays into numpy arrays,
    and returns a default array of zeros if the input array is `None`.
    """

    def __init__(self, T, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_in=0.0, r_out=np.inf, h_out=None,
                 N=None):
        self.T = self._to_array(T)  # (nl, )
        self.nl = len(self.T)
        if self.nl == 1:
            self.c = np.array([])
        else:
            self.c = self._to_array(c)  # (nl-1, )
        self.c_top = c_top
        self.h_top = h_top
        self.c_bot = c_bot
        self.h_bot = h_bot
        self.confined = np.all(np.isinf([self.c_top, self.c_bot]))  # bool
        self.r_in = r_in  # float
        self.r_out = r_out  # float
        self.h_out = self._check_array(h_out)
        self.N = self._check_array(N)
        self.no_warnings = True
        self._initialized = False  # becomes True the first time _out_() has been called

    def h(self, *rt):
        """
        Calculate hydraulic head h in each layer, at given distances r, and at given times t if the model is transient.

        Parameters
        ----------
        r : array_like
          Radial or horizontal distances [L].
        t : array_like
          Times [T]. Only if model is transient

        Returns
        -------
        h : ndarray
          Hydraulic heads [L] at given distances `r`, and at given times `t` if the model is transient.
          The shape of `h` is  `(nl, nr)` if the model is steady, and `(nl, nr, nt)` if the model is transient,
          where `nl` is the number of layers, `nr` the length of `r`, and `nt` the length of `t`.
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            return self._out_('_h_', *rt)

    def qh(self, *rt):
        """
        Calculate radial or horizontal discharge Qh in each layer, at given distances r,
        and at given times t if the model is transient.

        Parameters
        ----------
        r : array_like
          Radial or horizontal distances [L].
        t : array_like
          Times [T]. Only if model is transient

        Returns
        -------
        qh : ndarray
           Radial discharges [L³/T] or horizontal discharges [L³/T/L] at given distances `r`,
           and at given times `t` if the model is transient.
           The shape of `qh` is  `(nl, nr)` if the model is steady, and `(nl, nr, nt)` if the model is transient,
           where `nl` is the number of layers, `nr` the length of `r`, and `nt` the length of `t`.
        """
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            return self._out_('_qh_', *rt)

    def _out_(self, *args):
        """
        Called by methods `h` and `qh`.

        Subclasses override this method and extend it with code to calculate heads or discharges.

        Parameters
        ----------
        method : str
               String ``'_h_'`` or ``'_qh_'``.
        r : array_like
          Radial or horizontal distances [L].
        t : array_like
          Times [T]. Only if model is transient.

        Returns
        -------
        function
                 Method `_h_` or `_qh_`, depending on the value of input parameter `method`.
        """
        if not self._initialized:
            self._ini_()
            self._initialized = True
        return getattr(self, args[0])  # method _h_ or _qh_
        # SUBCLASSING: call super()._out_(*args) and add code to calculate h or qh

    @abstractmethod
    def _h_(self, r):
        """
        Calculate hydraulic head h in each layer at distance r.

        Abstract method.

        Parameters
        ----------
        r : float
          Radial or horizontal distance [L].

        Returns
        -------
        h : ndarray
          Hydraulic head [L] in each layer at distance `r`.
          The length of `h` is equal to the number of layers.
        """
        pass

    @abstractmethod
    def _qh_(self, r):
        """
        Calculate horizontal or radial discharge Qh in each layer at distance r.

        Abstract method.

        Parameters
        ----------
        r : float
          Radial or horizontal distance [L].

        Returns
        -------
        qh : ndarray
          Radial discharge [L³/T] or horizontal discharge [L³/T/L] in each layer at distance `r`.
          The length of `qh` is equal to the number of layers.
        """
        pass

    @abstractmethod
    def _ini_(self):
        """
        Initialize the calculation of hydraulic heads h and discharges Qh.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _Ab_(self, *p):
        """
        Construct system matrix A and vector b.

        Parameters
        ----------
        p : float
          Laplace variable p [1/T].
          Only required if model is transient.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _eig_(self):
        """
        Calculate eigenvalues and eigenvectors of system matrix A.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _bc_(self):
        """
        Apply boundary conditions.

        Returns
        -------
        None
        """
        pass

    @staticmethod
    def _to_array(arr, dtype=float):
        """
        Convert array-like input into numpy `ndarray`.

        Parameters
        ----------
        arr : array_like
            Input array.
        dtype : data-type, default: float
              Any object that can be interpreted as a numpy data type.

        Returns
        -------
        arr : ndarray
            Input array converted into numpy `ndarray`.
        """
        arr = np.array(arr, dtype=dtype)
        if arr.ndim == 0:
            arr = arr[np.newaxis]
        return arr

    def _check_array(self, arr, dtype=float, n=None):
        """
        Convert array-like input into numpy `ndarray`.
        If the input array is `None`, a default array containing zeros is returned with length n.
        If the length is not specified, the number of layers is taken.

        Parameters
        ----------
        arr : array_like or None
            Input array.
        dtype : data-type, default: float
              Any object that can be interpreted as a numpy data type.
        n : int, default: None
          Specified length of the default array.
          If `n` is not specified, it is set to the number of layers.

        Returns
        -------
        arr : ndarray
            Input array converted into numpy `ndarray`.
            If input array `arr` is `None`, an array with `n` zeros is returned.
        """
        if arr is None:
            return np.zeros(self.nl if n is None else n, dtype=dtype)
        else:
            return self._to_array(arr, dtype)


class Steady(Model):
    """
    Base class for classes that implement a steady state 2D groundwater flow model.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    c : array_like
      Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
    c_top : float, default: inf
          Vertical resistance [T] of the upper boundary of the aquifer system.
          By default, the upper boundary is impervious.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.
    c_bot : float, default: inf
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.
    h_bot : float, default: 0.0
          Constant head [L] of the lower boundary condition.
    r_in : float, default: 0.0
         Radial or horizontal distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial or horizontal distance [L] of the outer model boundary.
    h_out : array_like, default: None
          Constant head [L] at the outer model boundary for each layer.
          The length of `h_out` is equal to the number of layers.
          By default, the constant heads at the outer boundary are zero.
    N : array_like, default: None
      Recharge flux [L/T] for each layer.
      The length of `N` is equal to the number of layers.
      By default, the recharge in each layer is zero.

    Attributes
    ----------
    nl : int
       Number of layers
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    h(r) :
         Calculate hydraulic head h at given distances r.
    qh(r) :
          Calculate radial or horizontal discharge Qh at given distances r.

    Notes
    -----
    Subclasses override protected method `_bc_` and extend it.

    """

    def __init__(self, T, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_in=0.0, r_out=np.inf, h_out=None,
                 N=None):
        super().__init__(T=T, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out,
                         h_out=h_out, N=N)

    def h(self, r):
        """
        Calculate hydraulic head h in each layer at given distances r.

        Parameters
        ----------
        r : array_like
          Radial or horizontal distances [L].

        Returns
        -------
        h : ndarray
          Hydraulic heads [L] at given distances `r`.
          The shape of `h` is  `(nl, nr)`, where `nl` is the number of layers, and `nr` the length of `r`.
        """
        r = self._to_array(r)
        return super().h(r)

    def qh(self, r):
        """
        Calculate radial or horizontal discharge Qh in each layer at given distances r.

        Parameters
        ----------
        r : array_like
          Radial or horizontal distances [L].

        Returns
        -------
        qh : ndarray
           Radial discharges [L³/T] or horizontal discharges [L³/T/L] at given distances `r`.
           The shape of `qh` is  `(nl, nr)`, where `nl` is the number of layers, and `nr` the length of `r`.
        """
        r = self._to_array(r)
        return super().qh(r)

    def _out_(self, method, r):
        """
        Called by methods `h` and `qh`.

        Parameters
        ----------
        method : str
               String ``'_h_'`` or ``'_qh_'``.
        r : array_like
          Radial or horizontal distances [L].

        Returns
        -------
        out : ndarray
              Hydraulic heads `h` [L] or horizontal discharges `qh` [L³/T or L²/T/L] at distances `r`,
              depending on the value of input parameter `method`.
              The shape of `out` is  `(nl, nr)`, where `nl` is the number of layers, and `nr` the length of `r`.
        """
        method = super()._out_(method, r)
        nr = len(r)
        out = np.zeros((self.nl, nr))  # (nl, nr)
        for i in range(nr):
            out[:, i] = method(r[i])
        return out

    def _ini_(self):
        """
        Initialize the calculation of hydraulic heads h and discharges Qh.

        Calls methods `_Ab_`, `_eig_`, and `_bc_`.

        Returns
        -------
        None
        """
        self._Ab_()
        self._eig_()
        self._bc_()

    def _Ab_(self):
        """
        Construct system matrix A and vector b.

        Returns
        -------
        None
        """
        c = np.hstack((self.c_top, self.c, self.c_bot))
        Tc0 = 1 / (self.T * c[:-1])  # (nl, )
        Tc1 = 1 / (self.T * c[1:])  # (nl, )
        self._idx = np.diag_indices(self.nl)
        irow, icol = self._idx
        self._A = np.zeros((self.nl, self.nl))  # (nl, nl)
        self._A[irow, icol] = Tc0 + Tc1
        self._A[irow[:-1], icol[:-1] + 1] = -Tc1[:-1]
        self._A[irow[:-1] + 1, icol[:-1]] = -Tc0[1:]
        self._b = self.N / self.T  # (nl, )
        self._b[0] += Tc0[0] * self.h_top
        self._b[-1] += Tc1[-1] * self.h_bot

    def _eig_(self):
        """
        Calculate eigenvalues and eigenvectors of system matrix A.

        Applies functions `eig` and `inv` from the scipy `linalg` module.

        Returns
        -------
        None
        """
        self._d, self._V = eig(self._A)  # (nl, ), (nl, nl)
        self._d = np.real(self._d)
        self._inz = np.arange(self.nl)
        if self.confined:
            self._iz = np.argmin(np.abs(self._d))
            self._inz = np.setdiff1d(self._inz, self._iz)
        if len(self._inz) > 0:
            self._sd = np.sqrt(self._d[self._inz])
        self._iV = inv(self._V)
        self._v = np.dot(self._iV, self._b)

    def _bc_(self):
        """
        Apply boundary conditions.

        Subclasses call this method and extend it.

        Returns
        -------
        None
        """
        self._w = np.dot(self._iV, self.h_out)  # (nl, )
        self._alpha = np.zeros(self.nl)  # (nl, )
        self._beta = np.zeros(self.nl)  # (nl, )
        # SUBCLASSING: call super()._bc_(), transform inner boundary condition, and calculate alpha and beta


class Transient(Model):
    """
    Base class for classes that implement a transient state 2D groundwater flow model.

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
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.
    c_bot : float, default: inf
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.
    h_bot : float, default: 0.0
          Constant head [L] of the lower boundary condition.
    r_in : float, default: 0.0
         Radial or horizontal distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial or horizontal distance [L] of the outer model boundary.
    h_out : array_like, default: None
          Constant head [L] at the outer model boundary for each layer.
          The length of `h_out` is equal to the number of layers.
          By default, the constant heads at the outer boundary are zero.
    N : array_like, default: None
      Recharge flux [L/T] for each layer.
      The length of `N` is equal to the number of layers.
      By default, the recharge in each layer is zero.
    nstehfest: int, default: 16
             Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
             Must be a positive, even integer.

    Attributes
    ----------
    nu : ndarray
       Parameter `nu` is calculated as S/T.
    nl : int
       Number of layers
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    h(r, t) :
         Calculate hydraulic head h at given distances r and given times t.
    qh(r, t) :
          Calculate radial or horizontal discharge Qh at given distances r and given times t.

    Notes
    -----
    Subclasses initialize the corresponding steady state model during object instantation
    and assign this model to protected attribute `_steady`.
    Subclasses also override and extend protected method `_Ab_`.
    """

    def __init__(self, T, S, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_in=0.0, r_out=np.inf,
                 h_out=None, N=None, nstehfest=16):
        super().__init__(T=T, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out,
                         h_out=h_out, N=N)
        self.S = self._to_array(S)  # (nl, )
        self.nu = self.S / self.T
        self.nstehfest = int(nstehfest)
        self._steady = None  # SUBCLASSING: call super().__init__(*params) and initialize steady state model

    def h(self, r, t):
        """
        Calculate hydraulic head h in each layer, at given distances r, and at given times t.

        Parameters
        ----------
        r : array_like
          Radial or horizontal distances [L].
        t : array_like
          Times [T].

        Returns
        -------
        h : ndarray
          Hydraulic heads [L] at given distances `r`, and at given times `t`.
          The shape of `h` is  `(nl, nr, nt)`, where `nl` is the number of layers, `nr` the length of `r`,
          and `nt` the length of `t`.
        """
        r = self._to_array(r)
        t = self._to_array(t)
        return super().h(r, t)

    def qh(self, r, t):
        """
        Calculate radial or horizontal discharge Qh in each layer, at given distances r, and at given times t.

        Parameters
        ----------
        r : array_like
          Radial or horizontal distances [L].
        t : array_like
          Times [T].

        Returns
        -------
        qh : ndarray
           Radial discharges [L³/T] or horizontal discharges [L³/T/L] at given distances `r`, and at given times `t`.
           The shape of `qh` is `(nl, nr, nt)`, where `nl` is the number of layers, `nr` the length of `r`,
           and `nt` the length of `t`.
        """
        r = self._to_array(r)
        t = self._to_array(t)
        return super().qh(r, t)

    def _out_(self, method, r, t):
        """
        Called by methods `h` and `qh`.

        Parameters
        ----------
        method : str
               String ``'_h_'`` or ``'_qh_'``.
        r : array_like
          Radial or horizontal distances [L].
        t : array_like
          Times [T].

        Returns
        -------
        out : ndarray
              Hydraulic heads `h` [L] or horizontal discharges `qh` [L³/T or L²/T/L] at distances `r`,
              and at times `t`, depending on the value of input parameter `method`.
              The shape of `out` is  `(nl, nr, nt)`, where `nl` is the number of layers, `nr` the length of `r`,
              and `nt` is the length of `t`.
        """
        method = super()._out_(method, r, t)
        nr, nt = len(r), len(t)
        ln2t = log(2) / t
        out = np.zeros((self.nl, nr, nt))
        for i in range(nt):
            for k in range(self.nstehfest):
                p = ln2t[i] * (k + 1)
                self._Ab_(p)
                self._eig_()
                self._bc_()
                out[:, :, i] += self._W[k] * method(r)
            out[:, :, i] *= ln2t[i]
        return out

    def _h_(self, r):
        """
        Calculate the Laplace-transformed hydraulic head h in each layer at distance r and for inverse time p.

        Calls method `h` of the corresponding steady state model stored in attribute `_steady`.

        Parameters
        ----------
        r : float
          Radial or horizontal distance [L].

        Returns
        -------
        h : ndarray
          Laplace-transformed hydraulic head [L] in each layer at distance `r` and for inverse time `p`.
          The length of `h` is equal to the number of layers.
        """
        return self._steady.h(r)

    def _qh_(self, r):
        """
        Calculate the Laplace-transformed horizontal discharge Qh in each layer at distance r and for inverse time p.

        Calls method `qh` of the corresponding steady state model stored in attribute `_steady`.

        Parameters
        ----------
        r : float
          Radial or horizontal distance [L].

        Returns
        -------
        qh : ndarray
           Laplace-transformed horizontal discharge [L³/T or L³/T/L] in each layer at distance `r`
           and for inverse time `p`. The length of `qh` is equal to the number of layers.
        """
        return self._steady.qh(r)

    def _ini_(self):
        """
        Initialize the calculation of hydraulic heads h and discharges Qh.

        Calls method `_stehfest_weights`, initializes the corresponding steady state model `_steady`,
        calls its method `_Ab_`, and copies its system matrix `_A` and vector `_b`.

        Returns
        -------
        None
        """
        self._stehfest_weights()
        self._steady.confined = False
        self._steady._initialized = True
        self._steady._Ab_()
        self._A = self._steady._A.copy()
        self._b = self._steady._b.copy()

    def _Ab_(self, p):
        """
        Update system matrix A and vector b for a given value of Laplace variable p.

        Subclasses override this method and extend it.

        Parameters
        ----------
        p : float
          Laplace variable [1/T].

        Returns
        -------
        None
        """
        self._steady._A[self._steady._idx] = self._A[self._steady._idx] + self.nu * p
        self._steady._b = self._b / p + self.nu * self.h_out
        self._steady.h_out = self.h_out / p
        # SUBCLASSING: call super()._Ab_(p) and update inner boundary condition

    def _eig_(self):
        """
        Calculate eigenvalues and eigenvectors of system matrix A.

        Calls method `_eig_` of steady state model `_steady`.

        Returns
        -------
        None
        """
        self._steady._eig_()

    def _bc_(self):
        """
        Apply boundary conditions.

        Calls method `_bc_` of steady state model `_steady`.

        Returns
        -------
        None
        """
        self._steady._bc_()

    def _stehfest_weights(self):
        """
        Calculate weights required for applying the Stehfest algorithm.

        Returns
        -------
        None
        """
        fac = lambda x: float(factorial(x))
        ns2 = self.nstehfest // 2
        self._W = np.zeros(self.nstehfest)  # (nstehfest, )
        for j in range(1, self.nstehfest + 1):
            m = min(j, ns2)
            k_0 = (j + 1) // 2
            for k in range(k_0, m + 1):
                self._W[j - 1] += k ** ns2 * fac(2 * k) / fac(ns2 - k) / fac(k) / fac(k - 1) / fac(j - k) / fac(
                    2 * k - j)
            self._W[j - 1] *= (-1) ** (ns2 + j)
