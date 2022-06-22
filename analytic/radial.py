"""
Module containing classes for steady and transient state 2D radial groundwater flow models.
"""
import numpy as np
from scipy.special import i0, k0, k1, i1
from ._base import Steady as SteadyBase, Transient as TransientBase


class Steady(SteadyBase):
    """
    Base class for classes that implement a steady state 2D axi-symmetric groundwater flow model.

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
         Radial distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial distance [L] of the outer model boundary.
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
          Calculate radial discharge Qh at given distances r.

    Notes
    -----
    Subclasses override protected method `_bc_` and extend it.
    """

    def __init__(self, T, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_in=0.0, r_out=np.inf, h_out=None, N=None):
        super().__init__(T=T, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out, h_out=h_out, N=N)

    def _h_(self, r):
        """
        Calculate hydraulic head h in each layer at distance r.

        Parameters
        ----------
        r : float
          Radial distance [L].

        Returns
        -------
        h : ndarray
          Hydraulic head [L] in each layer at distance `r`.
          The length of `h` is equal to the number of layers.
        """
        g = np.zeros(self.nl)  # (nl, )
        if self.confined:
            g[self._iz] = self._alpha[self._iz] * np.log(r) + self._beta[self._iz] - self._v[self._iz] * r ** 2 / 4
        if len(self._inz) > 0:
            x = self._sd * r
            i0x = i0(x)
            i0x[np.isnan(i0x)] = np.inf  # x -> inf: i0(x) = inf
            alpha_i0x = self._alpha[self._inz] * i0x
            alpha_i0x[np.isnan(alpha_i0x)] = 0.0  # if alpha = 0 and I0 = Inf
            g[self._inz] = alpha_i0x + self._beta[self._inz] * k0(x) + self._v[self._inz] / self._d[self._inz]
        return np.dot(self._V, g)

    def _qh_(self, r):
        """
        Calculate radial discharge Qh in each layer at distance r.

        Parameters
        ----------
        r : float
          Radial distance [L].

        Returns
        -------
        qh : ndarray
           Radial discharge [L³/T] in each layer at distance `r`.
           The length of `qh` is equal to the number of layers.
        """
        dg = np.zeros(self.nl)  # (nl, )
        if self.confined:
            dg[self._iz] = self._alpha[self._iz] - self._v[self._iz] * r ** 2 / 2
        if len(self._inz) > 0:
            x = self._sd * r
            xi1x = x * i1(x)
            xk1x = x * k1(x)
            xk1x[np.isnan(xk1x)] = 1.0  # r -> 0: x_in * k1(x_in) = 1
            alpha_xi1x = self._alpha[self._inz] * xi1x
            alpha_xi1x[np.isnan(alpha_xi1x)] = 0.0  # if alpha = 0 and I1 = Inf
            dg[self._inz] = alpha_xi1x - self._beta[self._inz] * xk1x
        return 2 * np.pi * self.T * np.dot(self._V, -dg)

    def _bessel_(self):
        """
        Apply Bessel functions to inner and outer model boundary values.

        Subclasses call this method and extend it.

        Returns
        -------
        None
        """
        if len(self._inz) > 0:
            x_out = self._sd * self.r_out
            self._i0_out = i0(x_out)
            self._i0_out[np.isnan(self._i0_out)] = np.inf  # x_out -> inf: i0(x_out) = inf
            self._k0_out = k0(x_out)
        # SUBCLASSING: call super()._bessel_() and add _i*_in and _k*_in

    def _bc_(self):
        """
        Apply boundary conditions.

        Subclasses call this method and extend it.

        Returns
        -------
        None
        """
        self._bessel_()
        super()._bc_()
        # SUBCLASSING: call super()._bc_(), transform inner boundary condition, and calculate alpha and beta


class SteadyQ(Steady):
    """
    Class to build steady state 2D axi-symmetric groundwater flow models with constant discharges at the inner boundary.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    Q : array_like
      Layer discharges [L³/T] at the inner model boundary. The length of `Q` is equal to the number of layers.
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
         Radial distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial distance [L] of the outer model boundary.
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
          Calculate radial discharge Qh at given distances r.
    """

    def __init__(self, T, Q, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_in=0.0, r_out=np.inf, h_out=None, N=None):
        super().__init__(T=T, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out, h_out=h_out, N=N)
        self.Q = self._to_array(Q)  # (nl, )

    def _bessel_(self):
        """
        Apply Bessel functions to inner and outer model boundary values.

        Returns
        -------
        None
        """
        super()._bessel_()
        if len(self._inz) > 0:
            x_in = self._sd * self.r_in
            self._i1_in = x_in * i1(x_in)
            self._k1_in = x_in * k1(x_in)
            self._k1_in[np.isnan(self._k1_in)] = 1.0  # r_in -> 0: x_in * k1(x_in) = 1

    def _bc_(self):
        """
        Apply boundary conditions.

        Returns
        -------
        None
        """
        super()._bc_()
        self._q = np.dot(self._iV, self.Q / self.T / 2 / np.pi)  # (nl, )
        if self.confined:
            self._alpha[self._iz] = -self._q[self._iz] + self._v[self._iz] * self.r_in ** 2 / 2
            self._beta[self._iz] = self._w[self._iz] + self._v[self._iz] * self.r_out ** 2 / 4 - self._alpha[self._iz] * np.log(self.r_out)
        if len(self._inz) > 0:
            nominator = self._i1_in * self._k0_out + self._k1_in * self._i0_out
            wvd = self._w[self._inz] - self._v[self._inz] / self._d[self._inz]
            self._alpha[self._inz] = (wvd * self._k1_in - self._q[self._inz] * self._k0_out) / nominator
            self._beta[self._inz] = (wvd * self._i1_in + self._q[self._inz] * self._i0_out) / nominator
            i = np.isnan(self._beta[self._inz])
            if np.any(i):
                j = self._inz[i]
                self._beta[j] = self._q[j] / self._k1_in[i]  # if r_out -> inf


class SteadyH(Steady):
    """
    Class to build steady state 2D axi-symmetric groundwater flow models with constant heads at the inner boundary.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    h_in : array_like
         Constant heads [L] at the inner model boundary. The length of `h_in` is equal to the number of layers.
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
         Radial distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial distance [L] of the outer model boundary.
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
          Calculate Radial discharge Qh at given distances r.
    """

    def __init__(self, T, r_in, h_in, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_out=np.inf, h_out=None, N=None):
        super().__init__(T=T, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out, h_out=h_out, N=N)
        self.h_in = self._to_array(h_in)  # (nl, )

    def _bessel_(self):
        """
        Apply Bessel functions to inner and outer model boundary values.

        Returns
        -------
        None
        """
        super()._bessel_()
        if len(self._inz) > 0:
            x_in = self._sd * self.r_in
            self._i0_in = i0(x_in)
            self._k0_in = k0(x_in)

    def _bc_(self):
        """
        Apply boundary conditions.

        Returns
        -------
        None
        """
        super()._bc_()
        self._u = np.dot(self._iV, self.h_in)  # (nl, )
        if self.confined:
            r_in2, r_out2 = self.r_in ** 2, self.r_out ** 2
            ln_r_in, ln_r_out = np.log(self.r_in), np.log(self.r_out)
            u, w = self._u[self._iz], self._w[self._iz]
            v4 = self._v[self._iz] / 4
            nominator = ln_r_in - ln_r_out
            self._alpha[self._iz] = (u - w + v4 * (r_in2 - r_out2)) / nominator
            self._beta[self._iz] = (ln_r_in * (v4 * r_out2 + w) - ln_r_out * (v4 * r_in2 + u)) / nominator
        if len(self._inz) > 0:
            vd = self._v[self._inz] / self._d[self._inz]
            wvd, uvd = self._w[self._inz] - vd, self._u[self._inz] - vd
            nominator = self._i0_in * self._k0_out - self._k0_in * self._i0_out
            self._alpha[self._inz] = (uvd * self._k0_out - wvd * self._k0_in) / nominator
            self._beta[self._inz] = (wvd * self._i0_in - uvd * self._i0_out) / nominator
            i = np.isnan(self._beta[self._inz])
            if np.any(i):
                j = self._inz[i]
                self._beta[j] = uvd[j] / self._k0_in[i]  # if r_out -> inf


class TransientQ(TransientBase):
    """
    Class to build transient state 2D axi-symmetric groundwater flow models with constant discharges
    at the inner boundary.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    S : array_like
      Layer storativities [-]. The length of `S` is equal to the number of layers.
    Q : array_like
      Layer discharges [L³/T] at the inner model boundary. The length of `Q` is equal to the number of layers.
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
         Radial distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial distance [L] of the outer model boundary.
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
          Calculate radial discharge Qh at given distances r and given times t.
    """

    def __init__(self, T, S, Q, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_in=0.0, r_out=np.inf, h_out=None, N=None,
                 nstehfest=16):
        super().__init__(T=T, S=S, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out, h_out=h_out, N=N,
                         nstehfest=nstehfest)
        self.Q = self._to_array(Q)  # (nl, )
        self._steady = SteadyQ(T=T, Q=Q, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out, h_out=h_out, N=N)

    def _Ab_(self, p):
        """
        Update system matrix A and vector b for a given value of Laplace variable p.

        Parameters
        ----------
        p : float
          Laplace variable [1/T].

        Returns
        -------
        None
        """
        super()._Ab_(p)
        self._steady.Q = self.Q / p


class TransientH(TransientBase):
    """
    Class to build transient state 2D axi-symmetric groundwater flow models with constant heads at the inner boundary.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    S : array_like
      Layer storativities [-]. The length of `S` is equal to the number of layers.
    h_in : array_like
         Constant heads [L] at the inner model boundary. The length of `h_in` is equal to the number of layers.
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
         Radial distance [L] of the inner model boundary.
    r_out : float, default: inf
          Radial distance [L] of the outer model boundary.
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
          Calculate radial discharge Qh at given distances r and given times t.
    """

    def __init__(self, T, S, r_in, h_in, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0, r_out=np.inf,
                 h_out=None, N=None, nstehfest=16):
        super().__init__(T=T, S=S, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot, r_in=r_in, r_out=r_out,
                         h_out=h_out, N=N, nstehfest=nstehfest)
        self.h_in = self._to_array(h_in)  # (nl, )
        self._steady = SteadyH(T=T, r_in=r_in, h_in=h_in, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot,
                               r_out=r_out, h_out=h_out, N=N)

    def _Ab_(self, p):
        """
        Update system matrix A and vector b for a given value of Laplace variable p.

        Parameters
        ----------
        p : float
          Laplace variable [1/T].

        Returns
        -------
        None
        """
        super()._Ab_(p)
        self._steady.h_in = self.h_in / p
