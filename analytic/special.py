"""
Module containing functions to calculate well-known groundwater flow solutions for radial and parallel flow.
"""
import numpy as np
from scipy.special import exp1, k0, k1, erfc
from math import factorial, log


def darcy(r, T, Q, r_out, h_out=0.0):
    """
    Calculate hydraulic head at given distances r according to Darcy's law.
    The resulting flow is strictly horizontal and parallel.
    The aquifer has an infinitely large width.

    Parameters
    ----------
    r : array_like
      Horizontal distances [L], which should be smaller than `r_out`.
    T : float
      Transmissivity [L²/T].
    Q : float
      Horizontal discharge per unit width of the aquifer [L³/T/L].
    r_out : float
          Horizontal distance [L] of the outer aquifer boundary.
    h_out : float
          Hydraulic head [L] at the outer aquifer boundary at distance `r_out`.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.
    """
    r = np.array(r)
    return h_out + Q / T * (r_out - r)


def uniform_parallel(r, r_in, h_in, r_out, h_out=0.0):
    """
    Calculate hydraulic head at given distances r between two parallel constant-head boundaries.
    The resulting flow between the boundaries is strictly horizontal, parallel, and uniform.
    The aquifer has an infinitely large width.

    Parameters
    ----------
    r : array_like
      Horizontal distances [L], which should be between `r_in` and `r_out`
    r_in : float
         Horizontal distance [L] of the inner aquifer boundary.
    h_in : float
         Hydraulic head [L] at the inner aquifer boundary at distance `r_in`.
    r_out : float
          Horizontal distance [L] of the outer aquifer boundary.
    h_out : float, default: 0.0
          Hydraulic head [L] at the outer aquifer boundary at distance `r_out`.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.
    """
    r = np.array(r)
    return ((h_out - h_in) * r + h_in * r_out - h_out * r_in) / (r_out - r_in)


def thiem(r, T, Q, r_out, h_out=0.0):
    """
    Calculate hydraulic head at given distances r according to Thiem (1870).
    (See also Equation 3 in Louwyck et al., 2022)

    Parameters
    ----------
    r : array_like
      Radial distances [L], which should be smaller than `r_out`.
    Q : float
        Radial discharge [L³/T].
    r_out : float
          Radial distance [L] of the outer aquifer boundary.
    h_out : float, default: 0.0
          Hydraulic head [L] at the outer aquifer boundary at distance `r_out`.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.

    References
    ----------
    Thiem, A. (1870). Die Ergiebigkeit artesischer Bohrlöcher, Schachtbrunnen und Filtergalerien (in German).
    Journal für Gasbeleuchtung und Wasserversorgung 13, 450-467.

    Louwyck, A., Vandenbohede, A., Libbrecht, D., Van Camp, M., Walraevens, K. (2022). The radius of influence myth.
    Water 14(2): 149. https://doi.org/10.3390/w14020149.

    """
    r = np.array(r)
    return h_out + Q / 2 / np.pi / T * np.log(r_out / r)


def uniform_radial(r, r_in, h_in, r_out, h_out=0.0):
    """
    Calculate hydraulic head at given distances r between two parallel constant-head boundaries.
    The resulting flow between the boundaries is strictly horizontal, radial, and uniform.

    Parameters
    ----------
    r : array_like
      Radial distances [L], which should be between `r_in` and `r_out`
    r_in : float
         Radial distance [L] of the inner aquifer boundary.
    h_in : float
         Hydraulic head [L] at the inner aquifer boundary at distance `r_in`.
    r_out : float
          Radial distance [L] of the outer aquifer boundary.
    h_out : float, default: 0.0
          Hydraulic head [L] at the outer aquifer boundary at distance `r_out`.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.
    """
    r = np.array(r)
    return ((h_out - h_in) * np.log(r) + h_in * np.log(r_out) - h_out * np.log(r_in)) / np.log(r_out / r_in)


def haitjema(r, T, Q, N, r_out):
    """
    Calculate the solution for steady flow to a pumping well in an aquifer with recharge.

    The solution is obtained by superimposing the Thiem (1870) formula and the equation for a circular infiltration area
    (Haitjema, 1995; see also Equation A20 in Appendix A of Louwyck et al., 2022)

    Parameters
    ----------
    r : array_like
      Radial distances [L], which should be smaller than `r_out`.
    T : float
      Aquifer transmissivity [L²/T].
    Q : float
        Radial discharge [L³/T].
    N : float
      Recharge flux [L/T].
    r_out : float
          Radial distance [L] of the outer aquifer boundary.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.

    References
    ----------
    Thiem, A. (1870). Die Ergiebigkeit artesischer Bohrlöcher, Schachtbrunnen und Filtergalerien (in German).
    Journal für Gasbeleuchtung und Wasserversorgung 13, 450-467.

    Haitjema, H. (1995). Analytic Element Modeling of Groundwater Flow. San Diego: Academic Press.

    Louwyck, A., Vandenbohede, A., Libbrecht, D., Van Camp, M., Walraevens, K. (2022). The radius of influence myth.
    Water 14(2): 149. https://doi.org/10.3390/w14020149.

    """
    r = np.array(r)
    return -Q / 2 / np.pi / T * np.log(r / r_out) + N / 4 / T * (r_out**2 - r**2)


def deglee(r, T, Q, c_top, h_top=0.0):
    """
    Calculate the solution for steady flow to a pumping well in a leaky aquifer with impervious lower boundary
    (de Glee, 1930; Jacob, 1946; see also Equation A29 in Appendix A of Louwyck et. al., 2022).
    The well has an infinitesimal radius and extracts water at a constant pumping rate.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : float
      Aquifer transmissivity [L²/T].
    Q : float
      Pumping rate [L³/T] of the well.
    c_top : float
          Vertical resistance [T] of the aquitard overlying the aquifer.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.

    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r`.
      The length of `s` equals the length of `r`.

    References
    ----------
    De Glee, G.J. (1930). Over grondwaterstroomingen bij wateronttrekking door middel van putten (in Dutch).
    Thesis, J. Waltman Jr., Delft, 175pp.

    Jacob, C.E., (1946). Radial flow in a leaky artesian aquifer. Transactions American Geophysical Union 27, 198-205.

    Louwyck, A., Vandenbohede, A., Libbrecht, D., Van Camp, M., Walraevens, K. (2022). The radius of influence myth.
    Water 14(2): 149. https://doi.org/10.3390/w14020149.

    """
    r = np.array(r)
    return h_top + Q / 2 / np.pi / T * k0(r / np.sqrt(T * c_top))


def deglee_bruggeman(r, T, Q, r_in, c_top, h_top=0.0):
    """
    Calculate the solution for steady flow to a finite-diameter well in a leaky aquifer with impervious lower boundary
    (Equation 227.14 in Bruggeman, 1999; see also Equation A28 in Appendix A of Louwyck et. al., 2022).

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : float
      Aquifer transmissivity [L²/T].
    Q : float
      Pumping rate [L³/T] of the well.
    r_in : float
         Pumping well radius [L], which coincides with the inner model boundary.
    c_top : float
          Vertical resistance [T] of the aquitard overlying the aquifer.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.

    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r`.
      The length of `s` equals the length of `r`.

    References
    ----------
    Bruggeman, G.A. (1999). Analytical solutions of geohydrological problems. Developments in Water Science, 46.
    Elsevier, Amsterdam, 959pp.

    Louwyck, A., Vandenbohede, A., Libbrecht, D., Van Camp, M., Walraevens, K. (2022). The radius of influence myth.
    Water 14(2): 149. https://doi.org/10.3390/w14020149.

    """
    r = np.array(r)
    L = np.sqrt(1 / c_top / T)
    return h_top + Q / 2 / np.pi / T * k0(r * L) / r_in / L / k1(r_in * L)


def deglee_general(r, T, Q, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0):
    """
    Calculate the generalized solution for steady flow to a pumping well in a leaky aquifer
    (Equation A30 in Appendix A of Louwyck et. al., 2022).
    The well has an infinitesimal radius and extracts water at a constant pumping rate.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : float
      Aquifer transmissivity [L²/T].
    Q : float
      Pumping rate [L³/T] of the well.
    c_top : float, default: inf
          Vertical resistance [T] of the aquitard overlying the aquifer.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.
    c_bot : float, default: inf
          Vertical resistance [T] of the aquitard underlying the aquifer.
    h_bot : float, default: 0.0
          Constant head [L] of the lower boundary condition.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at distances `r`.
      The length of `h` equals the length of `r`.

    References
    ----------
    De Glee, G.J. (1930). Over grondwaterstroomingen bij wateronttrekking door middel van putten (in Dutch).
    Thesis, J. Waltman Jr., Delft, 175pp.

    Jacob, C.E., (1946). Radial flow in a leaky artesian aquifer. Transactions American Geophysical Union 27, 198-205.

    Louwyck, A., Vandenbohede, A., Libbrecht, D., Van Camp, M., Walraevens, K. (2022). The radius of influence myth.
    Water 14(2): 149. https://doi.org/10.3390/w14020149.

    """
    r = np.array(r)
    c_tot = c_top + c_bot
    return (c_bot * h_top + c_top * h_bot) / c_tot + Q / 2 / np.pi / T * k0(r * np.sqrt(c_tot / c_top / c_bot / T))


def theis(r, t, T, S, Q, h_out=0.0):
    """
    Calculate the solution for unsteady flow to a pumping well in a confined aquifer (Theis, 1935).
    The well has an infinitesimal radius and extracts water at a constant pumping rate.
    (see also Equation A33 in Appendix A of Louwyck et. al., 2022).

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : float
      Aquifer transmissivity [L²/T].
    S : float
      Aquifer storativity [-].
    Q : float
      Pumping rate [L³/T] of the well.
    h_out : float, default: 0.0
          Constant head [L] at the outer boundary condition, which is also the initial head in the aquifer.

    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r` and times `t`.
      Shape of `s` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.

    References
    ----------
    Theis C.V. (1935). The relation between the lowering of the piezometric surface and the rate and duration of
    discharge of a well using groundwater storage. Transactions of the American Geophysical Union, 2, 519 – 524.

    Louwyck, A., Vandenbohede, A., Libbrecht, D., Van Camp, M., Walraevens, K. (2022). The radius of influence myth.
    Water 14(2): 149. https://doi.org/10.3390/w14020149.

    """
    t, r = np.meshgrid(t, r)
    return h_out + Q / 4 / np.pi / T * exp1(r * r * S / 4 / t / T)


def theis_recovery(r, t, T, S, Q, t_end, S2=None):
    """
    Simulate pumping followed by recovery in a confined aquifer. The well has an infinitesimal radius and extracts water
    at a constant pumping rate during the pumping phase. The pump is shut down completely at the beginning of the
    recovery phase.

    The solution is obtained by applying the superposition principle to the Theis (1935) equation
    (see Chapter 13 in Kruseman & de Ridder, 1990).

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : float
      Aquifer transmissivity [L²/T].
    S : float
      Aquifer storativity [-] during pumping.
    Q : float
      Pumping rate [L³/T] of the well.
    t_end : float
          Time [T] at which the pumping stops and the recovery starts.
    S2 : float, default: None
       Aquifer storativity [-] during recovery. If `S2` is not given, `S2` is set to `S`.

    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r` and times `t`.
      Shape of `s` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.

    References
    ----------
    Theis C.V. (1935). The relation between the lowering of the piezometric surface and the rate and duration of
    discharge of a well using groundwater storage. Transactions of the American Geophysical Union, 2, 519 – 524.

    Kruseman, G.P., de Ridder, N.A. (1990). Analysis and Evaluation of Pumping Test Data (2nd ed). ILRI Publication,
    Wageningen 47, 377 pp.

    """
    h = theis(r=r, t=t, T=T, S=S, Q=Q)
    t = np.array(t)
    b = t > t_end
    if np.any(b):
        h[b, :] += theis(r=r, t=t[b] - t_end, T=T, Q=-Q, S=S if S2 is None else S2)
    return h


def edelman(r, t, T, S, h_in=None, Q=None):
    """
    Calculate the solution for unsteady parallel flow in an aquifer with an infinitely large width (Edelman, 1947).

    At the inner model boundary, a constant head `h_in` or a constant discharge `Q` is defined.
    This means either input parameter `h_in` or input parameter `Q` is assigned, but not both.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : float
      Aquifer transmissivity [L²/T].
    S : float
      Aquifer storativity [-].
    h_in : float
          Constant head [L] at the inner boundary condition.
    Q : float
      Constant discharge [L³/T/L] through the inner model boundary.

    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r` and times `t`.
      Shape of `s` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.

    References
    ----------
    Edelman, J.H. (1947). Over de berekening van grondwaterstroomingen (in Dutch). Thesis, TU Delft, 79pp.
    """
    t, r = np.meshgrid(t, r)
    u = r * np.sqrt(S / 4 / t / T)
    if Q is None:
        return h_in * erfc(u)
    else:
        return 2 * Q * np.sqrt(t / T / S) * (np.exp(-u**2) / np.sqrt(np.pi) - u * erfc(u))


def hantush_jacob(r, t, T, S, Q, c_top, h_top=0.0, ns=12):
    """
    Calculate the solution for unsteady flow to a pumping well in a leaky aquifer (Hantush & Jacob, 1955).
    The well has an infinitesimal radius and extracts water at a constant pumping rate.
    The solution is obtained from numerical inversion of the exact analytical solution in Laplace space
    (see Equation A34 in Appendix A of Louwyck et. al., 2022).

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : float
      Aquifer transmissivity [L²/T].
    S : float
      Aquifer storativity [-].
    Q : float
      Pumping rate [L³/T] of the well.
    c_top : float
          Vertical resistance [T] of the aquitard overlying the aquifer.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.
    ns : int, default: 12
       Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
       Must be a positive, even integer.

    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r` and times `t`.
      Shape of `s` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.

    References
    ----------
    Hantush, M.S., Jacob, C.E. (1955). Non-steady radial flow in an infinite leaky aquifer.
    Transactions American Geophysical Union 36, 95-100.

    Louwyck, A., Vandenbohede, A., Libbrecht, D., Van Camp, M., Walraevens, K. (2022). The radius of influence myth.
    Water 14(2): 149. https://doi.org/10.3390/w14020149.

    """
    r = np.array(r)
    if r.ndim == 0:
        r = r[np.newaxis]
    hp = lambda r, p: h_top / p + Q / 2 / np.pi / T / p * k0(r * np.sqrt(S * p / T + 1 / T / c_top))
    h = [stehfest(lambda p: hp(ri, p), t, ns) for ri in r]
    return np.array(h)


def hemker_steady(r, T, Q, c, c_top, axi=True):
    """
    Calculate the solution for steady well-flow in a leaky two-aquifer system with impervious lower boundary.
    The well has an infinitesimal radius and a separate fully penetrating screen in each aquifer.
    It extracts water at a constant pumping rate.
    The exact analytical solution is obtained applying the method described by Hemker (1984).

    It is also possible to obtain the solution for parallel flow under the same conditions. See input parameter `axi`.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : array_like
      Two-element array with the transmissivity [L²/T] of the upper and lower aquifer.
    Q : array_like
      Two-element array with the pumping rates [L³/T] of the wells located in the upper and lower aquifer.
    c : float
      Vertical resistance [T] of the aquitard separating the two aquifers.
    c_top : float
          Vertical resistance [T] of the aquitard overlying the upper aquifer.
    axi : bool, default: True
        Axi-symmetric flow is simulated if `axi` is `True`, parallel flow otherwise.

    Returns
    -------
    s : ndarray
      Drawdown [L] in each aquifer at distances `r`.
      Shape of `s` is `(2, nr)`, with `nr` the length of `r`.

    References
    ----------
    Hemker, C.J. (1984). Steady groundwater flow in leaky multiple-aquifer systems. Journal of Hydrology 72, 355-374.
    """
    r = np.array(r)
    if r.ndim == 0:
        r = r[np.newaxis]
    L1, L2, L = T[0] * c_top, T[1] * c, T[1] * c_top
    Lsum, Lprod = L1 + L2 + L, 2 * L1 * L2
    S = np.sqrt(Lsum**2 - 2 * Lprod)
    w1, w2 = (Lsum - S) / Lprod, (Lsum + S) / Lprod
    v11, v12, v21, v22 = 1 - w1*L2, 1 - w2*L2, 1.0, 1.0
    q1, q2 = Q[0] / T[0], Q[1] / T[1]
    x1, x2 = r*np.sqrt(w1), r*np.sqrt(w2)
    if axi:
        K1, K2, v = k0(x1), k0(x2), 2 * np.pi
    else:
        K1, K2, v = np.exp(-x1)/np.sqrt(w1), np.exp(-x2)/np.sqrt(w2), 1.0
    K, V = K1 - K2, v * (v12 - v11)
    s1 = (q2*v11*v12 * K - q1 * (v11*K1 - v12*K2)) / V
    s2 = (q2 * (v12*K1 - v11*K2) - q1 * K) / V
    return np.array([s1, s2])


def hemker_unsteady(r, t, T, S, Q, c, axi=True, ns=12):
    """
    Calculate the solution for unsteady flow to a pumping well in the lower aquifer of a confined two-aquifer system.
    The well has an infinitesimal radius and extracts water at a constant pumping rate.
    The solution is obtained from numerical inversion of the exact analytical solution in Laplace space according to
    Hemker (1985).

    It is also possible to obtain the solution for parallel flow under the same conditions. See input parameter `axi`.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : array_like
      Two-element array with the transmissivity [L²/T] of the upper and lower aquifer.
    S : array_like
      Two-element array with the storativity [-] of the upper and lower aquifer.
    Q : float
      Pumping rate [L³/T] of the well located in the lower aquifer.
    c : float
      Vertical resistance [T] of the aquitard separating the two aquifers.
    axi : bool, default: True
        Axi-symmetric flow is simulated if `axi` is `True`, parallel flow otherwise.
    ns : int, default: 12
       Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
       Must be a positive, even integer.

    Returns
    -------
    s : ndarray
      Drawdown [L] in each aquifer at distances `r` and times `t`.
      Shape of `s` is `(2, nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.

    References
    ----------
    Hemker, C.J. (1985). Transient well flow in leaky multiple-aquifer systems. Journal of Hydrology 81, 111-126.
    """
    r, t = np.array(r), np.array(t)
    if r.ndim == 0:
        r = r[np.newaxis]
    if t.ndim == 0:
        t = t[np.newaxis]
    a1, a2 = 1 / T[0] / c, 1 / T[1] / c
    b1, b2 = lambda p: (1 + S[0]*c*p) * a1, lambda p: (1 + S[1]*c*p) * a2
    D = lambda p: np.sqrt((b1(p) - b2(p))**2 + 4*a1*a2)
    d1, d2 = lambda p: (b1(p) + b2(p) - D(p)) / 2, lambda p: (b1(p) + b2(p) + D(p)) / 2
    x1, x2 = lambda p, r: r*np.sqrt(d1(p)), lambda p, r: r*np.sqrt(d2(p))
    if axi:
        K1, K2 = lambda p, r: k0(x1(p, r)), lambda p, r: k0(x2(p, r))
        q = lambda p: Q / 2 / np.pi / T[1] / p
    else:
        K1, K2 = lambda p, r: np.exp(-x1(p, r)) / np.sqrt(d1(p)), lambda p, r: np.exp(-x2(p, r)) / np.sqrt(d2(p))
        q = lambda p: Q / T[1] / p
    u1, u2 = lambda p: d1(p) - b2(p), lambda p: d2(p) - b2(p)
    s1p = lambda p, r: q(p)/D(p)/a2 * u1(p)*u2(p) * (K2(p, r) - K1(p, r))
    s2p = lambda p, r: q(p)/D(p) * (K1(p, r)*u2(p) - K2(p, r)*u1(p))
    s = np.zeros((2, len(r), len(t)))
    for i in range(len(r)):
        s[0, i, :] = stehfest(lambda p: s1p(p, r[i]), t, ns)
        s[1, i, :] = stehfest(lambda p: s2p(p, r[i]), t, ns)
    return s


def hunt_scott(r, t, T, S, Q, c, ns=12):
    """
    Calculate the solution for unsteady flow to a pumping well in the lower aquifer of a two-aquifer system.
    The well has an infinitesimal radius and extracts water at a constant pumping rate.
    The solution is obtained from numerical inversion of the exact analytical solution in Laplace space
    (Hunt & Scott, 2007).

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L] from the well.
    t : array_like
      One-dimensional array with the simulation times [T].
    T : array_like
      Two-element array with the transmissivity [L²/T] of the upper and lower aquifer.
    S : array_like
      Two-element array with the storativity [-] of the upper and lower aquifer.
    Q : float
      Pumping rate [L³/T] of the well located in the lower aquifer.
    c : float
      Vertical resistance [T] of the aquitard separating the two aquifers.
    ns : int, default: 12
       Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
       Must be a positive, even integer.

    Returns
    -------
    s : ndarray
      Drawdown [L] in each aquifer at distances `r` and times `t`.
      Shape of `s` is `(2, nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.

    References
    ----------
    Hunt, B., Scott, D. (2007). Flow to a Well in a Two-Aquifer System. Journal of Hydrologic Engineering 12(2),
    146-155. https://doi.org/10.1061/(ASCE)1084-0699(2007)12:2(146).
    """
    r, t = np.array(r), np.array(t)
    if r.ndim == 0:
        r = r[np.newaxis]
    if t.ndim == 0:
        t = t[np.newaxis]
    tdim = t * T[1] / S[1]
    K, T0, e = 1/c/T[1], T[0]/T[1], S[1]/S[0]
    b, c = lambda p: (p + K + (p/e + K)/T0)/2, lambda p: (p + K * (1 + e)) * p/T0/e
    D = lambda p: np.sqrt(b(p)**2 - c(p))
    d1, d2 = lambda p: b(p) + D(p), lambda p: b(p) - D(p)
    v1, v2 = lambda p: (p + K - d1(p))/K, lambda p: (p + K - d2(p))/K
    L1, L2 = lambda p: 1 + T0*v1(p)**2, lambda p: 1 + T0*v2(p)**2
    K1, K2 = lambda p, r: k0(r*np.sqrt(d1(p)))/L1(p), lambda p, r: k0(r*np.sqrt(d2(p)))/L2(p)
    s1p = lambda p, r: (K1(p, r)*v1(p) + K2(p, r)*v2(p)) / 2/np.pi/p
    s2p = lambda p, r: (K1(p, r) + K2(p, r)) / 2/np.pi/p
    s = np.zeros((2, len(r), len(t)))
    for i in range(len(r)):
        s[0, i, :] = stehfest(lambda p: s1p(p, r[i]), tdim, ns)
        s[1, i, :] = stehfest(lambda p: s2p(p, r[i]), tdim, ns)
    return s * Q / T[1]


def stehfest(F, t, ns=16):
    """
    Stehfest algorithm for numerical inversion of Laplace transforms.

    Parameters
    ----------
    F : callable
      Function that calculates the Laplace transform. It has frequency parameter `p` [1/T] as input
      and returns the Laplace-transform `F(p)`. Input parameter `p` is a one-dimensional numpy array,
      and the returned output is also a one-dimensional numpy array with the same length as `p`.
    t : array_like
      One-dimensional array with the real times `t` [T].
    ns : int, default: 16
       Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
       Must be a positive, even integer.

    Returns
    -------
    f : ndarray
      One-dimensional array with the numerically inverted values `f(t)`. The length of `f` equals the length of `t`.

    References
    ----------
    Stehfest, H. (1970). Algorithm 368: Numerical inversion of Laplace transform. Communication of the ACM 13(1),
    p. 47-49.
    """
    t = np.array(t)
    if t.ndim == 0:
        t = t[np.newaxis]
    nt = len(t)
    ns = int(ns)
    ln2t = log(2) / t
    W = stehfest_weights(ns)
    f = np.zeros(nt)
    for k in range(ns):
        p = ln2t * (k + 1)
        f += W[k] * F(p)
    return f * ln2t


def stehfest_weights(ns):
    """
    Calculate weights required for applying the Stehfest algorithm.

    Called by function `stehfest`.

    Parameters
    ----------
    ns : int
       Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
       Must be a positive, even integer.

    Returns
    -------
    W : ndarray
      One-dimensional array with weights, length of `W` is equal to `ns`.
    """
    fac = lambda x: float(factorial(x))
    ns2 = ns // 2
    W = np.zeros(ns)
    for j in range(1, ns + 1):
        m = min(j, ns2)
        k_0 = (j + 1) // 2
        for k in range(k_0, m + 1):
            W[j - 1] += k ** ns2 * fac(2 * k) / fac(ns2 - k) / fac(k) / fac(k - 1) / fac(j - k) / fac(2 * k - j)
        W[j - 1] *= (-1) ** (ns2 + j)
    return W
