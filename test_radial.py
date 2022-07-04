import matplotlib.pyplot as plt
from analytic.radial import SteadyQ, SteadyH, TransientQ, TransientH
from analytic.special import *


m1 = SteadyH(T=[100, 500, 10], c=[250, 500], r_in=0.1, h_in=[1, 0, 0], r_out=1000)
m2 = TransientH(T=[100, 500, 10], S=[0.1, 1e-3, 1e-4], c=[250, 500], r_in=0.1, h_in=[1, 0, 0], r_out=1000)
r = np.logspace(-1, 3, 100)
t = 1000
plt.semilogx(r, m1.h(r).T, 'k-')
plt.semilogx(r, m2.h(r, t).squeeze().T, 'r:')
plt.show()

m1 = SteadyQ(T=[100, 500, 10], c=[250, 500], Q=[0, 100, 0], r_out=1000)
m2 = TransientQ(T=[100, 500, 10], S=[0.1, 1e-3, 1e-4], c=[250, 500], Q=[0, 100, 0], r_out=1000)
r = np.logspace(-1, 3, 100)
t = 1000
plt.semilogx(r, m1.h(r).T, 'k-')
plt.semilogx(r, m2.h(r, t).squeeze().T, 'r:')
plt.show()


# uniform
T = 100
h_in = 10
r_in = 0.1
r_out = 1000
h_out = 1
r = np.logspace(np.log10(r_in), np.log10(r_out), 10)
m = SteadyH(T=T, h_in=h_in, r_in=r_in, r_out=r_out, h_out=h_out)
plt.semilogx(r, m.h(r).squeeze(), 'k-',
             r, uniform_radial(r=r, r_in=r_in, h_in=h_in, r_out=r_out, h_out=h_out), 'r:')
plt.title("uniform")
plt.show()

# uniform + recharge
N = 1e-4
m = SteadyH(T=T, h_in=h_in, r_in=r_in, r_out=r_out, h_out=h_out, N=N)
plt.semilogx(r, m.h(r).squeeze(), 'k-',
             r, uniform_radial(r=r, r_in=r_in, h_in=h_in, r_out=r_out, h_out=h_out, T=T, N=N), 'r:')
plt.title("uniform + recharge")
plt.show()

# Thiem
T = 100
Q = 200
r_out = 1000
h_out = 10
r = np.logspace(-1, np.log10(r_out), 10)
m = SteadyQ(T=T, Q=Q, r_out=r_out, h_out=h_out)
plt.semilogx(r, m.h(r).squeeze(), 'k-',
             r, thiem_dupuit(r=r, T=T, Q=Q, r_out=r_out, h_out=h_out), 'r:')
plt.title("Thiem")
plt.show()

# de Glee
T = 1000
Q = 5000
c_top = 1000
r_in = 0
r_out = np.inf
r = np.logspace(-1, 5, 100)
m = SteadyQ(T=T, Q=Q, c_top=c_top, r_in=r_in, r_out=r_out)
plt.semilogx(r, m.h(r).squeeze(), 'k-',
             r, deglee(r=r, T=T, Q=Q, c_top=c_top), 'r:')
plt.title("de Glee")
plt.show()

# de Glee (general)
T = 1000
Q = 5000
c_top, c_bot = 1000, 2000
h_top, h_bot = 10, 1
r = np.logspace(-1, 5, 100)
m = SteadyQ(T=T, Q=Q, c_top=c_top, c_bot=c_bot, h_top=h_top, h_bot=h_bot)
plt.semilogx(r, m.h(r).squeeze(), 'k-',
             r, deglee(r=r, T=T, Q=Q, c_top=c_top, h_top=h_top, c_bot=c_bot, h_bot=h_bot), 'r:')
plt.title("de Glee (general)")
plt.show()

# de Glee - Bruggeman
T = 200
Q = 10
c_top = 500
r_in = 10
h_top = 15
r = np.logspace(np.log10(r_in), 5, 100)
m = SteadyQ(T=T, Q=Q, c_top=c_top, r_in=r_in, h_top=h_top)
plt.semilogx(r, m.h(r).squeeze(), 'k-',
             r, deglee(r=r, T=T, Q=Q, r_in=r_in, c_top=c_top, h_top=h_top), 'r:')
plt.title("de Glee - Bruggeman")
plt.show()

# Bredehoeft's island
T = 400
Q = -10
N = 1e-4
r_in = 0.1
r_out = 1000
h_out = 10
r = np.logspace(np.log10(r_in), np.log10(r_out), 1000)
m = SteadyQ(T=T, Q=Q, r_in=r_in, r_out=r_out, N=N, h_out=h_out)
plt.semilogx(r, m.h(r).squeeze(), 'k-',
             r, bredehoeft(r=r, T=T, N=N, Q=Q, r_in=r_in, r_out=r_out, h_out=h_out), 'r:')
plt.title("Bredehoeft's island")
plt.show()

# Theis
T = 100
S = 0.2
Q = -500
t = 1
h_out = 100
r = np.logspace(-1, 4, 100)
m = TransientQ(T=T, S=S, Q=Q, h_out=h_out)
plt.semilogx(r, m.h(r, t).squeeze(), 'k-',
             r, theis(r=r, t=t, T=T, S=S, Q=Q, h_out=h_out).squeeze(), 'r:')
plt.title("Theis")
plt.show()

# Hantush-Jacob
T = 750
S = 1e-4
c_top = 2350
Q = -1500
h_top = 50
t = np.logspace(-5, 3, 100)
r = 0.1
m = TransientQ(T=T, S=S, Q=Q, c_top=c_top, h_top=h_top, h_out=h_top)
plt.semilogx(t, m.h(r, t).squeeze(), 'k-',
             t, hantush_jacob(r=r, t=t, T=T, S=S, Q=Q, c_top=c_top, h_top=h_top).squeeze(), 'r:')
plt.title("Hantush-Jacob")
plt.show()

# Hemker steady (2 layers)
T = [100, 500]
c_top, c = 1e4, 500
Q = [50, 100]
r = np.logspace(-1, 5, 100)
m = SteadyQ(T=T, Q=Q, c_top=c_top, c=c)
plt.semilogx(r, m.h(r).T, 'k-',
             r, hemker_steady(r=r, T=T, c_top=c_top, c=c, Q=Q).T, 'r:')
plt.title("Hemker steady 2 layers")
plt.show()

# Hemker unsteady (2 layers)
T = [250, 750]
S = [1e-3, 1e-4]
c = 1000
Q = 500
t = np.logspace(-5, 3, 100)
r = 0.03
m = TransientQ(T=T, S=S, Q=[0, Q], c=c)
plt.semilogx(t, m.h(r, t).squeeze().T, 'k-')
plt.semilogx(t, hemker_unsteady(r=r, t=t, T=T, S=S, c=c, Q=Q).squeeze().T, 'r:')
plt.title("Hemker steady 2 layers")
plt.show()

# Hunt & Scott
T = [100, 500]
S = [0.1, 1e-3]
c = 200
Q = 300
t = np.logspace(-5, 3, 100)
r = 0.5
m = TransientQ(T=T, S=S, Q=[0, Q], c=c)
plt.semilogx(t, m.h(r, t).squeeze().T, 'k-')
plt.semilogx(t, hunt_scott(r=r, t=t, T=T, S=S, c=c, Q=Q).squeeze().T, 'r:')
plt.title("Hunt & Scott")
plt.show()

# Bakker
T = [750, 1000]
c = 1450
N = 1e-3
Q = [1500, 3000]
r_out = 5e3
h_out = 100
r = np.logspace(-2, np.log10(r_out), 1000)
m = SteadyQ(T=T, c=c, Q=Q, N=[N, 0], r_out=r_out, h_out=[h_out, h_out])
plt.semilogx(r, m.h(r).T, 'k-',
             r, bakker(r=r, T=T, c=c, N=N, Q=Q, r_out=r_out, h_out=h_out).T, 'r:')
plt.title('Bakker')
plt.show()


