import matplotlib.pyplot as plt
from analytic.parallel import *
from analytic.special import *

# Darcy
T = 100
Q = 200
r_in = 1
r_out = 1000
h_out = 10
r = np.linspace(r_in, r_out, 10)
m = SteadyQ(T=T, Q=Q, r_out=r_out, h_out=h_out)
plt.plot(r, m.h(r).squeeze(), 'k-',
         r, darcy(r=r, T=T, Q=Q, r_out=r_out, h_out=h_out), 'r:')
plt.title("Darcy")
plt.show()

# uniform
T = 100
h_in = 10
r_in = 0.1
r_out = 1000
h_out = 1
r = np.linspace(r_in, r_out, 50)
m = SteadyH(T=T, h_in=h_in, r_in=r_in, r_out=r_out, h_out=h_out)
plt.plot(r, m.h(r).squeeze(), 'k-',
         r, uniform_parallel(r=r, r_in=r_in, h_in=h_in, r_out=r_out, h_out=h_out), 'r:')
plt.title("uniform")
plt.show()

# uniform + recharge
N = 1e-2
m = SteadyH(T=T, h_in=h_in, r_in=r_in, r_out=r_out, h_out=h_out, N=N)
plt.plot(r, m.h(r).squeeze(), 'k-',
         r, uniform_parallel(r=r, r_in=r_in, h_in=h_in, r_out=r_out, h_out=h_out, T=T, N=N), 'r:')
plt.title("uniform + recharge")
plt.show()

# Edelman H
T = 50
S = 0.25
h_in = -1.5
h_out = -15
r = np.linspace(0, 1000, 1000)
t = 5
m = TransientH(T=T, S=S, h_in=h_in, h_out=h_out)
plt.plot(r, m.h(r, t).squeeze(), 'k-',
         r, edelman(r=r, t=t, T=T, S=S, h_in=h_in, h_out=h_out).squeeze(), 'r:')
plt.title("Edelman H")
plt.show()

# Edelman H
T = 1500
S = 1e-3
H = -5
r = 1
t = np.logspace(-5, 3, 100)
m = TransientH(T=T, S=S, h_in=H)
plt.semilogx(t, m.h(r, t).squeeze(), 'k-',
             t, edelman(r=r, t=t, T=T, S=S, h_in=H).squeeze(), 'r:')
plt.title("Edelman H")
plt.show()

# Edelman Q
T = 500
S = 0.005
Q = 10
r = np.linspace(0, 1000, 1000)
h_out = 10
t = 5
m = TransientQ(T=T, S=S, Q=Q, h_out=10)
plt.plot(r, m.h(r, t).squeeze(), 'k-',
         r, edelman(r=r, t=t, T=T, S=S, Q=Q, h_out=10).squeeze(), 'r:')
plt.title("Edelman Q")
plt.show()

# Edelman Q
T = 150
S = 1e-1
Q = -5
r = 0
t = np.logspace(-5, 3, 100)
m = TransientQ(T=T, S=S, Q=Q)
plt.semilogx(t, m.h(r, t).squeeze(), 'k-',
             t, edelman(r=r, t=t, T=T, S=S, Q=Q).squeeze(), 'r:')
plt.title("Edelman Q")
plt.show()

# Hemker steady (2 layers)
T = [100, 500]
c_top, c = 1e4, 500
Q = [50, 100]
r = np.logspace(-1, 5, 100)
m = SteadyQ(T=T, Q=Q, c_top=c_top, c=c)
plt.semilogx(r, m.h(r).T, 'k-',
             r, hemker_steady(r=r, T=T, c_top=c_top, c=c, Q=Q, axi=False).T, 'r:')
plt.title("Hemker steady 2 layers")
plt.show()

# Hemker unsteady (2 layers)
T = [100, 500]
S = [0.1, 1e-3]
c = 500
Q = 10
r = np.logspace(-1, 5, 100)
t = 10
m = TransientQ(T=T, S=S, Q=[0, Q], c=c)
plt.semilogx(r, m.h(r, t).squeeze().T, 'k-',
             r, hemker_unsteady(r=r, t=t, T=T, S=S, c=c, Q=Q, axi=False).squeeze().T, 'r:')
plt.title("Hemker unsteady 2 layers")
plt.show()


m1 = SteadyH(T=[100, 500, 10], c=[250, 500], h_in=[1, 0, 0], r_out=1000)
m2 = TransientH(T=[100, 500, 10], S=[0.1, 1e-3, 1e-4], c=[250, 500], h_in=[1, 0, 0], r_out=1000)
r = np.linspace(0, 1000, 100)
t = 1000
plt.plot(r, m1.h(r).T, 'k-')
plt.plot(r, m2.h(r, t).squeeze().T, 'r:')
plt.show()

m1 = SteadyQ(T=[100, 500, 10], c=[250, 500], Q=[0, 100, 0], r_out=1000)
m2 = TransientQ(T=[100, 500, 10], S=[0.1, 1e-3, 1e-4], c=[250, 500], Q=[0, 100, 0], r_out=1000)
r = np.linspace(0, 1000, 100)
t = 1000
plt.plot(r, m1.h(r).T, 'k-')
plt.plot(r, m2.h(r, t).squeeze().T, 'r:')
plt.show()

m1 = SteadyQ(T=100, c_top=100, c_bot=50, Q=10, r_out=1000)
m2 = TransientQ(T=100, S=0.1, c_top=100, c_bot=50, Q=10, r_out=1000)
r = np.linspace(0, 1000, 100)
t = 10000
plt.plot(r, m1.h(r).T, 'k-')
plt.plot(r, m2.h(r, t).squeeze().T, 'r:')
plt.show()
