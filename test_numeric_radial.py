import numpy as np
import matplotlib.pyplot as plt
from analytic.radial import SteadyQ, TransientQ
from numeric.radial import Model
from analytic.superposition import VariableQ


# Thiem
r_in = 0.1
r_out = 1e3
nr = 100
rb = np.logspace(np.log10(r_in), np.log10(r_out), nr)
rb = np.insert(rb, len(rb), rb[-1] + 1e-3)
T = 100
Q = 100

mn = Model()
grid = mn.add_grid(rb=rb, D=1, constant=np.hstack((np.zeros(nr - 1), 1)))
period = mn.add_period()
period.add_kh(kh=T)
period.add_q(np.hstack((Q, np.zeros(nr - 1))))
mn.solve()

ma = SteadyQ(T=T, Q=Q, r_in=r_in, r_out=r_out)

plt.semilogx(grid.r, mn.h.squeeze(), 'k-',
             grid.r, ma.h(grid.r).squeeze(), 'r:')
plt.show()

# 3 layers in leaky system - steady flow
r_in = 1e-3
r_out = 1e5
nr = 100
rb = np.logspace(np.log10(r_in), np.log10(r_out), nr + 1)
T = [100, 500, 250]
c = [750, 1000]
c_top = 2000
Q = [0, 350, 0]

mn = Model()
grid = mn.add_grid(rb=rb, D=[1, 1, 1, 1], constant=[1, 0, 0, 0])
period = mn.add_period()
period.add_kh(kh=[0] + T)
period.add_kv(cv=[c_top] + c)
q = np.zeros((grid.nl, grid.nr))
q[1:, 0] = Q
period.add_q(q)
mn.solve()

ma = SteadyQ(T=T, c=c, c_top=c_top, Q=Q, r_in=r_in, r_out=r_out)

plt.semilogx(grid.r, mn.h.squeeze().T, 'k-',
             grid.r, ma.h(grid.r).squeeze().T, 'r:')
plt.show()

# same problem but now using head-dependent flux boundary condition
mn = Model()
grid = mn.add_grid(rb=rb, D=[1, 1, 1])
period = mn.add_period()
period.add_kh(kh=T)
period.add_kv(cv=c)
q = np.zeros((grid.nl, grid.nr))
q[:, 0] = Q
period.add_q(q)
period.add_hdep(dependent=[1, 0, 0], cdep=[c_top, 0, 0])
mn.solve()

plt.semilogx(grid.r, mn.h.squeeze().T, 'k-',
             grid.r, ma.h(grid.r).squeeze().T, 'r:')
plt.show()

# infiltration
r_in = 0.001
r_out = 1e3
nr = 500
rb = np.logspace(np.log10(r_in), np.log10(r_out), nr)
rb = np.insert(rb, len(rb), rb[-1] + 1e-3)
T = 100
N = 1e-4

mn = Model()
grid = mn.add_grid(rb=rb, D=1, constant=np.hstack((np.zeros(nr - 1), 1)))
period = mn.add_period()
period.add_kh(kh=T)
period.add_q(N * grid.hs)
mn.solve()

ma = SteadyQ(T=T, Q=0, N=N, r_in=rb[0], r_out=rb[-1])

plt.semilogx(grid.r, mn.h.squeeze(), 'k-',
             grid.r, ma.h(grid.r).squeeze(), 'r:')
plt.show()

# pumping + infiltration + nonzero constant head
r_in = 0.01
r_out = 1e3
h_out = [10, 8]
nr = 1000
rb = np.logspace(np.log10(r_in), np.log10(r_out), nr)
rb = np.insert(rb, len(rb), rb[-1] + 1e-3)
T = [100, 1000]
c = 500
Q = np.array([0, -100])
N = 1e-4

mn = Model()
constant = np.zeros((2, nr), dtype=bool)
constant[:, -1] = True
grid = mn.add_grid(rb=rb, D=[1, 1], constant=constant)
period = mn.add_period()
period.add_kh(kh=T)
period.add_kv(cv=c)
q = np.zeros((grid.nl, grid.nr))
q[0, :] = N * grid.hs
q[:, 0] = Q
period.add_q(q)
period.add_hc(hc=h_out)
mn.solve()

ma = SteadyQ(T=T, Q=Q, c=c, N=[N, 0], r_in=r_in, r_out=r_out, h_out=h_out)

plt.semilogx(grid.r, mn.h.squeeze().T, 'k-',
             grid.r, ma.h(grid.r).squeeze().T, 'r:')
plt.show()

# theis
r_in = 1e-5
r_out = 1e7
nr = 500
rb = np.logspace(np.log10(r_in), np.log10(r_out), nr + 1)
T = 100
S = 0.1
Q = 100
t = np.logspace(-5, 3, 100)

mn = Model()
grid = mn.add_grid(rb=rb, D=1)
period = mn.add_period(t)
period.add_kh(kh=T)
period.add_ss(ss=S)
period.add_q(np.hstack((Q, np.zeros(nr - 1))))
mn.solve()

ma = TransientQ(T=T, S=S, Q=Q)

i = [10, 100, 200]
plt.semilogx(t, mn.h[0, i, 1:].squeeze().T, 'k-',
             t, ma.h(grid.r[i], t).squeeze().T, 'r:')
plt.show()

# Theis recovery
r_in = 1e-5
r_out = 1e7
nr = 500
rb = np.logspace(np.log10(r_in), np.log10(r_out), nr + 1)
T = 100
S = 0.1
Q = 100
t = np.logspace(-5, 0, 100)

mn = Model()
grid = mn.add_grid(rb=rb, D=1)
pumping = mn.add_period(t)
pumping.add_kh(kh=T)
pumping.add_ss(ss=S)
pumping.add_q(np.hstack((Q, np.zeros(nr - 1))))
recovery = mn.add_period(t)
recovery.add_q(0.0)
mn.solve()

ma = VariableQ(TransientQ(T=T, S=S, Q=Q), t=1, Q=0)

i = [10, 100, 200]
plt.plot(mn.t, mn.h[0, i, :].squeeze().T, 'k-',
         mn.t[1:], ma.h(grid.r[i], mn.t[1:]).squeeze().T, 'r:')
plt.show()

# step-drawdown
r_in = 1e-3
r_out = 1e6
nr = 100
rb = np.logspace(np.log10(r_in), np.log10(r_out), nr + 1)
T = [100, 500]
c = 100
S = [0.1, 1e-4]
Q2 = np.array([100, 200, 300, 400])
t = np.logspace(-5, 0, 100)

mn = Model()
grid = mn.add_grid(rb=rb, D=[1, 1])
step = mn.add_period(t)
step.add_kh(kh=T)
step.add_kv(cv=c)
step.add_ss(ss=S)
q = np.zeros((grid.nl, grid.nr))
q[1, 0] = Q2[0]
step.add_q(q)
for i in range(1, 4):
    step = mn.add_period(t)
    q[1, 0] = Q2[i]
    step.add_q(q)
mn.solve()

Q = np.zeros((2, 4))
Q[1, :] = Q2
ma = VariableQ(TransientQ(T=T, c=c, S=S, Q=Q[:, 0]), t=[1, 2, 3], Q=Q[:, 1:])

i = 20
plt.plot(mn.t, mn.h[:, i, :].squeeze().T, 'k-',
         mn.t[1:], ma.h(grid.r[i], mn.t[1:]).squeeze().T, 'r:')
plt.show()
