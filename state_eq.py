from fenics import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def solve_state_eq(u_0, Q, V, T, num_steps, params):
	dt = T/num_steps
	a_const = params["a"]

	# Get initial value
	u_n = interpolate(u_0, V)
	q = Q[0]

	# Define variational problem
	u = TrialFunction(V)
	phi = TestFunction(V)

	a = u*phi*dx + dt*(dot(grad(u), grad(phi))*dx + u_n*u*phi*dx - a_const*u*phi*dx + q*u*phi*dx)
	l = u_n*phi*dx

	# Iterating in time
	u = Function(V)
	t = 0

	U = [interpolate(u_0, V)]

	for i in range(1, num_steps + 1):
	    t += dt
	    
	    solve(a == l, u)
	    u_n.assign(u)
	    q = Q[i]
	    
	    U.append(interpolate(u_n, V))

	return U
	    