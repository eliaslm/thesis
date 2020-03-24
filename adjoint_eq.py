from fenics import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def solve_adjoint_eq(p_end, U, Q, V, T, num_steps, params):
	rho = params["rho"]
	a_const = params["a"]
	t = T
	dt = T/num_steps

	# Get initial values
	p_n = interpolate(p_end, V)
	u = U[-1]
	q = Q[-1]

	# Define variational problem
	p = TrialFunction(V)
	phi = TestFunction(V)

	a = p*phi*dx + dt*(dot(grad(p), grad(phi)) + p*(2*u-a_const+q)*phi)*dx
	l = p_n*phi*dx - dt*exp(-1*rho*t)*q*phi*dx

	# Iterating in time
	p = Function(V)

	P = [interpolate(p_n, V)]

	for i in range(1, num_steps + 1):
	    solve(a == l, p)
	    p_n.assign(p)
	    
	    P.append(interpolate(p_n, V))
	    u = U[len(U) - i - 1]
	    q = Q[len(Q) - i - 1]

	P.reverse()

	return P