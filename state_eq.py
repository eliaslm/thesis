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

    return np.asarray(U)

    
def plot_states_and_control(U, Q, P, T, num_steps, num_plots=5):
    '''
        Function to plot states along with corresponding
        control and adjoint for each time.
    '''
    
    assert len(U) == len(Q) and len(U) == len(P),  "Lengths of arrays do not match"
    dt = T/num_steps
    indx = np.linspace(0, len(U)-1, num_plots, dtype=int)
    
    for i in indx:
        u_n = U[i]
        q_n = Q[i]
        p_n = P[i]
        t = i*dt
        
        if i == 0:
            utitle = "Initial state"
            qtitle = "Initial control"
            ptitle = "Initial adjoint state"
        else:
            utitle = "State at t = {} s".format(round(t, 2))
            qtitle = "Control at t = {} s".format(round(t, 2))
            ptitle = "Adjoint state at t = {} s".format(round(t, 2))
        
        plt.figure(figsize=(10, 10))
        
        # First plot is of the state
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_title(utitle)
        p = plot(u_n) #, **kw_u)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        for c in p.collections:
            c.set_edgecolor("face")
        plt.colorbar(p, cax=cax)
        
        # Second plot is of the control
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_title(qtitle)
        pp = plot(q_n) #, **kw_q)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        for c in pp.collections:
            c.set_edgecolor("face")
        plt.colorbar(pp, cax=cax)
        
        # Third plot is of the adjoint
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_title(ptitle)
        ppp = plot(p_n) #, **kw_q)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        for c in pp.collections:
            c.set_edgecolor("face")
        plt.colorbar(ppp, cax=cax)
        
        plt.show()
        print("Umax = {}".format(u_n.vector().max()))
        print("Umin = {}".format(u_n.vector().min()))
        
