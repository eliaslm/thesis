from fenics import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dolfin.fem.norms as norms


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

    
def plot_states_and_control(U, Q, P, T, num_steps, kw_q=None, num_plots=5):
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
        if kw_q != None:
            pp = plot(q_n, **kw_q)
        else:
            pp = plot(q_n)
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
        
        plt.tight_layout()
        plt.show()
        print("Umax = {}".format(u_n.vector().max()))
        print("Umin = {}".format(u_n.vector().min()))
        

def solve_adjoint_eq(p_end, U, Q, V, T, num_steps, params):
    a_const = params["a"]
    u_d = params["u_d"]
    t = T
    dt = T/num_steps

    # Get initial values
    p_n = interpolate(p_end, V)
    u = U[-1]
    q = Q[-1]
    u_d_n = u_d[-1]

    # Define variational problem
    p = TrialFunction(V)
    phi = TestFunction(V)

    a = p*phi*dx + dt*(dot(grad(p), grad(phi)) + p*(2*u-a_const+q)*phi)*dx
    l = p_n*phi*dx + 2*dt*(u - u_d_n)*phi*dx

    # Iterating in time
    p = Function(V)

    P = [interpolate(p_n, V)]

    for i in range(1, num_steps + 1):
        t -= dt
        
        solve(a == l, p)
        p_n.assign(p)
        
        
        P.append(interpolate(p_n, V))
        u = U[len(U) - i - 1]
        q = Q[len(Q) - i - 1]
        u_d_n = u_d[len(u_d) -i - 1]

    P.reverse()

    return np.asarray(P)


def j(U, Q, mesh, T, num_steps, params):
    """
    Function to evaluate the cost functional given functions u, q.
    """
    
    # Define parameters for cost functional
    alpha = params["alpha"]
    u_d = params["u_d"]
    
    # Compute integrals with time
    I1 = 0
    I3 = 0
    
    t = 0
    dt = T/num_steps
    for i in range(num_steps + 1):
        I1_int = assemble((U[i] - u_d[i])*(U[i] - u_d[i])*dx(mesh))
        I3_int = assemble(Q[i]*Q[i]*dx(mesh))
        
        if i == 0 or i == num_steps:
            I1_int *= 0.5
            I3_int *= 0.5
        
        I1 += I1_int
        I3 += I3_int
        
        t += dt
    
    
    I1 *= dt
    I3 *= dt*alpha/2
    
    # Compute end time integral
    
    print("Cost Functional Data")
    print("I1: {}".format(I1))
    print("I3: {}".format(I3))
    print()
    
    return I1 + I3


def j_reduced(Q, mesh, T, V, num_steps, params, u_0):
    U = solve_state_eq(u_0, Q, V, T, num_steps, params)
    return j(U, Q, mesh, T, num_steps, params)


def dj(U, Q, P, T, num_steps, params):
    """
    Function to compute gradient of cost functional.
    """
    
    # Parameters
    alpha = params["alpha"]
    
    grad = []
    
    t = 0
    dt = T/num_steps
    for i in range(num_steps + 1):
        grad.append(alpha*Q[i] - U[i]*P[i])
        t += dt
    
    grad = np.asarray(grad)
    return grad


# --- Function to compute norm of gradient ---
def grad_norm(grad, V, T, num_steps):
    norm_ = 0
    dt = T/num_steps
    
    for i in range(len(grad)):
        f = project(grad[i], V)
        int_ = norms.norm(f, "l2")**2
        
        if i == 0 or i == len(grad) - 1:
            int_ *= 0.5
            
        norm_ += int_
    
    norm_ *= dt
    return np.sqrt(norm_)


# --- Function to perform backtracking line search ---
def armijo(v_k, U, Q, P, V, mesh, T, num_steps, params, u_0):
    alpha = 1.0
    rho = 0.5
    c = 0.05
    max_iter = 10
    
    phi_0 = j(U, Q, mesh, T, num_steps, params)
    phi_k = j_reduced(Q + alpha*v_k, mesh, T, V, num_steps, params, u_0)
    m = -1*grad_norm(v_k, V, T, num_steps)**2
    it = 0
    while phi_k >= phi_0 + alpha*c*m and it < max_iter:
        alpha *= rho
        phi_k = j_reduced(Q + alpha*v_k, mesh, T, V, num_steps, params, u_0)
        it += 1
        
    return [Constant(alpha), phi_k, it]


def Max(a, b):
    return (a + b + abs(a-b))/Constant(2)


def Min(a, b):
    return (a + b - abs(a-b))/Constant(2)


"""
def project_to_admissible(Q, q_min, q_max, V):
    # for i in range(len(Q)):
    #    proj = Max(q_min, Min(Q[i], q_max))
    #    Q[i] = project(proj, V)
    Q_trunc = np.asarray([None]*len(Q))
    for i in range(len(Q)):
        foo = project(Q[i], V)
        foo.vector()[foo.vector() < q_min] = q_min
        foo.vector()[foo.vector() > q_max] = q_max
        
        Q_trunc[i] = foo
    return Q_trunc
"""

def project_to_admissible(Q, q_min, q_max, V):
    for i in range(len(Q)):
        proj = Max(q_min, Min(Q[i], q_max))
        Q[i] = project(proj, V)
    return Q


def pg_print(j_curr, j_prev, v_k_norm, it, armijo_it, s_k):
    print("-----------------------------------")
    print("Projected gradient iteration no. {} \n".format(it))
    
    print("Armijo search ended after {} iterations".format(armijo_it))
    print("Step length found: s_k = {}".format(float(s_k)))
    print("New function value: {}".format(j_curr))
    print("Decrease in function value: {}".format(j_prev - j_curr))
    print("Norm of gradient: {}".format(v_k_norm))
    print("-----------------------------------\n")