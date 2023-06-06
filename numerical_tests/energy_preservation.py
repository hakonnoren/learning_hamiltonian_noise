import autograd
import autograd.numpy as np

import numpy.linalg as la
import time

import matplotlib.pyplot as plt
import matplotlib as mpl



############ Defining double pendulum problem #############
def H(x):
    return (1/2*x[2]**2 + x[3]**2 - x[2]*x[3]*np.cos(x[0]-x[1]))/(1 + np.sin(x[0]-x[1])**2) - 2*np.cos(x[0]) - np.cos(x[1])
S = np.array([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]], np.float64)

DH = autograd.grad(H)
DDH = autograd.hessian(H)

def f(x):
    return np.matmul(S,DH(x))



############ Differentials for DGM and EDRK ############

Df = autograd.jacobian(f)


def Dff(x):
    return np.matmul(Df(x),f(x))
def DfDf(x):
    return np.matmul(Df(x),Df(x))
def DDff(x): # D^2 f (f, )
    return autograd.jacobian(Dff)(x) - DfDf(x)

def DDGH(x1, x2): # The derivative of the discrete gradient w.r.t. the second argument
    g2 = lambda xn: DGH(x1, xn)
    return autograd.jacobian(g2)(x2)

def DGH(x1, x2):
    xh = (x1+x2)/2
    xd = x2-x1
    if (x2==x1).all():
        return DH(xh)
    else:
        return DH(xh) + (H(x2)-H(x1) - np.matmul(DH(xh),xd))/np.dot(xd,xd)*xd

########## RK4, MIRK4, MIMP4, DGM4 ##############

def rk4(f, x1, dt):
    k1 = f(x1)
    k2 = f(x1+.5*dt*k1)
    k3 = f(x1+.5*dt*k2)
    k4 = f(x1+dt*k3)
    return 1/6*(k1+2*k2+2*k3+k4)

def mirk4(f, x1, x2, dt, df=None, Df=None):
    if df == None:
        df = f
    xh = (x1+x2)/2
    z = xh - dt/8*(f(x2)-f(x1))
    return 1/6*(df(x1)+df(x2))+2/3*df(z)


def mimp4(f, x1, x2, dt, df=None, Df=None): # Modified implicit midpoint method
    if df == None:
        df = f
    if Df == None:
        print('Error: Missing argument Df')
    xh = (x1+x2)/2
    return df(xh) + dt**2/12*(-np.matmul(Df(xh), np.matmul(Df(xh),df(xh))) + 1/2*np.matmul(DDff(xh),df(xh)))

def quasi_newton_dgm4(x, xn, dt, tol, max_iter):
    I = np.eye(x.shape[0])
    def St(x, xn, dt):
        xh = (x+xn)/2
        z1 = (xh+xn)/2
        z2 = (xh+x)/2
        SH = np.matmul(S,DDH(xh))
        SHS = np.matmul(SH,S)
        SHSHS = np.matmul(SH,SHS)
        DDGH1 = DDGH(x,1/3*x+2/3*xn) # Jacobian of discrete gradient DGH(x,1/3*x+2/3*xn)  w.r.t. (1/3*x+2/3*xn)
        Q1 = .5*(DDGH1.T-DDGH1)
        DDGH2 = DDGH(xn,1/3*xn+2/3*x)
        Q2 = .5*(DDGH2.T-DDGH2)
        Q = .5*(Q1-Q2)
        SQS = np.dot(S,np.matmul(Q,S))
        return (S + dt*SQS - 1/12*dt**2*SHSHS)
    F = lambda x_hat: 1/dt*(x_hat-x) - np.matmul(St(x,x_hat,dt), DGH(x,x_hat))
    J = lambda x_hat: 1/dt*I - np.matmul(St(x,x_hat,dt), DDGH(x,x_hat))
    err = la.norm(F(xn))
    it = 0
    while err > tol:
        xn = xn - la.solve(J(xn),F(xn))
        err = la.norm(F(xn))
        it += 1
        if it > max_iter:
            break
    return xn

def quasi_newton(integrator, x, xn, f, Df, dt, tol, max_iter):
    '''
    Integrating one step of the ODE x_t = f, from x to xn,
    with an integrator that we assume to be failry similar to
    the implicit midpoint rule
    Using a quasi-Newton method (i.e. with an approximated
    Jacobian that is exact for the implicit midpoint rule) to
    find xn
    '''
    I = np.eye(x.shape[0])
    F = lambda xn: 1/dt*(xn-x) - integrator(f, x, xn, dt, df=None, Df=Df)
    J = lambda xn: 1/dt*I - 1/2*integrator(f, x, xn, dt, df=Df, Df=Df)
    err = la.norm(F(xn))
    it = 0
    while err > tol:
        xn = xn - la.solve(J(xn),F(xn))
        err = la.norm(F(xn))
        it += 1
        if it > max_iter:
            break
    return xn




##################### Compute errors #######################

def get_ref_solution(f,rk4,x0,N=1000,T=500):
    dt = T/N
    ts = np.linspace(0,T,N+1, np.float64)
    xs = np.zeros((x0.shape[0],N+1), np.float64)
    xs[:,0] = x0

    Nref = 10*N
    dtref = T/Nref
    xsref = np.zeros((x0.shape[0],Nref+1), np.float64)
    xsref[:,0] = x0

    x = x0
    for i in range(Nref):
        x = x + dtref*rk4(f, x, dtref)
        xsref[:,i+1] = x
    xsref = xsref[:,0::10]

    return xsref

def get_errors_rk4(dt,x0,N,xsref):
    x = x0
    xs = np.zeros((x0.shape[0],N+1), np.float64)
    energies_rk4 = np.zeros(N+1, np.float64)
    energies_rk4[0] = H(x0)
    for i in range(N):
        x = x + dt*rk4(f,x,dt)
        xs[:,i+1] = x
        energies_rk4[i+1] = H(x)
    errors_rk4 = 1/xs.shape[0]*np.sqrt(np.sum((xs-xsref)**2, axis=0))

    return errors_rk4,energies_rk4

def get_errors_mirk4(dt,x0,N,xsref):
    x = x0
    xs = np.zeros((x0.shape[0],N+1), np.float64)
    energies_mirk4 = np.zeros(N+1, np.float64)
    energies_mirk4[0] = H(x0)
    for i in range(N):
        x = quasi_newton(mirk4, x, x, f, Df, dt, 1e-12, 5)
        xs[:,i+1] = x
        energies_mirk4[i+1] = H(x)
    errors_mirk4 = 1/xs.shape[0]*np.sqrt(np.sum((xs-xsref)**2, axis=0))

    return errors_mirk4,energies_mirk4

def get_errors_mimp4(dt,x0,N,xsref):
    x = x0
    xs = np.zeros((x0.shape[0],N+1), np.float64)
    energies_mimp4 = np.zeros(N+1, np.float64)
    energies_mimp4[0] = H(x0)
    for i in range(N):
        x = quasi_newton(mimp4, x, x, f, Df, dt, 1e-12, 5)
        xs[:,i+1] = x
        energies_mimp4[i+1] = H(x)
    errors_mimp4 = 1/xs.shape[0]*np.sqrt(np.sum((xs-xsref)**2, axis=0))

    return errors_mimp4,energies_mimp4

def get_errors_dgm4(dt,x0,N,xsref):
    x = x0
    xs = np.zeros((x0.shape[0],N+1), np.float64)
    energies_dgm4 = np.zeros(N+1, np.float64)
    energies_dgm4[0] = H(x0)
    for i in range(N):
        x = quasi_newton_dgm4(x, x, dt, 1e-15, 30)
        xs[:,i+1] = x
        energies_dgm4[i+1] = H(x)
    errors_dgm4 = 1/xs.shape[0]*np.sqrt(np.sum((xs-xsref)**2, axis=0))

    return errors_dgm4,energies_dgm4