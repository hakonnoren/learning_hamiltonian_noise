from .integrator_base import Integrator
import torch
import numpy as np


class RK(Integrator):

    def __init__(self,A,b):
        self.A = A
        self.b = b
        self.s = A.shape[0]

class MIRK(RK):
    def __init__(self,D=None,v=None,b=None,type=None):

        if type == 'mirk2':
            v,b,D = get_MIRK_2()

        elif type == 'mirk3':
            v,b,D = get_MIRK_3()
        elif type == 'mirk4':
            v,b,D = get_MIRK_4()
        elif type == 'mirk5':
            v,b,D = get_MIRK_5()
        elif type == 'mirk6':
            v,b,D = get_MIRK_6()
        elif type == 'rk4':
            v,b,D = get_RK4()

        self.type = type
        A = D + np.outer(v,b)
        self.v = v
        self.D = D

        super().__init__(A,b)
        self.set_torch_params()

    def set_torch_params(self):
        self.A_t = torch.from_numpy(self.A)
        self.D_t = torch.from_numpy(self.D)
        self.b_t = torch.from_numpy(self.b)
        self.v_t = torch.from_numpy(self.v)

    def J_t(self,y):
        d = y.shape[-1]//2
        y_q,y_p = torch.split(y,d,dim=-1)
        Jy = torch.cat([y_p,-y_q],axis=-1)
        return Jy

    def f_hat(self,y0,y1,hamiltonian,h):
        if len(y0.shape) == 1:
            y0 = y0.unsqueeze(0)
        if torch.any(self.v_t):
            if len(y1.shape) == 1:
                y1 = y1.unsqueeze(0)
        else:
            y1 = torch.zeros_like(y0)

        K = torch.zeros((y0.shape) + tuple([self.s]),dtype=torch.float64)
    
        def f(y):
            return hamiltonian.time_derivative(y)


        for i in range(self.s):
            K[:,:,i] = f(y0 + self.v_t[i]*(y1 - y0) + h*torch.einsum('bns,s -> bn',K,self.D_t[i]))
        return K@self.b_t

    def f_hat_np(self,y0,y1,hamiltonian,h):

        K = np.zeros((y0.shape) + tuple([self.s]))
    
        def f(y):
            return hamiltonian.time_derivative(y)

        for i in range(self.s):
            K[:,:,i] = f(y0 + self.v[i]*(y1 - y0) + h*np.einsum('bns,s -> bn',K,self.D[i]))
        return K@self.b

    def integrator_np(self,y0,y1,hamiltonian,h):
        return y0 + h*self.f_hat_np(y0,y1,hamiltonian,h)
    
    def integrator(self,y0,y1,hamiltonian,h):
        return y0 + h*self.f_hat(y0,y1,hamiltonian,h)

    def noise_term(self):
        return np.sum(self.b*(1-2*self.v))




class MIRK_symmetric(MIRK):
    def __init__(self,D=None,v=None,b=None,type=None):
        super().__init__(D,v,b,type)

    def f_hat_base(self,y0,y1,hamiltonian,h):
        if len(y0.shape) == 1:
            y0 = y0.unsqueeze(0)
        if len(y1.shape) == 1:
            y1 = y1.unsqueeze(0)

        K = torch.zeros((y0.shape) + tuple([self.s]),dtype=torch.float64)
    
        def f(y):
            return hamiltonian.time_derivative(y)

        for i in range(self.s):
            K[:,:,i] = f(y0 + self.v_t[i]*(y1 - y0) + h*torch.einsum('bns,s -> bn',K,self.D_t[i]))
        return K@self.b_t

    def f_hat(self,y0,y1,hamiltonian,h):
        return .5*self.f_hat_base(y0,y1,hamiltonian,h) + .5*self.f_hat_base(y1,y0,hamiltonian,-h)



def get_MIRK_2():
    D = np.array([[0]],dtype=np.float64)
    v = np.array([.5],dtype=np.float64)
    b = np.array([1],dtype=np.float64)
    return v,b,D


def get_MIRK_3(c=1):
    v1 = c
    v2 = (36*c**3 - 54*c**2 + 27*c - 4)/(9*(2*c-1)**3)
    d = -2*(3*c**2 - 3*c +1)/(9*(2*c-1)**3)
    b1 = 1/(4*(3*c**2 - 3*c + 1))
    b2 = 3*(4*c**2 - 4*c + 1)/(4*(3*c**2 - 3*c + 1))

    v = np.array([v1,v2],dtype=np.float64)
    b = np.array([b1,b2],dtype=np.float64)
    D = np.zeros((2,2),dtype=np.float64)
    D[1,0] = d
    return v,b,D



def get_MIRK_4():

    d31 = 1/8
    d32 = -1/8
    v = np.array([0,1,.5])
    b = np.array([1/6,1/6,2/3])
    D = np.zeros((3,3))

    D[2,0] = d31
    D[2,1] = d32

    return v,b,D

def get_RK4():

    D = np.zeros((4,4))
    D[1,0] = 1/3
    D[2,0] = -1/3
    D[3,0] = 1
    D[2,1] = 1
    D[3,1] = -1
    D[3,2] = 1

    v = np.zeros(4)
    b =  np.array([1,3,3,1])/8

    return v,b,D





def get_MIRK_5(c2=0,c3=3/2):

    
    delta = lambda x: 10*x**2 - 8*x + 1
    psi = lambda x,y: 12*(1-x)*(y-x)*(5*x*beta - alpha)
    alpha = 10*c2*c3 - 5*(c2+c3) + 3
    beta = 6*c2*c3 - 2*(c2+c3) + 1
    gamma = (5*beta - alpha)*(5*c2*beta - alpha)*(5*c3*beta - alpha)
    
    d32 = ((2*c2*c3 + c2 + c3 - 1)*(1-c3)*(c2-c3))/((3*c2-1)*(c2-1)*delta(c2))
    d21 = c2*(c2-1)
    d31 = c3*(c3-1) - d32*(2*c2 - 1)
    d43 = (gamma*delta(c2))/(625*beta**5*(c3-1)*(c2-c3))
    d42 = ((5*beta - alpha)*(5*c2*beta - alpha)*(5*c3 - 2) - 125*beta**3*d43*(3*c3 - 1)*(c3-1))/(125*beta**3*(3*c2 - 1)*(c2-1))
    d41 = (alpha/(5*beta))*(alpha/(5*beta) - 1) - d43*(2*c3 -1 ) - d42*(2*c2-1)

    D = np.zeros((4,4))
    D[1,0] = d21
    D[2,0] = d31
    D[2,1] = d32
    D[3,0] = d41
    D[3,1] = d42
    D[3,2] = d43

    v1 = 1
    v2 = c2*(2-c2)
    v3 = c3*(2-c3) + 2*d32*(c2 - 1)
    v4 = alpha/(5*beta) - d41 - d42 - d43
    b2 = delta(c3)/psi(c2,c3)
    b3 = delta(c2)/psi(c3,c2)
    b4 = (125*beta**4)/(12*gamma)
    b1 = 1 - b2 - b3 - b4

    v = np.array([v1,v2,v3,v4])
    b =  np.array([b1,b2,b3,b4])

    return v,b,D



def get_MIRK_6():
    sq = np.sqrt(21)
    f1 = 1/14

    v3 = 1/2 - 9*sq/98
    v4 = 1/2 + 9*sq/98
    b1 = 1/20
    b2 = 1/20
    b3 = 49/180
    b4 = 49/180
    b5 = 16/45

    b = np.array([b1,b2,b3,b4,b5],dtype=np.float64)
    v = np.array([0,1,v3,v4,1/2],dtype=np.float64)
    D = np.zeros((5,5),dtype=np.float64)

    D[2,0] = f1 + sq/98
    D[2,1] = -f1 + sq/98
    D[3,0] = f1 - sq/98
    D[3,1] = -f1 - sq/98
    D[4,0] = -5/128
    D[4,1] = 5/128
    D[4,2] = 7*sq/128
    D[4,3] = -7*sq/128

    return v,b,D



