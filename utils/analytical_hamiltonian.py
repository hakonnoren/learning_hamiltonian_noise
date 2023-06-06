
from sympy.matrices import Matrix
from sympy import symbols,IndexedBase,latex,Sum
from sympy.utilities import lambdify
from sympy.parsing.latex import parse_latex
import torch

#from utils.sympy_torch import SymPyModule


class HamiltonianSystem():
    def __init__(self,hamiltonian_latex,name = '',hamiltonian_sympy = None):
        if not hamiltonian_sympy:
            self.hamiltonian_sympy = parse_latex(hamiltonian_latex)
        else:
            self.hamiltonian_sympy = hamiltonian_sympy
        self.hamiltonian_lambda,z,q,p = self.make_vectorized_hamiltonian(self.hamiltonian_sympy)
        self.z = z
        self.dHdt_sympy = Matrix((self.hamiltonian_sympy.diff(p),-self.hamiltonian_sympy.diff(q)))
        self.dHdt_lambda = lambdify([z],self.dHdt_sympy)
        self.grad_sympy = Matrix(self.hamiltonian_sympy.diff(z))
        self.grad_lambda = lambdify([z],self.grad_sympy)
        self.jacobian_sympy = self.dHdt_sympy.jacobian(self.z)
        self.jacobian_lambda = lambdify([z],self.jacobian_sympy)
        self.hess_sympy = self.hamiltonian_sympy.diff(z).diff(z)[:,0,:,0]     
        self.hess_lambda = lambdify([z],self.hess_sympy)
        self.d = len(self.z)
        self.name = name

    def hamiltonian(self,z):
        return self.hamiltonian_lambda(z)

    def time_derivative(self,z):
        return self.dHdt_lambda(z)[:,0]

    def grad_sym(self,z):
        shape = z.shape
        if len(shape) > 1:
            z = z[0]
        if z.dtype.__class__ == torch.dtype:
            r = torch.stack(self.grad_lambda(z)[:,0].tolist())
            if len(shape) > 1:
                return r.unsqueeze(0)
            else:
                return r
        return self.grad_lambda(z)[:,0]

    def make_vectorized_hamiltonian(self,hamiltonian_sympy):
        d = len(hamiltonian_sympy.free_symbols)//2
        var_dict = {'q':['']*d,'p':['']*d}
        for a in hamiltonian_sympy.free_symbols:
            letter = str(a)[0]
            idx = str(a)[3]
            var_dict[letter][int(idx)-1] = a

        z,q,p = Matrix(var_dict['q'] + var_dict['p']),Matrix(var_dict['q']),Matrix(var_dict['p'])
        hamiltonian_lambda = lambdify([z],hamiltonian_sympy)

        return hamiltonian_lambda,z,q,p

    def grad(self,z):
        return self.hamiltonian_grad_torch(z)

def get_fermi_pasta_ulam_tsingou(m = 3,omega = 10):
    i = symbols("i")
    x = IndexedBase('q')
    y = IndexedBase('p')
    hlf = .5

    H1 = Sum( hlf*(y[i]**2 + y[i+m]**2),(i,1,m)).doit()
    H2 = Sum( omega**2*hlf*(x[i+m]**2),(i,1,m)).doit()
    H3 = hlf/2*((x[1] - x[1+m])**4 + (x[m] + x[2*m])**4 )
    H4 = Sum( ( x[i + 1] - x[m+i+1] - x[i] - x[m + i]  )**4,(i,1,m-1)).doit()
    H = H1 + H2 + H3 + H4
    latex_H = latex(H).replace("{","").replace("}","")
    return latex_H
