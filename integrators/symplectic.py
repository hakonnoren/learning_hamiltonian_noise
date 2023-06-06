from .integrator_base import Integrator
import torch


class Midpoint(Integrator):
    def __init__(self):
        super().__init__()

    def integrator(self,y1,y2,neural_net,h):
        return y1 + h*self.f_hat(y1,y2,neural_net)

    def f_hat(self,y1,y2,neural_net,h=None):
        Jdh = neural_net.time_derivative((y2+y1)/2)
        return Jdh

class SympEuler(Integrator):
    def __init__(self):
        super().__init__()

    def integrator(self,y1,y2,neural_net,h):
        Jdh = self.f_hat(y1,y2,neural_net)
        return y1 + h*Jdh

    def f_hat(self,y1,y2,neural_net,h=None):

        d = y1.shape[-1]//2
        q1,p1 = torch.split(y1,d,dim=-1)
        q2,p2 = torch.split(y2,d,dim=-1)
        Jdh = neural_net.time_derivative(torch.cat([q1,p2],dim=-1))
        return Jdh


class Stormer(Integrator):
    def __init__(self):
        super().__init__()

    def integrator(self,y1,y2,neural_net,h):
        d = y1.shape[-1]//2
        q1,p1 = torch.split(y1,d,dim=-1)

        p_hlf = p1 - .5*h*neural_net.grad_q(q1)
        q2 = q1 + h*neural_net.grad_p(p_hlf)
        p2 = p_hlf - .5*h*neural_net.grad_q(q2)

        return torch.cat([q2,p2],dim=-1)

    def f_hat(self,y1,y2,neural_net,h):
        d = y1.shape[-1]//2
        q1,p1 = torch.split(y1,d,dim=-1)
        p_hlf = p1 - .5*h*neural_net.grad_q(q1)
        p_incr =  -.5*(neural_net.grad_q(q1) + neural_net.grad_q(q1 + h*neural_net.grad_p(p_hlf)))
        q_incr = neural_net.grad_p(p_hlf)
        return torch.cat([q_incr,p_incr],dim=-1)

