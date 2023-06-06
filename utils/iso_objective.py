import torch
import torch.nn as nn


"""
This is a modification of the implementation of
initial state optimization from the following repository:

https://github.com/zhengdao-chen/SRNN/

(CC-BY-NC 4.0 license)

for the paper

@inproceedings{Chen2020Symplectic,
title={Symplectic Recurrent Neural Networks},
author={Zhengdao Chen and Jianyu Zhang and Martin Arjovsky and Léon Bottou},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BkgYPREtPr}
}


"""


class IsoObjective(object):

    def __init__(self, ys, neural_network, dt,integrator):
        self.mse = nn.MSELoss()
        self.n = ys.shape[1]
        self.b = ys.shape[0]
        self.ys_data = ys.requires_grad_()
        self.neural_network = neural_network
        self.d = ys.shape[-1]//2
        self.integrator = integrator
        self.dt = dt

    def fun(self,y0_np):
        y0 = torch.from_numpy(y0_np).reshape([self.b,1,self.d*2])
        y0.requires_grad = True
        ys_hat =  self.integrator.integrate_torch_explicit(y0[:,0,:],self.dt,self.neural_network,n_steps=self.n)
        loss = torch.mean((self.ys_data[:,1:,:] - ys_hat[:,1:,:])**2)
        self.loss = loss
        return loss.item()

    def grad(self,y0_np):
        y0 = torch.from_numpy(y0_np).reshape([self.b,1,self.d*2])
        y0.requires_grad = True
        ys_hat =  self.integrator.integrate_torch_explicit(y0[:,0,:],self.dt,self.neural_network,n_steps=self.n)

        loss = torch.mean((self.ys_data[:,1:,:] - ys_hat[:,1:,:])**2)
        loss.backward(retain_graph=True)
        grad = y0.grad.type(torch.DoubleTensor).detach().numpy().reshape([self.b*self.d*2])
        return grad
    

