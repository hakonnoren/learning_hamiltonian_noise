from torch import nn
import torch

class HNN(nn.Module):
    def __init__(self,input_dim,t_dtype,hidden_dim):
        super(HNN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        d = input_dim//2
        I = torch.diag(torch.ones(d))
        J = torch.zeros((input_dim,input_dim))
        J[0:d,d:],J[d:,0:d] = I,-I 
        self.J = J

    def grad(self, x,h=None):
        H = self.seq(x)
        dH = torch.autograd.grad(H,x,create_graph=True,grad_outputs=torch.ones_like(H))[0]
        return dH

    def forward(self,x):
        return self.H(x)

    def time_derivative(self,x):
        d = x.shape[-1]//2
        dh_q,dh_p = torch.split(self.grad(x),d,dim=-1)
        Jdh = torch.cat([dh_p,-dh_q],axis=-1)
        return Jdh

    def H(self,x):
        return self.seq(x)/(1-self.p)

    def hamiltonian(self,x):
        return self.H(x)
    

class sepHNN(nn.Module):
    def __init__(self,input_dim,t_dtype,hidden_dim):
        super(sepHNN, self).__init__()
        self.d = int(input_dim//2)
        self.Hq = nn.Sequential(
            nn.Linear(self.d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.Hp = nn.Sequential(
            nn.Linear(self.d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )



    def grad_q(self,q):
        Hq = self.Hq(q)
        dHq = torch.autograd.grad(Hq,q,create_graph=True,grad_outputs=torch.ones_like(Hq))[0]
        return dHq
    
    def grad_p(self,p):
        Hp = self.Hp(p)
        dHp = torch.autograd.grad(Hp,p,create_graph=True,grad_outputs=torch.ones_like(Hp))[0]
        return dHp
    
    def grad(self,x):
        q,p = torch.split(x,self.d,dim=-1)
        dH = torch.cat([self.grad_q(q),self.grad_p(p)],axis=-1)
        return dH

    def forward(self,x):
        q,p = torch.split(x,self.d,dim=-1)

        return  self.Hq(q) + self.Hp(p)

    def time_derivative(self,x):
        q,p = torch.split(x,self.d,dim=-1)
        JdH = torch.cat([self.grad_p(p),-self.grad_q(q)],axis=-1)
        return JdH

    def H(self,x):
        return self.forward(x)

    def hamiltonian(self,x):
        return self.H(x)


