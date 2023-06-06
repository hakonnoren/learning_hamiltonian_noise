import numpy as np

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from .iso_objective import IsoObjective
from scipy.optimize import minimize

def make_mean_matrices(n):
    mean_vectorfield = torch.ones((n,n-1))
    for i in range(n):
        for j in range(n-1):
            if j >= i:
                mean_vectorfield[i,j] = -(n-1) + j
            else:
                mean_vectorfield[i,j] = j+1
    mean_phase = torch.ones((n,n)) - torch.eye(n)
    return mean_vectorfield/(n-1),mean_phase/(n-1)


def hnn_step(batch,net):
    y,dy = batch[0],batch[1]
    dH_hat = net(y)
    loss = torch.mean((dH_hat[:,1] - dy[:,0])**2 + (dH_hat[:,0] + dy[:,1])**2)
    loss.backward()
    return loss.item()/y.shape[0]

def rk_step(batch,net,integrator,dt):
    y1,y2,dydt = batch[0],batch[1],batch[2]
    loss = torch.mean((y2 - integrator.integrator(y1,y2,net,dt))**2)
    loss.backward()
    return loss.item()/(dt**2)


def recurrent_step(batch,net,integrator,dt,epoch):
    ys = batch[0]
    n = ys.shape[1]

    if epoch > 60:
        objective = IsoObjective(ys, net, dt,integrator)
        result = minimize(objective.fun, ys[:,0,:].detach().numpy().reshape([objective.b,objective.d*2]), method='L-BFGS-B',
                        jac=objective.grad,options={'gtol': 1e-6, 'disp': False,
                    'maxiter':10})
        y0 = torch.tensor(result.x.reshape([objective.b,objective.d*2]))
        y0.requires_grad = True
    else:
        y0 = ys[:,0,:]

    ys_hat = integrator.integrate(y0,dt,net,n_steps=n,method = "explicit")
    loss = torch.mean((ys[:,1:,:] - ys_hat[:,1:,:])**2)
    loss.backward()
    test_loss.append(loss.item()/(dt**2))
    return loss.item()/(dt**2)


def rk_mean_step(batch,net,integrator,dt,mean_vectorfield,mean_phase):
    ys = batch[0]

    fs = torch.stack([integrator.f_hat(y1,y2,net,dt) for y1,y2 in zip(ys[:,0:-1],ys[:,1:])])
    ys_hat = (dt*mean_vectorfield@fs + mean_phase@ys)

    loss = torch.mean((ys - ys_hat)**2)
    loss.backward()
    return loss.item()/(dt**2)

def train_hnn(net,dataloader,optimizer,integrator=None,epochs=300,method="rk",mean=False,dt = None,plot=False,track_time=False):
    test_losses = []
    total_time = 0

    if mean and method != "recurrent":
        for batch in dataloader:
            n = batch[0].shape[1]
            break
            
        mean_vectorfield,mean_phase = make_mean_matrices(n)

    def epoch_loop(epochs):

        total_time_epochs = 0

        with tqdm(range(epochs), unit="epochs") as tepoch:
            for epoch in tepoch:

                test_loss = []

                for i,batch in enumerate(dataloader):
                    
                    T = time.time()
                    if method == "hnn":
                        epoch_loss = hnn_step(batch,net)

                    if method == "recurrent":
                        epoch_loss = recurrent_step(batch,net,integrator,dt,epoch)
                    
                    if method == "rk":
                        if mean:
                            epoch_loss = rk_mean_step(batch,net,integrator,dt,mean_vectorfield,mean_phase)
                        else:
                            epoch_loss = rk_step(batch,net,integrator,dt)


                    test_loss.append(epoch_loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    T = time.time() - T
                
                tepoch.set_postfix(loss=np.mean(test_loss))
                test_losses.append(np.mean(test_loss))
                total_time_epochs += T

        return np.mean(test_loss),total_time_epochs

    test_loss_final,total_time_epochs = epoch_loop(epochs)
    total_time = total_time_epochs
    while test_loss_final > 1.0:
        test_loss_final,total_time_epochs = epoch_loop(1)
        total_time += total_time_epochs

    if plot:
        plt.rcParams["figure.figsize"] = (10,5)
        plt.figure(dpi=100)
        plt.title("HNN Loss")
        plt.semilogy(test_losses,label="train loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if track_time:
        return total_time



def recurrent_step_closure(batch,net,integrator,optimizer,dt,epoch):
    ys = batch[0]
    n = ys.shape[1]

    if epoch > 10:
        objective = IsoObjective(ys, net, dt,integrator)
        result = minimize(objective.fun, ys[:,0,:].detach().numpy().reshape([objective.b,objective.d*2]), method='L-BFGS-B',
                        jac=objective.grad,options={'gtol': 1e-6, 'disp': False,
                    'maxiter':10})
        y0 = torch.tensor(result.x.reshape([objective.b,objective.d*2]))
        y0.requires_grad = True

    else:
        y0 = ys[:,0,:]

    def closure():
        optimizer.zero_grad()
        ys_hat = integrator.integrate(y0,dt,net,n_steps=n,method = "explicit")
        loss = torch.mean((ys[:,1:,:] - ys_hat[:,1:,:])**2)
        loss.backward(retain_graph=True)
        test_loss.append(loss.item()/(dt**2))
        return loss

    return closure




def rk_step_closure(batch,net,integrator,optimizer,dt):
    y1,y2,dydt = batch[0],batch[1],batch[2]

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((y2 - integrator.integrator(y1,y2,net,dt))**2)
        loss.backward(retain_graph=True)
        test_loss.append(loss.item()/(dt**2))
        return loss

    return closure


def rk_mean_step_closure(batch,net,integrator,optimizer,dt,mean_vectorfield,mean_phase):
    ys = batch[0]
    n = ys.shape[1]

    def closure():
        optimizer.zero_grad()
        fs = torch.stack([integrator.f_hat(y1,y2,net,dt) for y1,y2 in zip(ys[:,0:-1],ys[:,1:])])
        ys_hat = (dt*mean_vectorfield@fs + mean_phase@ys)
        loss = torch.mean((ys - ys_hat)**2)
        loss.backward(retain_graph=True)
        test_loss.append(loss.item()/(dt**2))
        return loss

    return closure

def derivative_step_closure(batch,net,optimizer,vectorfield):
    _,y2,_ = batch[0],batch[1],batch[2]

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((torch.tensor(vectorfield(y2.detach().numpy().T).T) - net.time_derivative(y2) )**2)
        loss.backward(retain_graph=True)
        test_loss.append(loss.item())
        return loss

    return closure


def train_hnn_closure(net,dataloader,optimizer,integrator=None,epochs=300,method="rk",mean=False,dt = None,plot=False,track_time = False,return_loss = False,vectorfield=None):
    test_losses = []
    total_time = 0
    
    if mean and method != "recurrent":
        for batch in dataloader:
            n = batch[0].shape[1]
            break


        mean_vectorfield,mean_phase = make_mean_matrices(n)


    def epoch_loop(epochs):
        global test_loss
        total_time_epochs = 0

        minimum_loss = np.inf
        with tqdm(range(epochs), unit="epochs") as tepoch:
        
            for epoch in tepoch:

                test_loss = []

                for batch in dataloader:
                    
                    T = time.time()
                    if vectorfield:
                        closure = derivative_step_closure(batch,net,optimizer,vectorfield)

                    if method == "recurrent":

                        closure = recurrent_step_closure(batch,net,integrator,optimizer,dt,epoch)

                    
                    if method == "rk":
                        if mean:
                            closure = rk_mean_step_closure(batch,net,integrator,optimizer,dt,mean_vectorfield,mean_phase)
                        else:
                            closure = rk_step_closure(batch,net,integrator,optimizer,dt)
                    optimizer.step(closure)

                    if method == "recurrent" and epoch > 10:
                        if test_loss[-1] < minimum_loss:
                            minimum_loss =  test_loss[-1]
                            torch.save({'epoch': epoch,'model_state_dict': net.state_dict()}, 'params/min_loss_model.pth')
        
                    T = time.time() - T
                
                tepoch.set_postfix(loss=np.mean(test_loss))
                test_losses.append(np.mean(test_loss))
                total_time_epochs += T
        if method == "recurrent":
            net.load_state_dict(torch.load('params/min_loss_model.pth')['model_state_dict'])

        return np.mean(test_loss),test_losses,total_time_epochs
    
    test_loss_final,test_losses,total_time_epochs = epoch_loop(epochs)
    total_time = total_time_epochs
    c = 0
    while test_loss_final > 1.0 and c < 4 and method != "recurrent":
        test_loss_final,test_losses,total_time_epochs = epoch_loop(1)
        c+=1
        total_time += total_time_epochs

    if plot:
        plt.rcParams["figure.figsize"] = (10,5)
        plt.figure(dpi=100)
        plt.title("HNN Loss")
        plt.semilogy(test_losses,label="train loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    if track_time:
        return total_time

    if return_loss:
        return test_losses

