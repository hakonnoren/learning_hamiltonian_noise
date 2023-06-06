from scipy.integrate import solve_ivp
import torch
from matplotlib import pyplot as plt
import numpy as np

from .visualization import set_plot_params 

set_plot_params()

def get_initial_values(n_trajectories,f,r_a,r_b,d,seed = 0,y0_center = False,volterra=False):

    if y0_center.__class__.__name__ != "ndarray":
        rng = np.random.default_rng(seed)
        if volterra:

            ts = np.linspace(0,7.0,100)
            ft = lambda t,y:f(y)
            y0s_init = rng.random((n_trajectories,d))*(r_b - r_a) + r_a
            y0 = []
            for y0_i in y0s_init:
                sol = solve_ivp(ft,(0,ts[-1]),y0_i,t_eval=ts)
                y0.append(sol.y[:,np.random.randint(sol.y.shape[-1])])
                y0.append(sol.y[:,np.random.randint(sol.y.shape[-1]//3,sol.y.shape[-1]//1.5) ])


        else:
        
            y0 = rng.random((n_trajectories,d))*2 - 1


            radius = r_a**2+rng.random(n_trajectories)*r_b**2
            radius = np.sqrt(radius)
            radius = np.array([radius]*d).T


            normalize = np.sqrt((y0**2).sum(axis=1))
            normalize = np.array([normalize]*d).T

            y0 = (y0/normalize)*radius

    else:
        std = 0.05
        rng = np.random.default_rng(seed)
        perturbations = rng.normal(size=(n_trajectories,d))*std
        y0 = np.stack([y0_center]*n_trajectories) + perturbations

    return y0


def integrate_ivps(f,y0,ts,return_dy = False,dt = None,sigma = 0,seed_noise=0,atol=1e-10):
    y = []
    dy = []

    ft = lambda t,y:f(y)
    rng = np.random.default_rng(seed_noise)

    if dt:
        y1,y2,dydt = [],[],[]
        ys = []
        for y0_i in y0:
            sol = solve_ivp(ft,(0,ts[-1]),y0_i,atol = atol,t_eval=ts,method='DOP853')
            sol_y = sol.y + rng.normal(size=sol.y.shape)*sigma
            y1.append(sol_y[:,:-1]);y2.append(sol_y[:,1:])
            ys.append(sol_y)
            dydt_y0 = (sol_y[:,1:]-sol_y[:,:-1])/dt
            dydt.append(dydt_y0)


        return y1,y2,dydt,ys

    else:

        for y0_i in y0:
            sol = solve_ivp(ft,(0,ts[-1]),y0_i,atol = atol,t_eval=ts)
            y.append(sol.y)
            if return_dy:
                dy.append(f(None,sol.y))

        if return_dy:
            return y,dy
        else:
            return y
    
def get_trajectories(f,n_trajectories,n_timesteps,dt,r_a,r_b,d,atol = 1e-10,seed = 0,sigma = 0,seed_noise=0,y0_center = False,volterra=False):
    y0 = get_initial_values(n_trajectories,f,r_a,r_b,d,seed = seed,y0_center=y0_center,volterra=volterra)

    ts = np.linspace(0,n_timesteps*dt,n_timesteps+1)

    return integrate_ivps(f,y0,ts,return_dy=True,dt=dt,sigma = sigma,seed_noise=seed_noise,atol=atol)


def get_differences(y,dt):
    y1,y2 = y[:-1],y[1:]
    dydt = (y2-y1)/dt
    return y1,y2,dydt

def scale_to_unit(array):
    t = [i for i in range(len(array.shape))]
    t.pop(1)
    axis = tuple(t)
    max_vals = np.max(np.abs(array),axis=axis)
    for i in range(len(max_vals)):
        array[:,i] = array[:,i]/max_vals[i]
    return array


def get_train_test_data(hamiltonian_system,
                        n_trajectories,
                        n_timesteps,
                        dt,
                        r_a,
                        r_b,
                        sigma = 0,
                        atol = 1e-10,
                        train_test = 0.95,
                        discrete_data = False,
                        dtype =torch.float32,
                        seed_init = 0,
                        seed_noise = 0,
                        y0_center = False,
                        plot=False,
                        save = False,
                        volterra=False):

    rng = np.random.default_rng(seed_noise)

    d = len(hamiltonian_system.z)
    f = hamiltonian_system.time_derivative

    if discrete_data:
        y1,y2,dydt,ys = get_trajectories(f,n_trajectories,n_timesteps,dt,r_a,r_b,d,atol = atol,seed = seed_init,sigma = sigma,seed_noise=seed_noise,y0_center=y0_center,volterra=volterra)
        y = y1

    else:
        y,dy = get_trajectories(f,n_trajectories,n_timesteps,dt,r_a,r_b,d,atol,y0_center=y0_center,volterra=volterra)

    n_train = int(len(y)*train_test)

    if plot:

        plt.rcParams["figure.figsize"] = (5,5)
        
        
        for j in range(d//2):

            plt.figure(dpi=300)

            plt.title("Flow sample " + hamiltonian_system.name + r" $\sigma = $" + " " + str(round(sigma,3)))
            for i in range(n_trajectories):
                plt.plot(y[i][j] + rng.normal(size=y[i][j].shape)*sigma,y[i][j+d//2] + rng.normal(size=y[i][j + d//2].shape)*sigma,'-')

            plt.xlabel("$q_" + str(j+1)+"$")
            plt.ylabel("$p_" + str(j+1)+"$")

            if save:
                file_name = "flow_sample_" + hamiltonian_system.name.replace(" ", "_").lower() + " d=" + str(j) + ".pdf"
                plt.savefig(save + file_name, format='pdf', bbox_inches='tight')
                plt.close()
            else:
                plt.show()



        
        y_train = np.concatenate(y[:n_train],axis=1).T
        y_test = np.concatenate(y[n_train:],axis=1).T
        
    data = {}
    

    if not discrete_data:
        dy_train = np.concatenate(dy[:n_train],axis=1).T
        dy_test = np.concatenate(dy[n_train:],axis=1).T
        
        y_train += rng.normal(size=y_train.shape)*sigma
        y_test += rng.normal(size=y_test.shape)*sigma

        y_train = torch.tensor(y_train,requires_grad=True,dtype=dtype)
        y_test = torch.tensor(y_test,requires_grad=True,dtype=dtype)
        dy_train = torch.tensor(dy_train,dtype=dtype)
        dy_test = torch.tensor(dy_test,dtype=dtype)

        data['y_train'],data['y_test'],data['dy_train'],data['dy_test'] = y_train,y_test,dy_train,dy_test
    
    if discrete_data:
        y1_train = np.concatenate(y1[:n_train],axis=1).T
        y2_train = np.concatenate(y2[:n_train],axis=1).T


        y1_test = np.array(y1[n_train:])
        y2_test = np.array(y2[n_train:])

        ys = np.stack(ys)
        ys = np.swapaxes(ys,1,2)
        ys_train = ys[:n_train]
        ys_test = ys[n_train:]

        dydt_train = np.concatenate(dydt[:n_train],axis=1).T
        dydt_test = np.concatenate(dydt[n_train:],axis=1).T



        y1_train = torch.tensor(y1_train,requires_grad=True,dtype=dtype)
        y2_train = torch.tensor(y2_train,requires_grad=True,dtype=dtype)
        ys_train = torch.tensor(ys_train,requires_grad=True,dtype=dtype)
        
        dydt_train = torch.tensor(dydt_train,dtype=dtype)

        y1_test = torch.tensor(y1_test,requires_grad=True,dtype=dtype)
        y2_test = torch.tensor(y2_test,requires_grad=True,dtype=dtype)
        ys_test = torch.tensor(ys_test,requires_grad=True,dtype=dtype)
        dydt_test = torch.tensor(dydt_test,dtype=dtype)
        data['y1_train'],data['y2_train'],data['dydt_train'],data['ys_train'] = y1_train,y2_train,dydt_train,ys_train
        data['y1_test'],data['y2_test'],data['dydt_test'],data['ys_test'] = y1_test,y2_test,dydt_test,ys_test

    return data

