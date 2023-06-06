
import numpy as np
import torch
from utils.training import make_mean_matrices
import numpy as np
from utils.generate_data import get_train_test_data
from integrators.mirk_integrators import MIRK




def compute_variance_mirk(hamiltonian_system,ys_perturbed,dt,n,mirk):
    matrix_norm = lambda x : np.linalg.norm(x,ord=2)
    ys_hat_mirk_target = np.stack(  [mirk.integrator_np(y1.T,y2.T,hamiltonian_system,dt).T for y1,y2 in zip(ys_perturbed[:,0:-1,:],ys_perturbed[:,1:,:])])
    variance_mirk_MC = []

    for i in range(0,n):
        target_os = np.cov(ys_hat_mirk_target[:,i,:].T - ys_perturbed[:,i+1,:].T)

        variance_mirk_MC.append(matrix_norm(target_os))

    return variance_mirk_MC

def compute_variance_mirk_mii(hamiltonian_system,ys_perturbed,dt,n,mirk):

    mean_vectorfield,mean_phase = make_mean_matrices(n+1)
    mean_vectorfield,mean_phase = mean_vectorfield.numpy(),mean_phase.numpy()

    fs = np.stack([mirk.f_hat_np(y1.T,y2.T,hamiltonian_system,dt).T for y1,y2 in zip(ys_perturbed[:,0:-1],ys_perturbed[:,1:])])

    ys_hat_mean_target = dt*np.einsum('ij,bjd->bid',mean_vectorfield,fs) + np.einsum('ii,bid->bid',mean_phase,ys_perturbed)

    variance_mii_MC = []

    matrix_norm = lambda x :  np.linalg.norm(x,ord=2)
    for i in range(0,n+1):
        target_mean = np.cov(ys_hat_mean_target[:,i,:].T - ys_perturbed[:,i,:].T)
        variance_mii_MC.append(matrix_norm(target_mean))

    return variance_mii_MC



def variance_mirk(hamiltonian_system,mirk,data,n_samples,args,sigma,seed):
    ys = torch.stack([data['ys_train'][0]]*n_samples)
    torch.manual_seed(seed)
    ys_perturbed = ys + torch.normal(mean=torch.zeros_like(ys))*sigma
    ys_perturbed = ys_perturbed.detach().numpy()

    n = args.n_timesteps_train
    dt = args.dt

    variance_mirk_MC = compute_variance_mirk(hamiltonian_system,ys_perturbed,dt,n,mirk)
    variance_mii_MC = compute_variance_mirk_mii(hamiltonian_system,ys_perturbed,dt,n,mirk)

    return variance_mirk_MC,variance_mii_MC



def get_variances_mirk4_rk4(hamiltonian_system,sigma,dts,args,n_mc_samples,seed):
    n = args.n_timesteps_train 
    mirk = MIRK(type='mirk4')
    os_mcs,mii_mcs =[],[]


    T = dts[-1]*args.n_timesteps_train
    Ns = np.int16(T/dts[:-1])
    dts = T/Ns

    for dt,N in zip(dts,Ns):
        args.dt = dt
        args.n_timesteps_train = N

        data = get_train_test_data(hamiltonian_system,args.n_trajectories_train,args.n_timesteps_train,args.dt,args.r_a,args.r_b,seed_init=seed,discrete_data=True,dtype = args.torch_dtype,save=False,plot=False)
        os_mc,mii_mc = variance_mirk(hamiltonian_system,mirk,data,n_samples=n_mc_samples,args=args,sigma=sigma,seed = seed)
        os_mcs.append(np.mean(os_mc,axis=-1))
        mii_mcs.append(np.mean(mii_mc,axis=-1))

    m4_mii_mcs = mii_mcs
    m4_os_mcs = os_mcs

    mirk = MIRK(type='rk4')
    args.n_timesteps_train = n
    os_mcs,mii_mcs =[],[]

    #T = dts[-1]*args.n_timesteps_train
    #Ns = np.int16(T/dts)
    #dts = T/Ns

    for dt,N in zip(dts,Ns):

        args.dt = dt
        args.n_timesteps_train = N

        data = get_train_test_data(hamiltonian_system,args.n_trajectories_train,args.n_timesteps_train,args.dt,args.r_a,args.r_b,seed_init=seed,discrete_data=True,dtype = args.torch_dtype,save=False,plot=False)

        os_mc,mii_mc = variance_mirk(hamiltonian_system,mirk,data,n_samples=n_mc_samples,args=args,sigma=sigma,seed = seed)

        os_mcs.append(np.mean(os_mc,axis=-1))
        mii_mcs.append(np.mean(mii_mc,axis=-1))

    rk4_os_mcs = os_mcs

    return np.array(m4_os_mcs),np.array(m4_mii_mcs),np.array(rk4_os_mcs)


def get_variances_mirk4_rk4_std(hamiltonian_system,n_runs,sigma,dts,args,n_mc_samples):
    m4_os_mcs,m4_mii_mcs,rk4_os_mcs = [],[],[]
    for seed in range(n_runs):
        m4_os_mc,m4_mii_mc,rk4_os_mc  = get_variances_mirk4_rk4(hamiltonian_system,sigma,dts,args,n_mc_samples,seed)
        m4_os_mcs.append(m4_os_mc)
        m4_mii_mcs.append(m4_mii_mc)
        rk4_os_mcs.append(rk4_os_mc)


    m4_os_mcs = np.stack(m4_os_mcs)
    m4_mii_mcs = np.stack(m4_mii_mcs)
    rk4_os_mcs = np.stack(rk4_os_mcs)

    return m4_os_mcs,m4_mii_mcs,rk4_os_mcs

