from integrators.symplectic import Midpoint,Stormer
from integrators.mirk_integrators import MIRK
import numpy as np
import torch
from utils.analytical_hamiltonian import get_fermi_pasta_ulam_tsingou

from argparse import ArgumentParser

parser = ArgumentParser(description=None)
parser.add_argument('--plot_dir',default = 'plots/', type=str)
parser.add_argument('--weight_dir',default = 'weight_dicts/', type=str)

parser.add_argument('--n_trajectories_train',default = 300, type=int)
parser.add_argument('--n_timesteps_train',default = 3, type=int)
parser.add_argument('--dt',default =.8, type=float)
parser.add_argument('--tolerance_training_integrator',default = 1e-15, type=float)
parser.add_argument('--r_a',default = 0.3, type=float)
parser.add_argument('--r_b',default = 0.6, type=float)
parser.add_argument('--torch_dtype',default = torch.float64, type=torch.dtype)
parser.add_argument('--hidden_dim',default = 200, type=int)
parser.add_argument('--training_epochs',default = 20, type=int)
parser.add_argument('--learning_rate',default = 1e-3, type=float) #Not used except if training with Adam
parser.add_argument('--non_linearity',default = 'tanh', type=str)
parser.add_argument('--seed',default = 0, type=int)
parser.add_argument('--sep',default = False, type=bool)
args = parser.parse_args(args=[])
torch.set_default_dtype(args.torch_dtype)


n_samples = 5
n_step_red = 4
sigma = 0.05

midpoint = Midpoint()
stormer = Stormer()
mirk4 = MIRK(type="mirk4")
rk4 = MIRK(type="rk4")


hamiltonian_double_pendulum = r' \frac{\frac{1}{2}p_1^2 + p_2^2 - p_1p_2\cos(q_1-q_2)}{1+\sin^2(q_1-q_2)} - 2\cos(q_1) - \cos(q_2)'
hamiltonian_henon_heiles = r' \frac{1}{2}(p_1^2 + p_2^2) + \frac{1}{2}(q_1^2 + q_2^2) + \frac{1}{1}(q_1^2q_2 - \frac{1}{3}q_2^3)'
hamiltonian_fput = get_fermi_pasta_ulam_tsingou(m = 1,omega = 2)


integrator_dict = {
            'Midpoint':{'integrator':midpoint},
            'ISO Störmer':{'integrator':stormer},
            'RK4':{'integrator':rk4},     
            'ISO RK4':{'integrator':rk4},
            'MIRK4':{'integrator':mirk4},
            'MII MIRK4':{'integrator':mirk4},
}


###############  Test comparing ISO, MIRK and MII  #############


test_dict_dp = {'integrator_dict' : integrator_dict,
            'n_trajectories_test':10,
            'hamiltonian_latex': hamiltonian_double_pendulum,
            'system_name':"Double pendulum",
            'scales': np.arange(n_step_red,step=1),
            'sigma':sigma,
}

test_dict_hh = {'integrator_dict' : integrator_dict,
            'n_trajectories_test':10,
            'hamiltonian_latex': hamiltonian_henon_heiles,
            'system_name':'Hénon-Heiles',
            'scales': np.arange(n_step_red,step=1),
            'sigma':sigma,
}


test_dict_fput = {'integrator_dict' : integrator_dict,
            'n_trajectories_test':10,
            'hamiltonian_latex': hamiltonian_fput,
            'system_name':'FPUT',
            'scales': np.arange(n_step_red,step=1),
            'sigma':sigma,
}
