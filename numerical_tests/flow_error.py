
import numpy as np
import torch

from torch.utils.data import TensorDataset,DataLoader
from torch import optim

from utils.generate_data import get_train_test_data
from utils.analytical_hamiltonian import HamiltonianSystem
from utils.training import train_hnn_closure,train_hnn
from utils.neural_nets import HNN,sepHNN
from utils.visualization import set_plot_params

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.integrate import solve_ivp
import torch

import pickle


set_plot_params()

def define_experiment(dt,n_steps,hamiltonian_latex,r_a,r_b,n_trajectories,sigma=0,seed=0,plot=False,args=None):


    hamiltonian_system = HamiltonianSystem(hamiltonian_latex)


    t_dtype = torch.float64
    np.random.seed(seed)
    d = len(hamiltonian_system.z)

    torch.set_default_dtype(t_dtype)


    data = get_train_test_data(hamiltonian_system,n_trajectories,n_steps,dt,r_a,r_b,seed_init=seed,seed_noise=seed,sigma=sigma,discrete_data=True,dtype = t_dtype,save=args.plot_dir,plot = plot)

    dataset = TensorDataset(data['y1_train'],data['y2_train'],data['dydt_train'])
    dataloader = DataLoader(dataset,batch_size=10_000,shuffle=False)
    dataset_mean = TensorDataset(data['ys_train'])
    dataloader_mean = DataLoader(dataset_mean,batch_size=10_000,shuffle=False)
    return hamiltonian_system,dataloader,dataloader_mean,dt,d,data


def get_trained_nn(args,dataloader,dataloader_mean,dt,d,epochs,method,hidden_dim,learning_rate,integrator = None,plot=False,fnn=False,vectorfield = None):
    t_dtype = torch.float64
    training_time = 0
    torch.manual_seed(args.seed)
    
    if (method == "rk" or method[0:3] == "mii" or method[0:3] == "rec" or method == "exact_derivatives") and not fnn:
        neural_net = HNN(input_dim=d,t_dtype=t_dtype,hidden_dim = hidden_dim)

    if method[0:3] == "ISO" or args.sep:
        neural_net = sepHNN(input_dim=d,t_dtype=t_dtype,hidden_dim = hidden_dim)
        if integrator.__class__.__name__ == "MIRK" and not args.sep:
            neural_net = HNN(input_dim=d,t_dtype=t_dtype,hidden_dim = hidden_dim)
    else:
        neural_net = HNN(input_dim=d,t_dtype=t_dtype,hidden_dim = hidden_dim)

    #optimizer = optim.Adam(neural_net.parameters(), lr=learning_rate)

    optimizer = optim.LBFGS(neural_net.parameters(), history_size=120,tolerance_grad=1e-9,tolerance_change=1e-9,line_search_fn="strong_wolfe")

    plot = False
    if method[0:3] == "mii":
            training_time_os = train_hnn_closure(neural_net,dataloader,optimizer,integrator = integrator,epochs=int(epochs)//2,method=method[4:],mean=False,dt=dt,plot = plot,track_time=True)
            training_time_mii = train_hnn_closure(neural_net,dataloader_mean,optimizer,integrator = integrator,epochs=int(epochs)//2,method=method[4:],mean=True,dt=dt,plot = plot,track_time=True)
            training_time = training_time_os + training_time_mii
    elif method[0:3] == "rec" or method[0:3] == "ISO":
        training_time = train_hnn_closure(neural_net,dataloader_mean,optimizer,integrator = integrator,epochs=epochs,method="recurrent",mean=False,dt=dt,plot = plot,track_time=True)

    elif method != "exact_derivatives":
        training_time = train_hnn_closure(neural_net,dataloader,optimizer,integrator = integrator,epochs=epochs,method=method,dt=dt,plot = plot,track_time=True)

    else:
        training_time = train_hnn_closure(neural_net,dataloader,optimizer,integrator = integrator,epochs=epochs,method="exact_derivatives",dt=dt,plot = plot,track_time=True,vectorfield=vectorfield)


    return neural_net,training_time


def experiment_flow(args,test_dict,plot=False):    
    measurements = {
                    'Flow error': {'data':{},'y_lab':r'$e(f_{\theta})$'},
                    'Training time': {'data':{},'y_lab':r'$\Delta t_c$'}
                    }


    for key in test_dict['integrator_dict']:
        for measurement in measurements:
            measurements[measurement]['data'][key] = []

    fnn = False
    if type(test_dict['hamiltonian_latex']) == list:
        fnn = True


    scales = test_dict['scales']
    dts = args.dt/2**scales
    n_stepss = args.n_timesteps_train*2**scales

    scale_n_steps = 1
    if plot:
        scale_n_steps = 8

    hamiltonian_system = HamiltonianSystem(test_dict['hamiltonian_latex'])
    
    np.random.seed(args.seed+1)
    y0_plot = np.array([0.3,0.2,0.25,0.1,0.3,0.4,0.1,0.2])
    y0_plot = y0_plot[:len(hamiltonian_system.z)]
    i = 0

    for dt,n_steps in zip(dts,n_stepss):
        i+=1
        hamiltonian_system,dataloader,dataloader_mean,dt,d,data = define_experiment(dt,
                                    n_steps,
                                    test_dict['hamiltonian_latex'],
                                    args.r_a,
                                    args.r_b,
                                    args.n_trajectories_train,
                                    test_dict['sigma'],
                                    args.seed,
                                    plot=plot,
                                    args=args)


        for key in test_dict['integrator_dict']:
            print(key)
            if test_dict['integrator_dict'][key]['integrator'].__class__.__name__ == 'str':
                method = test_dict['integrator_dict'][key]['integrator']
                integrator = None
                if key[0:3].lower() == "mii":
                    method = "mii " + test_dict['integrator_dict'][key]['integrator']
            elif key[0:3] != "ISO":
                integrator = test_dict['integrator_dict'][key]['integrator']
                method = "rk"
                if key[0:3].lower() == "mii":
                    method = "mii " + "rk"
                elif key[0:3].lower() == "rec":
                    method = "rec " + "rk"
            else:
                method = "ISO"
                integrator = test_dict['integrator_dict'][key]['integrator']
            if key.lower() == "exact derivatives":
                method = key.lower().replace(" ","_")
            
            
            test_dict['integrator_dict'][key]['nn'],training_time = get_trained_nn(args,dataloader,dataloader_mean,dt,d,args.training_epochs,method,
                            args.hidden_dim,
                            args.learning_rate,
                            integrator = integrator,
                            plot=plot,
                            fnn=fnn,
                            vectorfield=hamiltonian_system.time_derivative)
            measurements['Training time']['data'][key].append(training_time)

        if plot:
            plot = args.plot_dir
        error_flow = init_tests(data,hamiltonian_system,test_dict,scale_n_steps*n_steps,dt,test_dict['integrator_dict'],test_dict['n_trajectories_test'],y0_plot,plot)

        for key in test_dict['integrator_dict']:
            measurements['Flow error']['data'][key].append(error_flow[key])

    return measurements



def init_test(y0,hamiltonian_system,n_steps,dt,tests):

    ts = np.linspace(0,dt*(n_steps),n_steps+1)
    d= len(hamiltonian_system.z)
    flows = {}
    
    for key in tests:
        #Testing with scikit-integrator
        f = lambda t,y : tests[key]['nn'].time_derivative(torch.tensor(y,requires_grad=True)).detach().numpy()
        ys_hnn = solve_ivp(f,(0,ts[-1]),y0.detach().numpy(),atol = 1e-16,t_eval=ts,method='DOP853').y

        flows[key] = ys_hnn
    return flows

def init_tests(data,hamiltonian_system,test_dict,n_steps,dt,tests,n_inits,y0_plot,plot = False):
    d = len(hamiltonian_system.z)
    ts = np.linspace(0,dt*n_steps,n_steps+1)

    f = hamiltonian_system.time_derivative
    f_exact = lambda t,y:f(y)
    y0_test = data["y1_test"][...,0].detach()
    if plot: 
        y0_test[n_inits-1] = torch.tensor(y0_plot)
    y0_test.requires_grad = True

    errors_flow = {}
    for key in tests:
        errors_flow[key] = []

    rng = np.random.default_rng(0)

    for y0 in y0_test[0:n_inits]:
        sol_exact = solve_ivp(f_exact,(0,ts[-1]),y0.detach().numpy(),atol = 1e-16,t_eval=ts)
        ys_exact = sol_exact.y
        
        flows = init_test(y0,hamiltonian_system,n_steps,dt,tests)

        for key in tests:
            if plot:
                test_idx = flows[key].shape[-1]//8 + 1
            else:
                test_idx = -1
            errors_flow[key].append(np.linalg.norm(np.mean(ys_exact[:,:test_idx]-flows[key][:,:test_idx],axis=-1)))

    
    if plot:
        ys_noise = ys_exact + rng.normal(size=ys_exact.shape)*test_dict['sigma']
        x = np.linspace(0.0, 1.0, 10)
        rgb_all = cm.get_cmap("tab10")(x)
        rgb = np.concatenate([rgb_all[0:2],rgb_all[3:]])



        plt.rcParams["figure.figsize"] = (8,3)
        for j in range(2):
            
            title = "Flow roll-out " + " " + test_dict['system_name'] + r" $h$ = " + str(round(dt,3)) + r", $\sigma$ = " +str(round(test_dict['sigma'],3))
            plt.title(title)
            plt.rcParams["figure.figsize"] = (8,3)

            for i,key in enumerate(tests):
                plt.plot(ts,flows[key][j],label=key,color = rgb[i],zorder=2)
            plt.plot(ts,ys_exact[j],ls="--",color='tab:green',lw=2,label="Ground truth",zorder=1)
            plt.scatter(ts[:test_idx],ys_noise[j][:test_idx],marker = "o",color='tab:red',label="Given data",alpha = 0.8,zorder=3)

            plt.xlabel("$t$")
            plt.ylabel("$y_" + str(j)+"$")
            plt.legend(bbox_to_anchor=(1, 1))
            

            file_name = "roll-out" + " " + test_dict['system_name'] + "_j=" + str(j) + r"_sigma = " +str(round(test_dict['sigma'],3)) + r"_dt=" +str(round(dt,3)  )
            plt.savefig(plot + file_name + ".pdf", format='pdf', bbox_inches='tight')
            plt.close()

    for key in tests:
        errors_flow[key] = np.mean(errors_flow[key])
    return errors_flow



def compute_measurement_dist(measurements):
    measurements_c = measurements.copy()
    ms = measurements_c[0]['Flow error']['data']
    for key in ms:
        ms[key] = [ms[key]]

    for m in measurements_c[1:]:
        data = m['Flow error']['data']
        for key in data:
            ms[key].append(data[key])
    for key in ms:
        ms[key] = np.stack(ms[key])


    aggregated_measurements = measurements[0].copy()

    aggregated_measurements['Flow error']['mean'] = {}
    aggregated_measurements['Flow error']['sd'] = {}

    for key in aggregated_measurements['Flow error']['data']:
        aggregated_measurements['Flow error']['mean'][key] = np.mean(ms[key],axis=0)
        aggregated_measurements['Flow error']['sd'][key] = np.std(ms[key],axis=0)
    
    return aggregated_measurements


def experiment_flow_error(args,experiment_name,test_dict,n_samples,plot=False):
    measurements = []
    print("Running experiment for " + test_dict['system_name'])
    for i in range(n_samples):
        print("****************      EXPERIMENT NUMBER " + str(i+1) + "      *********************")
        args.seed = i
        m = experiment_flow(args,test_dict,plot=plot)
        measurements.append(m)
    aggregated_measurements = compute_measurement_dist(measurements)
    results_to_pickle(aggregated_measurements,test_dict,args,experiment_name)
    return aggregated_measurements


def results_to_pickle(aggregated_measurements,test_dict,args,experiment_name):
    filename = experiment_name + "_Flow error" + " " + test_dict['system_name'] + r", $\sigma$ = " +str(round(test_dict['sigma'],3))
    file = open(args.plot_dir + filename.lower().replace("$", "").replace("\\", "") + ".pkl","wb")
    pickle.dump(aggregated_measurements, file) 
    file.close()
 

