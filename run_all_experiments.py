from set_variables import *
from utils.analytical_hamiltonian import HamiltonianSystem
from numerical_tests.flow_error import experiment_flow_error
from numerical_tests.variance_optimization import get_variances_mirk4_rk4_std
from numerical_tests.energy_preservation import *
from utils.plotting import plot_error_bars,plot_variances_mirk4_rk4,plot_global_error,plot_energy_error


save_plots = True
plot_roll_out_time = True

test_dicts = [test_dict_fput,test_dict_hh,test_dict_dp]
measurements = []


print("Computing flow error:")

for test_dict in test_dicts:
    if test_dict['system_name'] in ['Hénon-Heiles',"FPUT"]:
        args.sep = True
    else:
        args.sep = False
    measurments = experiment_flow_error(args,"mii",test_dict,n_samples=n_samples,plot=plot_roll_out_time)
    measurements.append(measurments)

plot_error_bars(measurements,test_dicts,args,save=save_plots)


print("Computing variance in optimization target:")

n_mc_samples = 10_000
n_runs = 10
dts = np.linspace(0.1,0.8,num=7)

hamiltonian_system = HamiltonianSystem(hamiltonian_double_pendulum)
hamiltonian_name = "Double pendulum"

m4_os_mcs,m4_mii_mcs,rk4_os_mcs = get_variances_mirk4_rk4_std(hamiltonian_system,n_runs,sigma,dts,args,n_mc_samples)
plot_variances_mirk4_rk4(hamiltonian_name,dts,args,m4_os_mcs,m4_mii_mcs,rk4_os_mcs,save = save_plots)


print("Computing error in energy preservation with fourth order method:")

N=1000
T=500

dt = T/N
x0 = np.array([0.1,0.3,-0.4,0.2], np.float64)
xsref = get_ref_solution(f,rk4,x0,N,T)
ts = np.linspace(0,T,N+1, np.float64)

errors_rk4,energies_rk4 = get_errors_rk4(dt,x0,N,xsref)
errors_mirk4,energies_mirk4 = get_errors_mirk4(dt,x0,N,xsref)
errors_mimp4,energies_mimp4 = get_errors_mimp4(dt,x0,N,xsref)
errors_dgm4,energies_dgm4 = get_errors_dgm4(dt,x0,N,xsref)

plot_global_error(ts,errors_rk4,errors_mirk4,errors_dgm4,errors_mimp4,save=save_plots)
plot_energy_error(ts,energies_rk4,energies_mirk4,energies_dgm4,energies_mimp4,save=save_plots)