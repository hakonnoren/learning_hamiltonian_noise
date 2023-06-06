import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm 

from numerical_tests.variance_optimization import *
from .visualization import set_plot_params


def plot_error_bars(measurements_list,test_dict_list,args,save=False):

    for i in range(len(test_dict_list)):
        test_dict_list[i]['dts_testing'] = args.dt/2**(test_dict_list[i]['scales'])

    step_sizes = test_dict_list[0]['dts_testing']
    scales = test_dict_list[0]['scales']

    x = np.linspace(0.0, 1.0, 20)

    rgb = cm.get_cmap("tab20")(x)

    variable = "Flow error"
    n_ticks = len(test_dict_list[0]['dts_testing'])
    n_integrators = len(measurements_list[0][variable]['data'].keys())
    n = n_integrators
    n_experiments = len(measurements_list)

    if n%2 == 0:
        ncol = n//2
    else:
        ncol = n//2 + 1
    
    set_plot_params()
    plt.rcParams["figure.figsize"] = (int(6/1.5)*n_ticks,3*n_experiments)


    fig, axs = plt.subplots(n_experiments, 2, sharex='col', gridspec_kw={'width_ratios': [5, 3]},sharey='row')
    fig.tight_layout()
    fig.subplots_adjust(top=0.8,wspace=0,hspace=0)
    xs = []
    
    def add_one_exp(measurements,ax1,ax2):
        i = 0
        min_mean = 100

        for integrator in measurements[variable]['data']:
            if integrator == "ISO Stormer":
                integrator_label = "ISO Störmer"
            else:
                integrator_label = integrator
            j = 0
            for mean,std in zip(measurements[variable]['mean'][integrator],measurements[variable]['sd'][integrator]):
                x = [0.25*(i+j*(n+1))]
                xs.append(x[0])
                if j == 0:
        
                    ax1.bar(x,height = [mean],width=0.15,color=rgb[i],label = integrator_label)
                else: 
                    ax1.bar(x,height = [mean],width=0.15,color=rgb[i])
                ax1.errorbar(x,[mean],yerr = [std],color="black",ls="none",capsize=1)
                j+=1 

            i += 1

        for i,integrator in enumerate(measurements["Training time"]['data']):
            time = measurements["Training time"]['data'][integrator]
            mean = measurements['Flow error']['mean'][integrator]
            std = measurements['Flow error']['sd'][integrator]
            ax2.plot(time,mean,marker="o",color=rgb[i],linewidth=2)

            if np.min(mean) < min_mean:
                min_mean = np.min(mean)


        ax1.set_ylabel(measurements[variable]['y_lab'])



        if True:
            ax1.yaxis.grid(color="0.9")
            ax2.yaxis.grid(color="0.9")
            ax2.xaxis.grid(color="0.9")
            ax1.set_axisbelow(True)
            ax2.set_axisbelow(True)
            ax1.set_yscale('log')
            ax2.set_yscale('log')



    for i,m in enumerate(measurements_list):
        xs = []
        add_one_exp(m,axs[i,0],axs[i,1])
        if i == 0:
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),
            fancybox=True, shadow=True, ncol=ncol)

        axs[i,0].legend([],[],title=test_dict_list[i]['system_name'] + r', $\sigma = $ ' + str(round(test_dict_list[i]['sigma'],3)),loc=2)

    title_2 = "Time and accuracy"
    title_1 = "Flow error"

    axs[0,0].set_title(title_1)
    axs[0,1].set_title(title_2)

    xs = np.sort(xs)
    ticks = [np.mean(xs[n_integrators*i:n_integrators*(i+1)]) for i in range(n_ticks)]
    xticks = ( r'$h = $' + str(round(h,2)) for h in step_sizes)

    xticks = ( r'$h = $ '+ str(round(h,2)) + r',  $N_1 =$ ' + str(args.n_timesteps_train*2**sc) for h,sc in zip(step_sizes,scales))

    axs[n_experiments-1,1].set_xlabel("Training time")
    axs[n_experiments-1,0].set_xticks(ticks, xticks)

    if save:
        file_name = "flow_error_and_time"  + "_" + ".pdf"
        fig.savefig(args.plot_dir + file_name, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_variances_mirk4_rk4(hamiltonian_name,dts,args,m4_os_mcs,m4_mii_mcs,rk4_os_mcs,save = False):

    set_plot_params()

    T = dts[-1]*args.n_timesteps_train
    Ns = np.int16(T/dts)
    dts = T/Ns
    dts = dts[:-1]

    title = "Propagation of noise, "+ hamiltonian_name


    fill_plot = lambda x,y,c : plt.fill_between(x,np.mean(y,axis=0) - np.std(y,axis=0),np.mean(y,axis=0) + np.std(y,axis=0),color=c,alpha=0.2)


    plt.rcParams["figure.figsize"] = (5,3.5)

    fill_plot(dts,rk4_os_mcs,"tab:blue")
    fill_plot(dts,m4_os_mcs,"tab:red")
    fill_plot(dts,m4_mii_mcs,"tab:orange")


    plt.plot(dts,np.mean(rk4_os_mcs,axis=0),color='tab:blue',label="RK4 OS",linestyle='--')
    plt.plot(dts,np.mean(m4_os_mcs,axis=0),color='tab:red',label="MIRK4 OS",linestyle='--')
    plt.plot(dts,np.mean(m4_mii_mcs,axis=0),color='tab:orange',label="MIRK4 MII",linestyle='--')

    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))

    plt.title(title)

    plt.xlabel(r'Step size $h$')
    plt.ylabel(r'Spectral radius $\overline \rho$')
    plt.xticks(np.round(dts,2))

    plt.legend()
    if not save:
        plt.show()
    else:
        file_name = args.plot_dir + title + ".pdf"
        plt.savefig(file_name, format='pdf', bbox_inches='tight')
        plt.close()


def plot_global_error(ts,errors_rk4,errors_mirk4,errors_dgm4,errors_mimp4,save):
    fig = plt.figure(figsize=(6.5, 4))

    plt.semilogy(ts, errors_rk4, label='RK4')
    plt.semilogy(ts, errors_mirk4, label='MIRK4')
    plt.semilogy(ts, errors_dgm4, label='DGM4')
    plt.semilogy(ts, errors_mimp4, label='MIMP4')
    plt.xlabel("$t$")
    plt.ylabel("$L_2$-error")
    plt.legend()

    if save:
        fig.savefig('plots/global_error_double_pendulum.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_energy_error(ts,energies_rk4,energies_mirk4,energies_dgm4,energies_mimp4,save):
    fig = plt.figure(figsize=(6.5, 4))
    plt.title('Energy error')
    plt.semilogy(ts, np.abs(energies_rk4-energies_rk4[0]), label='RK4')
    plt.semilogy(ts, np.abs(energies_mirk4-energies_mirk4[0]), label='MIRK4')
    plt.semilogy(ts, np.abs(energies_dgm4-energies_dgm4[0]), label='DGM4')
    plt.semilogy(ts, np.abs(energies_mimp4-energies_mimp4[0]), label='MIMP4')
    plt.xlabel("$t$")
    plt.ylabel("$|H(t)-H(t_0)|$")
    plt.legend()

    if save:
        fig.savefig('plots/energy_error_double_pendulum.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()