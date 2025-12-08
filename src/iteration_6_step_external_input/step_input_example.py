from brian2 import *

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True



# 1s UP, 1 s Down
# What is different between the Up and the Down input? the number of Neurons that get activated
def generate_step_input(experiment):
    P_high = PoissonGroup(experiment.network_params.N, rates=50 * Hz)
    P_low = PoissonGroup(experiment.network_params.N, rates=5 * Hz)

    G = NeuronGroup(experiment.network_params.N, '''
        dv/dt = -v*Hz : 1
                                                ''', threshold='v>10', reset='v=0',
                    refractory=experiment.neuron_params.tau_rp, method='euler')

    G.v[:] = 0

    S_high = Synapses(P_high, G, model="w: 1", on_pre='v_post += w * 0.05')
    S_low = Synapses(P_low, G, model="w: 1", on_pre='v_post -= w * 0.05')
    S_high.connect(p=experiment.network_params.epsilon)
    S_low.connect(i=list(S_high.i), j=list(S_high.j))

    # Control which group is active via w values

    S_high.w = 1
    S_low.w = 0

    @network_operation(dt=100 * ms)
    def toggle_inputs():
        if int(defaultclock.t / second) % 2 == 0:
            S_high.w[:] = 1
            S_low.w[:] = 0
        else:
            S_high.w[:] = 0
            S_low.w[:] = 1

    rate_monitor = PopulationRateMonitor(G)
    spike_monitor = SpikeMonitor(G)
    v_monitor = StateMonitor(source=G[
                                    experiment.network_params.N_E - experiment.network_params.neurons_to_record: experiment.network_params.N_E + experiment.network_params.neurons_to_record],
                             variables="v", record=True)
    run(experiment.sim_time)

    fig = plt.figure(figsize=(10, 12))

    fig.suptitle(f''' Developing a step input model ''')

    gs = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[4, 1, 2])

    ax_spikes, ax_rates, ax_voltages = gs.subplots(sharex="col")

    ax_spikes.plot(spike_monitor.t / ms, spike_monitor.i, "|")
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz)

    for i in range(0, 5):
        ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v, label=f"Exc {i}")
        ax_voltages.plot(v_monitor.t / ms, v_monitor[experiment.network_params.neurons_to_record + i].v,
                         label=f"Inh {experiment.network_params.neurons_to_record + i}")

    ax_spikes.set_yticks([])
    ax_rates.set_xlabel("t [ms]")
    ax_voltages.legend(loc="best")

    plt.subplots_adjust(hspace=0)
    plt.show(block=False)
    plt.close(fig)