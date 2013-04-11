import random
import time
import inspyred
import numpy as np
from matplotlib import pyplot as plt


from waveforms import a_g_comp, rothman2012_AMPA_signal

DATA_DIR = "/home/ucbtepi/doc/jason_data/JasonsIAFmodel/Jason_Laurence_AMPA_NMDA_Trains"
NMDA_DIR = DATA_DIR + "/NMDA"
AMPA_DIR = DATA_DIR + "/AMPA"
TIME_DIR = DATA_DIR + "/StimTimes"
PULSE_CUTOFF = 200

class Rothman_AMPA_STP(inspyred.benchmarks.Benchmark):
    def __init__(self):
        # parameters: for both direct and spillover components, 7 for
        # the waveform and 2 for the plasticity.
        inspyred.benchmarks.Benchmark.__init__(self, dimensions=18)
        # load experimental data
        self.frequencies = [5, 10, 20, 30, 50, 80, 100, 150]
        self.n_trials = 4
        self.exp_data = []
        self.exp_pulses = []
        self.single_waveform_lengths = []
        self.timestep_sizes = []
        for freq in self.frequencies:
            for trial in range(self.n_trials):
                self.exp_pulses.append(np.loadtxt(TIME_DIR + "/gp{0}_{1}hz_times.txt".format(trial,
                                                                                             freq)))
                self.exp_data.append(np.loadtxt(AMPA_DIR + "/Avg_AMPA_{0}hz_G{1}.txt".format(freq,
                                                                                             trial)))
                self.single_waveform_lengths.append(np.searchsorted(self.exp_data[-1][:,0],
                                                                    PULSE_CUTOFF))
                self.timestep_sizes.append(np.diff(self.exp_data[-1][:100,0]).mean())
        # needed to generate random values for tau_rec in the _open_
        # interval (0,2)
        self.epsilon = 1e-15

        self.maximize = False

        self.bounds = [(0.1, 0.41), # d_tau_rise **direct (waveform)**
                       (0.4, 4.5), # d_a1
                       (0.07, 0.65), # d_a2
                       (0.0001, 0.055), # d_a3
                       (0.05, 0.35), # d_tau_dec1
                       (1.35, 2.45), # d_tau_dec2
                       (15., 23.), # d_tau_dec3
                       (0.01, 0.8), # d_u_se   **direct (STD)**
                       (9., 150.), # d_tau_dep
                       (0.1, 1.2), # s_tau_rise **spillover (waveform)**
                       (0.005, 0.21), # s_a1
                       (0.005, 0.42), # s_a2
                       (0.005, 0.104), # s_a3
                       (0.5, 1.53), # s_tau_dec1
                       (7.0, 8.0), # s_tau_dec2
                       (28.1, 50.), #s_tau_dec3
                       (0.001, .8), # s_u_se   **spillover (STD)**
                       (2., 80.)] # s_tau_dep

problem = Rothman_AMPA_STP()

bounder = inspyred.ec.Bounder([b[0] for b in problem.bounds],
                              [b[1] for b in problem.bounds])

def dep_table(pulse_train, u_se, tau_rec):
    # r[n,k] is the remaining fraction of synaptic resource before
    # nth pulse. The remaing fraction before the 0-th pulse is
    # supposed to be 1.
    r =  np.ones(shape=pulse_train.shape, dtype=np.float)
    pulse_intervals = np.diff(pulse_train)
    # evolve according to tsodyks-markram model
    for n in range(1, r.shape[0]):
        r[n] = 1 - np.exp(-(pulse_intervals[n-1])/tau_rec) * (1 - r[n-1]*(1-u_se))
    # by the (facilitation-less) tsodyks-markram model, the
    # plasticity factors are obtained by multiplying r by u_se, to
    # give the actual fraction of synaptic resource utilised at
    # any given pulse, and by another constant representing the
    # absolute synaptic efficacy (ie the response amplitude when
    # all the synaptic resource is used). We skip the last step,
    # as we are already optimising over the shape of the base
    # synaptic waveform.
    return r * u_se

def synthetic_conductance_signal(time_points, pulse_train, single_waveform_length, timestep_size, tau_rise, a1, a2, a3, tau_dec1, tau_dec2, tau_dec3, u_se, tau_rec):
    # this is meant to be used for a single component (direct or
    # spillover). The 'delay' compensates the offset that is present
    # between the experimental pulse times and the times when the
    # responses start to appear on the experimental recordings. It is
    # present in both the AMPA and the NMDA case, but with a different
    # value.
    delay = 0.5
    delay_time_points = int(round(delay/timestep_size))
    n_time_points = time_points.shape[0]
    dep_factors = dep_table(pulse_train, u_se, tau_rec)
    single_pulse_waveform = a_g_comp(time_points[0:single_waveform_length], tau_rise, a1, tau_dec1, a2, tau_dec2, a3, tau_dec3)
    signal = np.zeros(shape=n_time_points)
    for n, pulse in enumerate(pulse_train):
        start = np.searchsorted(time_points, pulse) + delay_time_points
        end = start + single_waveform_length
        signal[start:end] += (single_pulse_waveform * dep_factors[n])[:single_pulse_waveform.shape[0]-(end - n_time_points)]
    return signal

def fitness_to_experiment(cs):
    # fitness is defined as the average (across experimental datasets)
    # of the L2 norm between the synthetic and the experimental signal
    # divided by the square root of the number of time points, to
    # avoid unfair weighing in favour of longer recordings (in other
    # words, this should give equal weight to every time point in
    # every recording).
    distances = []
    for k, ep in enumerate(problem.exp_pulses):
        timepoints = problem.exp_data[k][:,0]
        signal_direct = synthetic_conductance_signal(timepoints,
                                                     ep,
                                                     problem.single_waveform_lengths[k],
                                                     problem.timestep_sizes[k],
                                                     *cs[:9])
        signal_spillover = synthetic_conductance_signal(timepoints,
                                                        ep,
                                                        problem.single_waveform_lengths[k],
                                                        problem.timestep_sizes[k],
                                                        *cs[9:])
        distances.append(np.linalg.norm(signal_direct+signal_spillover-problem.exp_data[k][:,1])/np.sqrt(timepoints.shape[0]))# + 0.15 * np.abs(normalised_difference_trace.sum()))
    return sum(distances)/len(distances)

def generator(random, args):
    return [random.uniform(bounder.lower_bound[k],
                           bounder.upper_bound[k]) for k in range(len(bounder.lower_bound))]

def evaluator(candidates, args):
    return [fitness_to_experiment(cs) for cs in candidates]


def main(plot=False):
    prng = random.Random()
    prng.seed(int(time.time()))
    max_evaluations = 2100
    pop_size = 70

    algorithm = inspyred.swarm.PSO(prng)
    #algorithm = inspyred.ec.EDA(prng)
    algorithm.topology = inspyred.swarm.topologies.ring_topology
    algorithm.terminator = inspyred.ec.terminators.evaluation_termination
    #algorithm.terminator = [inspyred.ec.terminators.evaluation_termination,
    #                        inspyred.ec.terminators.diversity_termination]
    final_pop = algorithm.evolve(evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                                 mp_evaluator=evaluator,
                                 mp_nprocs=7,
                                 generator=generator,
                                 maximize=problem.maximize,
                                 bounder=bounder,
                                 max_evaluations=max_evaluations,
                                 pop_size=pop_size,
                                 neighborhood_size=5,
                                 num_elites=1)
    # Sort and print the best individual
    final_pop.sort(reverse=True)
    selected_candidate = final_pop[0].candidate[:18]
    print("direct:    {0}".format(selected_candidate[:9]))
    print("spillover: {0}".format(selected_candidate[9:]))
    print("fitness:   {0}".format(final_pop[0].fitness))
    if plot:
        plot_optimisation_results(problem,
                                  selected_candidate,
                                  final_pop[0].fitness,
                                  max_evaluations,
                                  pop_size)

def plot_optimisation_results(problem, candidate, fitness, max_evaluations, pop_size):
    fig, ax = plt.subplots(nrows=8, ncols=4, figsize=(160,80), dpi=500)
    rothman_fitness = 0
    for k, ep in enumerate(problem.exp_pulses):
        timepoints = problem.exp_data[k][:,0]
        signal_direct = synthetic_conductance_signal(timepoints,
                                                     ep,
                                                     problem.single_waveform_lengths[k],
                                                     problem.timestep_sizes[k],
                                                     *candidate[0:9])
        signal_spillover = synthetic_conductance_signal(timepoints,
                                                        ep,
                                                        problem.single_waveform_lengths[k],
                                                        problem.timestep_sizes[k],
                                                        *candidate[9:])
        rothman_signal = rothman2012_AMPA_signal(timepoints,
                                                 ep,
                                                 problem.single_waveform_lengths[k],
                                                 problem.timestep_sizes[k])
        rothman_fitness += np.linalg.norm(rothman_signal - problem.exp_data[k][:,1])/np.sqrt(timepoints.shape[0])
        ax.flat[k].plot(timepoints, problem.exp_data[k][:,1], color='k', linewidth=3)
        ax.flat[k].scatter(ep, np.zeros(shape=ep.shape)-0.05, color='k')
        ax.flat[k].plot(timepoints, rothman_signal, linewidth=1, color='r')
        ax.flat[k].plot(timepoints, signal_direct+signal_spillover, linewidth=1, color='g')
        #ax.flat[k].plot(timepoints, signal_direct, color='r')
        #ax.flat[k].plot(timepoints, signal_spillover, color='c')
    rothman_fitness /= len(problem.exp_pulses)
    fig.suptitle('parameters: {0}\n fitness: {1} max_evaluations: {2} pop_size: {3}\nRothman2012 fitness: {4}'.format(candidate, fitness, max_evaluations, pop_size, rothman_fitness))
    plt.savefig('Rothman_AMPA_TM_fit_{0}.png'.format(time.time()))

def plot_example_trace(problem=None, candidate=None):
    #a = np.loadtxt(NMDA_DIR + "/NMDA_10hz_G2_181103.txt")
    a = np.loadtxt(AMPA_DIR + "/Avg_AMPA_10hz_G2.txt")
    t = np.loadtxt(TIME_DIR + "/gp2_10hz_times.txt")

    fig, ax = plt.subplots()
    ax.plot(a[:,0], a[:,1])
    ax.scatter(t, np.zeros(shape=t.shape)-0.05, color='r')

    if problem and candidate:
        signal_direct = synthetic_conductance_signal(a[:,0],
                                                     t,
                                                     problem.single_waveform_lengths[5],
                                                     problem.timestep_sizes[5],
                                                     *candidate[1:9])
        signal_spillover = synthetic_conductance_signal(a[:,0],
                                                        t,
                                                        problem.single_waveform_lengths[5],
                                                        problem.timestep_sizes[5],
                                                        *candidate[9:])
        ax.plot(a[:,0], signal_direct+signal_spillover, linewidth=2.5)
        ax.plot(a[:,0], signal_direct)
        ax.plot(a[:,0], signal_spillover)

    plt.show()

if __name__ == '__main__':
    main(plot=True)

#to profile, from shell:
#python -m cProfile -o output.pstats tsodyks_markram_plasticity_fit.py
#gprof2dot -f pstats output.pstats | dot -Tpng -o output.png
