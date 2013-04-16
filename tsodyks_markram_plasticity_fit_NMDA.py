import random
import time
import glob
import inspyred
import h5py
import numpy as np
from matplotlib import pyplot as plt


from waveforms import n_g_comp, rothman2012_NMDA_signal

DATA_DIR = "/home/ucbtepi/doc/jason_data/JasonsIAFmodel/Jason_Laurence_AMPA_NMDA_Trains"
NMDA_DIR = DATA_DIR + "/NMDA"
AMPA_DIR = DATA_DIR + "/AMPA"
TIME_DIR = DATA_DIR + "/StimTimes"
PULSE_CUTOFF = 500

class Rothman_NMDA_STP(inspyred.benchmarks.Benchmark):
    def __init__(self):
        # parameters: 5 for the waveform and 3 for the plasticity.
        inspyred.benchmarks.Benchmark.__init__(self, dimensions=8)
        # load experimental data
        self.frequencies = [5, 10, 20, 30, 50, 80, 100, 150]
        self.n_protocols = 4
        self.exp_data = []
        self.exp_pulses = []
        #self.binary_exp_pulses = []
        self.single_waveform_lengths = []
        self.timestep_sizes = []
        with h5py.File("NMDA_experimental_data.hdf5") as data_repo:
            for freq in self.frequencies:
                for prot in range(self.n_protocols):
                    data_group = data_repo['/{0}/{1}'.format(freq, prot)]
                    self.exp_pulses.append(np.array(data_group['pulse_times']))
                    self.exp_data.append(np.array(data_group['average_waveform']))
                    self.single_waveform_lengths.append(np.searchsorted(self.exp_data[-1][:,0],
                                                                        PULSE_CUTOFF))
                    self.timestep_sizes.append(np.diff(self.exp_data[-1][:100,0]).mean())

        self.maximize = False

        self.bounds = [(0.4, 2.), # d_tau_rise **direct (waveform)**
                       (0.4, 4.5), # d_a1
                       (0.07, 1.20), # d_a2
                       (4, 15), # d_tau_dec1
                       (25, 150), # d_tau_dec2
                       (0.01, 0.5), # d_u_se   **direct (STD)**
                       (9., 300.), # d_tau_dep
                       (0.5, 12.)] # d_tau_fac

problem = Rothman_NMDA_STP()

bounder = inspyred.ec.Bounder([b[0] for b in problem.bounds],
                              [b[1] for b in problem.bounds])

def plast_table(pulse_train, u_0, tau_rec, tau_fac):
    # r[n] is the remaining fraction of synaptic resource before nth
    # pulse. The remaing fraction before the 0-th pulse is supposed to
    # be 1.
    r =  np.ones(shape=pulse_train.shape, dtype=np.float)
    # u[n] is the fraction of existing synaptic resource used for the
    # nth pulse. u[0] is u_0.
    u = np.zeros(shape=pulse_train.shape, dtype=np.float)
    u[0] = u_0
    pulse_intervals = np.diff(pulse_train)
    # evolve according to tsodyks-markram model
    for n in range(1, r.shape[0]):
        u[n] = u_0 + np.exp(-pulse_intervals[n-1]/tau_fac) * u[n-1] * (1-u_0)
        r[n] = 1 - np.exp(-pulse_intervals[n-1]/tau_rec) * (1 - r[n-1]*(1-u[n-1]))
    # by the tsodyks-markram model, the plasticity factors are
    # obtained by multiplying r by u, to give the actual fraction of
    # synaptic resource utilised at any given pulse, and by another
    # constant representing the absolute synaptic efficacy (ie the
    # response amplitude when all the synaptic resource is used). We
    # skip the last step, as we are already optimising over the shape
    # of the base synaptic waveform.
    return r * u

def synthetic_conductance_signal(time_points, pulse_train, single_waveform_length, timestep_size, delay, tau_rise, a1, a2, tau_dec1, tau_dec2, u_se, tau_rec, tau_fac):
    # the 'delay' compensates the offset that is present between the
    # experimental pulse times and the times when the responses start
    # to appear on the experimental recordings. It is present in both
    # the AMPA and the NMDA case, but with a different value.

    n_time_points = time_points.shape[0]
    signal = np.zeros(shape=n_time_points)
    # compute a table of all the plasticity-scaled waveforms that we
    # will need to add to form the total conductance train.
    plast_factors = plast_table(pulse_train, u_se, tau_rec, tau_fac)
    single_pulse_waveform = n_g_comp(time_points[0:single_waveform_length],
                                     tau_rise,
                                     a1,
                                     tau_dec1,
                                     a2,
                                     tau_dec2)
    pulse_waveforms = np.outer(plast_factors, single_pulse_waveform)
    # find the indexes of the 'signal' array that correspond to the
    # pulse times, as they are where we need to start summing each of
    # the scaled waveforms we have precomputed.
    start_points = np.floor((pulse_train + delay - time_points[0])/timestep_size)
    end_points = start_points + single_waveform_length
    # sum all waveforms with the appropriate offsets
    for n, waveform in enumerate(pulse_waveforms):
        signal[start_points[n]:end_points[n]] += waveform[:single_waveform_length-(end_points[n] - n_time_points)]
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
        signal = synthetic_conductance_signal(timepoints,
                                              ep,
                                              problem.single_waveform_lengths[k],
                                              problem.timestep_sizes[k],
                                              1.,
                                              *cs)
        distances.append(np.linalg.norm(signal-problem.exp_data[k][:,1])/np.sqrt(timepoints.shape[0]))# + 0.15 * np.abs(normalised_difference_trace.sum()))
    return sum(distances)/len(distances)

def generator(random, args):
    return [random.uniform(bounder.lower_bound[k],
                           bounder.upper_bound[k]) for k in range(len(bounder.lower_bound))]

def evaluator(candidates, args):
    return [fitness_to_experiment(cs) for cs in candidates]


def main(plot=False):
    prng = random.Random()
    prng.seed(int(time.time()))
    max_evaluations = 22800
    pop_size = 140

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
    selected_candidate = final_pop[0].candidate[:8]
    print("direct:    {0}".format(selected_candidate[:8]))
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

    java_fit_time_points = []
    java_fit_signals = []
    with h5py.File("NMDA_jason_fit_traces.hdf5") as java_fit_data_repo:
        for freq in problem.frequencies:
            for prot in range(problem.n_protocols):
                data_group = java_fit_data_repo['/{0}/{1}'.format(freq, prot)]
                # jason's modeled traces have their first peak
                # normalised to 1 and they don't have any offset, so
                # we must add 1ms to the time dimension and scale by
                # the value used for the maximum NMDA peak conductance
                # (0.18nS)
                java_fit_time_points.append(np.array(data_group['average_waveform'][:-1,0]) + 1.)
                java_fit_signals.append(np.array(data_group['average_waveform'][:-1,1]) * 0.18)

    for k, ep in enumerate(problem.exp_pulses):
        timepoints = problem.exp_data[k][:,0]
        signal = synthetic_conductance_signal(timepoints,
                                              ep,
                                              problem.single_waveform_lengths[k],
                                              problem.timestep_sizes[k],
                                              1.,
                                              *candidate)
        # we use the trace we compute in Python to calculate the
        # fitness of Jason's model, whereas the plotted trace is taken
        # straight from the synthetic data he generated in Java. The
        # two coincide anyway, and this is just to avoid headaches
        # with time axis misalignments.
        rothman_signal_python = rothman2012_NMDA_signal(timepoints,
                                                        ep,
                                                        problem.single_waveform_lengths[k],
                                                        problem.timestep_sizes[k])
        rothman_signal_java = java_fit_signals[k]
        rothman_fitness += np.linalg.norm(rothman_signal_python - problem.exp_data[k][:,1])/np.sqrt(timepoints.shape[0])
        ax.flat[k].plot(timepoints, problem.exp_data[k][:,1], color='k', linewidth=3)
        ax.flat[k].scatter(ep, np.zeros(shape=ep.shape)-0.1, color='k')
        ax.flat[k].plot(java_fit_time_points[k], rothman_signal_java, linewidth=1, color='r')
        ax.flat[k].plot(timepoints, signal, linewidth=1, color='g')
    rothman_fitness /= len(problem.exp_pulses)
    fig.suptitle('parameters: {0}\n fitness: {1} max_evaluations: {2} pop_size: {3}\nRothman2012 fitness: {4}'.format(candidate, fitness, max_evaluations, pop_size, rothman_fitness))
    plt.savefig('Rothman_NMDA_TM_fit_{0}.png'.format(time.time()))

if __name__ == '__main__':
    main(plot=True)

#to profile, from shell:
#python -m cProfile -o output.pstats tsodyks_markram_plasticity_fit.py
#gprof2dot -f pstats output.pstats | dot -Tpng -o output.png
