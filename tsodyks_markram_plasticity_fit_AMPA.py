#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model fitting for J Rothman's data on the mossy fibre to granule cell
synapse, AMPA component.

The model is the sum of a direct (fast) and spillover (slow)
component, and both of them exhibit short term depression (STD). The
basic waveforms are multiexponential (one rise time and two decay
times for the direct component, one rise time and three decay times
for the spillover component), and STD is modeled after Tsodyks and
Markram 1997.

The experimental data consists of 32 conductance waveforms (8
frequencies, 4 different Poisson spike trains generated at any given
frequency; each trace an average across four cells; expressed in nS)
paired with the sequences of stimulation times used to elicit
them. The core idea of the fitting method is to measure how bad a
given model (ie, set of parameter defining basic waveform shapes and
plasticity) is by using it to generate 32 synthetic conductance traces
corresponding to the experimental ones, and then measuring the average
distance between the synthetic and the experimental traces.

Once the fits have been done, the basic waveforms have to undergo
another scaling step before being used in a actual simulation. This is
because the experimental data we are fitting to have been taken on a
small number of cells (n=4), and better estimates for the peak values
of the AMPA conductance are available in the literature. Effectively,
we use our fits to determine the basic waveform shapes and plasticity
properties but we refer to the peak AMPA value in Sargent2005 (0.63
nS) for the final scaling.
"""
import random
import time
import h5py
import inspyred
import numpy as np
from matplotlib import pyplot as plt

from waveforms import a_g_comp, rothman2012_AMPA_signal
from waveforms import synthetic_conductance_signal_direct_AMPA as synthetic_conductance_signal_direct
from waveforms import synthetic_conductance_signal_spillover_AMPA as synthetic_conductance_signal_spillover


DATA_DIR = "/home/ucbtepi/doc/jason_data/JasonsIAFmodel/Jason_Laurence_AMPA_NMDA_Trains"
NMDA_DIR = DATA_DIR + "/NMDA"
AMPA_DIR = DATA_DIR + "/AMPA"
TIME_DIR = DATA_DIR + "/StimTimes"
PULSE_CUTOFF = 300

class Rothman_AMPA_STP(inspyred.benchmarks.Benchmark):
    def __init__(self):
        # parameters: for the plasticity, 2 parameters for each
        # (direct and spillover) component. For the basic waveforms, 5
        # parameters for the direct component and 7 parameters for the
        # spillover.
        inspyred.benchmarks.Benchmark.__init__(self, dimensions=16)
        # load experimental data
        self.frequencies = [5, 10, 20, 30, 50, 80, 100, 150]
        self.n_protocols = 4
        self.exp_data = []
        self.exp_pulses = []
        self.single_waveform_lengths = []
        self.timestep_sizes = []
        with h5py.File("AMPA_experimental_data.hdf5") as data_repo:
            for freq in self.frequencies:
                for prot in range(self.n_protocols):
                    data_group = data_repo['/{0}/{1}'.format(freq, prot)]
                    self.exp_pulses.append(np.array(data_group['pulse_times']))
                    self.exp_data.append(np.array(data_group['average_waveform']))
                    self.single_waveform_lengths.append(np.searchsorted(self.exp_data[-1][:,0],
                                                                        PULSE_CUTOFF))
                    self.timestep_sizes.append(np.diff(self.exp_data[-1][:100,0]).mean())

        self.maximize = False

        self.bounds = [(0.1, 0.41), # d_tau_rise **direct (waveform)**
                       (0.4, 6.5), # d_a1
                       (0.07, 0.65), # d_a2
                       (0.05, 0.90), # d_tau_dec1
                       (1.35, 4.2), # d_tau_dec2
                       (0.01, 0.8), # d_u_se   **direct (STD)**
                       (9., 200.), # d_tau_dep
                       (0.1, 1.2), # s_tau_rise **spillover (waveform)**
                       (0.005, 0.31), # s_a1
                       (0.005, 0.72), # s_a2
                       (0.005, 0.353), # s_a3
                       (0.23, 2.53), # s_tau_dec1
                       (3.0, 8.0), # s_tau_dec2
                       (28.1, 60.), #s_tau_dec3
                       (0.001, .8), # s_u_se   **spillover (STD)**
                       (2., 80.)] # s_tau_dep

# this is not very elegant, but needed for simple parallelisation
# (multiprocessing) with inspyred
problem = Rothman_AMPA_STP()
bounder = inspyred.ec.Bounder([b[0] for b in problem.bounds],
                              [b[1] for b in problem.bounds])

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
        signal_direct = synthetic_conductance_signal_direct(timepoints,
                                                            ep,
                                                            problem.single_waveform_lengths[k],
                                                            problem.timestep_sizes[k],
                                                            0.54,
                                                            *cs[:7])
        signal_spillover = synthetic_conductance_signal_spillover(timepoints,
                                                                  ep,
                                                                  problem.single_waveform_lengths[k],
                                                                  problem.timestep_sizes[k],
                                                                  0.54,
                                                                  *cs[7:])
        distances.append(np.linalg.norm(signal_direct+signal_spillover-problem.exp_data[k][:,1])/np.sqrt(timepoints.shape[0]))# + 0.15 * np.abs(normalised_difference_trace.sum()))
    return sum(distances)/len(distances)

def generator(random, args):
    return [random.uniform(bounder.lower_bound[k],
                           bounder.upper_bound[k]) for k in range(len(bounder.lower_bound))]

def evaluator(candidates, args):
    return [fitness_to_experiment(cs) for cs in candidates]


def main(plot=False):
    """Perform the main Tsodyks-Markram fit."""
    prng = random.Random()
    prng.seed(int(time.time()))
    max_evaluations = 63000
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
    selected_candidate = final_pop[0].candidate[:16]
    print("direct:    {0}".format(selected_candidate[:7]))
    print("spillover: {0}".format(selected_candidate[7:]))
    print("fitness:   {0}".format(final_pop[0].fitness))
    if plot:
        plot_optimisation_results(selected_candidate,
                                  final_pop[0].fitness,
                                  max_evaluations,
                                  pop_size)

def plot_optimisation_results(candidate, fitness, max_evaluations, pop_size):
    """
    Plot a comparison of the model, the experimental data and the
    original model by Rothman (Schwartz2012) across all experimental
    traces. The figure gets saved to disk in the current folder.
    """
    fig, ax = plt.subplots(nrows=8, ncols=4, figsize=(160,80), dpi=500)
    rothman_fitness = 0

    java_fit_time_points = []
    java_fit_signals = []
    with h5py.File("AMPA_jason_fit_traces.hdf5") as java_fit_data_repo:
        for freq in problem.frequencies:
            for prot in range(problem.n_protocols):
                data_group = java_fit_data_repo['/{0}/{1}'.format(freq, prot)]
                java_fit_time_points.append(np.array(data_group['average_waveform'][:-1,0]))
                java_fit_signals.append(np.array(data_group['average_waveform'][:-1,1]))

    for k, ep in enumerate(problem.exp_pulses):
        timepoints = problem.exp_data[k][:,0]
        signal_direct = synthetic_conductance_signal_direct(timepoints,
                                                            ep,
                                                            problem.single_waveform_lengths[k],
                                                            problem.timestep_sizes[k],
                                                            0.54,
                                                            *candidate[0:7])
        signal_spillover = synthetic_conductance_signal_spillover(timepoints,
                                                                  ep,
                                                                  problem.single_waveform_lengths[k],
                                                                  problem.timestep_sizes[k],
                                                                  0.54,
                                                                  *candidate[7:])
        # rothman_signal = rothman2012_AMPA_signal(timepoints,
        #                                          ep,
        #                                          problem.single_waveform_lengths[k],
        #                                          problem.timestep_sizes[k])
        rothman_signal = java_fit_signals[k]
        rothman_fitness += np.linalg.norm(rothman_signal - problem.exp_data[k][:,1])/np.sqrt(timepoints.shape[0])
        ax.flat[k].plot(timepoints, problem.exp_data[k][:,1], color='k', linewidth=3)
        ax.flat[k].scatter(ep, np.zeros(shape=ep.shape)-0.05, color='k')
        ax.flat[k].plot(timepoints, rothman_signal, linewidth=1, color='r')
        #ax.flat[k].plot(java_fit_time_points[k], java_fit_signals[k], color='c')
        ax.flat[k].plot(timepoints, signal_direct+signal_spillover, linewidth=1, color='g')
        #ax.flat[k].plot(timepoints, signal_direct, color='r')
        #ax.flat[k].plot(timepoints, signal_spillover, color='c')
    rothman_fitness /= len(problem.exp_pulses)
    fig.suptitle('parameters: {0}\n fitness: {1} max_evaluations: {2} pop_size: {3}\nRothman2012 fitness: {4}'.format(candidate, fitness, max_evaluations, pop_size, rothman_fitness))
    plt.savefig('Rothman_AMPA_TM_fit_{0}.png'.format(time.time()))

def scale_to_sargent(candidate):
    """Scale fit values to match peak AMPA reported in Sargent2005."""
    sargent_peak = 0.63 # (nS)

    timestep = 0.01
    timepoints = np.arange(0, 300, timestep)
    pulse_times = np.array([10.])
    single_waveform_length = timepoints.shape[0]
    signal_direct = synthetic_conductance_signal_direct(timepoints,
                                                        pulse_times,
                                                        single_waveform_length,
                                                        timestep,
                                                        0.,
                                                        *candidate[:7])
    signal_spillover = synthetic_conductance_signal_spillover(timepoints,
                                                              pulse_times,
                                                              single_waveform_length,
                                                              timestep,
                                                              0.,
                                                              *candidate[7:])
    signal = signal_direct + signal_spillover
    scaling_factor = sargent_peak/signal.max()
    scaled_signal = scaling_factor * signal
    scaled_candidate = candidate[:]
    for k in [1,2,8,9,10]:
        # scale 'amplitude' parameters
        scaled_candidate[k] *= scaling_factor
    print(scaled_candidate)
    # rounded scaled candidate should be [0.3274, 3.724, 0.3033,
    # 0.3351, 1.651, 0.1249, 131.0, 0.5548, 0.2487, 0.2799, 0.1268,
    # 0.4, 4.899, 43.1, 0.2792, 14.85]

    fig, ax = plt.subplots()
    ax.plot(timepoints, signal, label="fit to Rothman data")
    ax.plot(timepoints, scaled_signal, label="scaled to peak value by Sargent")
    ax.legend(loc="best")
    fig.suptitle("direct {0}\nspillover {1}".format(scaled_candidate[:7],
                                                    scaled_candidate[7:]))
    plt.show()

def plot_lems_comparison(candidate):
    """
    Plot a comparison between python and LEMS model implementation for
    a specific trace: 20Hz, pulse train #0. The LEMS data is for the
    synapse with Sargent scaling.
    """
    lems_data = np.loadtxt("gAMPA_LEMS_20Hz_G0.dat")
    timepoints = lems_data[:,0] * 1e3 # transform to ms
    lems_trace = lems_data[:,1] * 1e9 # transform to nS

    timestep = 0.025 # has to match what was used in the LEMS simulation
    pulse_times = problem.exp_pulses[8]
    single_waveform_length = problem.single_waveform_lengths[8]

    signal_direct = synthetic_conductance_signal_direct(timepoints,
                                                        pulse_times,
                                                        single_waveform_length,
                                                        timestep,
                                                        0.,
                                                        *candidate[:7])
    signal_spillover = synthetic_conductance_signal_spillover(timepoints,
                                                              pulse_times,
                                                              single_waveform_length,
                                                              timestep,
                                                              0.,
                                                              *candidate[7:])

    fig, ax = plt.subplots()
    ax.scatter(pulse_times, np.zeros(shape=pulse_times.shape)-0.05, color='k')
    ax.plot(timepoints, lems_trace, linewidth=1.5)
    ax.plot(timepoints, signal_direct+signal_spillover, linewidth=1.5)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit",
                        help=main.__doc__,
                        action="store_true")
    parser.add_argument("--scale",
                        help=scale_to_sargent.__doc__,
                        action="store_true")
    parser.add_argument("--compare-exp",
                        help=plot_optimisation_results.__doc__,
                        action="store_true")
    parser.add_argument("--compare-lems",
                        help=plot_lems_comparison.__doc__,
                        action="store_true")

    # this is the official result of the optimisation. It's used for
    # comparisons etc
    candidate = [0.3274, 4.492, 0.3659, 0.3351, 1.651, 0.1249, 131.0, 0.5548, 0.3000, 0.3376, 0.153, 0.4, 4.899, 43.10, 0.2792, 14.85]
    scaled_candidate = [0.3274, 3.724, 0.3033, 0.3351, 1.651, 0.1249, 131.0, 0.5548, 0.2487, 0.2799, 0.1268, 0.4, 4.899, 43.1, 0.2792, 14.85]
    fitness = 0.021
    max_evaluations = 21000
    pop_size = 140

    args = parser.parse_args()
    if args.fit:
        main(plot=True)
    if args.scale:
        scale_to_sargent(candidate)
    if args.compare_exp:
        plot_optimisation_results(candidate,
                                  fitness,
                                  max_evaluations,
                                  pop_size)
    if args.compare_lems:
        plot_lems_comparison(scaled_candidate)

#to profile, from shell:
#python -m cProfile -o output.pstats tsodyks_markram_plasticity_fit.py
#gprof2dot -f pstats output.pstats | dot -Tpng -o output.png
