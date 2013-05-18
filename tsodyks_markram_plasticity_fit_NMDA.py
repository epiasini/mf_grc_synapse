#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model fitting for J Rothman's data on the mossy fibre to granule cell
synapse, NMDA component.

The model is a multiexponential basic waveform (one rise time and two
decay times) with short term plasticity (depression and facilitation)
modeled after Tsodyks and Markram 1998.

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
of the NMDA conductance are available in the literature. Effectively,
we use our fits only to determine the basic waveform shapes and
plasticity properties. The final scaling for the waveforms, assuming
total absence of Mg2+ block, is given following what Schwartz2012 did
in the case when the parameter 'phi' was set to 1 (assumed to be a
'physiological' value). That is to say, the peak of the unblocked NMDA
waveform is set to the value reported in Sargent2005 for the peak of
the AMPA waveform (0.63 nS). Finally, note that the actual value of
peak gNMDAR will always be far less than gAMPAR during simulations
because of the block: using the blockage model in Schwartz2012, at
-40mV (spiking threshold for the granule cell) the actual conductance
is about 20% of the unblocked value.
"""
import random
import time
import glob
import inspyred
import h5py
import numpy as np
from matplotlib import pyplot as plt

from waveforms import rothman2012_NMDA_signal
from waveforms import synthetic_conductance_signal_NMDA as synthetic_conductance_signal

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
        distances.append(np.linalg.norm(signal-problem.exp_data[k][:,1])/np.sqrt(timepoints.shape[0]))
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
    with h5py.File("NMDA_jason_fit_traces.hdf5") as java_fit_data_repo:
        for freq in problem.frequencies:
            for prot in range(problem.n_protocols):
                data_group = java_fit_data_repo['/{0}/{1}'.format(freq, prot)]
                # jason's modeled traces have their first peak
                # normalised to 1 and they don't have any offset, so
                # we must add 1ms to the time dimension and
                # 'denormalise' them by scaling them by the value he
                # measured as the experimental NMDA peak conductance
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

def scale_to_sargent(candidate):
    """
    Scale fit values to match peak AMPA reported in Sargent2005.
    (unblocked, and phi=1!)
    """
    sargent_peak = 0.63 # (nS)

    timestep = 0.01
    timepoints = np.arange(0, 300, timestep)
    pulse_times = np.array([10.])
    single_waveform_length = timepoints.shape[0]
    signal = synthetic_conductance_signal(timepoints,
                                          pulse_times,
                                          single_waveform_length,
                                          timestep,
                                          0.,
                                          *candidate)
    scaling_factor = sargent_peak/signal.max()
    scaled_signal = scaling_factor * signal
    scaled_candidate = candidate[:]
    for k in [1,2]:
        # scale 'amplitude' parameters
        scaled_candidate[k] *= scaling_factor
    print(scaled_candidate)
    # rounded scaled candidate should be [0.8647, 17.00, 2.645, 13.52,
    # 121.9, 0.0322, 236.1, 6.394]

    fig, ax = plt.subplots()
    ax.plot(timepoints, signal, label="fit to Rothman data")
    ax.plot(timepoints, scaled_signal, label="scaled to peak value by Sargent (unblocked, phi=1)")
    ax.legend(loc="best")
    fig.suptitle("parameters {0}".format(scaled_candidate))
    plt.show()

def plot_lems_comparison(candidate):
    """
    Plot a comparison between python and LEMS model implementation for
     a specific trace: 20Hz, pulse train #0. The LEMS data is for a
     completely unblocked synapse with Sargent scaling.
    """
    lems_data = np.loadtxt("gNMDA_LEMS_20Hz_G0.dat")
    timepoints = lems_data[:,0] * 1e3 # transform to ms
    lems_trace = lems_data[:,1] * 1e9 # transform to nS

    timestep = 0.025 # has to match what was used in the LEMS simulation
    pulse_times = problem.exp_pulses[8]
    single_waveform_length = problem.single_waveform_lengths[8]

    signal = synthetic_conductance_signal(timepoints,
                                          pulse_times,
                                          single_waveform_length,
                                          timestep,
                                          0.,
                                          *candidate)

    fig, ax = plt.subplots()
    ax.scatter(pulse_times, np.zeros(shape=pulse_times.shape)-0.05, color='k')
    ax.plot(timepoints, lems_trace, linewidth=1.5)
    ax.plot(timepoints, signal, linewidth=1.5)
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
    candidate = [0.8647, 3.683, 0.5730, 13.52, 121.9, 0.03220, 236.1, 6.394]
    scaled_candidate = [0.8647, 17.00, 2.645, 13.52, 121.9, 0.0322, 236.1, 6.394]
    fitness = 0.035
    max_evaluations = 22800
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
