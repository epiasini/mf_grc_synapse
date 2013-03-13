import random
import time
import inspyred
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt

from waveforms import a_g_comp

import pdb

DATA_DIR = "/home/ucbtepi/doc/jason_data/JasonsIAFmodel/Jason_Laurence_AMPA_NMDA_Trains"
NMDA_DIR = DATA_DIR + "/NMDA"
AMPA_DIR = DATA_DIR + "/AMPA"
TIME_DIR = DATA_DIR + "/StimTimes"
PULSE_CUTOFF = 200


class Rothman_AMPA_STP(inspyred.benchmarks.Benchmark):
    def __init__(self):
        # parameters: for both direct and spillover components, 7 for
        # the waveform and 2 for the plasticity
        inspyred.benchmarks.Benchmark.__init__(self, dimensions=18)
        # load experimental data
        self.frequencies = [5, 10, 30, 50, 80, 100, 150]
        self.n_trials = 4
        self.exp_data = []
        self.exp_pulses = []
        self.binary_exp_pulses = []
        self.single_waveform_lengths = []
        for freq in self.frequencies:
            for trial in range(self.n_trials):
                self.exp_pulses.append(np.loadtxt(TIME_DIR + "/gp{0}_{1}hz_times.txt".format(trial,
                                                                                             freq)))
                self.exp_data.append(np.loadtxt(AMPA_DIR + "/Avg_AMPA_{0}hz_G{1}.txt".format(freq,
                                                                                             trial)))
                self.binary_exp_pulses.append(np.zeros(shape=self.exp_data[-1].shape[0]))
                self.binary_exp_pulses[-1][np.searchsorted(self.exp_data[-1][:,0],
                                                          self.exp_pulses[-1])] = 1
                self.single_waveform_lengths.append(np.searchsorted(self.exp_data[-1][:,0],
                                                                    PULSE_CUTOFF))
        # needed to generate random values for tau_rec in the _open_
        # interval (0,2)
        self.epsilon = 1e-15

        self.bounds = [(0.2, 0.3), # d_tau_rise **direct (waveform)**
                       (0.4, 0.5), # d_a1
                       (0.06, 0.10), # d_a2
                       (0.01, 0.02), # d_a3
                       (0.2, 0.3), # d_tau_dec1
                       (2., 3.), # d_tau_dec2
                       (18., 25.), # d_tau_dec3
                       (0.2, 1.), # d_u_se   **direct (STD)**
                       (30., 100.), # d_tau_dep
                       (0.6, 1.0), # s_tau_rise **spillover (waveform)
                       (0.08, 0.10), # s_a1
                       (0.05, 0.2), # s_a2
                       (0.005, 0.010), # s_a3
                       (0.5, 1.0), # s_tau_dec1
                       (5., 10.), # s_tau_dec2
                       (27., 42.), #s_tau_dec3
                       (0.2, 1.), # s_u_se   **spillover (STD)**
                       (30., 100.)] # s_tau_dep

        self.bounder = inspyred.ec.Bounder([b[0] for b in self.bounds],
                                           [b[1] for b in self.bounds])
        self.maximize = False

    def dep_table(self, pulse_train, u_se, tau_rec):
        # r[n,k] is the remaining fraction of synaptic resource before
        # nth pulse. The remaing fraction before the 0-th pulse is
        # supposed to be 1.
        r =  np.ones(shape=pulse_train.shape, dtype=np.float)
        pulse_intervals = np.diff(pulse_train)
        # evolve according to tsodyks-markram model
        for n in range(1, r.shape[0]):
            r[n] = 1 - np.exp(-(pulse_intervals[n-1])/tau_rec) * (1 - r[n-1]*(1-u_se))
        # by the (facilitation-less) tsodyks-markram model, the
        # plasticity factors are obtained by multiplying r by u_se (to
        # give the actual fraction of synaptic resource utilised at
        # any givn pulse) and by another constant representing the
        # absolute synaptic efficacy (ie the response amplitude when
        # all the synaptic resource is used). We omit this step, as we
        # are only interested in relative scaling with respect to the
        # first pulse.
        return r

    def synthetic_conductance_signal(self, time_points, pulse_train, binary_pulse_train, single_waveform_length, tau_rise, a1, a2, a3, tau_dec1, tau_dec2, tau_dec3, u_se, tau_rec):
        # this is meant to be used for a single component (direct or
        # spillover)
        dep_factors = self.dep_table(pulse_train, u_se, tau_rec)
        single_pulse_waveform = np.array([a_g_comp(t, tau_rise, a1, tau_dec1, a2, tau_dec2, a3, tau_dec3) for t in time_points[0:single_waveform_length]])
        raise Exception()
        #signal = np.zeros(shape=time_points.shape[0])
        #for n, pulse in enumerate(pulse_train):
        #    print n, pulse
        #    start = np.searchsorted(time_points, pulse)
        #    signal[start:start+single_pulse_waveform.shape[0]] += single_pulse_waveform * dep_factors[n]
        #return signal
        return scipy.signal.fftconvolve(binary_pulse_train, single_pulse_waveform)[:time_points.shape[0]] #this doesn't consider plasticity!

    def fitness_to_experiment(self, cs):
        print 'a'
        distances = []
        for k, ep in enumerate(self.exp_pulses):
            signal_direct = self.synthetic_conductance_signal(self.exp_data[k][:,0],
                                                              ep,
                                                              self.binary_exp_pulses[k],
                                                              self.single_waveform_lengths[k],
                                                              *cs[:9])
            signal_spillover = self.synthetic_conductance_signal(self.exp_data[k][:,0],
                                                                 ep,
                                                                 self.binary_exp_pulses[k],
                                                                 self.single_waveform_lengths[k],
                                                                 *cs[9:])
            distances.append(np.linalg.norm(signal_direct+signal_spillover-self.exp_data[k][:,1]))
        return sum(distances)

    def generator(self, random, args):
        return [random.uniform(self.bounder.lower_bound[k],
                               self.bounder.upper_bound[k]) for k in range(self.dimensions)]

    def evaluator(self, candidates, args):
        return [self.fitness_to_experiment(cs) for cs in candidates]


def main():
    #a = np.loadtxt(NMDA_DIR + "/NMDA_10hz_G2_181103.txt")
    #a = np.loadtxt(AMPA_DIR + "/Avg_AMPA_10hz_G2.txt")
    #t = np.loadtxt(TIME_DIR + "/gp2_10hz_times.txt")

    #fig, ax = plt.subplots()
    #ax.plot(a[:,0], a[:,1])
    #ax.scatter(t, np.zeros(shape=t.shape)-0.05, color='r')
    #plt.show()
    prng = random.Random()
    prng.seed(int(time.time()))

    problem = Rothman_AMPA_STP()
    algorithm = inspyred.swarm.PSO(prng)
    algorithm.topology = inspyred.swarm.topologies.ring_topology
    algorithm.terminator = inspyred.ec.terminators.evaluation_termination
    final_pop = algorithm.evolve(evaluator=problem.evaluator,
                                 generator=problem.generator,
                                 maximize=problem.maximize,
                                 bounder=problem.bounder,
                                 max_evaluations=18,
                                 pop_size=6,
                                 neighborhood_size=5)
    # Sort and print the best individual
    final_pop.sort(reverse=True)
    print(final_pop[0])

if __name__ == '__main__':
    main()
