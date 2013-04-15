import h5py
from NeuroTools import stgen
import numpy as np
from matplotlib import pyplot as plt

from tsodyks_markram_plasticity_fit_AMPA import synthetic_conductance_signal as signal_generator_AMPA
from tsodyks_markram_plasticity_fit_AMPA import PULSE_CUTOFF as PULSE_CUTOFF_AMPA
from tsodyks_markram_plasticity_fit_NMDA import synthetic_conductance_signal as signal_generator_NMDA
from tsodyks_markram_plasticity_fit_NMDA import PULSE_CUTOFF as PULSE_CUTOFF_NMDA

from waveforms import n_g_peak_J

# jason's experimental data
jason_exp_frequencies = np.array([6.0, 9.5, 19.5, 29.5, 42.8, 70.0, 85.6, 138.0])
jason_exp_gAMPA_mean = np.array([0.017, 0.022, 0.055, 0.071, 0.110, 0.178, 0.190, 0.241])
jason_exp_gAMPA_err = np.array([0.003, 0.002, 0.011, 0.010, 0.008, 0.028, 0.021, 0.027])
jason_exp_gNMDA_mean = np.array([0.154, 0.242, 0.371, 0.612, 0.821, 1.325, 1.637, 2.225])
jason_exp_gNMDA_err = np.array([0.030, 0.045, 0.148, 0.111, 0.086, 0.174, 0.198, 0.401])
# jason's fit
jason_fit_frequencies = np.array([7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0, 105.0, 120.0, 135.0, 150.0])
jason_fit_gAMPA = np.array([0.0255, 0.0492, 0.0734, 0.0912, 0.1118, 0.1289, 0.1487, 0.1636, 0.1810, 0.1951, 0.2090, 0.2189, 0.2458, 0.2717, 0.2945, 0.3121])
jason_fit_gNMDA = np.array([0.1678, 0.3272, 0.4976, 0.6215, 0.7695, 0.8851, 1.0344, 1.1419, 1.2830, 1.3759, 1.4940, 1.5669, 1.7775, 1.9963, 2.2018, 2.3852])

frequencies = np.arange(7.5, 151., 7.5)
sim_length = 100000.
dt = 0.01
time_points = np.arange(0, sim_length, dt)
single_waveform_length_AMPA = np.searchsorted(time_points,
                                              PULSE_CUTOFF_AMPA)
single_waveform_length_NMDA = np.searchsorted(time_points,
                                              PULSE_CUTOFF_NMDA)

candidate_AMPA = [0.3275, 4.5, 0.5011, 0.0001329, 0.3493, 2.45, 23.0, 0.1260, 141.1, 0.2189, 0.1700, 0.3089, 0.123, 1.53, 7.0, 49.97, 0.2849, 12.66]
candidate_NMDA = [0.9501, 2.870, 0.8586, 10.31, 78.38, 0.03734, 178.34, 7.124]
st_gen = stgen.StGen()

average_conductances_AMPA = np.zeros(shape=frequencies.shape[0])
average_conductances_NMDA = np.zeros(shape=frequencies.shape[0])
for k, freq in enumerate(frequencies):
    pulse_train = st_gen.poisson_generator(rate=freq,
                                           t_start=0.,
                                           t_stop=sim_length,
                                           array=True)
    signal_AMPA_direct = signal_generator_AMPA(time_points,
                                               pulse_train,
                                               single_waveform_length_AMPA,
                                               dt,
                                               0.,
                                               *candidate_AMPA[0:9])
    signal_AMPA_spillover = signal_generator_AMPA(time_points,
                                                  pulse_train,
                                                  single_waveform_length_AMPA,
                                                  dt,
                                                  0.,
                                                  *candidate_AMPA[9:])
    signal_NMDA = signal_generator_NMDA(time_points,
                                        pulse_train,
                                        single_waveform_length_NMDA,
                                        dt,
                                        0.,
                                        *candidate_NMDA)
    average_conductances_AMPA[k] = (signal_AMPA_direct+signal_AMPA_spillover).sum()*dt/sim_length
    average_conductances_NMDA[k] = (1./n_g_peak_J)*signal_NMDA.sum()*dt/sim_length


fig_AMPA, ax_AMPA = plt.subplots()
ax_AMPA.errorbar(jason_exp_frequencies,
            jason_exp_gAMPA_mean,
            yerr=jason_exp_gAMPA_err,
            marker='o',
            ls='',
            color='k')
ax_AMPA.plot(jason_fit_frequencies,
        jason_fit_gAMPA,
        marker='s',
        color='r')
ax_AMPA.plot(frequencies,
        average_conductances_AMPA,
        marker='d',
        color='g')
ax_AMPA.set_xlabel('MF input rate (Hz)')
ax_AMPA.set_ylabel('time-averaged AMPAR-mediated conductance')

fig_NMDA, ax_NMDA = plt.subplots()
ax_NMDA.errorbar(jason_exp_frequencies,
            jason_exp_gNMDA_mean,
            yerr=jason_exp_gNMDA_err,
            marker='o',
            ls='',
            color='k')
ax_NMDA.plot(jason_fit_frequencies,
        jason_fit_gNMDA,
        marker='s',
        color='r')
ax_NMDA.plot(frequencies,
        average_conductances_NMDA,
        marker='d',
        color='g')
ax_NMDA.set_xlabel('MF input rate (Hz)')
ax_NMDA.set_ylabel('time-averaged NMDAR-mediated conductance')

plt.show()


