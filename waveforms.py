import numpy as np
from numpy import exp, log
from functools import partial
from matplotlib import pyplot as plt

# ad : ampa direct
# as : ampa spillover
# ...

a_g_peak_J = 0.62 # nS
a_sd_ratio_J = 0.34

ad_N_J = 1.94
ad_tau_rise_J = 0.161
ad_a1_J = .89382
ad_tau_dec_1_J = 0.317
ad_a2_J = .09568
ad_tau_dec_2_J = 1.729
ad_a3_J = .01050
ad_tau_dec_3_J = 19.685

as_N_J = 1.7406
as_tau_rise_J = 0.3811
as_a1_J = .427811
as_tau_dec_1_J = 1.3838
as_a2_J = .535912
as_tau_dec_2_J = 7.2695
as_a3_J = .0362763
as_tau_dec_3_J = 30.856

n_g_peak_J = 0.18 # this takes into account Mg2+ block at the voltage
                  # used for the conductance recordings (carried out
                  # in voltage clamp). Actual value to be used in the
                  # model should be 0.37 (Rothman2009, methods
                  # section)

n_N_J = 1.
n_tau_rise_J = 1.14
n_a1_J = .641176
n_tau_dec_1_J = 8.10
n_a2_J = .358824
n_tau_dec_2_J = 37.

n_K1slope_J = 38.427
n_K2slope_J = 28.357
n_VK1on_J = 84.784
n_VK1off_J = -119.51
n_VK2off_J = -45.895

pulse_duration = 2*as_tau_dec_3_J
time_points = np.arange(0.01, pulse_duration, 0.001)

ampa_fit=True
nmda_fit=True
p7_9_fit=False

def peak_time(tau_rise, tau_dec):
    return np.divide(tau_dec*tau_rise, tau_dec-tau_rise)*log(tau_dec/tau_rise)

def norm_factor(tau_rise, tau_dec):
    return 1./(exp(-peak_time(tau_dec, tau_rise)/tau_dec) - exp(-peak_time(tau_dec, tau_rise)/tau_rise))

def g_subcomp(t, tau_rise, a, tau_dec):
    return a*norm_factor(tau_rise, tau_dec)* (exp(-t/tau_dec) - exp(-t/tau_rise))

def a_g_comp(t, tau_rise, a1, tau_dec_1, a2, tau_dec_2, a3, tau_dec_3):
    return g_subcomp(t, tau_rise, a1, tau_dec_1) + g_subcomp(t, tau_rise, a2, tau_dec_2) + g_subcomp(t, tau_rise, a3, tau_dec_3)

def n_g_comp(t, tau_rise, a1, tau_dec_1, a2, tau_dec_2):
    return g_subcomp(t, tau_rise, a1, tau_dec_1) + g_subcomp(t, tau_rise, a2, tau_dec_2)

def a_g_J_unnorm_comp(t, N, tau_rise, a1, tau_dec_1, a2, tau_dec_2, a3, tau_dec_3):
    return pow(1-exp(-t/tau_rise), N) * (a1*exp(-t/tau_dec_1) + a2*exp(-t/tau_dec_2) + a3*exp(-t/tau_dec_3))

def n_g_unblock_J_unnorm(t, N, tau_rise, a1, tau_dec_1, a2, tau_dec_2):
    return pow(1-exp(-t/tau_rise), N) * (a1*exp(-t/tau_dec_1) + a2*exp(-t/tau_dec_2))

def n_block_eric(v, K1slope, K2slope, VK1on, VK1off, VK2off):
     K1ON = exp( -( v - VK1on ) / K1slope )
     K1OFF = exp( ( v - VK1off ) / K1slope )
     K2OFF = exp( -( v - VK2off ) / K2slope )
     block = ( K1OFF + K2OFF ) / ( K1OFF + K2OFF + K1ON )
     return block

def block_takahashi_1996(v):
    z = 2
    F = 96.4853415
    R = 8.3144621
    T = 24.5 + 273.15

    S = (z * F * v) / (2 * R * T)

    #mgOut = 1.5 * 1000 # Eric's extracellular concentration
    mgOut = 1.0 * 1000 # matches Takahashi 1996 data


    # Jason's fit to P08 data
    k1on = 0.95334
    k1off = 1876.33
    k2off = 3.33674
    Fbinding = 0.852014
    Fpermeation = 0.966919

    K1ON = mgOut * k1on * exp( -Fbinding * S )
    K1OFF = k1off * exp( Fbinding * S )
    K2OFF = k2off * exp( -Fpermeation * S )

    block = ( K1OFF + K2OFF ) / ( K1OFF + K2OFF + K1ON )
    return block

def n_block(v, eta, gamma):
    return 1/(1 + eta * exp(-gamma*v))

def rothman2012_plast_table(spikes, D, TauD, Dmin, F, TauF, Fmax):
    d = 1.
    f = 1.
    intvl = 0.
    DF = np.ones(shape=spikes.shape[0])
    for i, pt in enumerate(spikes):
        if i>0:
            intvl = spikes[i] - spikes[i-1]
        if i == 0 or D == 1:
            d = 1.
        else:
            d = 1. + (d - 1.) * np.exp(-intvl / TauD)
            d = max(d, Dmin)

        if i == 0 or F == 1:
            f = 1.
        else:
            f = 1. + (f - 1.) * np.exp(-intvl / TauF)
            f = min(f, Fmax)

        DF[i] = d * f
        d *= D
        f *= F
    return DF

def rothman2012_NMDA_signal(time_points, pulse_train, single_waveform_length, timestep_size):
    n_time_points = time_points.shape[0]
    # there seems to be an offset between the experimental pulse times
    # and the response recordings
    offset = 1.0 # (ms)
    offset_time_points = int(round(offset/timestep_size))

    # plasticity parameters
    D = 0.9
    TauD = 70.
    Dmin = 0.1
    F = 1.7
    TauF = 3.5
    Fmax = 3.4
    plast_factors = rothman2012_plast_table(pulse_train, D, TauD, Dmin, F, TauF, Fmax)
    # waveform shape
    n_g_unblock_J_values = n_g_unblock_J_unnorm(time_points[0:single_waveform_length], n_N_J, n_tau_rise_J, n_a1_J, n_tau_dec_1_J, n_a2_J, n_tau_dec_2_J)
    max_n_g_unblock_J_values = n_g_unblock_J_values.max()
    n_g_unblock_J_norm_values = n_g_unblock_J_values/max_n_g_unblock_J_values
    single_pulse_waveform = n_g_unblock_J_norm_values
    # construct signal
    signal = np.zeros(shape = n_time_points)
    for n, pulse in enumerate(pulse_train):
        start = np.searchsorted(time_points, pulse) + offset_time_points
        end = start + single_waveform_length
        signal[start:end] += (single_pulse_waveform * plast_factors[n])[:single_pulse_waveform.shape[0]-(end - n_time_points)]
    return n_g_peak_J * signal

def rothman2012_AMPA_signal(time_points, pulse_train, single_waveform_length, timestep_size):
    n_time_points = time_points.shape[0]
    # there seems to be an offset between the experimental pulse times
    # and the response recordings
    offset = 0.5
    offset_time_points = int(round(offset/timestep_size))

    # -- DIRECT --
    # plasticity
    D = 0.6
    TauD = 50
    Dmin = 0.1
    F = 1
    TauF = None
    Fmax = None
    plast_factors = rothman2012_plast_table(pulse_train, D, TauD, Dmin, F, TauF, Fmax)
    # waveform shape
    ad_g_J_values = a_g_J_unnorm_comp(time_points[0:single_waveform_length], ad_N_J, ad_tau_rise_J, ad_a1_J, ad_tau_dec_1_J, ad_a2_J, ad_tau_dec_2_J, ad_a3_J, ad_tau_dec_3_J)
    ad_g_J_norm_values = ad_g_J_values/ad_g_J_values.max()
    single_pulse_waveform = ad_g_J_norm_values
    # construct signal
    signal_direct = np.zeros(shape = n_time_points)
    for n, pulse in enumerate(pulse_train):
        start = np.searchsorted(time_points, pulse) + offset_time_points
        end = start + single_waveform_length
        signal_direct[start:end] += (single_pulse_waveform * plast_factors[n])[:single_pulse_waveform.shape[0]-(end - n_time_points)]

    # -- SPILLOVER --
    # spillover component plasticity
    D = 0.95
    TauD = 50
    Dmin = 0.6
    F = 1
    TauF = None
    Fmax = None
    plast_factors = rothman2012_plast_table(pulse_train, D, TauD, Dmin, F, TauF, Fmax)
    # waveform shape
    as_g_J_values = a_g_J_unnorm_comp(time_points[0:single_waveform_length], as_N_J, as_tau_rise_J, as_a1_J, as_tau_dec_1_J, as_a2_J, as_tau_dec_2_J, as_a3_J, as_tau_dec_3_J)
    as_g_J_norm_values = as_g_J_values/as_g_J_values.max()
    single_pulse_waveform = as_g_J_norm_values
    # construct signal
    signal_spillover = np.zeros(shape = n_time_points)
    for n, pulse in enumerate(pulse_train):
        start = np.searchsorted(time_points, pulse) + offset_time_points
        end = start + single_waveform_length
        signal_spillover[start:end] += (single_pulse_waveform * plast_factors[n])[:single_pulse_waveform.shape[0]-(end - n_time_points)]

    # sum direct and spillover components and normalise again
    signal = signal_direct + a_sd_ratio_J*signal_spillover
    return signal/signal.max()

def plast_table_AMPA(pulse_train, u_se, tau_rec):
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

def synthetic_conductance_signal_direct_AMPA(time_points, pulse_train, single_waveform_length, timestep_size, delay, tau_rise, a1, a2, tau_dec1, tau_dec2, u_se, tau_rec):
    # this is meant to be used for a single component (direct or
    # spillover). The 'delay' compensates the offset that is present
    # between the experimental pulse times and the times when the
    # responses start to appear on the experimental recordings. It is
    # present in both the AMPA and the NMDA case, but with a different
    # value.

    n_time_points = time_points.shape[0]
    signal = np.zeros(shape=n_time_points)
    # compute a table of all the plasticity-scaled waveforms that we
    # will need to add to form the total conductance train.
    plast_factors = plast_table_AMPA(pulse_train, u_se, tau_rec)
    single_pulse_waveform = a_g_comp(time_points[0:single_waveform_length],
                                     tau_rise,
                                     a1,
                                     tau_dec1,
                                     a2,
                                     tau_dec2,
                                     0,
                                     1)
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

def synthetic_conductance_signal_spillover_AMPA(time_points, pulse_train, single_waveform_length, timestep_size, delay, tau_rise, a1, a2, a3, tau_dec1, tau_dec2, tau_dec3, u_se, tau_rec):
    # this is meant to be used for a single component (direct or
    # spillover). The 'delay' compensates the offset that is present
    # between the experimental pulse times and the times when the
    # responses start to appear on the experimental recordings. It is
    # present in both the AMPA and the NMDA case, but with a different
    # value.

    n_time_points = time_points.shape[0]
    signal = np.zeros(shape=n_time_points)
    # compute a table of all the plasticity-scaled waveforms that we
    # will need to add to form the total conductance train.
    plast_factors = plast_table_AMPA(pulse_train, u_se, tau_rec)
    single_pulse_waveform = a_g_comp(time_points[0:single_waveform_length],
                                     tau_rise,
                                     a1,
                                     tau_dec1,
                                     a2,
                                     tau_dec2,
                                     a3,
                                     tau_dec3)
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

def plast_table_NMDA(pulse_train, u_0, tau_rec, tau_fac):
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

def synthetic_conductance_signal_NMDA(time_points, pulse_train, single_waveform_length, timestep_size, delay, tau_rise, a1, a2, tau_dec1, tau_dec2, u_se, tau_rec, tau_fac):
    # the 'delay' compensates the offset that is present between the
    # experimental pulse times and the times when the responses start
    # to appear on the experimental recordings. It is present in both
    # the AMPA and the NMDA case, but with a different value.

    n_time_points = time_points.shape[0]
    signal = np.zeros(shape=n_time_points)
    # compute a table of all the plasticity-scaled waveforms that we
    # will need to add to form the total conductance train.
    plast_factors = plast_table_NMDA(pulse_train, u_se, tau_rec, tau_fac)
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


if __name__ == '__main__':
    from scipy.optimize import curve_fit
    from scipy.integrate import trapz
    param_file = open('fitted_parameters.txt', 'w')

    #====AMPA====
    if ampa_fit:
        # prepare the two 'scaled' components (direct and s.over) to fit the multi_decay_syn expressions to
        ad_g_J_values = np.array([a_g_J_unnorm_comp(t, ad_N_J, ad_tau_rise_J, ad_a1_J, ad_tau_dec_1_J, ad_a2_J, ad_tau_dec_2_J, ad_a3_J, ad_tau_dec_3_J) for t in time_points])
        as_g_J_values = np.array([a_g_J_unnorm_comp(t, as_N_J, as_tau_rise_J, as_a1_J, as_tau_dec_1_J, as_a2_J, as_tau_dec_2_J, as_a3_J, as_tau_dec_3_J) for t in time_points])

        max_ad_g_J_values = ad_g_J_values.max()
        max_as_g_J_values = as_g_J_values.max()

        a_g_J_unnorm_values = ad_g_J_values/max_ad_g_J_values + a_sd_ratio_J*(as_g_J_values/max_as_g_J_values)
        max_a_g_J_unnorm_values = a_g_J_unnorm_values.max()
        a_g_J_values_norm = a_g_J_unnorm_values/max_a_g_J_unnorm_values
        a_g_J_values_scaled_to_max = a_g_peak_J*a_g_J_values_norm

        ad_g_J_scaled_component = a_g_peak_J*ad_g_J_values/(max_ad_g_J_values*max_a_g_J_unnorm_values)
        as_g_J_scaled_component = a_g_peak_J*a_sd_ratio_J*as_g_J_values/(max_as_g_J_values*max_a_g_J_unnorm_values)

        # fitting
        ad_params_f = curve_fit(a_g_comp, time_points, ad_g_J_scaled_component, [ad_tau_rise_J, ad_a1_J, ad_tau_dec_1_J, ad_a2_J, ad_tau_dec_2_J, ad_a3_J, ad_tau_dec_3_J])[0]
        ad_g_fitted = partial(a_g_comp, tau_rise=ad_params_f[0], a1=ad_params_f[1], tau_dec_1=ad_params_f[2], a2=ad_params_f[3], tau_dec_2=ad_params_f[4], a3=ad_params_f[5], tau_dec_3=ad_params_f[6])
        as_params_f = curve_fit(a_g_comp, time_points, as_g_J_scaled_component, [as_tau_rise_J, as_a1_J, as_tau_dec_1_J, as_a2_J, as_tau_dec_2_J, as_a3_J, as_tau_dec_3_J])[0]
        as_g_fitted = partial(a_g_comp, tau_rise=as_params_f[0], a1=as_params_f[1], tau_dec_1=as_params_f[2], a2=as_params_f[3], tau_dec_2=as_params_f[4], a3=as_params_f[5], tau_dec_3=as_params_f[6])

        def a_g(t):
            return ad_g_fitted(t) + as_g_fitted(t)

        # estimate error on waveform integral
        a_g_values = np.array([a_g(t) for t in time_points])
        integral_diff = np.abs(trapz(a_g_J_values_scaled_to_max - a_g_values, dx=0.001))
        print(integral_diff)

        param_file.write("AMPA direct parameters: {params}\n".format(params=zip(['tau_rise', 'a1', 'tau_dec_1', 'a2', 'tau_dec_2', 'a3', 'tau_dec_a3'], ad_params_f)))
        param_file.write("AMPA spillover parameters: {params}\n".format(params=zip(['tau_rise', 'a1', 'tau_dec_1', 'a2', 'tau_dec_2', 'a3', 'tau_dec_a3'], as_params_f)))
        a_fig = plt.figure()
        a_ax = a_fig.add_subplot(111)
        a_ax.plot(time_points, a_g_J_values_scaled_to_max, label="Rothman 2009")
        a_ax.plot(time_points, a_g_values, label="NeuroML")
        a_ax.set_title('AMPA conductance waveform: direct+spillover')
        a_ax.set_xlabel('time (ms)')
        a_ax.set_ylabel('conductance (nS)')
        a_ax.legend()
        a_ax.grid()

    #====NMDA====
    if nmda_fit:
        voltage_points = np.arange(-80, -40, 0.001)

        n_g_unblock_J_values = np.array([n_g_unblock_J_unnorm(t, n_N_J, n_tau_rise_J, n_a1_J, n_tau_dec_1_J, n_a2_J, n_tau_dec_2_J) for t in time_points])
        max_n_g_unblock_J_values = n_g_unblock_J_values.max()
        n_g_unblock_J_norm_values = n_g_unblock_J_values/max_n_g_unblock_J_values
        n_g_unblock_J_scaled = n_g_peak_J*n_g_unblock_J_norm_values

        n_block_eric_values = np.array([n_block_eric(v, n_K1slope_J, n_K2slope_J, n_VK1on_J, n_VK1off_J, n_VK2off_J) for v in voltage_points])

        n_g_unblock_params_f = curve_fit(n_g_comp, time_points, n_g_unblock_J_scaled, [n_tau_rise_J, n_g_peak_J*n_a1_J, n_tau_dec_1_J, n_g_peak_J*n_a2_J, n_tau_dec_2_J])[0]
        n_block_params_f = curve_fit(n_block, voltage_points, n_block_eric_values, [1, 0.0035])[0]

        n_g_unblock_fitted = partial(n_g_comp, tau_rise=n_g_unblock_params_f[0], a1=n_g_unblock_params_f[1], tau_dec_1=n_g_unblock_params_f[2], a2=n_g_unblock_params_f[3], tau_dec_2=n_g_unblock_params_f[4])
        n_block_fitted = partial(n_block, eta=n_block_params_f[0], gamma=n_block_params_f[1])

        n_g_unblock_values = np.array([n_g_unblock_fitted(t) for t in time_points])
        n_block_values = np.array([n_block_fitted(v) for v in voltage_points])

        integral_diff = np.abs(trapz(n_g_unblock_J_scaled - n_g_unblock_values, dx=0.001))
        print(integral_diff)


        param_file.write("NMDA (unbblocked) parameters: {params}\n".format(params=zip(['tau_rise', 'a1', 'tau_dec_1', 'a2', 'tau_dec_2'], n_g_unblock_params_f)))
        param_file.write("NMDA block expression: {params}\n".format(params=zip(['eta', 'gamma'], n_block_params_f)))

        n_fig = plt.figure()
        n_ax = n_fig.add_subplot(111)
        n_ax.plot(time_points, n_g_unblock_J_scaled, label="Rothman 2009")
        n_ax.plot(time_points, n_g_unblock_values, label="NeuroML")
        n_ax.set_title('NMDA waveform (unblocked)')
        n_ax.set_xlabel('time (ms)')
        n_ax.set_ylabel('conductance (nS)')
        n_ax.legend()
        n_ax.grid()

        nb_fig = plt.figure()
        nb_ax = nb_fig.add_subplot(111)
        nb_ax.plot(voltage_points, n_block_eric_values, label="Rothman 2009")
        nb_ax.plot(voltage_points, n_block_values, label="NeuroML")
        nb_ax.set_title('NMDA block')
        nb_ax.set_xlabel('membrane potential (mV)')
        nb_ax.set_ylabel('block factor')
        nb_ax.legend()
        nb_ax.grid()

    #===P7_9 blocking factor====
    if p7_9_fit:

        lower_limit = -80
        upper_limit = 0
        plot_voltage_points = np.arange(-100, 40, 0.001) # plot region
        voltage_points = np.arange(lower_limit, upper_limit, 0.001) # fit region
        takahashi_block_values = np.array([block_takahashi_1996(v) for v in voltage_points])

        p7_9_block_params_f = curve_fit(n_block, voltage_points, takahashi_block_values, [1, 0.0035])[0]
        n_block_fitted = partial(n_block, eta=p7_9_block_params_f[0], gamma=p7_9_block_params_f[1])
        n_block_values = np.array([n_block_fitted(v) for v in voltage_points])

        p7_9_file = open('p7_9_fit.txt', 'w')
        p7_9_file.write("NeuroMLv1 fit to Jason's Woodhull Fit of Takahashi 1996 results for p7-9:\n")
        p7_9_file.write("Fitting region: from {lower}mV to {upper}mV\n".format(lower=lower_limit, upper=upper_limit))
        p7_9_file.write("NMDA block expression: {params}\n".format(params=zip(['eta', 'gamma'], p7_9_block_params_f)))
        p7_9_file.close()

        p7_9_fig = plt.figure()
        p7_9_ax = p7_9_fig.add_subplot(111)
        p7_9_ax.plot(plot_voltage_points, [block_takahashi_1996(v) for v in plot_voltage_points], label="Jason's fit to Takahashi1996 p7-9")
        p7_9_ax.plot(plot_voltage_points, [n_block_fitted(v) for v in plot_voltage_points], label="NeuroMLv1\n%s" % str(zip(['eta', 'gamma'], p7_9_block_params_f)))
        p7_9_ax.set_title('NMDA block (fitted between %dmV and %dmV)' % (lower_limit, upper_limit))
        p7_9_ax.set_xlabel('membrane potential (mV)')
        p7_9_ax.set_ylabel('unblock factor')
        p7_9_ax.legend(loc='upper left')
        p7_9_ax.grid()



    param_file.close()
    plt.show()
