import numpy as np
from numpy import exp, log
from scipy.optimize import curve_fit
from scipy.integrate import trapz
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

n_g_peak_J = 0.37

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

if __name__ == '__main__':
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
