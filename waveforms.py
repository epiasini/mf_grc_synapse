import numpy as np
from numpy import exp, log
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from functools import partial

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

pulse_duration = 2*as_tau_dec_3_J

def peak_time(tau_rise, tau_dec):
    return (tau_dec*tau_rise)/(tau_dec-tau_rise)*log(tau_dec/tau_rise)

def norm_factor(tau_rise, tau_dec):
    return 1./(exp(-peak_time(tau_dec, tau_rise)/tau_dec) - exp(-peak_time(tau_dec, tau_rise)/tau_rise))

def g_subcomp(t, tau_rise, tau_dec):
    return norm_factor(tau_rise, tau_dec)* (exp(-t/tau_dec) - exp(-t/tau_rise))

def a_g_comp(t, tau_rise, a1, tau_dec_1, a2, tau_dec_2, a3, tau_dec_3):
    return a1*g_subcomp(t, tau_rise, tau_dec_1) + a2*g_subcomp(t, tau_rise, tau_dec_2) + a3*g_subcomp(t, tau_rise, tau_dec_3)



def a_g_J_unnorm_comp(t, N, tau_rise, a1, tau_dec_1, a2, tau_dec_2, a3, tau_dec_3):
    return pow(1-exp(-t/tau_rise), N) * (a1*exp(-t/tau_dec_1) + a2*exp(-t/tau_dec_2) + a3*exp(-t/tau_dec_3))

time_points = np.arange(0, pulse_duration, 0.001)
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

