from numpy import exp, log
from scipy.optimize import fmin

# ad : ampa direct
# as : ampa spillover
# ...

ad_tau_rise_J = 0.161
ad_dec_taus = [0.317, 1.729, 19.685]
ad_rise_taus = [ad_tau_rise_J*(dec_tau/(ad_tau_rise_J+dec_tau)) for dec_tau in ad_dec_taus]
ad_weight_factors = [.89382, .09568, .01050]
ad_norm_factors = [norm_factor(tau_rise, tau_dec) for tau_rise, tau_dec in zip(ad_rise_taus,ad_dec_taus)]


as_tau_rise_J = 0.38





def peak_time(tau_rise, tau_dec):
    return (tau_dec*tau_rise)/(tau_dec-tau_rise)*log(tau_dec/tau_rise)

def norm_factor(tau_rise, tau_dec):
    return 1./(exp(-peak_time(tau_dec, tau_rise)/tau_dec) - exp(-peak_time(tau_dec, tau_rise)/tau_rise))


def ad_g_1(t):
    return ad_norm_factors[0]*(exp(-t/ad_dec_taus[0]) - exp(-t/ad_tau_rise))

def ad_g_2(t):
    return ad_norm_factors[1]*(exp(-t/ad_dec_taus[1]) - exp(-t/ad_tau_rise))

def ad_g_3(t):
    return ad_norm_factors[2]*(exp(-t/ad_dec_taus[2]) - exp(-t/ad_tau_rise))


def ad_g_unnorm(t):
    return ad_weight_factors[0]*ad_g_1(t) + ad_weight_factors[1]*ad_g_2(t) + ad_weight_factors[2]*ad_g_3(t)

max_ad_g_unnorm = ad_g_unnorm(fmin(lambda x: -ad_g_unnorm(x), 0))
ad_global_norm_factor = 1./max_ad_g_unnorm

def ad_g_norm(t):
    return ad_global_norm_factor * ad_g_unnorm(t)

# In [2]: fmin(lambda x: -ad_g_norm(x), 0)
# Optimization terminated successfully.
#          Current function value: -1.000000
#          Iterations: 21
#          Function evaluations: 42
# Out[2]: array([ 0.2304375])
# In [3]: max_ad_g_unnorm
# Out[3]: array([ 0.98706784])
# To recap, and taking into account the different shapes for Jason's and Padraig's waveforms (APART from the "power of N" term), for each component we have:

tau_decay_nC = tau_decay_J
tau_rise_nC = tau_rise_J * (tau_decay_J/(tau_rise_J+tau_decay_J))


