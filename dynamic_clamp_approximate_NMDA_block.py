#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import exp

def block_factor_conductance_box(v, k1, k2):
    return 1./(1 + k1 * exp(-k2*v))

def block_factor_rothman(v, z=2., T=35+273.15, block_concentration=1., delta_bind=0.35, delta_perm=0.53, c1=2.07, c2=0.015):
    F = 96485.3365 # C/mol (Faraday constant)
    R = 8314.4621 # mJ/(mol K) (ideal gas constant)
    theta = (z * F)/(R *T)
    return (c1 * exp(delta_bind * theta * v) + c2 * exp(-delta_perm * theta * v)) / (c1 * exp(delta_bind * theta * v) + c2 * exp(-delta_perm * theta * v) + block_concentration * exp(-delta_bind * theta * v))

def fit_simple_block_to_rothman_model(lower_bound, upper_bound):
    from scipy.optimize import curve_fit
    voltage_points = np.arange(lower_bound, upper_bound, 0.01)
    rothman_block_values = block_factor_rothman(voltage_points)
    k1, k2 = curve_fit(block_factor_conductance_box,
                       voltage_points,
                       rothman_block_values,
                       p0=[exp(-12.8/22.4), 1/22.4])[0]
    return k1, k2

def plot_fit_results(k1, k2, lower_bound, upper_bound):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    voltage_points = np.arange(-100, 40, 0.01)
    block_values_rothman = block_factor_rothman(voltage_points)
    block_values_simple = block_factor_conductance_box(voltage_points, k1=k1, k2=k2)

    fig, ax = plt.subplots()
    ax.fill_between(voltage_points, block_values_rothman, color="g")
    ax.fill_between(voltage_points, block_values_simple, color="r", alpha=0.5)
    ax.set_xlabel('membrane voltage (mV)')
    ax.set_ylabel('gNMDA unblocked fraction')
    ax.set_xlim(left=-92, right=32)
    ax.set_ylim(bottom=0, top=1)
    # proxy artists for legend
    p1 = Rectangle((0, 0), 1, 1, fc="g")
    p2 = Rectangle((0, 0), 1, 1, fc="r", alpha=0.5)
    ax.legend([p1, p2], ["Jason (Schwartz2012)", "Conductance box"], loc="upper left")
    ax.grid()
    fig.suptitle("Conductance box blocking function fitted between {}mV and {}mV\nB = 1/(1 + k1 exp(-V * k2))    where    k1: {:0.4}  and  k2: {:0.4}mV$^{{-1}}$".format(lower_bound, upper_bound, k1, k2))
    plt.show()

if __name__=="__main__":
    lower_bound = -80
    upper_bound = -40
    k1, k2 = fit_simple_block_to_rothman_model(lower_bound, upper_bound)
    plot_fit_results(k1, k2, lower_bound, upper_bound)
