import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

def f(x, a, b, c):
    return a + b*x + c*x*x

input_frequencies = np.array([7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0, 105.0, 120.0, 135.0, 150.0])
ampa_factors = np.array([0.8933192982, 0.8634754386, 0.8582502924, 0.8003701754, 0.784554386, 0.7538421053, 0.7454987469, 0.7177017544, 0.7056530214, 0.6846526316, 0.6666666667, 0.6401783626, 0.6160802005, 0.5957324561, 0.5741500975, 0.5474929825])
nmda_factors = np.array([0.9133768707, 0.8904353741, 0.9027628118, 0.845555102, 0.8375053061, 0.8027981859, 0.8042215743, 0.7768163265, 0.7758125472, 0.7487891156, 0.7391465677, 0.7105986395, 0.6909776482, 0.6790204082, 0.6656870748, 0.6490394558])


ampa_a, ampa_b, ampa_c = curve_fit(f, input_frequencies, ampa_factors)[0]
nmda_a, nmda_b, nmda_c = curve_fit(f, input_frequencies, nmda_factors)[0]

frequencies_plot = np.arange(input_frequencies[0], input_frequencies[-1], .01)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(frequencies_plot, [f(x, ampa_a, ampa_b, ampa_c) for x in frequencies_plot], color='r', label=r'AMPA: ${a:.2g} {b:+.2g}\nu {c:+.2g}\nu^2$'.format(a=ampa_a, b=ampa_b, c=ampa_c))
ax.plot(input_frequencies, ampa_factors, linestyle='', marker='o', color='r')
ax.plot(frequencies_plot, [f(x, nmda_a, nmda_b, nmda_c) for x in frequencies_plot], color='g', label=r'NMDA: ${a:.2g} {b:+.2g}\nu {c:+.2g}\nu^2$'.format(a=nmda_a, b=nmda_b, c=nmda_c))
ax.plot(input_frequencies, nmda_factors, linestyle='', marker='o', color='g')

ax.set_xlabel('input frequency (Hz)')
ax.set_ylabel('scaling factor')
ax.set_title('Average plasticity scaling factors')
ax.legend()
ax.grid()




plt.show()
