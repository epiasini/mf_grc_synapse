import numpy as np
from matplotlib import pyplot as plt

m_x = np.arange(10,150,10)
m_y_noplast = np.array([0.2, 4.6, 17.8, 41.3, 81, 128.2, 162.8, 200.8, 240.5, 257.5, 279.5, 294.6, 315.3, 325.3])
m_y_plast = np.array([0.1, 0.8, 6.4, 16.7, 33.5, 56.8, 89.2, 102.9, 132.4, 152.4, 173.1, 187.2, 202.6, 226.1])
# j_x = np.array([10,20,30,45, 70, 90, 140])
# j_y_noplast = np.array([0, 5, 20, 55, 130, 170, 220]) # this is from exp data, but it doesn't really make sense to compare it to the model given how the 9 grcs were chosen in the gain paper (i.e.: this is not the average cell)
# j_y_plast = np.array([0, 5, 18, 40, 100, 110, 145])
j_x = np.arange(15, 165, 15)
j_y_plast_phi06 = np.array([0.00, 0.12, 0.53, 4.06, 13.04, 31.52, 56.33, 93.56, 134.29, 165.93])
j_y_plast_phi10 = np.array([0.00, 1.65, 8.67, 34.44, 72.50, 117.76, 168.61, 215.88, 250.14, 271.57])
j_y_plast_phi12 = np.array([0.06, 3.53, 18.98, 59.19, 112.81, 162.91, 213.61, 254.25, 282.64, 299.29])
j_y_plast_phi14 = np.array([0.22, 7.10, 33.00, 88.94, 153.15, 200.91, 247.17, 282.75, 306.14, 320.79])

j_x_noinh = np.arange(15, 165, 15)
j_y_plast_phi06_noinh = np.array([0.020408163265306124, 1.8163265306122447, 9.744186046511627, 33.4375, 66.92307692307693, 105.14285714285715, 149.27777777777777, 190.125, 221.07142857142858, 240.7142857142857])
j_y_noplast_phi06_noinh = np.array([0.40816326530612246, 6.938775510204081, 43.25581395348837, 92.5, 159.6153846153846, 222.38095238095238, 263.8888888888889, 304.375, 335.71428571428567, 343.57142857142856])



fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(j_x_noinh, j_y_noplast_phi06_noinh, marker='o', color='r', label=r"J, $\varphi=0.597$, -inh, -STD")
ax.plot(j_x_noinh, j_y_plast_phi06_noinh, marker='o', color='b', label=r"J, $\varphi=0.597$, -inh, +STD")
ax.plot(m_x, m_y_noplast, marker='s', color='m', label='NML, -inh, -STD')
ax.plot(m_x, m_y_plast, marker='s', color='c', label='NML, -inh, +STD sc. fact.')


#ax.plot(j_x, j_y_plast_phi10, marker='o', color='b', label=r"J, $\varphi=1.0$")
#ax.plot(j_x, j_y_plast_phi12, marker='o', color='c', label=r"J, $\varphi=1.2$")
#ax.plot(j_x, j_y_plast_phi14, marker='o', color='g', label=r"J, $\varphi=1.4$")
#ax.plot(j_x, j_y_plast, marker='o', color='b', label="J, with STD")


ax.set_title('IaF granule cell (4 MFs)')
ax.set_xlabel('Input frequency on each MF (Hz) (Poisson)')
ax.set_ylabel('Output frequency (Hz)')
ax.legend(loc='upper left')
plt.show()
