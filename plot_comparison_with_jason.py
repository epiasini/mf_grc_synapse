import numpy as np
from matplotlib import pyplot as plt

m_x = np.arange(10,150,10)
m_y = np.array([0.7, 3.8, 7.7, 16.6, 27.4, 40.4, 60.2, 72.6, 104.5, 115.7, 135.5, 156.5, 173.7, 208.4])
j_x = np.array([10,20,30,45, 70, 90, 140])
j_y = np.array([0, 5, 20, 55, 130, 170, 220])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(m_x, m_y, marker='s', color='b', label='my model (no inhibition)')
ax.plot(j_x, j_y, marker='o', color='r', label="Jason's (2009, no STD, no inhibition)")

ax.set_title('IaF granule cell')
ax.set_xlabel('Input frequency (Hz)')
ax.set_ylabel('Output frequency (Hz)')
ax.legend(loc='upper left')
plt.show()
