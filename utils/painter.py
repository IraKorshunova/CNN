import matplotlib.pyplot as plt
import numpy as np

skip = 14
b1 = np.genfromtxt('../plots_data/bs1', delimiter=' ', skiprows=skip)
b5= np.genfromtxt('../plots_data/bs5', delimiter=' ', skiprows=skip)
plt.plot(b1[:,1], b1[:,6], 'm', label='1')
plt.plot(b5[:,1], b5[:,6], 'g', label='5')
plt.legend(loc=1)
plt.show()
