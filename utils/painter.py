import matplotlib.pyplot as plt
import numpy as np

skip = 20
p1 = np.genfromtxt('../plots_data/test0_patient24_drop05', delimiter=' ', skiprows=skip)
p2= np.genfromtxt('../plots_data/test0_patient24_drop00', delimiter=' ', skiprows=skip)
plt.plot(p1[:,1], p1[:,6], 'm', label='1')
#plt.plot(p2[:,1], p2[:,6], 'g', label='5')
plt.legend(loc=1)
plt.show()
