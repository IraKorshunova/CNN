import cPickle
import matplotlib.pyplot as plt
import numpy as np

f = open('out.pkl', 'rb')
cost, gnorms = cPickle.load(f)
plt.plot(cost, 'g', linewidth=2.0)
plt.plot(gnorms[:,0], 'b')
plt.plot(gnorms[:,1], 'b--')
plt.plot(gnorms[:,2], 'r')
plt.plot(gnorms[:,3], 'r--')
plt.plot(gnorms[:,4], 'm')
plt.plot(gnorms[:,5], 'm--')
plt.plot(gnorms[:,6], 'k')
plt.plot(gnorms[:,7], 'k--')
plt.show()
