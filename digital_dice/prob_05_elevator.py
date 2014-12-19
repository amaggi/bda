import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

NIT = 10000
N_FLOORS = 6

elevator_pos = uniform.rvs(0, N_FLOORS, size=NIT)
my_pos = np.arange(N_FLOORS+1)

def going_up(e_pos, m_pos):
    up = 0
    if e_pos < m_pos:
        up = 1
    return up, np.abs(e_pos - m_pos)

ups = np.empty((NIT, N_FLOORS+1), dtype=int)
dist = np.empty((NIT, N_FLOORS+1), dtype=float)

# iterate over all samples and floors
for i in xrange(NIT):
    for j in xrange(N_FLOORS+1):
        ups[i, j], dist[i, j] = going_up(elevator_pos[i], my_pos[j])

# get the numers of up
p_up = np.sum(ups, 0) / float(NIT) * 100
for j in xrange(N_FLOORS+1):
    print "Floor %d : going up = %.2f %%" % (j, p_up[j]) 

fig, axes = plt.subplots(1, N_FLOORS+1)
for j in xrange(N_FLOORS+1):
    plt.sca(axes[j])
    plt.hist(dist[:, j], 2*(N_FLOORS+1), normed=True)
    plt.xlabel('Distance')
    plt.title('Floor %d'%j)
    plt.xlim([0, 6])
plt.show()
plt.close()
