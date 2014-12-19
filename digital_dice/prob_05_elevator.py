import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import gaussian_kde

NIT = 10000
N_FLOORS = 7
NPTS = 100


elevator_pos = uniform.rvs(0, N_FLOORS, size=NIT)
my_pos = np.arange(N_FLOORS+1)
d = np.linspace(0, N_FLOORS, NPTS)

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
# create a probability density for distance for each each floor
p_dist = np.empty(N_FLOORS+1, dtype=object)
gp_dist = np.empty(N_FLOORS+1, dtype=object)
for j in xrange(N_FLOORS+1):
    print "Floor %d : going up = %.2f %%" % (j, p_up[j]) 
    gp_dist[j] = gaussian_kde(dist[:, j])

# gaussians are not a good approximation
# the first and last floors have uniform distance distribution
p_dist[0] = uniform(0, N_FLOORS)
p_dist[N_FLOORS] = uniform(0, N_FLOORS)

# now try to deal with the two-step probabilites
fl = np.arange(N_FLOORS-1) + 1
upper_dist = np.max([fl, N_FLOORS-fl], 0)
print upper_dist

fig, axes = plt.subplots(1, N_FLOORS+1)
for j in xrange(N_FLOORS+1):
    plt.sca(axes[j])
    plt.hist(dist[:, j], 2*(N_FLOORS+1), normed=True)
    plt.plot(d, gp_dist[j].evaluate(d), lw=3, c='green')
    if j==0 or j==N_FLOORS:
        plt.plot(d, p_dist[j].pdf(d), lw=3, c='red')
    plt.xlabel('Distance')
    plt.title('Floor %d'%j)
    plt.xlim([0, N_FLOORS])

plt.show()
plt.close()
