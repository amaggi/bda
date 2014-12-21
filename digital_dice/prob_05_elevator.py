import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, beta
from scipy.stats import gaussian_kde
from pickle import dump

NIT_DATA = 10000
NIT_BAYES = 20
N_FLOORS = 7
NPTS = 100
EPS = 1e-5


elevator_pos = uniform.rvs(0, N_FLOORS, size=NIT_DATA)
my_pos = np.arange(N_FLOORS+1)
d = np.linspace(0, N_FLOORS, NPTS)

def going_up(e_pos, m_pos):
    up = 0
    if e_pos < m_pos:
        up = 1
    return up, np.abs(e_pos - m_pos)

ups = np.empty((NIT_DATA, N_FLOORS+1), dtype=int)
dist = np.empty((NIT_DATA, N_FLOORS+1), dtype=float)

# iterate over all samples and floors
for i in xrange(NIT_DATA):
    for j in xrange(N_FLOORS+1):
        ups[i, j], dist[i, j] = going_up(elevator_pos[i], my_pos[j])

# get the numers of up
p_up = np.sum(ups, 0) / float(NIT_DATA) * 100
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
fl = np.arange(N_FLOORS+1)
dmax = np.max([fl, N_FLOORS-fl], 0)
d0 = np.min([fl, N_FLOORS-fl], 0)
print dmax, d0

# save the iteration data to disk
f_ = open('prob_05_elevator.dat', 'w')
dump((dist, dmax, d0), f_)
f_.close()

# the probability model
def prob_dist(th0, d0, dmax, dist):
    if dist > dmax:
        return EPS 
    elif dist < d0:
        return th0
    else:
        return th0/2.0

# get reasonable uniform priors for th0 and d0
th0_0_exp = 1/float(N_FLOORS)
th0_1_exp = 2/float(N_FLOORS)
#prior_th0_0 = uniform(0.8*th0_0_exp, 0.4*th0_0_exp)  # for ground and last floor
#prior_th0_1 = uniform(0.8*th0_1_exp, 0.4*th0_1_exp)  # for other floors
#th0_0 = prior_th0_0.rvs(size=NIT_BAYES)
#th0_1 = prior_th0_1.rvs(size=NIT_BAYES)
#th0_0.sort()
#th0_1.sort()

p_th0_exp = np.empty((NPTS, N_FLOORS+1), dtype=float) 
for i in xrange(NPTS):
    for j in xrange(N_FLOORS+1):
        if j==0 or j==N_FLOORS+1:
            p_th0_exp[i, j] = th0_0_exp
        else:
            p_th0_exp[i, j] = prob_dist(th0_1_exp, d0[j], dmax[j], d[i])

# get the unnormalized joint posterior density
#post_th0 = np.empty((NIT_BAYES, N_FLOORS+1), dtype=float)
#for j in xrange(N_FLOORS+1):
#    if j==0 or j==N_FLOORS+1:
#        th0 = th0_0
#    else:
#        th0 = th0_1
#    for i in xrange(NIT_BAYES):
#        post_th0[i, j] = np.sum(np.array([np.log(prob_dist(th0[i], d0[j],
#                                                 dmax[j], dist[i2, j]))
#                                          for i2 in xrange(NIT_DATA)])) 

# do plots
fig, axes = plt.subplots(1, N_FLOORS+1)
for j in xrange(N_FLOORS+1):
    plt.sca(axes[j])
    plt.hist(dist[:, j], 2*(N_FLOORS+1), normed=True)
    plt.plot(d, gp_dist[j].evaluate(d), lw=3, c='green')
    plt.plot(d, p_th0_exp[:, j], c='r', lw=3)
    plt.xlabel('Distance')
    plt.title('Floor %d'%j)
    plt.xlim([0, N_FLOORS])

    #plt.sca(axes[1, j])
    #if j==0 or j==N_FLOORS+1:
    #    th0 = th0_0
    #else:
    #    th0 = th0_1
    #plt.plot(th0, post_th0[:, j])
    #plt.xlabel('theta0')
    #plt.ylabel('log(P(th | data) - unnorm')
    #plt.xlim([th0.min(), th0.max()])

plt.show()
plt.close()
