import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# given a number of points on a line, how many mutual nearest neighbours are
# there ?

NIT = 1000

n_cars = np.arange(20)+4
prob_nn = np.empty(n_cars.max()+1, dtype=float)
prob_nn[0] = 0.
prob_nn[1] = 0.
prob_nn[2] = 1.
prob_nn[3] = 2./3.

# get an estimate of the probability
for n in n_cars:
    # do NIT iterations
    n_mn = 0
    for it in xrange(NIT):
        cars = uniform.rvs(size=n)
        cars.sort()
        nn = np.empty(n, dtype=int)
        # set the nearest neighbours
        nn[0] = 1
        nn[-1] = n-1
        for j in np.arange(n-2)+1:
            if cars[j]-cars[j-1] < cars[j+1]-cars[j]:
                nn[j] = j-1
            else:
                nn[j] = j+1
        # count the mutual nearest neighbours
        if nn[1] == 0:
            n_mn += 1
        j = 1
        while j < n-1:
            if nn[j] == j+1 and nn[j+1] == j:
                n_mn += 1
                j += 2
            else:
                j += 1
    prob_nn[n] = 2*n_mn / float(n*NIT)


# plot
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.plot(prob_nn, '*')

plt.show()
plt.close()
