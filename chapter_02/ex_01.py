import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

NPTS = 1000
N_THROWS = 10
N_HEADS_MAX = 2

prior = beta(4, 4)
theta = np.linspace(prior.ppf(0.0001), prior.ppf(0.9999), NPTS)

likely = binom(10, theta)
prob_likely = likely.pmf(0) 
for i in range(N_HEADS_MAX):
    print i+1
    prob_likely += likely.pmf(i+1)

plt.figure()
plt.plot(theta, prior.pdf(theta), 'r')
plt.plot(theta, prob_likely*prior.pdf(theta), 'k')
plt.show()
