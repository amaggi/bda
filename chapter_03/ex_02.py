import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, uniform
from sample_via_cdf import sample_via_cdf


NPTS = 100
NSAMP = 2000

pre_debate = {}
pre_debate['Bush'] = 294
pre_debate['Dukakis'] = 307
pre_debate['None'] = 38

post_debate = {}
post_debate['Bush'] = 288
post_debate['Dukakis'] = 332
post_debate['None'] = 19

# are only interested in two of the categories -> binomial distribution ok
prior_theta = uniform()
theta = np.linspace(prior_theta.ppf(0), prior_theta.ppf(1), NPTS)

# p(y|theta)
def p_y_theta(y, n, theta):
    rv = binom(n, theta)
    return rv.pmf(y)

post_theta1 = np.empty(NPTS, dtype=float)
post_theta2 = np.empty(NPTS, dtype=float)

# un-normalized posterior
for i in xrange(NPTS):
    post_theta1[i] = p_y_theta(294, 294+307, theta[i])
    post_theta2[i] = p_y_theta(288, 288+332, theta[i])

# take samples from posteriors
samp1 = sample_via_cdf(theta, post_theta1, NSAMP)
samp2 = sample_via_cdf(theta, post_theta2, NSAMP)

# after - before
dif = samp2 - samp1

# prob of shift towards bush = proportion of dif>0
print np.mean(dif[dif > 0])
print np.sum([dif > 0])/np.float(NSAMP)

fig, axes = plt.subplots(1, 3)
plt.sca(axes[0])
plt.plot(theta, post_theta1, label='pre')
plt.plot(theta, post_theta2, label='post')
plt.xlabel('theta')
plt.ylabel('P(theta | y)')
plt.legend()

plt.sca(axes[1])
plt.hist(samp1, label='pre')
plt.hist(samp2, label='post')
plt.xlabel('theta')
plt.legend()

plt.sca(axes[2])
plt.hist(dif, label='diff')
plt.xlabel('theta difference')
plt.legend()

plt.show()
plt.close()


