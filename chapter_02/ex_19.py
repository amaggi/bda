import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon
from scipy.integrate import trapz

NPTS = 100
NSAMP = 30

# exponential with unknown rate theta
# scale = 1/theta

# prior distribution for theta
# gamma distribution with coefficient of variation = V/E = 0.5
# p(theta) = Gamma(theta|a,b)
# a>0
# b = inverse scale > 0
# E(theta) = a/b
# var(theta) = a/b**2

# set up the prior for theta
a = 2.
b = 2.
scale = 1/b
prior = gamma(a, scale=b)
theta = np.linspace(prior.ppf(0.01), prior.ppf(0.99), NPTS)

# simulate NSAMP life measures 
theta_samp = prior.rvs(size=NSAMP)
life_samp = np.empty(NSAMP, dtype=float)
for i in xrange(NSAMP):
    life_samp[i] = expon.rvs(scale=1/theta_samp[i])

# get probability of samples
def likely_life(life_samp, theta):
    p = np.empty(NSAMP, dtype=float)
    for i in xrange(NSAMP):
        p[i] = expon.pdf(life_samp[i], scale=1/theta) 
    return np.prod(p)

# non-normalized posterior
u_post_theta = np.empty(NPTS, dtype=float)
for i in xrange(NPTS):
    u_post_theta[i] = likely_life(life_samp, theta[i]) * prior.pdf(theta[i])

# normalize
Z = trapz(u_post_theta, theta)
post_theta = u_post_theta / Z
print Z

# get coefficient of variation
E = trapz(theta * post_theta, theta)
V = trapz((theta - E)**2 * post_theta, theta)
print V/E

# plot
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.plot(theta, prior.pdf(theta), label='prior')
plt.plot(theta, post_theta, label='posterior')
plt.xlabel('theta')

plt.sca(axes[1])
plt.hist(life_samp, label='samples')
plt.xlabel('Life-time')
plt.show()
plt.close()
