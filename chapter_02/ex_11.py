import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.integrate import trapz
from scipy.stats import norm, cauchy

NPTS = 1000
N_SAMP = 1000

y_samples = np.array([43, 44, 45, 46.5, 47.5])

# Cauchy 
def P_yi_theta(yi, theta):
    return 1/(1. + (yi-theta)**2)

# likelihood
def likelihood(y_samples, theta):
    result = 1.
    for yi in y_samples:
        result = result * P_yi_theta(yi, theta)
    return result

# set up prior for theta
theta_prior = uniform(0, 100)
theta = np.linspace(theta_prior.ppf(0), theta_prior.ppf(1), NPTS)

# get unnormalised posterior
theta_post_unnorm = np.empty(NPTS, dtype=float)
for i in xrange(NPTS):
    th = theta[i]
    theta_post_unnorm[i] = likelihood(y_samples, th) * theta_prior.pdf(th)

# get integral of unnormalised posterior
Z = trapz(theta_post_unnorm, theta)

# posterior
theta_post = theta_post_unnorm / Z

# approximate the posterior distribution
theta_mean = trapz(theta * theta_post, theta)
theta_var = trapz((theta-theta_mean)**2 * theta_post, theta)
theta_post_approx = norm(theta_mean, np.sqrt(theta_var))

# get samples from approximate posterior
theta_samples = theta_post_approx.rvs(size=N_SAMP)

# sample the posterior predictive distribution for y6
y6_samples = np.empty(N_SAMP, dtype=float)
for ith in xrange(N_SAMP):
    y6_samples[ith] = cauchy.rvs(loc=theta_samples[ith], scale=1) 

# plot
fig, axes = plt.subplots(1, 3)
plt.sca(axes[0])
plt.plot(theta, theta_post, label='actual')
plt.plot(theta, theta_post_approx.pdf(theta), label='approx')
plt.legend()
plt.xlabel('theta')
plt.ylabel('P(theta | y1...y5)')
plt.sca(axes[1])
plt.hist(theta_samples)
plt.xlabel('theta')
plt.sca(axes[2])
plt.hist(y6_samples)
plt.xlabel('y6')
plt.show()
plt.close()
