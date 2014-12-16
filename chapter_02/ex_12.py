import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, uniform
from scipy.integrate import trapz
from am_bda import get_pdf_quantiles

NPTS = 100
NSIM = 1000

year = np.arange(10)+1976
accid = np.array([24, 25, 31, 31, 22, 21, 26, 20, 16, 22])
deaths = np.array([734, 516, 754, 877, 814, 362, 764, 809, 223, 1066])
drate = np.array([0.19, 0.12, 0.15, 0.16, 0.14, 0.06, 0.13, 0.13, 0.03, 0.15])
pmiles = deaths/drate * 100e6

# number of accidents is a Poission distribution with scale theta
def P_yi_theta(yi, theta):
    return poisson.pmf(yi, mu=theta)

# likelihood of y_values given theta
def likelihood(y_values, theta):
    p = np.empty(len(y_values), dtype=float)
    for i in xrange(len(y_values)):
        p[i] = P_yi_theta(y_values[i], theta)
    return np.prod(p)

# start with a gaussian prior for theta
#theta_prior = norm(np.average(accid), np.std(accid))
theta_prior = uniform(600, 200)
theta = np.linspace(theta_prior.ppf(0.001), theta_prior.ppf(0.999), NPTS)

# post_unnormalized
u_post = np.empty(NPTS, dtype=float)
for i in xrange(NPTS):
    u_post[i] = theta_prior.pdf(theta[i]) * likelihood(deaths, theta[i])

# norm factor
Z = trapz(u_post, theta)

# posterior
theta_post = u_post / Z

# get approximate posterior
mu = trapz(theta * theta_post, theta)
var = trapz((theta-mu)**2 * theta_post, theta)
theta_approx = norm(mu, np.sqrt(var))

# simulate NSIM values of theta and y
ysim = np.empty(NSIM, dtype=int)
for i in xrange(NSIM):
    th = theta_approx.rvs()
    ysim[i] = poisson.rvs(mu = th)

# get quantiles
yi_out = np.percentile(ysim, [2.5, 97.6])
#yi_out, pr_out = get_pdf_quantiles(y_post, y_range, [0.025, 0.975])
print yi_out

# plots
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.plot(theta, theta_prior.pdf(theta), label='prior')
plt.plot(theta, theta_post, label='posterior')
plt.xlabel('theta')
plt.ylabel('P(theta) or P(theta | data)')
plt.legend()
plt.sca(axes[1])
plt.hist(ysim)
plt.vlines(yi_out, 0, 100, 'k', lw=2, label='95% interval')
plt.xlabel('y_1986')
plt.ylabel('Hist(y_1986)')
plt.legend()

plt.show()
plt.close()
