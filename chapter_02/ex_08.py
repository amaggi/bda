import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from am_bda import get_pdf_quantiles

N = 100
S_Y = 20

NPTS = 100

y = norm(150, S_Y)
y_samples = y.rvs(size = N)
y_range = np.linspace(y.ppf(0.001), y.ppf(0.999), NPTS)

theta_prior = norm(180, 40)
theta = np.linspace(theta_prior.ppf(0.001), theta_prior.ppf(0.999), NPTS)

def likelihood(y_samples, theta):
    result = 1.
    yprob = norm(theta, S_Y)
    for yi in y_samples:
        result = result * yprob.pdf(yi)
    return result

def evidence(y_samples, theta_prior, theta):
    integrand = np.empty(NPTS, dtype=float)
    for i in xrange(NPTS):
        th = theta[i]
        integrand[i] = likelihood(y_samples, th) * theta_prior.pdf(th)
    return trapz(integrand, theta)
ev = evidence(y_samples, theta_prior, theta)

# get the posterior distribution for theta
Pr_theta_given_data = np.empty(NPTS, dtype = float)
for i in xrange(NPTS):
    th = theta[i]
    Pr_theta_given_data[i] = likelihood(y_samples, th)*theta_prior.pdf(th)/ev


th_quant, pr_th_quant = get_pdf_quantiles(Pr_theta_given_data, theta, [0.025, 0.5, 0.975])

# get the posterior predictive function for y
post_y = np.empty(NPTS, dtype=float)
for iy in xrange(NPTS):
    integrand = np.empty(NPTS, dtype=float)
    for ith in xrange(NPTS):
        integrand[ith] = norm.pdf(y_range[iy], loc=theta[ith], scale=S_Y) *\
                         Pr_theta_given_data[ith]
    post_y[iy] = trapz(integrand, theta)
        
y_quant, pr_y_quant = get_pdf_quantiles(post_y, y_range, [0.025, 0.5, 0.975])

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.plot(theta, Pr_theta_given_data)
plt.vlines(th_quant, 0, pr_th_quant, 'r', lw=1)
plt.xlabel('theta')
plt.ylabel('Pr(theta | y)')

plt.sca(axes[1])
plt.plot(y_range, post_y)
plt.vlines(y_quant, 0, pr_y_quant, 'r', lw=1)
plt.xlabel('y')
plt.ylabel('Pr(y\')')
plt.savefig('ex_08_n%03d.png'%N)
plt.close()

print th_quant
print y_quant
