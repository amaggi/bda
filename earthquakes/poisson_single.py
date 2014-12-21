import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, uniform, norm
from scipy.integrate import trapz

# invent a "recurrence time"
T = 150
L = 1000
NPTS = 100
NSAMP = 10000
ones = np.ones(NSAMP)

# set poission recurrence for earthquakes
T_prior = poisson(T)
X_prior = uniform(0, L)

# get the samples
Dt = T_prior.rvs(size=NSAMP)
x = X_prior.rvs(size=NSAMP)
t = np.cumsum(Dt)

def likelihood_data_mu(data, mu):
    return np.prod([poisson.pmf(d, mu) for d in data])

def post_mu(data):
    mu_mean = np.average(data)
    mu_std = np.std(data)
    mu_prior = norm(mu_mean, mu_std)
    mu = np.linspace(mu_prior.ppf(0.01), mu_prior.ppf(0.99), NPTS)
    post = np.empty(NPTS, dtype=float)
    for i in xrange(NPTS):
        post[i] = likelihood_data_mu(data, mu[i]) * mu_prior.pdf(mu[i])
    Z = trapz(post, mu)
    post = post/Z
    return mu, post

fig, axes = plt.subplots(3, 2)
plt.sca(axes[0, 0])
plt.hist(Dt, normed=True)
plt.xlabel('Time interval')
plt.sca(axes[0, 1])
plt.hist(x, normed=True)
plt.xlabel('Epicenter location')
plt.sca(axes[1, 0])
plt.scatter(t[t<20*T], x[t<20*T])
plt.xlabel('Time (years)')
plt.ylabel('Epicenter (km)')
plt.sca(axes[1, 1])
plt.xlabel('Time (years)')
plt.ylabel('Epicenter (km)')
plt.scatter(5*T*ones[t<5*T], x[t<5*T], c='r')
plt.scatter(10*T*ones[t<10*T], x[t<10*T], c='g')
plt.scatter(15*T*ones[t<15*T], x[t<15*T], c='b')
plt.scatter(20*T*ones[t<20*T], x[t<20*T], c='k')
plt.sca(axes[2, 0])
mu, post = post_mu(Dt[t<20*T])
plt.plot(mu, post)
plt.xlabel('T (years)')
plt.ylabel('P(T|data)')
plt.sca(axes[2, 1])
mu, post = post_mu(Dt[t<5*T])
plt.plot(mu, post, 'r', label='5T')
mu, post = post_mu(Dt[t<10*T])
plt.plot(mu, post, 'g', label='10T')
mu, post = post_mu(Dt[t<15*T])
plt.plot(mu, post, 'b', label='15T')
mu, post = post_mu(Dt[t<20*T])
plt.plot(mu, post, 'k', label='20T')
plt.legend()
plt.xlabel('T (years)')
plt.ylabel('P(T|data)')

plt.show()
plt.close()
