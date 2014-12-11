import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

N_ROLLS = 1000
NPTS = 100

# unfair die
Pr12 = 0.25
Pr6 = 0.5
Pr4 = 0.25

m_12 = N_ROLLS / 12.
s_12 = np.sqrt(N_ROLLS*(1/12.)*(11/12.))
m_6 = N_ROLLS / 6.
s_6 = np.sqrt(N_ROLLS*(1/6.)*(5/6.))
m_4 = N_ROLLS / 4.
s_4 = np.sqrt(N_ROLLS*(1/4.)*(3/4.))

print m_12, s_12
print m_6, s_6
print m_4, s_4

n_12 = norm(m_12, s_12)
n_6 = norm(m_6, s_6)
n_4 = norm(m_4, s_4)

min_y = np.min([n_12.ppf(0.001), n_6.ppf(0.001), n_4.ppf(0.001)])
max_y = np.max([n_12.ppf(0.999), n_6.ppf(0.999), n_4.ppf(0.999)])
y = np.linspace(min_y, max_y, NPTS)

# the prior is a mixture of three gaussian distributions
prior_y = Pr12*n_12.pdf(y) + Pr6*n_6.pdf(y) + Pr4*n_4.pdf(y)

# the cumulative prior is necessary to get quantiles
cum_prior = cumtrapz(prior_y, x=y, initial=0)
q = interp1d(cum_prior, y)

plt.figure()
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(12, 6)
plt.sca(axes[0])
plt.plot(y, prior_y, 'b')
plt.xlabel('Number of throws of a 6')
plt.ylabel('Prior PDF')
plt.sca(axes[1])
plt.plot(y, cum_prior, 'r')
plt.xlabel('Number of throws of a 6')
plt.ylabel('Prior CDF')
plt.savefig('ex_04.png')
plt.close()

# get quantiles:
print q(0.05), q(0.25), q(0.5), q(0.75), q(0.95)
