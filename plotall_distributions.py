import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta


NPTS = 100
COLS = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']

# normal distribution
mu_vals = [-1, 0, +1]
sig_vals = [0.5, 1, 2]

fig, axes = plt.subplots(1, 2)

plt.sca(axes[0])
n_vals = len(mu_vals)
for i in xrange(n_vals):
    rv = norm(mu_vals[i], 1)
    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), NPTS)
    plt.title('Norm, sigma = 1')
    plt.plot(x, rv.pdf(x), color=COLS[i], label='mu=%d'%mu_vals[i])
    plt.legend()

plt.sca(axes[1])
n_vals = len(sig_vals)
for i in xrange(n_vals):
    rv = norm(0, sig_vals[i])
    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), NPTS)
    plt.title('Norm, mu = 0')
    plt.plot(x, rv.pdf(x), color=COLS[i], label='sigma=%d'%sig_vals[i])
    plt.legend()


plt.savefig('norm.png')
plt.close()

# beta distribution
a_vals = [0.9, 1., 2., 3., 4.]
b_vals = [0.9, 1., 2., 3., 4.]

fig, axes = plt.subplots(1, 2)

plt.sca(axes[0])
n_vals = len(a_vals)
for i in xrange(n_vals):
    rv = beta(a_vals[i], 1)
    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), NPTS)
    plt.title('Beta, b = 1')
    plt.plot(x, rv.pdf(x), color=COLS[i], label='a=%.1f'%a_vals[i])
    plt.legend()

plt.sca(axes[1])
n_vals = len(b_vals)
for i in xrange(n_vals):
    rv = beta(1, b_vals[i])
    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), NPTS)
    plt.title('Beta, a = 1')
    plt.plot(x, rv.pdf(x), color=COLS[i], label='b=%.1f'%b_vals[i])
    plt.legend()


plt.savefig('beta.png')
plt.close()


