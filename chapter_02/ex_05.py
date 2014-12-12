import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb
from scipy.stats import uniform
from scipy.integrate import trapz

N_THROWS = 50
NPTS = 100

k_vals = np.arange(N_THROWS+1)

def Pr_y_is_k_given_theta(k, theta):
    # coin toss probability
    return comb(N_THROWS, k) * theta**k * (1-theta)**(N_THROWS-k)

def Pr_y_is_k(k, theta):
    # integrate over all values of theta
    y = Pr_y_is_k_given_theta(k, theta)
    return trapz(y, theta)

# uniform prior for theta
theta_prior = uniform(0, 1)
theta = np.linspace(theta_prior.ppf(0), theta_prior.ppf(1), NPTS)
#theta = np.linspace(0, 1, NPTS)

Pr_y = np.empty(N_THROWS+1, dtype=float)
for k in xrange(N_THROWS+1):
    Pr_y[k] = Pr_y_is_k(k, theta)
expected = np.ones(N_THROWS+1) * 1/np.float(N_THROWS+1)

#plt.figure()
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(15, 5)
plt.sca(axes[0])
plt.plot(k_vals, Pr_y_is_k_given_theta(k_vals, 0.2), 'b', label='theta=0.2')
plt.plot(k_vals, Pr_y_is_k_given_theta(k_vals, 0.8), 'r', label='theta=0.8')
plt.plot(k_vals, Pr_y_is_k_given_theta(k_vals, 0.4), 'g', label='theta=0.4')
plt.legend()
plt.xlabel('k')
plt.ylabel('Pr(y=k|theta)')

plt.sca(axes[1])
plt.plot(theta, Pr_y_is_k_given_theta(0, theta), 'b', label='k=0')
plt.plot(theta, Pr_y_is_k_given_theta(N_THROWS/2, theta), 'r',
         label='k=%d'%(N_THROWS/2))
plt.plot(theta, Pr_y_is_k_given_theta(N_THROWS, theta), 'g',
         label='k=%d'%(N_THROWS))
plt.legend()
plt.xlabel('theta')
plt.ylabel('Pr(y=k|theta)')

plt.sca(axes[2])
plt.plot(k_vals, Pr_y, 'b', label='computed')
plt.plot(k_vals, expected, 'r', label='expected')
plt.xlabel('k')
plt.ylabel('Pr(y=k)')
plt.legend()

plt.show()
plt.close()
