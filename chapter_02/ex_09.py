import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from scipy.integrate import trapz

NPTS = 100
N_SAMPLES = 1000
N_FOR = 650

# mean and variance of a beta distribution are known
E = 0.6
V = 0.3**2

# get likelihood
def likelihood(n_for, n_samples, theta):
    return theta**n_for * (1-theta)**(n_samples-n_for)

def get_post_summary(E, V):
    # do the math to get values of a and b
    A = (1-E)/E
    a = (A - V * (1+A)**2) / (V * (1+A)**3)
    b = A*a

    prior = beta(a, b)
    theta = np.linspace(prior.ppf(0.00001), prior.ppf(0.99999), NPTS)

    # get evidence
    integrand = likelihood(N_FOR, N_SAMPLES, theta)*prior.pdf(theta)
    evidence = trapz(integrand, theta)

    # get posterior theta
    post_theta = integrand / evidence

    # get summary
    post_mean = trapz(theta * post_theta, theta)
    post_variance = trapz((theta-post_mean)**2 * post_theta, theta)

    return post_mean, post_variance, theta, post_theta, prior

post_mean, post_variance, theta, post_theta, prior = get_post_summary(E, V)
print post_mean, post_variance

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.plot(theta, prior.pdf(theta))
plt.title('Prior on theta')
plt.sca(axes[1])
plt.plot(theta, post_theta)
plt.title('Posterior on theta')
plt.show()
plt.close()

N_E = 10
V = 0.2**2
E = norm(0.6, 0.1)

E_samples = E.rvs(size=N_E)
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.title('Prior on theta')
plt.sca(axes[1])
plt.title('Posterior on theta')
for Ei in E_samples:
    post_mean, post_variance, theta, post_theta, prior = get_post_summary(Ei, V)
    plt.sca(axes[0])
    plt.plot(theta, prior.pdf(theta))
    plt.sca(axes[1])
    plt.plot(theta, post_theta)
# uniformative prior
Ei = 1/2.
V = 1/12.
post_mean, post_variance, theta, post_theta, prior = get_post_summary(Ei, V)
plt.sca(axes[0])
plt.plot(theta, prior.pdf(theta))
plt.sca(axes[1])
plt.plot(theta, post_theta)
plt.show()
plt.close()
