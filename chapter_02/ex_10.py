import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.stats import uniform

N_MEAN = 100.
Y_SEEN = 203
FACT = 10

# set a starting range for N
n_values = np.arange(FACT*N_MEAN)+1

def prior_N(n):
    return (1./N_MEAN) * ((N_MEAN-1)/N_MEAN)**(n-1)

# get the prior pdf
prior_pdf = np.empty(len(n_values), dtype=float)
for i in xrange(len(n_values)):
    prior_pdf[i] = prior_N(i)

# get the likelihood
def likelihood(y, N):
    if y <= N:
        return 1./N
    else:
        return 0

# get the evidence
def evidence(y, n_values, prior_pdf):
    integrand = np.empty(len(n_values), dtype=float)
    for i in xrange(len(n_values)):
        integrand[i] = prior_pdf[i] * likelihood(y, n_values[i])
    return np.sum(integrand)
ev = evidence(Y_SEEN, n_values, prior_pdf)

# get the posterior
post_pdf = np.empty(len(n_values), dtype=float)
for i in xrange(len(n_values)):
    post_pdf[i] = likelihood(Y_SEEN, n_values[i]) * prior_pdf[i] / ev

post_mean = np.int(np.round(trapz(n_values * post_pdf, n_values)))
post_stdev = np.int(np.round(np.sqrt(trapz((n_values-post_mean)**2 * post_pdf, n_values))))

print post_mean, post_stdev

# redo for non-informative prior
prior_pdf_uniform = uniform.pdf(n_values, loc=1, scale=FACT*N_MEAN)
ev = evidence(Y_SEEN, n_values, prior_pdf_uniform)

# get the posterior
post_pdf_uniform = np.empty(len(n_values), dtype=float)
for i in xrange(len(n_values)):
    post_pdf_uniform[i] = likelihood(Y_SEEN, n_values[i]) * prior_pdf_uniform[i] / ev

post_mean = np.int(np.round(trapz(n_values * post_pdf_uniform, n_values)))
post_stdev = np.int(np.round(np.sqrt(trapz((n_values-post_mean)**2 * post_pdf_uniform, n_values))))

print post_mean, post_stdev

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.plot(n_values, prior_pdf, label='geometric')
plt.plot(n_values, prior_pdf_uniform, label='uniform')
plt.xlabel('N')
plt.ylabel('P(N)')
plt.title('Prior')
plt.legend()
plt.sca(axes[1])
plt.plot(n_values, post_pdf, label='geometric')
plt.plot(n_values, post_pdf_uniform, label='uniform')
plt.xlabel('N')
plt.ylabel('P(N|y=203)')
plt.title('Posterior')
plt.legend()
plt.show()
plt.close()
