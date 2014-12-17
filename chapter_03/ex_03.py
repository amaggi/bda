import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
from scipy.integrate import trapz
from sample_via_cdf import sample_via_cdf

NT = 36
NC = 32

MEAN_T = 1.173
MEAN_C = 1.013

SIG_T = 0.20
SIG_C = 0.24

NPTS = 100
NSAMP = 1000

# assume uniform prior distribution for mu_c, mu_t, log(sig_c), log(sig_t)

mu_prior = uniform(0.5, 1)
log_sig_prior = uniform(-2, 1)

# get samples of mu and log_sigma
mu = mu_prior.rvs(NSAMP)
log_sig = log_sig_prior.rvs(NSAMP)
sig = np.exp(log_sig)

# from the measured statistics of the data, simulte the observations
y_t = norm.rvs(loc=MEAN_T, scale=SIG_T, size=NT)
y_c = norm.rvs(loc=MEAN_C, scale=SIG_C, size=NC)

# get unnormalized posterior
post_c = np.empty(NSAMP, dtype=float)
post_t = np.empty(NSAMP, dtype=float)
for i in xrange(NSAMP):
    post_c[i] = np.prod([norm.pdf(y_c[ii], loc=mu[i], scale=sig[i])
                      for ii in xrange(NC)])
    post_t[i] = np.prod([norm.pdf(y_t[ii], loc=mu[i], scale=sig[i])
                      for ii in xrange(NT)])


# sample NSAMP draws over the posterior distribution
# compute the marginal posterior of mu by numerically summing over log_sig
mu_x = np.linspace(mu_prior.ppf(0), mu_prior.ppf(1), NPTS)
dmu = mu_x[1]-mu_x[0]
p_mu_c = np.empty(NPTS, dtype=float)
p_mu_t = np.empty(NPTS, dtype=float)
for i in xrange(NPTS):
    p_mu_c[i] = np.sum(post_c[:][np.abs(mu[:]-mu_x[i]) < 2*dmu])
    p_mu_t[i] = np.sum(post_t[:][np.abs(mu[:]-mu_x[i]) < 2*dmu])

# integrate
Z_c = trapz(p_mu_c, mu_x)
Z_t = trapz(p_mu_t, mu_x)

# normalize
p_mu_c = p_mu_c/Z_c
p_mu_t = p_mu_t/Z_t

# sample
mu_c_samp = sample_via_cdf(mu_x, p_mu_c, NSAMP)
mu_t_samp = sample_via_cdf(mu_x, p_mu_t, NSAMP)
dif = mu_t_samp - mu_c_samp


fig, axes = plt.subplots(2, 3)
plt.sca(axes[0, 0])
plt.hist(y_t, label='treatment')
plt.hist(y_c, label='control')
plt.xlabel('measurement')
plt.title('Simulated data')
plt.legend()

plt.sca(axes[0, 1])
plt.scatter(mu, log_sig, c=post_c)
plt.xlabel('mu')
plt.ylabel('log(sigma)')
plt.title('Posterior for control')
plt.xlim([mu.min(), mu.max()])
plt.ylim([log_sig.min(), log_sig.max()])

plt.sca(axes[0, 2])
plt.scatter(mu, log_sig, c=post_t)
plt.xlabel('mu')
plt.ylabel('log(sigma)')
plt.title('Posterior for treatment')
plt.xlim([mu.min(), mu.max()])
plt.ylim([log_sig.min(), log_sig.max()])

plt.sca(axes[1, 0])
plt.plot(mu_x, p_mu_t, label='treatment')
plt.plot(mu_x, p_mu_c, label='control')
plt.xlabel('mu')
plt.ylabel('P(mu|data)')
plt.legend()

plt.sca(axes[1, 1])
plt.hist(mu_t_samp, label='treatment')
plt.hist(mu_c_samp, label='control')
plt.xlabel('mu')
plt.legend()

plt.sca(axes[1, 2])
plt.hist(dif, label='mu_t - mu_c')
plt.xlabel('mu_t - mu_c')
plt.legend()


plt.show()
plt.close()

