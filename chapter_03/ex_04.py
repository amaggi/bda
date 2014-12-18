import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.integrate import trapz
from sample_via_cdf import sample_via_cdf

N0 = 674
N0D = 39

N1 = 680
N1D = 22

NPTS = 100
NSAMP = 1000

# set the same uniform prior for the two cases
prior = uniform(0, 0.2)
p0 = np.linspace(prior.ppf(0), prior.ppf(1), NPTS)
p1 = np.linspace(prior.ppf(0), prior.ppf(1), NPTS)

# set up the posterior
post_p0 = np.empty(NPTS, dtype=float)
post_p1 = np.empty(NPTS, dtype=float)

for i in xrange(NPTS):
    post_p0[i] = (p0[i]**N0D * (1-p0[i])**(N0-N0D)) * prior.pdf(p0[i])
    post_p1[i] = (p1[i]**N1D * (1-p1[i])**(N1-N1D)) * prior.pdf(p1[i])

Z0 = trapz(post_p0, p0)
Z1 = trapz(post_p1, p1)

post_p0 = post_p0/Z0
post_p1 = post_p1/Z1

# get samples from posterior
p0_samp = sample_via_cdf(p0, post_p0, NSAMP)
p1_samp = sample_via_cdf(p1, post_p1, NSAMP)

# get odds_ratio distribution
odds_ratio = (p1_samp/(1-p1_samp)) / (p0_samp/(1-p0_samp))
odds_summary = np.percentile(odds_ratio, [2.5, 50, 97.5])
print odds_summary

# plot

fig, axes = plt.subplots(1, 3)
plt.sca(axes[0])
plt.plot(p0, post_p0, label='control')
plt.plot(p1, post_p1, label='beta-blocker')
plt.xlabel('Probability of death')
plt.ylabel('P(p | data)')
plt.legend()

plt.sca(axes[1])
plt.hist(p0_samp, label='control')
plt.hist(p1_samp, label='treatment')
plt.xlabel('Probability of death')
plt.legend()

plt.sca(axes[2])
plt.hist(odds_ratio, label='odds ratio')
plt.xlabel('odds ratio')
plt.legend()



plt.show()
plt.close()
