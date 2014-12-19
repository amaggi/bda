import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.misc import comb
from scipy.integrate import trapz
from sample_via_cdf import sample_via_cdf

# 5 dishwashers, with two probabilities of breaking a dish in one week : four
# with p0 and one with pc
# the dishwasher with prob pc breaks four or five out of 5 dishes in a week.

NPTS = 100
NSAMP = 10000

# assign a uniform prior to both pc and p0
prior_p0 = uniform(0, 1)
prior_pc = uniform(0, 1)

p0 = np.linspace(prior_p0.ppf(0), prior_p0.ppf(1), NPTS)
pc = np.linspace(prior_pc.ppf(0), prior_pc.ppf(1), NPTS)

p0_samp = prior_p0.rvs(NSAMP)
pc_samp = prior_pc.rvs(NSAMP)

def P_data_given_prob(p0, pc):
    # prob that pc breaks only 4 and p0 breaks exactly 1 + prob that pc breaks 5
    return comb(5, 4)*pc**4*(1-pc) * 4*p0*comb(5, 4)*(1-p0)**4 +\
           comb(5, 5)*pc**5 *4*(1-p0)**5

# get un-normalized joint posterior
post_p0_pc = np.empty(NSAMP, dtype=float)
for i in xrange(NSAMP):
    post_p0_pc[i] = P_data_given_prob(p0_samp[i], pc_samp[i])

# get un-normalized posterior marginal for pc
marg_pc = np.empty(NPTS, dtype=float)
dpc = pc[1]-pc[0]
for i in xrange(NPTS):
    marg_pc[i] = np.sum(post_p0_pc[:][np.abs(pc_samp[:]-pc[i]) < 2*dpc])

# take samples from the marginal
marg_pc_samp = sample_via_cdf(pc, marg_pc, NSAMP)

# for each sample of pc, get p(p0|pc) then sample to get p0
marg_p0_samp = np.empty(NSAMP, dtype=float)
for i in xrange(NSAMP):
    indexes = [np.abs(pc_samp[:]-marg_pc_samp[i]) < 2*dpc]
    p0_ = p0_samp[indexes]
    pp0_ = post_p0_pc[indexes]
    indexes_sorted = p0_.argsort()
    p0_ = p0_[indexes_sorted]
    pp0_ = pp0_[indexes_sorted]
    marg_p0_samp[i] = sample_via_cdf(p0_, pp0_, 1)

dif = marg_pc_samp - marg_p0_samp
dif2 = marg_pc_samp - 2*marg_p0_samp
dif4 = marg_pc_samp - 4*marg_p0_samp

print('Prob(pc > p0) = %.2f %%' % (np.sum([dif >0]) / float(NSAMP) * 100))
print('Prob(pc > 2p0) = %.2f %%' % (np.sum([dif2 >0]) / float(NSAMP) * 100))
print('Prob(pc > 4p0) = %.2f %%' % (np.sum([dif4 >0]) / float(NSAMP) * 100))

fig, axes = plt.subplots(2, 2)
plt.sca(axes[0, 0])
plt.scatter(p0_samp, pc_samp, c=post_p0_pc)
plt.xlim([p0.min(), p0.max()])
plt.ylim([pc.min(), pc.max()])
plt.xlabel('p0')
plt.ylabel('pc')
plt.title('Posterior P(p0,pc)')

plt.sca(axes[0, 1])
plt.scatter(marg_p0_samp, marg_pc_samp)
plt.xlim([p0.min(), p0.max()])
plt.ylim([pc.min(), pc.max()])
plt.xlabel('p0')
plt.ylabel('pc')
plt.title('Samples drawn from P(p0,pc)')

plt.sca(axes[1, 0])
plt.hist(marg_p0_samp, label='p0')
plt.hist(marg_pc_samp, label='pc')
plt.xlabel('Beakage probability')
plt.legend()

plt.sca(axes[1, 1])
plt.hist(dif, label='pc-p0')
plt.xlabel('pc-p0')
plt.legend()

plt.show()
plt.close()
