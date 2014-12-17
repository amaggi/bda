import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, uniform
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

NSAMP = 1000
NPTS = 100

# dose xi (log g/ml)
dose = np.array([-0.86, -0.30, -0.05, 0.73])
# number of animals tested ni
nani = np.ones(4)*5
# number of deaths yi
nded = np.array([0, 1, 3, 5])

ntests = len(dose)


# model animals within each group as interchangeable and independent
# P(yi\thetai,ni) = binom(ni, thetai)

# expect a relation between nded and dose (the higher dose, the higher
# mortality), so theta could be linearly related to dose. But theta is
# a probability, and so must be between 0 and 1, so set logit(theta)=a+bx,
# where logit(theta)=log(theta/(1-theta)) = logistic regression model
# so theta=inverse_logit(a+bx) and 
# p(yi\a, b, ni, xi) propto theta^yi * (1-theta)^(ni-yi)
# posterior on (a,b) propto prior(a,b)*prod(p(yi\a,b,ni,xi))
# what is prior(a,b)?? probably independent and locally uniform...
# maximum likelihood estimate of a, b... 

prior_a = uniform(-5, 15)
prior_b = uniform(-10, 50)

# get samples of a and b
a_samp = prior_a.rvs(size=NSAMP)
b_samp = prior_b.rvs(size=NSAMP)

def theta_a_b_x(a, b, x):
    e = np.exp(a+b*x)
    return e/(1+e)

def p_y_theta(y, n, theta):
    rv = binom(n, theta)
    return rv.pmf(y)

# get the corresponding values of theta for each xi
# get the corresponding values of p for each xi
# calculate prior(a,b) * prod(p(yi\theta]
upost_a_b_samp = np.empty(NSAMP, dtype=float)
for i in xrange(NSAMP):
    p = np.empty(ntests, dtype=float)
    for ix in xrange(ntests):
        theta = theta_a_b_x(a_samp[i], b_samp[i], dose[ix])
        p[ix] = p_y_theta(nded[ix], nani[ix], theta)
    upost_a_b_samp[i] = np.prod(p) * prior_a.pdf(a_samp[i])\
                                   * prior_b.pdf(b_samp[i])
    


# sample NSAMP draws over the posterior distribution
# compute the marginal posterior of a by numerically summing over b
a = np.linspace(a_samp.min(), a_samp.max(), NPTS)
da = a[1]-a[0]
pa = np.empty(NPTS, dtype=float)
for i in xrange(NPTS):
    pa[i] = np.sum(upost_a_b_samp[:][np.abs(a_samp[:]-a[i]) < 2*da])

def sample_via_cdf(a, pa, nsamp):
    # get normalized cumulative distribution
    cpa = cumtrapz(pa, a, initial=0)
    cpa = cpa/cpa.max()
    # get interpolator
    interp = interp1d(cpa, a)
    # get uniform samples over cpa
    cpa_samp = uniform.rvs(size=nsamp)
    return interp(cpa_samp)

# get samples of posterior for a
a_s = sample_via_cdf(a, pa, NSAMP)

# for each sample of posterior for a
# get p(b|a)
b_s = np.empty(NSAMP, dtype=float)
for i in xrange(NSAMP):
    # get the indexes of (a,b) that correspond to a_s
    indexes = [np.abs(a_samp[:]-a_s[i]) < 2*da]
    b = b_samp[indexes]
    pb = upost_a_b_samp[indexes]
    # get b and pb
    indexes_sorted = b.argsort()
    b = b[indexes_sorted]
    pb = pb[indexes_sorted]
    # get one sample from pb
    b_s[i] = sample_via_cdf(b, pb, 1)
    if i==100:
        b_100=b
        pb_100=pb

# get corresponding probability distribution of LD50 : a+bx = logit(0.5)=0
# x = -a/b

x_s = -a_s/b_s

fig, axes = plt.subplots(2, 3)
# do scatter plot
plt.sca(axes[0, 0])
plt.scatter(a_samp, b_samp, c=upost_a_b_samp)
plt.xlim([a_samp.min(), a_samp.max()])
plt.ylim([b_samp.min(), b_samp.max()])
plt.xlabel('a value')
plt.ylabel('b value')
plt.title('Posterior P(a,b)')

# posterior marginal P(a)
plt.sca(axes[0, 1])
plt.plot(a, pa)
plt.xlabel('a')
plt.ylabel('P(a) - unnorm')

# samples over posterior marginal P(a)
plt.sca(axes[0, 2])
plt.hist(a_s)
plt.xlabel('a')

# corresponding samples over P(b|a)
plt.sca(axes[1, 0])
plt.hist(b_s)
plt.xlabel('b')

# scatterplot
plt.sca(axes[1, 1])
plt.scatter(a_s, b_s)
plt.xlim([a_samp.min(), a_samp.max()])
plt.ylim([b_samp.min(), b_samp.max()])
plt.xlabel('a value')
plt.ylabel('b value')
plt.title('Samples from P(a,b)')

# posterior for LD50
plt.sca(axes[1, 2])
plt.hist(x_s)
plt.xlabel('LD50')

plt.show()
plt.close()
