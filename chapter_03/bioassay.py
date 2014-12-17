import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# dose xi (log g/ml)
dose = np.array([-0.86, -0.30, -0.05, 0.73])
# number of animals tested ni
nani = np.ones(4)*5
# number of deaths yi
nded = np.array([0, 1, 3, 5])

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

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.plot(dose, nded, '*')

plt.show()
plt.close()
