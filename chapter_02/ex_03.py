import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N_ROLLS = 1000
NPTS = 100

# number of 6s for a fair die
E_y = N_ROLLS / 6.
sig_y = np.sqrt(N_ROLLS * (1/6.) * (5/6.))

y_distrib = norm(E_y, sig_y)
y = np.linspace(y_distrib.ppf(0.001), y_distrib.ppf(0.999), NPTS)

plt.figure()
plt.plot(y, y_distrib.pdf(y))
plt.show()

print y_distrib.ppf(0.05), y_distrib.ppf(0.25), y_distrib.ppf(0.5),\
      y_distrib.ppf(0.75), y_distrib.ppf(0.95)
