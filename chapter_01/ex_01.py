import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


PT1 = 0.5
PT2 = 0.5
NPTS = 100


yt1 = norm(loc=1, scale=2)
yt2 = norm(loc=2, scale=2)

x = np.linspace(yt1.ppf(0.01), yt2.ppf(0.99), NPTS)
y1pdf = yt1.pdf(x)
y2pdf = yt2.pdf(x)
ypdf = y1pdf*PT1 + y2pdf*PT2

plt.figure()
plt.plot(x, ypdf, 'k')
plt.show()

py1t1 = yt1.pdf(1)
py1t2 = yt2.pdf(1)
pt1y1 = py1t1*PT1 / (py1t1*PT1 + py1t2*PT2)
print 'Pr(theta=1|y=1) = ', pt1y1

thetas = np.linspace(0.01, 4, NPTS)
pt1 = np.empty(NPTS, dtype=float)

for i in xrange(NPTS):
    py1t1 = norm.pdf(1, loc=1, scale=thetas[i])
    py1t2 = norm.pdf(1, loc=2, scale=thetas[i])
    pt1[i] = py1t1*PT1 / (py1t1*PT1 + py1t2*PT2)

plt.figure()
plt.plot(thetas, pt1, 'r')
plt.show()
