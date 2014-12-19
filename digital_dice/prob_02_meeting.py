import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

TIME_SPAN = 30.
L_WAIT = 5.
B_WAIT = 7.

NSAMP = 1000

lil_arrival = uniform.rvs(0, 30, size=NSAMP)
bil_arrival = uniform.rvs(0, 30, size=NSAMP)

bil_minus_lil = bil_arrival - lil_arrival

n_meet = 0
lil_wait_times = []
bil_wait_times = []
p_meet = np.zeros(NSAMP)
for i in xrange(NSAMP):
    if bil_minus_lil[i] > 0:  # bill arrives after lil
        if np.abs(bil_minus_lil[i]) < L_WAIT:
            n_meet += 1
            p_meet[i] = 1.
            lil_wait_times.append(np.abs(bil_minus_lil[i]))
        else:
            lil_wait_times.append(L_WAIT)
            bil_wait_times.append(np.min([B_WAIT, TIME_SPAN-bil_arrival[i]]))

    elif bil_minus_lil[i] < 0:  # bill arrives before lil
        if np.abs(bil_minus_lil[i]) < B_WAIT:
            n_meet += 1
            p_meet[i] = 1.
            bil_wait_times.append(np.abs(bil_minus_lil[i]))
        else:
            bil_wait_times.append(B_WAIT)
            lil_wait_times.append(np.min([L_WAIT, TIME_SPAN-lil_arrival[i]]))
    else:  # they both arrive at the same time
        n_meet +=1
        p_meet[i] = 1.
        lil_wait_times.append(0.)
        bil_wait_times.append(0.)

print('Probability of meeting: %.2f %%' % (n_meet/float(NSAMP)*100))
print('Median wait time for Lil: %.2f minutes' % np.median(lil_wait_times))
print('Median wait time for Bil: %.2f minutes' % np.median(bil_wait_times))
        
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.hist(lil_wait_times, label='Lil\'s wait times')
plt.hist(bil_wait_times, label='Bill\'s wait times')
plt.legend()
plt.xlabel('Minutes')
plt.sca(axes[1])
plt.scatter(lil_arrival, bil_arrival, c=p_meet)
plt.xlabel('Lil\'s arrival time')
plt.ylabel('Bill\'s arrival time')
plt.xlim([lil_arrival.min(), lil_arrival.max()])
plt.ylim([bil_arrival.min(), bil_arrival.max()])
plt.show()
plt.close()
