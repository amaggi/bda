import numpy as np
from scipy.stats import poisson, uniform

NPTS = 100
LENGTH_OF_DAY = 7*60
N_DOCS = 3

arrival_intervals = poisson.rvs(mu=10, size=NPTS)
visit_durations = uniform.rvs(loc=5, scale=15, size=NPTS)

arrival_times = np.empty(NPTS, dtype=float)
entry_times = np.empty(NPTS, dtype=float)
wait_times = np.empty(NPTS, dtype=float)

doc_end_times = np.zeros(N_DOCS)

for i in xrange(NPTS):
    # first_doc = np.argmin(doc_end_times)
    # first_doc_time = doc_end_times[first_doc]
    if i==0:
        arrival_times[i] = arrival_intervals[i]
        entry_times[i] = arrival_times[i]
    else:
        arrival_times[i] = arrival_times[i-1] + arrival_intervals[i]
        entry_times[i] = np.max([entry_times[i-1] + visit_durations[i-1],
                                 arrival_times[i]])
        # entry_times[i] = np.max([first_doc_time, arrival_times[i]])

    # doc_end_times[first_doc] = first_doc_time + visit_durations[i]
    wait_times[i] = entry_times[i] - arrival_times[i]
    if arrival_times[i] > LENGTH_OF_DAY:
        n_clients = i
        break

arrival_times.resize(n_clients)
entry_times.resize(n_clients)
wait_times.resize(n_clients)

print entry_times
print wait_times
print 'Number of clients =', n_clients
print 'Number of clients who waited = ', np.sum(wait_times > 0.)
print 'Average wait time = ', np.average(wait_times[wait_times > 0.])
print 'Closing time', entry_times[-1] + visit_durations[-1]
