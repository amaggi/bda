import numpy as np
from scipy.stats import poisson, uniform

NPTS = 100
LENGTH_OF_DAY = 7*60
N_DOCS = 3
N_ITERATIONS = 500
T_SCALE = 5
MIN_VISIT_TIME = 5
MAX_VISIT_TIME = 20

n_clients = np.empty(N_ITERATIONS, dtype=int)
n_wait = np.empty(N_ITERATIONS, dtype=int)
av_wait_time = np.empty(N_ITERATIONS, dtype=float)
close_time = np.empty(N_ITERATIONS, dtype=float)

for it in xrange(N_ITERATIONS):
    # arrival intervals are poisson distributed
    arrival_intervals = poisson.rvs(mu=T_SCALE, size=NPTS)
    # visit durations are distributed uniformly
    visit_durations = uniform.rvs(loc=MIN_VISIT_TIME,
                                  scale=MAX_VISIT_TIME-MIN_VISIT_TIME,
                                  size=NPTS)

    # create emtpy arrays
    arrival_times = np.empty(NPTS, dtype=float)
    entry_times = np.empty(NPTS, dtype=float)
    wait_times = np.empty(NPTS, dtype=float)
    doc_end_times = np.zeros(N_DOCS)

    # loop over clients
    for i in xrange(NPTS):
        # get the next free doctor
        first_doc = np.argmin(doc_end_times)
        first_doc_time = doc_end_times[first_doc]
        if i==0:
            # special case of the first client
            arrival_times[i] = arrival_intervals[i]
            entry_times[i] = arrival_times[i]
        else:
            # get arrival time and entry time of this client
            arrival_times[i] = arrival_times[i-1] + arrival_intervals[i]
            entry_times[i] = np.max([first_doc_time, arrival_times[i]])

        # update the doc time (time at which the doc will be free again)
        doc_end_times[first_doc] = entry_times[i] + visit_durations[i]
        # save the wait time
        wait_times[i] = entry_times[i] - arrival_times[i]

        # when the last client arrives, exit the loop
        if arrival_times[i] > LENGTH_OF_DAY:
            n_clients[it] = i
            break

    # resize on the number of clients
    arrival_times.resize(n_clients[it])
    entry_times.resize(n_clients[it])
    wait_times.resize(n_clients[it])
    n_wait[it] = np.sum(wait_times > 0.)
    if n_wait[it] == 0:
        av_wait_time[it] = 0.
    else:
        av_wait_time[it] = np.average(wait_times[wait_times > 0.])
    close_time[it] = np.max(doc_end_times)

print 'Median number of clients = %d (%d)' % (np.percentile(n_clients, 50),
      np.percentile(n_clients, 75) - np.percentile(n_clients, 25))
print 'Median number of clients who waited = %d (%d)' % (
      np.percentile(n_wait, 50),
      np.percentile(n_wait, 75) - np.percentile(n_wait, 25))
print 'Median average wait time = %.2f (%.2f)' % (
       np.percentile(av_wait_time, 50),
       np.percentile(av_wait_time, 75) - np.percentile(av_wait_time, 25))
print 'Median closing time = %.2f (%.2f)' % (np.percentile(close_time, 50),
       np.percentile(close_time, 75) - np.percentile(close_time, 25))
