import numpy as np

NPTS = 100

PI_1 = 0.6
PI_2 = 0.4

# probability of needing N throws to get a head
def prob_n(pi, N):
    return pi * (1-pi)**(N-1)

# expected value of N = Sum_N N*prob_n
N = np.arange(NPTS)
e_n1 = np.sum(N * prob_n(PI_1, N))
e_n2 = np.sum(N * prob_n(PI_2, N))

print e_n1, 1/PI_1
print e_n2, 1/PI_2

# two Tails are thrown
Pr_TT = ((1-PI_1)**2 * 0.5 + (1-PI_2)**2 * 0.5)

Pr_C1 = ((1-PI_1)**2 * 0.5) / Pr_TT
Pr_C2 = ((1-PI_2)**2 * 0.5) / Pr_TT
print Pr_C1, Pr_C2

# expected value of N given two Tails and no info on which coin was chosen.
E_N = Pr_C1*e_n1 + Pr_C2*e_n2
print E_N
