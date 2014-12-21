import numpy as np
import matplotlib.pyplot as plt
from pickle import load
from scipy.stats import uniform

# open and read the distance data
f_ = open('prob_05_elevator.dat', 'r')
dist, dmax, d0 = load(f_)
f_.close()

# get the number of samples and floors
nsamp, nstops =  dist.shape
print dmax, d0

# deal with floors 0 and nfloors-1
dist_0 = dist[:, 0]

th0_exp = 1/float(nstops-1)
prior_th0 = uniform(0.8*th0_exp, 0.4*th0_exp)


