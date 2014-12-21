import numpy as np
import matplotlib.pyplot as plt

N_START = 40
N_IT = 10000

n_until_one_empty = np.empty(N_IT, dtype=int)
n_left_when_one_empty = np.empty(N_IT, dtype=int)
for i in xrange(N_IT):
    books = np.ones(2) * N_START
    choices = np.random.rand(2*N_START)
    ic = 0
    while np.min(books) > 0:
        if choices[ic] < 0.5:
            books[0] -= 1
        else:
            books[1] -= 1
        ic = ic+1
    n_until_one_empty[i] = ic
    n_left_when_one_empty[i] = np.max(books)

print np.average(n_until_one_empty)
print np.average(n_left_when_one_empty)

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.hist(n_until_one_empty, normed=True)
plt.xlabel('N until one empty')
plt.sca(axes[1])
plt.hist(n_left_when_one_empty, normed=True)
plt.xlabel('N left when one empty')

plt.show()
plt.close()
