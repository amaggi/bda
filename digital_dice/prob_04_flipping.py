import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

l = 1
m = 2
n = 3
p = 0.4

NIT = 10000

def game_play(p, l, m, n):
    n_tosses = 0
    coins = np.array([l, m, n])
    # while all players still have coins play the game
    while coins.min() > 0:
        n_tosses += 1
        flips = np.random.rand(3) >= p 
        not_flips = (flips == False)
        # if ==1, then the True wins, if == 2 then the False wins
        res = np.sum(flips) 
        if res==1:
            coins[flips] += 2
            coins[not_flips] -= 1
        if res==2:
            coins[not_flips] += 2
            coins[flips] -= 1
        
    return n_tosses, np.argmax(coins)

n_tosses1 = np.empty(NIT, dtype=int)
n_tosses2 = np.empty(NIT, dtype=int)
n_wins1 = np.empty(NIT, dtype=int)
n_wins2 = np.empty(NIT, dtype=int)
for i in xrange(NIT):
    n_tosses1[i], n_wins1[i] = game_play(p, 1, 2, 3)
    n_tosses2[i], n_wins2[i] = game_play(p, 3, 4, 5)
n_av1 =  np.average(n_tosses1)
n_av2 =  np.average(n_tosses2)

fig, axes = plt.subplots(2, 2)
plt.sca(axes[0, 0])
n, bins, patches = plt.hist(n_tosses1, np.arange(5*n_av1)+0.5, normed=True)
plt.xlabel('n (duration of game / throws)')
plt.ylabel('P(n)')
plt.title('1, 2, 3')
plt.sca(axes[0, 1])
n, bins, patches = plt.hist(n_tosses2, np.arange(5*n_av2)+0.5, normed=True)
plt.xlabel('n (duration of game / throws)')
plt.ylabel('P(n)')
plt.title('2, 3, 4')
plt.sca(axes[1, 0])
n, bins, patches = plt.hist(n_wins1, np.arange(4)-0.5, normed=True)
plt.xlabel('winning player')
plt.ylabel('P(n)')
plt.sca(axes[1, 1])
n, bins, patches = plt.hist(n_wins2, np.arange(4)-0.5, normed=True)
plt.xlabel('winning player')
plt.ylabel('P(n)')

plt.show()
plt.close()
