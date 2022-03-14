import numpy as np

import numpy.random as random
def get_factors(n, m):
    factors = []
    factor = int(n**(1.0/m) + .1) # fudged to deal with precision problem with float roots
    while n % factor != 0:
        factor = factor - 1
    factors.append(factor)
    if m > 1:
        factors = factors + get_factors(n / factor, m - 1)
    return factors

# print(get_factors(729, 2))

memo = {}
def dp(n, left): # returns tuple (cost, [factors])
    if (n, left) in memo: return memo[(n, left)]

    if left == 1:
        return (n, [n])

    i = 2
    best = n
    bestTuple = [n]
    while i * i <= n:
        if n % i == 0:
            rem = dp(n / i, left - 1)
            if rem[0] + i < best:
                best = rem[0] + i
                bestTuple = [i] + rem[1]
        i += 1

    memo[(n, left)] = (best, bestTuple)
    return memo[(n, left)]



mids = np.load('E:\\Thesis\\DomainVis\\reductor\\data\\new\\init_patheval_map.npy')

lle =np.load('E:\\Thesis\\DomainVis\\reductor\\data\\graphs\\lle\\init_path_mode_pixel_thresh_None_cut_None_n_5(128, 128).npy')
shape = lle.shape
lle = lle.reshape(6,-1,2)
lle_id = np.sort(random.choice(np.arange(lle.shape[1]), 64*64, replace=False))
lle_new = np.zeros((6,int(lle.shape[1]/4),2))

for i in range(lle.shape[0]):
    lle_new[i]=lle[i][1::4]

lle_new = lle_new.reshape(6,3,64,64,2)
np.save('E:\\Thesis\\DomainVis\\reductor\\data\\graphs\\lle\\init_path_mode_pixel_thresh_None_cut_None_n_5(128, 128)1.npy',lle_new)
