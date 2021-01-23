#!/usr/bin/env python

import numpy as np
import scipy.linalg as linalg
MEPS = 1.0e-10

def lp_RevisedSimplex(c, A, b):
    np.seterr(divide='ignore')
    (nrows, ncols) = A.shape
    AI = np.hstack((A, np.identity(nrows)))
    c0 = np.r_[c, np.zeros(nrows)]
    basis = [ncols+i for i in range(nrows)]
    nonbasis = [j for j in range(ncols)]
    while True:
        y = linalg.solve(AI[:, basis].T, c0[basis])
        cc = c0[nonbasis] - np.dot(y, AI[:, nonbasis])
    
        if np.all(cc <= MEPS): 
            x = np.zeros(nrows + ncols)
            x[basis] = linalg.solve(AI[:, basis], b)
            print('Oprimal')
            print(f'Optimal value = {np.dot(c0[basis], x[basis])}')
            for i in range(nrows):
                print(f'x{i:d} = {x[i]}')
            break
        else:
            s = np.argmax(cc)
        d = linalg.solve(AI[:, basis], AI[:, nonbasis[s]])
        if np.all(d <= MEPS):
            print('Unbounded')
            break
        else:
            bb = linalg.solve(AI[:, basis], b)
            ratio = bb/d
            ratio[ratio < -MEPS] = np.inf
            r = np.argmin(ratio)
            # pivot
            nonbasis[s], basis[r] = basis[r], nonbasis[s]

def main():
    A = np.array([[2, 2, -1], [2, -2, 3], [0, 2, -1]])
    c = np.array([4, 3, 5])
    b = np.array([6, 8, 4])

    lp_RevisedSimplex(c, A, b)

if __name__=='__main__':
    main()