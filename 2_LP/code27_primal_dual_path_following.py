#!/usr/bin/env python
import numpy as np
MEPS = 1.0e-10  # --> machine epsilon
# def make_Mq_from_cAb(c, A, b):
def convert_to_selfdual_problem(c, A, b):
    nrows, ncols = A.shape
    m1 = np.hstack((np.zeros((nrows, nrows)), -A, b.reshape(nrows, -1)))
    m2 = np.hstack((A.T, np.zeros((ncols, ncols)), -c.reshape(ncols, -1)))
    m3 = np.append(np.append(-b, c), 0)
    M = np.vstack((m1, m2, m3))
    q = np.zeros(nrows+ncols+1)
    return M, q


def make_artProb_initialPoint(M, q):
    nrows, nrows = M.shape  # --> nrows == ncols
    x0 = np.ones(nrows)
    mu0 = np.dot(q, x0)/(nrows+1)+1
    z0 = mu0/x0
    r = z0 - M @ x0 - q
    qn1 = (nrows+1)*mu0 + q @ x0

    M = np.hstack((M, r.reshape((-1, 1))))
    M = np.vstack((M, np.append(-r, 0)))
    q = np.append(q, qn1)
    x0 = np.append(x0, 1)
    z0 = np.append(z0, mu0)
    return M, q, x0, z0


def get_binarysearch_stepsize(x, z, dx, dz, beta=0.5, precision=0.001):
    n = np.alen(x)

    th_low = 0.0
    th_high = 1.0
    if np.alen(-x[dx<0]/dx[dx<0]) > 0:
        th_high = min(th_high, np.min(-x[dx<0]/dx[dx<0]))
    if np.alen(-z[dz<0]/dz[dz<0]) > 0:
        th_high = min(th_high, np.min(-z[dz<0]/dz[dz<0]))
    
    x_low = x + th_low * dx
    z_low = z + th_low*dz
    x_high = x + th_high * dx
    z_high = z + th_high * dz
    mu_high = x_high @ z_high / n

    if (beta * mu_high >= np.linalg.norm(x_high*z_high - mu_high*np.ones(n))):
        return th_high
    while th_high - th_low > precision:
        th_mid = (th_high + th_low)/2
        x_mid = x + th_mid*dx
        z_mid = z + th_mid*dz
        mu_mid = x_mid @ z_mid / n
        if (beta * mu_mid) >= np.linalg.norm(x_mid*z_mid - mu_mid*np.ones(n)):
            th_low = th_mid
        else:
            th_high = th_mid
    return th_low


def PrimalDualPathFollowing(c, A, b):
    (M0, q0) = convert_to_selfdual_problem(c, A, b)
    (M, q, x, z) = make_artProb_initialPoint(M0, q0)

    (nrows, ncols) = A.shape
    (n, n) = M.shape

    count = 0
    mu = x @ z / n
    print(f'Initail objective function : {mu}')
    while mu > MEPS:
        count += 1
        print(f'It {count:5d} ', end='')
        # dx = (M + X^-1 Z)^-1 (delta mu X^-1 e - z)
        # dz = delta mu X^-1 e - z - X^-1 Z dx
        # mu = x^T z / n , delta in [0, 1]

        # --> Predictor, delta=0.0
        dx = np.linalg.inv(M + np.diag(z/x)) @ (-z)
        dz = - z - np.diag(1/x) @ (z * dx)
        theta = get_binarysearch_stepsize(x, z, dx, dz, 0.5, 0.001)
        print(f'theta = {theta:.4e}', end=', ')
        x += theta * dx
        z += theta * dz
        mu = x @ z / n

        # --> Corrector, delta=1.0
        dx = np.linalg.inv(M + np.diag(z/x)) @ (mu * (1/x) - z)
        dz = mu/x - z - np.diag(1/x) @ z*dx
        x += dx
        z += dz
        mu = x @ z / n
        print(f'Objective function {mu:.4e}')

    print(f'Converged.')

    if x[n-2] > MEPS:
        sols = x[nrows:nrows+ncols]/x[n-2]
        val = c @ x[nrows:nrows+ncols]/x[n-2]
        print(f'Optimal solution: {sols} has been found.')
        print(f'Optimal value = {val}')
        print(f'Optimal solution(dual) {x[:nrows]/x[n-2]} has been found.')
        print(f'Optimal value = {b @ x[:nrows]/x[n-2]}')
        return sols, val
    else:
        return None

def main():
    c = np.array([150, 200, 300])
    A = np.array([[3,1,2], [1,3,0], [0,2,4]])
    b = np.array([60, 36, 48])

    PrimalDualPathFollowing(c, A, b)

if __name__=="__main__":
    main()
