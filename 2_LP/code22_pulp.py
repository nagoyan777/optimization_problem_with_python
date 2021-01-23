#!/usr/bin/env python
from pulp import *
import numpy as np
name='Production'
A = np.array([[3,1,2],[1,3,0],[0,2,4]])
c = np.array([150, 200, 300])
b = np.array([60, 36, 48])
(nrows, ncols) = A.shape
prob = LpProblem(name=name, sense=LpMaximize)
x = [LpVariable(f'x{i+1}', lowBound=0) for i in range(ncols)]
prob += lpDot(c,x)
for i in range(nrows):
    prob += lpDot(A[i], x) <= b[i], f'ineq{i:d}'
print(prob)
prob.solve()
print(LpStatus[prob.status])
print(f'Optimal value = {value(prob.objective)}')
for v in prob.variables():
    print(f'{v.name} = {v.varValue}')

# --> Varidate the result.
X = np.array([v.varValue for v in prob.variables()])
print('Is Fiesible: {np.all(np.abs(b-np.dot(A, X)) <= 1.0e-5)}')
