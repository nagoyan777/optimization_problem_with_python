#!/usr/bin/env python

from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, value
#  from pulp import *
name='LP-sample'
prob = LpProblem(name=name, sense=LpMaximize)
x1 = LpVariable('x1', lowBound=0.0)
x2 = LpVariable('x2', lowBound=0.0)
prob += 2*x1 + 3 * x2  # objective function
prob += x1 + 3*x2 <= 9, 'ineq1'
prob += x1 + x2 <= 4, 'ineq2'
prob += x1 + x2 <= 6, 'ineq3'
print(f'Problem definition: \n{prob}')
print(f'Solving problem "{name}" ...')
prob.solve()
print(f'Done.')

print(f'\nStatus:  {LpStatus[prob.status]}')
print('Optimal value =', value(prob.objective))
for v in prob.variables():
    print(f'varible {v.name} = {value(v)}')
