#!/usr/bin/env python

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from pulp import *

@dataclass
class KnapsakProblem:
    """ The difinition of KnapSackProblem """

    name : str
    capacity : float
    items : set = field(default_factory=set)
    costs : list = field(default_factory=list)
    weights : list = field(default_factory=list)
    ones : set = field(default_factory=set)   # --> Items in the knapsak.
    zeros : set = field(default_factory=set)  # --> Excessive items.
    lb :  float = -100.0
    ub : float = -100.0
    ratio : dict = field(init=False)
    sortedItemList : list = field(init=False)
    xlb : dict = field(init=False)  # --> number of items for lower bound. 0 <= x <= 1 for each items
    xub : dict = field(init=False)  # --> number of items for upper bound. 0 or 1 for each items.
    bi : int = None

    def __post_init__(self):
        self.ratio = {j:self.costs[j]/self.weights[j] for j in self.items}
        self.sortedItemList = [k for k, v in sorted(self.ratio.items(), key=lambda x:x[1], reverse=True)]
        self.xlb = dict.fromkeys(self.items, 0)
        self.xub = dict.fromkeys(self.items, 0)

    def getbounds(self):
        """ Calculate the upper and lower bounds. """

        for j in self.zeros:
            self.xlb[j] = self.xub[j] = 0  # --> discarded items
        for j in self.ones:
            self.xlb[j] = self.xub[j] = 1  # --> stored items
        
        total_weight = np.sum(self.weights[j] for j in self.ones)

        if self.capacity < total_weight:
            self.lb = self.ub = -100  # --> Too heavy.
            return 0
        
        # --> Greedy upper bound and
        # --> Linear relaxation lower bound.
        remaining_items = self.items - self.zeros - self.ones
        sorted_items = [j for j in self.sortedItemList if j in remaining_items]
        cap = self.capacity - total_weight
        for j in sorted_items:
            if self.weights[j] <= cap:
                # --> Store j in knapsak.
                cap -= self.weights[j]
                self.xlb[j] = self.xub[j] = 1
            else:
                # --> j is the bisection point. Set partial amount for xub.
                self.xub[j] = cap/self.weights[j]
                self.bi = j
                break
        self.lb = np.sum(self.costs[j] * self.xlb[j] for j in self.items)
        self.ub = np.sum(self.costs[j] * self.xub[j] for j in self.items)

def KnapsakProblemSolve(capacity, items, costs, weights):
    from collections import deque
    queue = deque()
    root = KnapsakProblem('KP', capacity=capacity, items=items, costs=costs,
                          weights=weights, zeros=set(), ones=set())
    root.getbounds()
    best = root
    queue.append(root)
    while queue != deque([]):
        p = queue.popleft()
        p.getbounds()
        if p.ub > best.lb:  # --> Not optimal.
            if p.lb > best.lb:  # --> Trial p is better than stored best. 
                best = p  # --> update the best.
            if p.ub > p.lb:  # --> p can be improved. 
                # --> Devide p into partial problem p1 and p2 
                # --> at the heviest item in the remaining.
                k = p.bi
                p1 = KnapsakProblem(f'{p.name}+{k}', capacity=p.capacity, 
                                    items=p.items, costs=p.costs, 
                                    weights=p.weights,
                                    zeros=p.zeros, ones=p.ones.union({k}))
                queue.append(p1)
                p2 = KnapsakProblem(f'{p.name}-{k}', capacity=p.capacity, items=p.items,
                costs=p.costs, weights=p.weights,
                zeros=p.zeros.union({k}), ones=p.ones)
                queue.append(p2)
    return 'Optimal', best.lb, best.xlb

def main():
    capacity = 15
    items = {1,2,3,4,5}
    c = {1:50, 2:40, 3:10, 4:70, 5:55}
    w = {1:7, 2:5, 3:1, 4:9, 5:6}

    res = KnapsakProblemSolve(capacity=capacity,
    items=items, costs=c, weights=w)
    print(f'Optimal value = {res[1]}')
    print(f'Optimal solution = {res[2]}')

if __name__=='__main__':
    main()