import numpy as np
import cvxpy as cp
from numpy import linalg as la
from Cournot import cournot

sep = 1  # put 1 if you want separable production costs, anything else for non-separable costs

q0 = np.random.rand(2, 2)

f0 = np.random.rand(1)
c = 8
B = np.array([-1, 1])
cost = np.array([[7, 11], [5, 2]])
a = np.array([12, 33])
b = np.array([9, 3])

s = cournot(q0, f0, B, cost, a, b, c)

eq, peq = s.equilibrium(sep)

q1 = np.random.rand(1, 2)
cost1 = np.array([7, 4])

s1 = cournot(q1, f0, B, cost1, a, b, c)

eq1, peq1 = s1.equilibrium(sep)

# 4 players

q4 = np.random.rand(4, 2)
cost4 = np.array([[7, 9999], [9999, 11], [5, 9999], [9999, 2]])

s4 = cournot(q4, f0, B, cost4, a, b, c)

eq4, peq4 = s4.equilibrium(sep)

# q1 = (np.diag(1 / (b + cost[0, :])) @ (a - b * (B * eq[-1] + (np.delete(q0, 0, 0)).sum(axis=0)))) / 2
