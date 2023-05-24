import numpy as np  # linear algebra
import cvxpy as cp
from numpy import linalg as la
from Cournot import cournot

q0 = np.random.rand(2, 2)

f0 = np.random.rand(1)
c = 8
B = np.array([-1, 1])
cost = np.array([[7, 7], [7, 7]])
a = np.array([12, 33])
b = np.array([9, 3])

s = cournot(q0, f0, B, cost, a, b, c)

eq, peq = s.equilibrium()

q1 = np.random.rand(1, 2)
cost1 = np.array([7, 7])

s1 = cournot(q1, f0, B, cost1, a, b, c)

eq1, peq1 = s1.equilibrium()

# q1 = (np.diag(1 / (b + cost[0, :])) @ (a - b * (B * eq[-1] + (np.delete(q0, 0, 0)).sum(axis=0)))) / 2
