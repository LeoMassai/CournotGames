import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
from numpy import linalg as la
from Cournot import cournot, cournotStack

m = 2
B = np.array([[-1], [1]])


a = 33 * np.random.rand(m,1)
b = 7 * np.random.rand(m,1)
f0 = np.random.rand(1)
q1 = np.random.rand(2, 1)
cost1 = np.array([7, 11])
c = 99 * np.ones(1)

H = np.array([[1, 0], [0, 1]])

s1 = cournotStack(q1, f0, B, H, cost1, a, b, c)

eq1= s1.equilibrium()

ft=(-a[0]+a[1]+b[0]*eq1[0]-b[1]*eq1[1])/(b[0]+b[1])


# 4 players

q4 = np.random.rand(4, 2)
cost4 = np.array([[7, 9999], [9999, 13], [8, 9999], [9999, 11]])

s4 = cournot(q4, f0, B, cost4, a, b, c)

eq4, peq4 = s4.equilibrium(sep)

# four producers, two markets

q4 = np.random.rand(4, 2)
cost4 = np.array([[7, 12], [8, 13], [8.8, 10], [9, 11]])

s4 = cournot(q4, f0, B, cost4, a, b, c)

eq4, peq4 = s4.equilibrium(sep)

q8 = np.random.rand(8, 2)
cost8 = np.array([[7, 9999], [9999, 12], [8, 9999], [9999, 13], [8.8, 9999], [9999, 10], [9, 9999], [9999, 11]])

s8 = cournot(q8, f0, B, cost8, a, b, c)

eq8, peq8 = s8.equilibrium(sep)

q11 = (a[0] + b[0] * eq2[-1]) / (2 * b[0] + 2 * cost[0, 0])

fe = (b[0] * eq2[0] - b[1] * eq2[3] + a[1] - a[0]) / (b[0] + b[1])

num_nodes = 4

# Number of edges in the graph
num_edges = 6

# Define the edges of the graph
edges = [(1, 4), (2, 1), (2, 3), (3, 1), (3, 4), (4, 2)]

# Create an empty incidence matrix
incidence_matrix = np.zeros((num_nodes, num_edges))

# Fill the incidence matrix
for i, edge in enumerate(edges):
    start_node, end_node = edge
    incidence_matrix[start_node - 1, i] = 1

    incidence_matrix[end_node - 1, i] = -1

B = incidence_matrix

cost = np.array([[7, 4, 11, 5], [9, 12, 6, 7]])
a = np.array([22, 33, 23, 21])
b = np.array([4, 2, 3, 3])

f0 = np.random.rand(6)
q0 = np.random.rand(2, 4)
c = 99 * np.ones(6)

s = cournot(q0, f0, B, cost, a, b, c)

eq2, peq2 = s.equilibrium(sep)

cost = np.array([[7, 9999, 9999, 9999], [9999, 4, 9999, 9999], [9999, 9999, 11, 9999], [9999, 9999, 9999, 5],
                 [9, 9999, 9999, 9999], [9999, 12, 9999, 9999], [9999, 9999, 6, 9999], [9999, 9999, 9999, 7]])

q0 = np.random.rand(8, 4)

s = cournot(q0, f0, B, cost, a, b, c)

eq22, peq22 = s.equilibrium(sep)
