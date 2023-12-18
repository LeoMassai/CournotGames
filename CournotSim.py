import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
from numpy import linalg as la
from Cournot import cournot, cournot1, cournotStack


def generate_connected_graph(n, l):
    G = nx.DiGraph()

    # Generate a connected graph with n nodes
    G.add_nodes_from(range(1, n + 1))
    for i in range(1, n):
        G.add_edge(i, i + 1)

    # Add remaining edges until l links are reached
    while G.number_of_edges() < l:
        u = np.random.randint(1, n)
        v = np.random.randint(1, n)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G


def incidence_matrix(G):
    nodes = sorted(G.nodes())
    edges = sorted(G.edges())

    node_index = {node: i for i, node in enumerate(nodes)}

    incidence_mat = np.zeros((len(nodes), len(edges)), dtype=int)
    for j, edge in enumerate(edges):
        u, v = edge
        incidence_mat[node_index[u]][j] = -1
        incidence_mat[node_index[v]][j] = 1

    return incidence_mat


def costreshape(C):
    n = C.shape[0]
    m = C.shape[1]
    Ce = 999 * np.ones((n * m, m))
    for k in range(n):
        matrix = 9999 * np.ones((m, m))
        np.fill_diagonal(matrix, C[k, :])
        Ce[k * m:k * m + m, :] = matrix
    return Ce


# Set the number of nodes and links
n = 8
m = 11
l = 18

# Generate the random connected graph
G = generate_connected_graph(m, l)

# Generate the incidence matrix
inc_mat = incidence_matrix(G)
# Plot the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


plt.title("Random Connected Graph")
plt.axis('off')
plt.show()

sep = 1  # put 1 if you want separable production costs, anything else for non-separable costs

q0 = np.random.rand(n, m)

f0 = np.random.rand(l)
c = 3 * np.ones(l)
B = inc_mat
cost = 9 * np.random.rand(n, m)
a = 33 * np.random.rand(m)
b = 7 * np.random.rand(m)


# coste = costreshape(cost)
#
# q2 = np.random.rand(n * m, m)
#
# s2 = cournot(q2, f0, B, coste, a, b, c)
#
# eq2, peq2, d2 = s2.equilibrium(sep)

# print(peq, peq2)
# print(d, d2)
edges = sorted(G.edges())
H = np.eye(m)
for k in range(n - 1):
    H = np.vstack((H, np.eye(m)))

cost1 = cost.flatten()
qs = np.random.rand(m * n)
qs = np.expand_dims(qs, 1)
#a = np.expand_dims(a, 1)
#b = np.expand_dims(b, 1)
cost1=cost1.squeeze()

ss = cournot1(B, H, cost1, a, b, c)

#eqs, peqs, ds, w = ss.equilibrium(c)


cap=np.linspace(0.7, 190, num=200)
h=0
w=np.zeros(200)
for i in cap:
    aa, w[h], aa, aa, aa = ss.capopt(i)
    h=h+1

plt.figure()
plt.plot(cap, w, label='Welfare')
plt.show()



