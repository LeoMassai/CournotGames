import matplotlib.pyplot as plt
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
import networkx as nx
import numpy as np
import pickle
import random
import copy

from CournotModels import (cournot1, generate_connected_graph, incidence_matrix, generate_connected_directed_graph,
                           generate_row_stochastic_matrix)

random.seed(3)
np.random.seed(0)

# Set the number of nodes and links
n = 9  # producers
m = 7  # markets (nodes)
l = 14  # links

# Generate the random connected graph
G = generate_connected_directed_graph(m, l)

# Generate a directed random graph
# G = nx.gnp_random_graph(m, 0.5, directed=True)
#
# # Create a DAG from the random graph
# DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
# # Ensure the generated DAG is connected
# while not nx.is_weakly_connected(DAG):
#     G = nx.gnp_random_graph(m, 0.5, directed=True)
#     DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
#
# G = DAG
m = G.number_of_nodes()
l = G.number_of_edges()

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
# cost = 9 * np.random.rand(n, m)
a = 33 * np.random.rand(m)
b = 7 * np.random.rand(m)
cost = 9 * np.random.rand(n)

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
# H = np.zeros((n, m))
# for k in range(n - 1):
#     H = np.vstack((H, np.eye(m)))

H = generate_row_stochastic_matrix(n, m)

cost1 = cost.flatten()
qs = np.random.rand(m * n)
qs = np.expand_dims(qs, 1)
# a = np.expand_dims(a, 1)
# b = np.expand_dims(b, 1)
cost1 = cost1.squeeze()

ss = cournot1(B, H, cost, a, b, c)
eqs, peqs, ds, ws = ss.equilibrium(c)

fla = copy.deepcopy(eqs[1])

# fla[10] = fla[10] - fla[5]
# # fla[9] = fla[9] - fla[6]
# fla[5] = 0
# #
# # eqsa, peqsa, dsa, wsa = ss.equilibrium(fla)
# #
# fl = eqs[1]
# qo = eqs[0]
# w = np.sum(np.multiply(a, (B @ fla + H.T @ qo))
#            - np.multiply(b / 2, np.square((B @ fla + H.T @ qo))) -
#            np.sum(np.multiply(cost, np.square(qo))))

# %% Equilibrium Computations


sample = 18
cap = np.linspace(2, 37, num=sample)
h = 0
w = np.zeros(sample)
wm = np.zeros(sample)
links = np.zeros((l, sample))
peq = np.zeros((m, sample))
fe = np.zeros((l, sample))
for i in cap:
    links[:, h], w[h], ft, peq[:, h], d, wm[h] = ss.capopt(i)
    fe[:, h] = ft[1]
    h = h + 1

r = B @ fe
diff = wm - w

plt.figure()
plt.plot(cap, w, label='Welfare')
plt.show()
plt.plot(cap, wm, label='Welfare')
plt.show()

plt.figure()
plt.plot(cap, links.T)
plt.show()

plt.figure()
plt.plot(cap, r.T)
plt.show()

plt.figure()
plt.plot(cap, peq.T)
plt.show()

plt.figure()
plt.plot(cap, diff)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
lines = ax1.plot(cap, r.T, label='Open values')
fig.set_size_inches(3.16, 3.5, forward=True)
plt.savefig('test.pgf')
plt.show()

# %%
fl = links[:, -1]
eqs, peqs, ds, ws = ss.equilibrium(fl)

lt, wt, ftt, peqt, dt, wmt = ss.capopt(130)

# %%
