import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

plt.rcParams["figure.autolayout"] = True
import networkx as nx
import numpy as np
import pickle
import random
import copy

from CournotModels import (cournot1, generate_connected_graph, incidence_matrix, generate_connected_directed_graph,
                           generate_row_stochastic_matrix)

# 3
# 0
random.seed(3)
np.random.seed(0)

# 9, 7, 14

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

# 3
# 56
random.seed(3)
np.random.seed(56)

q0 = np.random.rand(n, m)

f0 = np.random.rand(l)
c = 3 * np.ones(l)
B = inc_mat
# cost = 9 * np.random.rand(n, m)
a = 28 * np.random.rand(m)
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
# w2 = np.zeros(sample)
wm = np.zeros(sample)
links = np.zeros((l, sample))
peq = np.zeros((m, sample))
# peq2 = np.zeros((m, sample))
fe = np.zeros((l, sample))
for i in cap:
    links[:, h], w[h], ft, peq[:, h], d, wm[h] = ss.capopt(i)
    #   solnew, peq2[:, h], d2, w2[h] = ss.equilibrium2(i)
    fe[:, h] = ft[1]
    h = h + 1

r = B @ fe
diff = wm - w
# diff2 = w - w2

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
eqs, peqs, ds, ws = ss.equilibrium(130 * np.ones(l))
f = eqs[1]
qsim = eqs[0]

lt, wt, ftt, peqt, dt, wmt = ss.capopt(130)

# quantity maximizer when there is no capacity constraint (theta goes to infinity)
qteor = np.linalg.inv(np.diag(H @ b) @ H @ H.T + np.diag(H @ b) + np.diag(2 * cost)) @ (
        H @ a - np.diag(H @ b) @ H @ B @ f)

qteor2 = np.linalg.inv(H @ np.diag(b) @ H.T + np.diag(H @ b) + np.diag(2 * cost)) @ (
        H @ a - H @ np.diag(b) @ B @ f)

# flow maximizer when there is no capacity constraint (theta goes to infinity)

aa = B.T @ np.diag(b) @ B
bb = B.T @ a - B.T @ np.diag(b) @ H.T @ qteor

fteor = np.linalg.pinv(aa) @ bb + (np.eye(l) - aa @ np.linalg.pinv(aa)) @ (5 * np.random.rand(l))

aa2 = B.T @ np.diag(b)

bfteor = B @ (np.eye(l) - aa @ np.linalg.pinv(aa)) @ (5 * np.random.rand(l))

bff = B @ fteor

pteor = a - np.diag(b) @ (B @ fteor + H.T @ qteor)

# %%
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# lines = ax1.plot(cap, peq.T, label='Open values')
# fig.set_size_inches(3.16, 1.8, forward=True)
# plt.xlabel(r'$\theta$')
# ax1.set_ylabel(r'$\phi^*$', rotation=0)
# plt.savefig('test.pgf')
# plt.show()
#
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(cap, w, label=r'$w_e(\theta)$')
# ax1.plot(cap, wm, label=r'$w_m(\theta)$')
# ax1.legend()
# fig.set_size_inches(3.16, 1.8, forward=True)
# plt.xlabel(r'$\theta$')
# #ax1.set_ylabel(r'$\phi^*$', rotation=0)
# plt.savefig('test.pgf')
# plt.show()
#
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# #fig.suptitle('Aligning x-axis using sharex')
# ax1.plot(cap, w, label=r'$w_e(\theta)$')
# ax1.plot(cap, wm, label=r'$w_m(\theta)$')
# ax2.plot(cap, peq.T, label='Open values')
# plt.xlabel(r'$\theta$')
# ax2.set_ylabel(r'$\phi^*$', rotation=0)
# ax1.legend()
# fig.set_size_inches(3.16, 3.7, forward=True)
# plt.savefig('test.pgf')
# plt.show()
