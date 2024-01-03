import numpy as np  # linear algebra
import cvxpy as cp
from numpy import linalg as la
from scipy.optimize import LinearConstraint, minimize, shgo, basinhopping

import networkx as nx
import random
import numpy as np

random.seed(3)
np.random.seed(0)


def generate_connected_directed_graph(num_nodes, num_edges):
    # Ensure num_edges is valid for a directed graph
    if num_edges < num_nodes - 1:
        raise ValueError("The number of edges must be at least num_nodes - 1 for a connected directed graph.")

    connected = False

    while not connected:
        # Generate a directed random graph with a probability that ensures the desired average number of edges
        probability = (2 * num_edges) / (num_nodes * (num_nodes - 1))
        G = nx.gnp_random_graph(num_nodes, probability, directed=True)

        # Ensure the graph has exactly num_edges
        while G.number_of_edges() != num_edges:
            # Add or remove edges to match the desired number
            current_edges = G.number_of_edges()

            if current_edges < num_edges:
                # Add a random edge
                u = random.choice(range(num_nodes))
                v = random.choice(range(num_nodes))
                G.add_edge(u, v)
            elif current_edges > num_edges:
                # Remove a random edge
                edge_to_remove = random.choice(list(G.edges()))
                G.remove_edge(*edge_to_remove)
                # Ensure every node has at least degree 2
            for node in G.nodes():
                while G.out_degree(node) < 2:
                    target_node = random.choice(list(set(range(num_nodes)) - {node}))
                    G.add_edge(node, target_node)

        # Check if the graph is weakly connected
        connected = nx.is_weakly_connected(G)

    return G


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


def costreshape(C):
    n = C.shape[0]
    m = C.shape[1]
    Ce = 999 * np.ones((n * m, m))
    for k in range(n):
        matrix = 9999 * np.ones((m, m))
        np.fill_diagonal(matrix, C[k, :])
        Ce[k * m:k * m + m, :] = matrix
    return Ce


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


# Multiple markets Cournot --------------------------
class cournot:
    def __init__(self, Q, f, B, cost, a, b, c):
        self.f = f
        self.Q = Q
        self.B = np.atleast_2d(B)
        self.cost = np.atleast_2d(cost)
        self.a = a
        self.b = b
        self.c = c
        self.m = self.B.shape[0]
        self.n = Q.shape[0]
        self.l = self.B.shape[1]

    def equilibrium(self, sep):
        q0 = self.Q.copy()
        f0 = self.f.copy()
        B = self.B
        n = self.n
        m = self.m
        a = self.a
        b = self.b
        cost = self.cost
        c = self.c
        l = self.l
        err = 5
        sol = np.append(q0.flatten(), f0)
        sol = np.atleast_2d(sol)
        sol = sol.reshape((n * m + l, 1))
        k = 0
        qt = q0
        ft = f0
        while err > 1e-07:
            k += 1
            for j in range(n):
                q = cp.Variable(m)
                qim = np.delete(qt, j, 0)
                if sep == 1:
                    objective = cp.Maximize(q @ (a - cp.multiply(b, (B @ ft + qim.sum(axis=0)))) -
                                            cp.sum(
                                                cp.multiply(b, cp.square(q)) + cp.multiply(cost[j, :], cp.square(q))))
                else:
                    objective = cp.Maximize(q @ (a - cp.multiply(b, (B @ ft + qim.sum(axis=0)))) -
                                            cp.sum(
                                                cp.multiply(b, cp.square(q))) - cp.square(
                        cp.sum(cp.multiply(cost[j, :], q))))
                constraints = [0 <= q]
                prob = cp.Problem(objective, constraints)
                qs = prob.solve()
                qt[j, :] = q.value

            f = cp.Variable(l)
            if sep == 1:
                objective2 = cp.Maximize(cp.sum(cp.multiply(a, (B @ f + qt.sum(axis=0)))
                                                - cp.multiply(b / 2, cp.square((B @ f + qt.sum(axis=0)))) -
                                                cp.sum(cp.multiply(cost, cp.square(qt)))))
            else:
                objective2 = cp.Maximize(cp.sum(cp.multiply(a, (B @ f + qt.sum(axis=0)))
                                                - cp.multiply(b / 2, cp.square((B @ f + qt.sum(axis=0))))) -
                                         cp.sum(cp.square(cp.sum(cp.multiply(cost, qt), 1))))
            constraints2 = [-c <= f, f <= c]
            prob2 = cp.Problem(objective2, constraints2)
            ft = prob2.solve()
            ft = f.value
            temp = np.atleast_2d(np.append(qt.flatten(), ft))
            temp = temp.reshape((n * m + l, 1))
            sol = np.hstack((sol, temp))
            err = la.norm(sol[:, k] - sol[:, k - 1], 2)

            peq = a - b * (B @ ft + qt.sum(axis=0))

            d = B @ ft + qt.sum(axis=0)

        return sol[:, -1], peq, d


# Single market Cournot with quadratic costs, affine demand and Marshallian welfare
class cournot1:
    def __init__(self, B, H, cost, a, b, c):
        self.B = np.atleast_2d(B)
        self.cost = cost
        self.H = H
        self.a = a
        self.b = b
        self.c = c
        self.m = self.B.shape[0]
        self.n = self.H.shape[0]
        self.l = self.B.shape[1]
        self.H = H

    def equilibrium(self, cap):  # computes game equilibrium via potential maximization given a capacity vector cap
        B = self.B
        n = self.n
        a = self.a
        b = self.b
        H = self.H
        m = self.m
        cost = self.cost
        l = self.l
        pinvB = np.linalg.pinv(B)
        c = cap
        r = cp.Variable(m)
        q = cp.Variable(n)
        w = cp.Variable(l)
        objective = cp.Maximize(cp.sum(cp.multiply(a, (r + H.T @ q))
                                       - cp.multiply(b / 2, cp.square((r + H.T @ q))) -
                                       cp.sum(cp.multiply(cost, cp.square(q)))) -
                                1 / 2 * cp.sum(cp.multiply(b, H.T @ cp.square(q))))
        constraints = [0 <= q, 0 <= pinvB @ r + (np.identity(l) - pinvB@B) @ w,
                       pinvB @ r + (np.identity(l) - pinvB@B) @ w <= c,
                       cp.sum(r) == 0]
        prob = cp.Problem(objective, constraints)
        qs = prob.solve()
        ro = r.value
        qo = q.value
        wo = w.value
        w = np.sum(np.multiply(a, (ro + H.T @ qo))
                   - np.multiply(b / 2, np.square((ro + H.T @ qo))) -
                   np.sum(np.multiply(cost, np.square(qo))))  # Marshallian welfare at equilibrium
        sol = np.array((qo, ro), dtype=object)  # Equilibrium point
        peq = a - b * (ro + H.T @ qo)  # Equilibrium prices

        d = ro + H.T @ qo  # Consumption in each market at equilibrium

        return sol, peq, d, w

    def capopt(self, theta):  # Computes welfare-optimal capacity allocation at the equilibrium given a budget theta...
        # and the maximum possible welfare
        B = self.B
        n = self.n
        a = self.a
        b = self.b
        H = self.H
        cost = self.cost
        l = self.l

        # Objective function to maximize
        def objective(x):
            aa, aa, aa, w = self.equilibrium(x)
            return -w  # Negate the objective function for maximization

        # Inequality constraint functions
        def inequality_constraint_1(x):
            return theta - np.sum(x)

        linear_constraint = LinearConstraint(np.ones((1, l)), -np.inf, theta)

        bounds = [[0, theta] for i in range(l)]

        # Initial guess
        x0 = np.random.rand() * np.ones(self.l)

        # Define the inequality constraints
        inequality_constraints = [
            {'type': 'ineq', 'fun': inequality_constraint_1}
        ]

        kwargs = {'method': 'SLSQP', 'constraints': inequality_constraints, 'bounds': bounds}

        # Solve the constrained optimization problem
        result = basinhopping(objective, x0, minimizer_kwargs=kwargs, niter=10)
        # result = basinhopping(objective, bounds=bounds, constraints=inequality_constraints)

        sol, peq, d, w = self.equilibrium(result.x)

        # computes maximum possible welfare ------------------------------
        f = cp.Variable(l)
        q = cp.Variable(n)

        objective = cp.Maximize(cp.sum(cp.multiply(a, (B @ f + H.T @ q))
                                       - cp.multiply(b / 2, cp.square((B @ f + H.T @ q))) -
                                       cp.sum(cp.multiply(cost, cp.square(q)))))
        constraints = [0 <= q, 0 <= f, cp.sum(f) <= theta]
        prob = cp.Problem(objective, constraints)
        wm = prob.solve()

        return result.x, -result.fun, sol, peq, d, wm


class cournotStack:
    def __init__(self, Q, f, B, H, cost, a, b, c):
        self.f = f
        self.Q = Q
        self.B = np.atleast_2d(B)
        self.cost = np.atleast_2d(cost)
        self.a = a
        self.b = b
        self.c = c
        self.m = self.B.shape[0]
        self.n = Q.shape[0]
        self.l = self.B.shape[1]
        self.H = H

    def equilibrium(self):
        q0 = self.Q.copy()
        f0 = self.f.copy()
        B = self.B
        n = self.n
        m = self.m
        a = self.a
        b = self.b
        H = self.H
        cost = self.cost
        c = self.c
        l = self.l
        err = 5
        qt = q0
        ft = f0

        f = cp.Variable(l)
        objective2 = cp.Maximize(cp.sum(cp.multiply(a, (cp.multiply(B, f) + H.T @ qt))
                                        - cp.multiply(b / 2, cp.square((cp.multiply(B, f) + H.T @ qt))) -
                                        cp.sum(cp.multiply(cost, cp.square(qt)))))

        constraints2 = [-c <= f, f <= c]

        prob2 = cp.Problem(objective2, constraints2)
        ft = prob2.solve()
        ft = f.value
        k = 0
        sol = q0
        while err > 1e-09:
            k += 1
            for j in range(n):
                q = cp.Variable(1)
                qim = np.delete(qt, j)
                ind = np.nonzero(H[j, :])
                Htemp = np.delete(H, j, 0)
                objective = cp.Maximize(
                    cp.multiply(q, a[ind] - cp.multiply(b[ind], (cp.multiply(B, ft))[ind] + (Htemp.T @ qim)[ind]))
                    - cp.multiply(b[ind], cp.square(q))
                    - cp.multiply(cost[0, j], cp.square(q)))
                constraints = [0 <= q]
                prob = cp.Problem(objective, constraints)
                qs = prob.solve()
                qt[j] = q.value
            sol = np.hstack((sol, qt))
            err = la.norm(sol[:, k] - sol[:, k - 1], 2)

        sol = np.vstack((qt, ft))

        return sol, qt, ft
