import numpy as np  # linear algebra
import cvxpy as cp
from numpy import linalg as la
from scipy.optimize import minimize


# Multiple markets Cournot
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
        cost = self.cost
        l = self.l
        c = cap
        f = cp.Variable(l)
        q = cp.Variable(n)
        objective = cp.Maximize(cp.sum(cp.multiply(a, (B @ f + H.T @ q))
                                       - cp.multiply(b / 2, cp.square((B @ f + H.T @ q))) -
                                       cp.sum(cp.multiply(cost, cp.square(q)))) -
                                1 / 2 * cp.sum(cp.multiply(b, H.T @ cp.square(q))))
        constraints = [0 <= q, 0 <= f, f <= c]
        prob = cp.Problem(objective, constraints)
        qs = prob.solve()
        fo = f.value
        qo = q.value
        w = np.sum(np.multiply(a, (B @ fo + H.T @ qo))
                   - np.multiply(b / 2, np.square((B @ fo + H.T @ qo))) -
                   np.sum(np.multiply(cost, np.square(qo))))  # Marshallian welfare at equilibrium
        sol = np.array((qo, fo), dtype=object)  # Equilibrium point
        peq = a - b * (B @ fo + H.T @ qo)  # Equilibrium prices

        d = B @ fo + H.T @ qo  # Consumption in each market at equilibrium

        return sol, peq, d, w

    def capopt(self, theta):  # Computes welfare-optimal capacity allocation at the equilibrium given a budget theta
        # Objective function to maximize
        def objective(x):
            aa, aa, aa, w = self.equilibrium(x)
            return -w  # Negate the objective function for maximization

        # Inequality constraint functions
        def inequality_constraint_1(x):
            return theta - np.sum(x)

        def inequality_constraint_2(x):
            return x - 0.05

        # Initial guess
        x0 = np.random.rand(self.l) + 2

        # Define the inequality constraints
        inequality_constraints = [
            {'type': 'ineq', 'fun': inequality_constraint_1},
            {'type': 'ineq', 'fun': inequality_constraint_2}
        ]

        # Solve the constrained optimization problem
        result = minimize(objective, x0, method='SLSQP', constraints=inequality_constraints)

        sol, peq, d, w = self.equilibrium(result.x)

        return result.x, -result.fun, sol, peq, d


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
