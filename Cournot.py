import numpy as np  # linear algebra
import cvxpy as cp
from numpy import linalg as la


class cournot:
    def __init__(self, Q, f, B, cost, a, b, c):
        self.f = f
        self.Q = Q
        self.B = np.atleast_2d(B)
        self.cost = np.atleast_2d(cost)
        self.a = a
        self.b = b
        self.c = c
        self.m = self.B.shape[1]
        self.n = Q.shape[0]
        self.l = self.B.shape[0]

    def equilibrium(self, sep):
        q0 = self.Q.copy()
        f0 = self.f.copy()
        B = self.B
        B = B.squeeze()
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
        sol = sol.reshape((n * m + l, l))
        k = 0
        qt = q0
        ft = f0
        while err > 1e-07:
            k += 1
            for j in range(n):
                q = cp.Variable(m)
                qim = np.delete(qt, j, 0)
                if sep == 1:
                    objective = cp.Maximize(q @ (a - cp.multiply(b, (cp.multiply(B, ft) + qim.sum(axis=0)))) -
                                            cp.sum(
                                                cp.multiply(b, cp.square(q)) + cp.multiply(cost[j, :], cp.square(q))))
                else:
                    objective = cp.Maximize(q @ (a - cp.multiply(b, (cp.multiply(B, ft) + qim.sum(axis=0)))) -
                                            cp.sum(
                                                cp.multiply(b, cp.square(q))) - cp.square(
                        cp.sum(cp.multiply(cost[j, :], cp.square(q)))))
                constraints = [0 <= q]
                prob = cp.Problem(objective, constraints)
                qs = prob.solve()
                qt[j, :] = q.value

            f = cp.Variable(l)
            if sep == 1:
                objective2 = cp.Maximize(cp.sum(cp.multiply(a, (cp.multiply(B, f) + qt.sum(axis=0)))
                                                - cp.multiply(b / 2, cp.square((cp.multiply(B, f) + qt.sum(axis=0)))) -
                                                cp.sum(cp.multiply(cost, cp.square(qt)))))
            else:
                objective2 = cp.Maximize(cp.sum(cp.multiply(a, (cp.multiply(B, f) + qt.sum(axis=0)))
                                                - cp.multiply(b / 2, cp.square((cp.multiply(B, f) + qt.sum(axis=0))))) -
                                         cp.sum(cp.square(cp.sum(cp.multiply(cost, cp.square(qt)), 1))))
            constraints2 = [-c <= f, f <= c]
            prob2 = cp.Problem(objective2, constraints2)
            ft = prob2.solve()
            ft = f.value
            temp = np.atleast_2d(np.append(qt.flatten(), ft))
            temp = temp.reshape((n * m + 1, l))
            sol = np.hstack((sol, temp))
            err = la.norm(sol[:, k] - sol[:, k - 1], 2)

            peq = a - b * (B * ft + qt.sum(axis=0))

        return sol[:, -1], peq
