import numpy as np
from scipy.optimize import minimize

# Objective function to maximize
def objective(x):
    return -(x[0]**2 - x[1]**2)  # Negate the objective function for maximization

# Inequality constraint function
def inequality_constraint(x):
    return x[0] - 2*x[1] - 2  # Example inequality constraint: x_0 + 2*x_1 - 2 >= 0

# Initial guess
x0 = np.array([0.0, 0.0])

# Define the inequality constraints
inequality_constraints = [{'type': 'ineq', 'fun': inequality_constraint}]

# Solve the constrained optimization problem (maximization)
result = minimize(objective, x0, method='SLSQP',constraints=inequality_constraints)

# Display the result
print("Optimal solution:", result.x)
print("Optimal objective value (negated):", -result.fun)  # Negate the result for the actual objective value
