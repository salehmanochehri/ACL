import numpy as np


def dynamics(t, x, u):
    # Physical parameters (typical small robotic arm)
    m1 = 1.0  # Mass of link 1 (kg)
    m2 = 0.8  # Mass of link 2 (kg)
    L1 = 0.5  # Length of link 1 (m)
    L2 = 0.4  # Length of link 2 (m)
    Lc1 = 0.25  # Distance to center of mass of link 1 (m)
    Lc2 = 0.2  # Distance to center of mass of link 2 (m)
    I1 = 0.02  # Moment of inertia of link 1 (kg*m^2)
    I2 = 0.01  # Moment of inertia of link 2 (kg*m^2)
    g = 9.81  # Gravitational acceleration (m/s^2)
    b1 = 0.1  # Viscous friction coefficient joint 1 (N*m*s/rad)
    b2 = 0.1  # Viscous friction coefficient joint 2 (N*m*s/rad)

    # State variables
    theta1 = x[0]
    theta2 = x[1]
    theta1_dot = x[2]
    theta2_dot = x[3]

    # Inputs
    tau1 = u[0]
    tau2 = u[1]

    # Shorthand notations
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s2 = np.sin(theta2)
    c2 = np.cos(theta2)
    s12 = np.sin(theta1 + theta2)
    c12 = np.cos(theta1 + theta2)

    # Mass matrix M(q)
    M11 = m1 * Lc1 ** 2 + m2 * (L1 ** 2 + Lc2 ** 2 + 2 * L1 * Lc2 * c2) + I1 + I2
    M12 = m2 * (Lc2 ** 2 + L1 * Lc2 * c2) + I2
    M21 = M12
    M22 = m2 * Lc2 ** 2 + I2

    M = np.array([[M11, M12], [M21, M22]])

    # Coriolis and centrifugal terms C(q,q_dot)*q_dot
    h = -m2 * L1 * Lc2 * s2
    C11 = h * theta2_dot
    C12 = h * (theta1_dot + theta2_dot)
    C21 = -h * theta1_dot
    C22 = 0

    C = np.array([[C11, C12], [C21, C22]])

    # Gravity vector G(q)
    G1 = (m1 * Lc1 + m2 * L1) * g * s1 + m2 * g * Lc2 * s12
    G2 = m2 * g * Lc2 * s12

    G = np.array([G1, G2])

    # Friction
    B = np.diag([b1, b2])

    # Control torques
    tau = np.array([tau1, tau2])

    # Equation of motion: M(q)*q_ddot + C(q,q_dot)*q_dot + G(q) + B*q_dot = tau
    # Solve for q_ddot
    theta_dot = np.array([theta1_dot, theta2_dot])
    theta_ddot = np.linalg.solve(M, tau - C @ theta_dot - G - B @ theta_dot)

    # State derivatives
    dq1 = theta1_dot
    dq2 = theta2_dot
    ddq1 = theta_ddot[0]
    ddq2 = theta_ddot[1]
    xdot = [dq1, dq2, ddq1, ddq2]
    return np.array(xdot)