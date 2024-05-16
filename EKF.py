"""
ExtendedKalmanFilter class

This class implements the Extended Kalman Filter (EKF) algorithm for nonlinear systems.
The EKF is a recursive estimator that estimates the state of a dynamic system from a series
of noisy measurements. It is an extension of the Kalman Filter for nonlinear systems.

Attributes:
    state_equations (list): List of symbolic expressions representing the state equations.
    output_equations (list): List of symbolic expressions representing the output equations.
    state_symbols (list): List of symbolic variables representing the state variables.
    input_symbols (list): List of symbolic variables representing the input variables.
    params_dict (dict): Dictionary containing the numerical values of the system parameters.
    A_num (sympy.Matrix): Numerical Jacobian matrix of the state equations with respect to the state variables.
    B_num (sympy.Matrix): Numerical Jacobian matrix of the state equations with respect to the input variables.
    C_num (sympy.Matrix): Numerical Jacobian matrix of the output equations with respect to the state variables.

Methods:
    __init__(self, state_equations, output_equations, state_symbols, input_symbols, params_dict):
        Initializes the ExtendedKalmanFilter object with the given system equations, symbols, and parameters.

    linearize_and_substitute_params(self):
        Computes the numerical Jacobian matrices A_num, B_num, and C_num by substituting the parameter values
        in the symbolic Jacobian matrices.

    evaluate_and_update_states(self, x_k_minus_1, u_k_minus_1):
        Evaluates the state equations and computes the numerical Jacobian matrices A_num_eval, B_num_eval, and
        C_num_eval by substituting the previous state and input values. Returns the updated state vector x_k.

    ekf(self, x_k_minus_1, u_k_minus_1, y_k, P_k_minus_1, Q_k, R_k):
        Implements the Extended Kalman Filter algorithm. Takes the previous state estimate x_k_minus_1, input
        u_k_minus_1, measurement y_k, previous covariance P_k_minus_1, process noise covariance Q_k, and
        measurement noise covariance R_k as inputs. Returns the updated state estimate x_k and covariance P_k.
"""

import sympy as sp
import numpy as np
from itertools import zip_longest

class ExtendedKalmanFilter:
    def __init__(self, state_equations, output_equations, state_symbols, input_symbols, params_dict):
        self.state_equations = [sp.sympify(eq) for eq in state_equations]
        self.output_equations = [sp.sympify(eq) for eq in output_equations]
        self.state_symbols = [sp.symbols(sym) for sym in state_symbols]
        self.input_symbols = [sp.symbols(sym) for sym in input_symbols]
        self.params_dict = params_dict

        self.A_num, self.B_num, self.C_num, _ = self.linearize_and_substitute_params()

    def linearize_and_substitute_params(self):
        # Create a sympy Matrix for the system of equations
        sys = sp.Matrix(self.state_equations)
        output_sys = sp.Matrix(self.output_equations)

        # Calculate the Jacobian matrices
        A = sys.jacobian(self.state_symbols)
        B = sys.jacobian(self.input_symbols)
        C = output_sys.jacobian(self.state_symbols)

        # Substitute parameter values in the Jacobian matrices
        A_num = A.subs(self.params_dict)
        B_num = B.subs(self.params_dict)
        C_num = C.subs(self.params_dict)

        return A_num, B_num, C_num, self.state_equations

    def evaluate_and_update_states(self, x_k_minus_1, u_k_minus_1):
        # Substitute previous state and input values into the Jacobian matrices
        params_dict_eval = dict(zip_longest(self.state_symbols, x_k_minus_1, fillvalue=0))
        params_dict_eval.update(dict(zip_longest(self.input_symbols, u_k_minus_1, fillvalue=0)))

        A_num_eval = self.A_num.subs(params_dict_eval).evalf()
        B_num_eval = self.B_num.subs(params_dict_eval).evalf()
        C_num_eval = self.C_num.subs(params_dict_eval).evalf()

        # Evaluate the state equations to get the updated state vector (numerical values)
        x_k = np.array([eq.subs(params_dict_eval).evalf() for eq in self.state_equations]).astype(float)

        return A_num_eval, B_num_eval, C_num_eval, x_k

    def ekf(self, x_k_minus_1, u_k_minus_1, y_k, P_k_minus_1, Q_k, R_k):
        ######################### Predict #############################
        # Predict the state estimate at time k based on the state
        # estimate at time k-1 and the control input applied at time k-1.

        A_k_minus_1, _, C_num_eval, x_k = self.evaluate_and_update_states(x_k_minus_1, u_k_minus_1)

        # Predict the state covariance estimate based on the previous
        # covariance and some noise
        P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + Q_k

        ################### Update (Correct) ##########################
        # Calculate the difference between the actual sensor measurements
        # at time k minus what the measurement model predicted
        # the sensor measurements would be for the current timestep k.

        e_k = y_k - (C_num_eval @ x_k)

        # Calculate the measurement residual covariance
        S_k = C_num_eval @ P_k @ C_num_eval.T + R_k

        S_k = np.array(S_k, dtype=float)  # Convert S_k to a NumPy array of floats

        # Calculate the near-optimal Kalman gain
        K_k = P_k @ C_num_eval.T @ np.linalg.inv(S_k)

        # Calculate an updated state estimate for time k
        x_k = x_k + (K_k @ e_k)

        # Update the state covariance estimate for time k
        P_k = P_k - (K_k @ C_num_eval @ P_k)

        # Return the updated state and covariance estimates
        return x_k, P_k