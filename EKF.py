import sympy as sp
import numpy as np
class ExtendedKalmanFilter:

    def __init__(self, state_equations, output_equations, state_symbols, input_symbols, params_dict, P_k_minus_1):
        self.state_equations = state_equations
        self.output_equations = output_equations
        self.state_symbols = state_symbols
        self.input_symbols = input_symbols
        self.params_dict = params_dict
        self.P_k_minus_1 = P_k_minus_1

        self.A_num, self.B_num, self.C_num = self.linearize_and_substitute_params()

    def set_QR(self, Q_k, R_k):
        self.Q_k = Q_k
        self.R_k = R_k
        

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

        # Substitute parameter values in equations
        self.state_equations = [eq.subs(self.params_dict) for eq in self.state_equations]

        return A_num, B_num, C_num

    def evaluate_and_update_states(self, x_k_minus_1, u_k_minus_1):
        # Substitute previous state and input values into the Jacobian matrices

        params_dict_eval = dict(zip(self.state_symbols, x_k_minus_1))
        params_dict_eval.update(zip(self.input_symbols, u_k_minus_1[0]))

        A_num_eval = np.array(self.A_num.subs(params_dict_eval).evalf().tolist()).astype(np.float64)
        B_num_eval = np.array(self.B_num.subs(params_dict_eval).evalf().tolist()).astype(np.float64)
        C_num_eval = np.array(self.C_num.subs(params_dict_eval).evalf().tolist()).astype(np.float64)

        # Evaluate the state equations to get the updated state vector (numerical values)
        x_k = np.array([eq.subs(params_dict_eval).evalf() for eq in self.state_equations])
        x_k = np.array(x_k, dtype=float)  # Convert x_k to a NumPy array of floats

        return A_num_eval, B_num_eval, C_num_eval, x_k

    def ekf(self, x_k_minus_1, u_k_minus_1, y_k):
        ######################### Predict #############################
        # Predict the state estimate at time k based on the state
        # estimate at time k-1 and the control input applied at time k-1.

        A_k_minus_1, _, C_num_eval, x_k = self.evaluate_and_update_states(x_k_minus_1, u_k_minus_1)

        # Predict the state covariance estimate based on the previous
        # covariance and some noise
        P_k = A_k_minus_1 @ self.P_k_minus_1 @ A_k_minus_1.T + self.Q_k

        ################### Update (Correct) ##########################
        # Calculate the difference between the actual sensor measurements
        # at time k minus what the measurement model predicted
        # the sensor measurements would be for the current timestep k.

        e_k = y_k - (C_num_eval @ x_k)

        # Calculate the measurement residual covariance
        S_k = C_num_eval @ P_k @ C_num_eval.T + self.R_k

        # Calculate the near-optimal Kalman gain
        K_k = P_k @ C_num_eval.T @ np.linalg.inv(S_k)

        # Calculate an updated state estimate for time k
        x_k = x_k + (K_k @ e_k)

        # Update the state covariance estimate for time k
        P_k = P_k - (K_k @ C_num_eval @ P_k)

        self.P_k_minus_1 = P_k

        # Return the updated state and covariance estimates
        return x_k