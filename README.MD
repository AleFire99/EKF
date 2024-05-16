# ExtendedKalmanFilter

The `ExtendedKalmanFilter` class implements the Extended Kalman Filter (EKF) algorithm for nonlinear systems. The EKF is a recursive estimator that estimates the state of a dynamic system from a series of noisy measurements. It is an extension of the Kalman Filter for nonlinear systems.

The class has the following methods:

- `__init__(self, state_equations, output_equations, state_symbols, input_symbols, params_dict)`: Initializes the `ExtendedKalmanFilter` object with the given system equations, symbols, and parameters.

- `linearize_and_substitute_params(self)`: Computes the numerical Jacobian matrices `A_num`, `B_num`, and `C_num` by substituting the parameter values in the symbolic Jacobian matrices.

- `evaluate_and_update_states(self, x_k_minus_1, u_k_minus_1)`: Evaluates the state equations and computes the numerical Jacobian matrices `A_num_eval`, `B_num_eval`, and `C_num_eval` by substituting the previous state and input values. Returns the updated state vector `x_k`.

- `ekf(self, x_k_minus_1, u_k_minus_1, y_k, P_k_minus_1, Q_k, R_k)`: Implements the Extended Kalman Filter algorithm. It takes the previous state estimate `x_k_minus_1`, input `u_k_minus_1`, measurement `y_k`, previous covariance `P_k_minus_1`, process noise covariance `Q_k`, and measurement noise covariance `R_k` as inputs. It returns the updated state estimate `x_k` and covariance `P_k`.

The EKF algorithm is used to estimate the state of a nonlinear system from noisy measurements. It linearizes the nonlinear system equations around the current state estimate and then applies the standard Kalman Filter equations to update the state estimate and covariance.