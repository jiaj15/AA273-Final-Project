clc; clear; close all;
rng(0);

r = 2;                          % number of dimensions
lamda = diag([1, 1]);           % lamda in MRD model
D = [5, 6, 8; 1, 6, 9];         % destination Set
sig = diag([0.1, 0.1]);         % small sigma in MRD model
h = 0.001;                      % time step
Vn = [2, 1; 1, 5] / 10;         % covariance matrix for measurement noise
x = [0; 0];                     % initial state
Sigma = diag([0.4, 0.6, 3, 3]); % initial value for state covariance (note: first two terms should not be greater than 0.76)
pd = D(:, 1);                   % for testing only, select one of the nominal destinations
Z = [[-2; -2]; pd];             % initial state mean and destination state, concatenated as a state vector
T = 5;                          % time horizon
X_his = [];                     % true system state container
Z_his = [];                     % concatenated state container

for t = 0 : h : T - h           % loop for testing MRD model and algorithm 1 (KF)
    % recording
    X_his = [X_his, x];
    Z_his = [Z_his, Z];
    
    % roll out through model
    [x_next, y, R_t, U_t, G_til, m_til] = MRD(x, pd, h, lamda, sig, Vn, r, T, t);
    
    % a single KF iteration 
    [l_n, Z_next, Sigma_next] = KF(y, Z, Sigma, R_t, U_t, G_til, m_til, Vn, r, t);
    
    % update
    x = x_next;
    Z = Z_next;
    Sigma = Sigma_next;
end

% plotting
plot(X_his(1, :), X_his(2, :), 'linewidth', 1.5);
hold on;
plot(Z_his(1, :), Z_his(2, :), 'linewidth', 1.5);
plot(pd(1), pd(2), 'g.', 'MarkerSize', 35);
grid on;
legend('True Traj', 'Estimated Traj', 'Goal State');
axis equal;