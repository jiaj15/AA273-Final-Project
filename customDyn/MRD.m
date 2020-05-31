function [x_next, y, R_t, U_t, G_til, m_til] = MRD(x, pd, h, lamda, sig, Vn, r, T, t)

% MRD model parameters (Eq. (6))
I_s = eye(r);
F = @(dt) expm(-lamda * dt);
M = @(dt) (I_s - expm(-lamda * dt)) * pd;
Q = @(dt) 0.5 * (I_s - expm(-2 * lamda * dt)) * inv(lamda) * sig * sig;
G = I_s;

% process matrices
P_T = [zeros(r, r), eye(r)];                                                    
C_t = inv(inv(Q(h)) + F(T - t - h)' * inv(Q(T - t - h)) * F(T - t - h));            % Eq.(17)
H_t = [C_t * inv(Q(h)) * F(h), C_t * F(T - t - h)' * inv(Q(T - t - h))];            % between Eq.(17) & (18)
m_t = C_t * (inv(Q(h)) * M(h) - F(T - t - h)' * inv(Q(T - t - h)) * M(T - t - h));  % between Eq.(17) & (18)
R_t = [H_t; P_T];                                                                   % Eq.(19)
m_til = [m_t; zeros(r, 1)];                                                         % Eq.(19)
U_t = [C_t, zeros(r, r); zeros(r, r), zeros(r, r)];                                 % Eq.(19)
G_til = [G, zeros(r, r)];                                                           % Eq.(20)

% generate measurement and process noise
v = mvnrnd([0; 0], Vn, 1);
v = v';
w = mvnrnd([0;0], Q(h), 1);
w = w';

% propagate state & generate measurement of current state
x_next = F(h) * x + M(h) + w;
y = G * x + v;
end