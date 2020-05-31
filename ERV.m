function [x_next, y, R_t, U_t, G_til, m_til] = ERV(x, pd, h, sig, Vn, r, T, t, ita, rho)

% MRD model parameters (Eq. (6))
s = r/2;                         %number of initial state(not include derivative)
I_s = eye(s);
I_2s = eye(2*s);

mu_d = [pd;zeros(s,1)];
A = [zeros(s,s) -I_s; ita rho];
F = @(dt) expm(-A * dt);
M = @(dt) (I_2s - expm(-A * dt)) * mu_d;

JK = @(dt) expm([-A sig*sig'; zeros(2*s, 2*s) A']*dt)*[zeros(2*s, 2*s); I_2s];

%  Q = @(dt)JK(1:2*s,:)*inv(JK(2*s+1:end,:));

G = [I_s zeros(s,s)];                                              % s*2r dimension

JK_h = JK(h);                                  
JK_Th = JK(T - t - h);
Q_h = JK_h(1:2*s,:)*inv(JK_h(2*s+1:end,:));                        %Q(h)
Q_Th = JK_Th(1:2*s,:)*inv(JK_Th(2*s+1:end,:));                     %Q(T - t - h)
% process matrices
P_T = [zeros(r, r), eye(r)];                                                    
C_t = inv(inv(Q_h) + F(T - t - h)' * inv(Q_Th) * F(T - t - h));            % Eq.(17)
H_t = [C_t * inv(Q_h) * F(h), C_t * F(T - t - h)' * inv(Q_Th)];            % between Eq.(17) & (18)
m_t = C_t * (inv(Q_h) * M(h) - F(T - t - h)' * inv(Q_Th) * M(T - t - h));  % between Eq.(17) & (18)
R_t = [H_t; P_T];                                                                   % Eq.(19)
m_til = [m_t; zeros(r, 1)];                                                         % Eq.(19)
U_t = [C_t, zeros(r, r); zeros(r, r), zeros(r, r)];                                 % Eq.(19)
G_til = [G, zeros(s, r)];                                                           % Eq.(20)

% generate measurement and process noise

v = mvnrnd([0;0], Vn, 1);
v = v';

w = mvnrnd([0;0; 0; 0], Q_h, 1);
w = w';
% x
% F(h)
% M(h)
% w
% propagate state & generate measurement of current state
x_next = F(h) * x + M(h) + w;
y = G * x + v;
end