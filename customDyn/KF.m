function [l_n, Z_hat, Sig] = KF(y_n, Z_hat_prev, Sig_prev, Rt, Ut, G_til, m_til, Vn, r, t)

% Predict step
if t == 0                   % according to the notes at the bottom of algorithm 1
    Z_pred = Z_hat_prev;
    Sig_pred = Sig_prev;
else                        % normal update
    Z_pred = Rt * Z_hat_prev + m_til;
    Sig_pred = Rt * Sig_prev * Rt' + Ut;
end

% PED calculation
disp(G_til * Sig_pred * G_til');
l_n = mvnpdf(y_n, G_til * Z_pred, G_til * Sig_pred * G_til');

% Correct step
K = Sig_pred * G_til' * inv(G_til * Sig_pred * G_til' + Vn);
Z_hat = Z_pred + K * (y_n - G_til * Z_pred);
%disp(K * G_til);
%disp("!!!");
%disp(Sig_pred);
Sig = (eye(2 * r) - K * G_til) * Sig_pred;
end