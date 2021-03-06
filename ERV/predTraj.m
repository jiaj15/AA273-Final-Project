function St_predpd = predTraj(Z, Sigma, h_step, sig, Vn, r, quadPoints, tn, desNum, q, Pred_steps, u_id, D, stateDim,  ita, rho)
% Number of steps between the time step to predicted to the current time step tn
h = Pred_steps * h_step;

% Similar to main loop in online.m
for d = 1 : desNum
    pd = D(:,d);
    for i = 1 : q
        Ti = quadPoints(i);
        z = Z(stateDim*(2*d-2)+1:stateDim*2*d,i);
        sigma = Sigma(stateDim*(2*d-2)+1:stateDim*2*d,stateDim*(2*i-2)+1:stateDim*2*i);
        
        [~, ~, R_t, U_t, ~, m_til] = ERV(z(1:stateDim,1), pd, h, sig, Vn, r, Ti, tn,  ita, rho);
        
        % Only the prediction step of Kalman filter
        Z_next = R_t * z + m_til;
        Sigma_next = R_t * sigma * R_t' + U_t;
       
        Z(stateDim * (2 * d - 2) + 1 : stateDim * 2 * d, i) = Z_next;
        Sigma(stateDim*(2*d-2)+1:stateDim*2*d,stateDim*(2*i-2)+1:stateDim*2*i) = Sigma_next;
    end
end
% Formulate Gaussian mixture model and return
St_predpd = genDist(desNum, q, stateDim, u_id, Z, Sigma);
end
    
function predgmmdist = genDist(desNum, q, stateDim, u_id, Z, Sigma)
mu_pd = zeros(desNum * q, stateDim);
sigma_pd = zeros(stateDim, stateDim, desNum * 2);

for d = 1 : desNum
    for i = 1 : q
        mu_pd_temp = Z(stateDim * (2 * d - 2) + 1 : stateDim * 2 * d , i);
        sigma_pd_temp = Sigma(stateDim * (2 * d - 2) + 1 : stateDim * 2 * d, stateDim*(2 * i - 2) + 1 : stateDim * 2 * i);
        mu_pd((d - 1) * q + i, :) = mu_pd_temp(1 : 4);
        sigma_pd(:, :, (d - 1) * q + i) = sigma_pd_temp(1 : 4, 1 : 4);
    end
end
bool_mask = u_id ~= 0;
predgmmdist = gmdistribution(mu_pd(bool_mask, :), sigma_pd(:, :, bool_mask), u_id(bool_mask));
end