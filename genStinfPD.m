function [gmmdist, u_id] = genStinfPD(L, p_ati, Z, Sigma, stateDim)
desNum = size(L, 1);
q = size(L, 2);
u_id = zeros(desNum * q, 1);
mu_pd = zeros(desNum * q, stateDim);
sigma_pd = zeros(stateDim, stateDim, desNum * 2); % All follow from eq.(32) and above

L_vec = reshape(L, [], 1);
p_ati_vec = reshape(p_ati', [], 1);
normalizer = L_vec' * p_ati_vec / desNum; % Calculate the normalizer in eq.(32)

% Format parameters
for d = 1 : desNum
    for i = 1 : q
        mu_pd_temp = Z(stateDim * (2 * d - 2) + 1 : stateDim * 2 * d , i);
        sigma_pd_temp = Sigma(stateDim * (2 * d - 2) + 1 : stateDim * 2 * d, stateDim*(2 * i - 2) + 1 : stateDim * 2 * i);
        mu_pd((d - 1) * q + i, :) = mu_pd_temp(1 : 2);
        sigma_pd(:, :, (d - 1) * q + i) = sigma_pd_temp(1 : 2, 1 : 2);
        u_id((d - 1) * q + i) = L(d, i) * p_ati(d, i) * (1 / desNum) / normalizer;
    end
end
% Generate Gaussian mixture model and return
bool_mask = u_id ~= 0;
gmmdist = gmdistribution(mu_pd(bool_mask, :), sigma_pd(:, :, bool_mask), u_id(bool_mask));
end