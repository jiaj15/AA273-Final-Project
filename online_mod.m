% D is the destination set [2,desDim]
% Y is measurement history [2,N]
% time respond to measurement history, e.g. at time[1], we get the measurement Y[1]
% quadPoints is [T1,T2,...,Tq], [1,q]
clc;
clear all;
close all;

%% Get all the parameter
N = 100;                         %% measurement step
h = 0.01;                        %% the input of dynamics model, smaller means smoother curve
RATIO = 10;                      %% the measurement frequency 
measure_t = RATIO * h;           %% every measure_t, get a new measurement

D = [4,  1.5,   4;
     5,    2,   4];
stateDim = size(D,1);
desNum = size(D,2);

Tq = 15;
quadPoints = Tq-2:0.2:Tq+2;
q = size(quadPoints,2);

trueDes = D(:,1);
x0 = [0;0];

Pred_flag = true;
Pred_steps = 30;

%% intialize all the history container
% a table which stores d rows and q columns concatenated state vectors at time tn
Z = zeros(2 * stateDim * desNum, q); 
for d = 1:1:desNum
%     A = repmat(D(:,d),1,q);
%     B = Z(stateDim*(2*d-1)+1:2*stateDim*d,:);
    Z(stateDim*(2*d-1)+1:2*stateDim*d,:) = repmat(D(:,d),1,q);
end

% The true trajectory
X = zeros(stateDim,N);
X(:,1) = x0;
Y = zeros(stateDim,N);

% a table which stores d rows and q columns covariances at time tn
sigma0 = diag([0.4, 0.6, 3, 3]);
Sigma = repmat(sigma0,desNum,q);

% a table which stores d rows and q columns likelihood at time tn
L = ones(desNum, q);

% a table which stores d rows and q columns P(T = Ti | D = d) at time tn
P_ati = ones(desNum, q)/q;

% a row which stores all the unnormalized probablitis of pd, at time tn
P = zeros(1, desNum);

% a table which stores all the arrival time posterior
P_arrival_t =  zeros(N, q);

% For part (e.1) -- state inference: 
St_infpd = cell(N, 1);                % A cell array storing all states as Gaussian mixture distributions at N time steps
St_predpd = cell(Pred_steps, 1);      % A cell array storing all states as Gaussian mixture distributions at all prediction steps

prior = ones(1,desNum)/desNum;

U = zeros(N, desNum);

%% Parameter Setting for the MRD model
sig = diag([0.1, 0.1]);         % small sigma in MRD model
Vn = [2, 1; 1, 5] / 10; 
lamda = diag([1, 1]);
r = stateDim;

%% main loop for destination inference
for t=1:1:N
    % responding time for a measurement
    %tn = time(t);
    tn = (t-1)* measure_t;
    x = X(:,t);
    for interval = 1:1:RATIO
        [x_next, y, R_t, U_t, G_til, m_til] = MRD(x, trueDes, h, lamda, sig, Vn, r, Tq, tn);
        tn = tn + h;
        x = x_next;
    end
    
    if t<N
        X(:,t+1) = x_next;
    end
    
    for d=1:1:desNum
        % probable destination
        pd = D(:,d);

        % probable arrival time, used for quad function
        for i=1:1:q
            Ti = quadPoints(i);

            z = Z(stateDim*(2*d-2)+1:stateDim*2*d,i);

            sigma = Sigma(stateDim*(2*d-2)+1:stateDim*2*d,stateDim*(2*i-2)+1:stateDim*2*i);

            %y = Y(:,t);

            % use MRD model to predict the next position
            [~, ~, R_t, U_t, G_til, m_til] = MRD(z(1:stateDim,1), pd, h, lamda, sig, Vn, r, Ti, tn);

            % use kF to update
            [l_n, Z_next, Sigma_next] = KF(y, z, sigma, R_t, U_t, G_til, m_til, Vn, r, tn);

            Z(stateDim*(2*d-2)+1:stateDim*2*d,i) = Z_next;
            Sigma(stateDim*(2*d-2)+1:stateDim*2*d,stateDim*(2*i-2)+1:stateDim*2*i) = Sigma_next;
            L(d, i) = l_n * L(d, i);
        end
        p = simpsonQuad(L(d,:),quadPoints);
        P(1,d) = p;        
    end
    
    L = L./sum(sum(L));
    U(t,:) = P .* prior/sum(P .* prior);
    [St_infpd{t}, u_id] = genStinfPD(L, P_ati, Z, Sigma, stateDim);   % Store state inference Gausian Mixture distribution into cell
    P_arrival_t(t,:) = sum(L.*P_ati, 1)/sum(sum(L.*P_ati)); % store arrival time probabilities into table
    
end

% Store state prediction Gausian Mixture distribution into cell
if Pred_flag
    for idx = 1 : Pred_steps
        St_predpd{idx} = predTraj(Z, Sigma, h, lamda, sig, Vn, r, quadPoints, tn, desNum, q, idx, u_id, D, stateDim);
    end
end

%% Plot

time = measure_t * (1:1:N);

figure;
plot(time,U(:,1),'r','linewidth',2);
hold on;
plot(time,U(:,2),'b','linewidth',2);
hold on;
plot(time,U(:,3),'y','linewidth',2);

figure;
plot(X(1,:),X(2,:),'k','linewidth',2);
hold on;
scatter(D(1,1),D(2,1),  'r','filled');
hold on;
scatter(D(1,2),D(2,2),  'b','filled');
hold on;
scatter(D(1,3),D(2,3),  'y','filled');
