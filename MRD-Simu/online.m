% D is the destination set [2,desDim]
% Y is measurement history [2,N]
% time respond to measurement history, e.g. at time[1], we get the measurement Y[1]
% quadPoints is [T1,T2,...,Tq], [1,q]

clear all;close all;


%% Get all the parameter
N = 100;                                %% time step
D = [4,  1.5,   4;
     5,    2,   4];
stateDim = size(D,1);
desNum = size(D,2);

Tq = 20;
quadPoints = Tq-2:0.5:Tq+2;
q = size(quadPoints,2);

trueDes = D(:,1);
x0 = [0;0];


%% intialize all the history container
% a table which stores d rows and q columns concatenated state vectors at time tn
Z = zeros(2 * stateDim * desNum, q); 
for d = 1:1:desNum
    A = repmat(D(:,d),1,q);
    B = Z(stateDim*(2*d-1)+1:2*stateDim*d,:);
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

% a row which which stores all the unnormalized probablitis of pd, at time tn
P = zeros(1, desNum);

prior = ones(1,desNum)/desNum;

U = zeros(N, desNum);


%% Parameter Setting for the MRD model
sig = diag([0.1, 0.1]);         % small sigma in MRD model
h = 0.1;                      % time step
Vn = [2, 1; 1, 5] / 10; 
lamda = diag([1, 1]);
r = stateDim;

%% main loop for destination inference
for t=1:1:N
    % responding time for a measurement
    %tn = time(t);
    tn = t*h;
    x = X(:,t);
    [x_next, y, R_t, U_t, G_til, m_til] = MRD(x, trueDes, h, lamda, sig, Vn, r, Tq, tn);
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
    L = L./sum(L);
    U(t,:) = P .* prior/sum(P .* prior);

end

%% Plot

time = h * (1:1:N);

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
