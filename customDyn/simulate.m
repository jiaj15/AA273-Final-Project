clear all;
close all;

dt = 0.01;

V = @(t) 1;
FI = @(t) sin(t);
% true state noise
Q1 = 0.1 * dt * eye(3);
R1 = 0.1;

% noises used for EKF
Q2 = 0.1 * dt * eye(3);
R2 = 0.1;


lamba = 2;
p_num = 1000;

 
w0 =  [0; 0; 0]; %true state
mu0 = [0 ;0; 0];    %estimate state
cov0 = 0.01 * eye(3);

num_steps =1000;
W = zeros(3,num_steps+1);
W(:,1) = w0;
Y = zeros(2,num_steps+1);


for t=2:1:num_steps+1
    v = V((t-1)*dt);
    a = (t-1)*dt;
    fi = FI(a);
    
    w = W(:,t-1); 
    
    w_next = dyn(w,dt,v,fi, Q1,true);
    y = sensor(w,R1,false);
    W(:,t) = w_next;
    Y(:,t-1) = y;
end

plot(W(1,:),W(2,:));
quadPoints = 8:0.2:12;
D = [4,2,2.5;7,0,5];
 time = linspace(0,10,10);
U = inference(D,Y(:,1:100:1000),time,quadPoints);