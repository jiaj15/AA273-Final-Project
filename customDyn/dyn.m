function x_next = dyn(x,dt,v,fi, Q,pred)
    n = size(x,2);
    x_next = zeros(3,n);
    px = x(1,:);  py = x(2,:); theta = x(3,:);
    x_next(1,:) = px+dt*v*cos(theta);
    x_next(2,:) = py+dt*v*sin(theta);
    x_next(3,:) = theta + dt*fi;
    
    if pred == false
        v  = randn([3,n]);
        x_next = x_next + Q^(0.5)*v;
    end
    
end