function y = sensor(x,R,pred)
    n = size(x,2);
    px = x(1,:);
    py = x(2,:);
    y = [px; py];
    if pred == false
        v = randn([2,n]);
        y = y + R^(0.5)*v;
    end
    
end