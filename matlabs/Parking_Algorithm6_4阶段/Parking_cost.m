function [Mayer,Lagrange]=Parking_cost(sol)

Width_Car=1.868;
Hx_Car=0.974; %ºóÐü
Qx_Car=4.596-Hx_Car;

iphase = sol.phase;
tf = sol.terminal.time;
xf = sol.terminal.state;
t = sol.time;
x = sol.state;
u = sol.control;

X = x(:,1);
Y = x(:,2);
fai = x(:,3);
v = u(:,1);
delta  = u(:,2);

Mayer    = tf; 
Lagrange = zeros(size(t));

end

    
    
