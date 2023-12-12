function dae =Parking_dae(sol)

Length_Car=4.588;
Width_Car=1.868;  % 原程序写成1.878了
Hr_Car=2.705; %轴距
Hx_Car=0.974; %后悬
Qx_Car=Length_Car-Hx_Car;

iphase=sol.phase;
t = sol.time;
x = sol.state;
u = sol.control;

X = x(:,1);
Y = x(:,2);
fai = x(:,3);
v = u(:,1);
delta  = u(:,2);

%------------------车辆状态方程---------------------------------
x1dot = v.*cos(fai);
x2dot = v.*sin(fai);
x3dot = v.*tan(delta)/Hr_Car;

%---------------计算车辆四个顶点坐标----------------------------
x_r1=X-Hx_Car*cos(fai)-Width_Car/2*sin(fai);
y_r1=Y-Hx_Car*sin(fai)+Width_Car/2*cos(fai);

x_r2=X-Hx_Car*cos(fai)+Width_Car/2*sin(fai);
y_r2=Y-Hx_Car*sin(fai)-Width_Car/2*cos(fai);

x_f1=X+Qx_Car*cos(fai)-Width_Car/2*sin(fai);
y_f1=Y+Qx_Car*sin(fai)+Width_Car/2*cos(fai);

x_f2=X+Qx_Car*cos(fai)+Width_Car/2*sin(fai);
y_f2=Y+Qx_Car*sin(fai)-Width_Car/2*cos(fai);

if iphase==1
    path(:,1)=x_f2.*(y_f2<=5)+10*(y_f2>5);
    path(:,2)=y_r2;
elseif iphase==2
    path(:,1)=y_f1;
    path(:,2)=y_r2;
elseif iphase==3
    path(:,1)=y_f1;
    path(:,2)=y_r2;
    path(:,3)=y_f2;
else
    path=[];
end

dae = [x1dot x2dot x3dot path];

end

