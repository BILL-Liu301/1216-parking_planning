function Y = Car_plot(x,y,fai,c)

Length_Car=4.588;
Width_Car=1.868;
Hr_Car=2.705; %轴距取为2.8 m
Hx_Car=0.974; %后悬取为0.9m
Qx_Car=Length_Car-Hx_Car;

x_r1=x-Hx_Car*cos(fai)-Width_Car/2*sin(fai);
y_r1=y-Hx_Car*sin(fai)+Width_Car/2*cos(fai);

x_r2=x-Hx_Car*cos(fai)+Width_Car/2*sin(fai);
y_r2=y-Hx_Car*sin(fai)-Width_Car/2*cos(fai);

x_f1=x+Qx_Car*cos(fai)-Width_Car/2*sin(fai);
y_f1=y+Qx_Car*sin(fai)+Width_Car/2*cos(fai);

x_f2=x+Qx_Car*cos(fai)+Width_Car/2*sin(fai);
y_f2=y+Qx_Car*sin(fai)-Width_Car/2*cos(fai);

LINE([x_r1,y_r1],[x_r2,y_r2],c)
LINE([x_f1,y_f1],[x_r1,y_r1],c)
LINE([x_f1,y_f1],[x_f2,y_f2],c)
LINE([x_f2,y_f2],[x_r2,y_r2],c)

end

