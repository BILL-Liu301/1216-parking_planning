function dae = optim_dae(sol)
    state = sol.state;
    control = sol.control;
    iphase = sol.phase;
    load('paras.mat')
    
    xdot = state(:, 5) .* cos(state(:, 3));
    ydot = state(:, 5) .* sin(state(:, 3));
    thetadot = state(:, 5) .* tan(state(:, 4)) / paras.Car_L;
    deltadot = control(:, 1);
    vdot = control(:, 2);

    xmid_r = state(:, 1);
    ymid_r = state(:, 2);
    xmid_f = xmid_r + cos(state(:, 3)) .* paras.Car_L;
    ymid_f = ymid_r + sin(state(:, 3)) .* paras.Car_L;
    xmid_r = xmid_r - cos(state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;
    ymid_r = ymid_r - sin(state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;
    xmid_f = xmid_f + cos(state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;
    ymid_f = ymid_f + sin(state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;

    xr_r = xmid_r + cos(state(:, 3) - pi/2) .* paras.Car_Width/2;
    yr_r = ymid_r + sin(state(:, 3) - pi/2) .* paras.Car_Width/2;
    xr_l = xmid_r + cos(state(:, 3) + pi/2) .* paras.Car_Width/2;
    yr_l = ymid_r + sin(state(:, 3) + pi/2) .* paras.Car_Width/2;
    xf_r = xmid_f + cos(state(:, 3) - pi/2) .* paras.Car_Width/2;
    yf_r = ymid_f + sin(state(:, 3) - pi/2) .* paras.Car_Width/2;
    xf_l = xmid_f + cos(state(:, 3) + pi/2) .* paras.Car_Width/2;
    yf_l = ymid_f + sin(state(:, 3) + pi/2) .* paras.Car_Width/2;
    
    if iphase ~= 3
        path_x = min([xr_r - paras.limits(1, 1), paras.limits(1, 2) - xr_r, ...
                      xr_l - paras.limits(1, 1), paras.limits(1, 2) - xr_l, ...
                      xf_r - paras.limits(1, 1), paras.limits(1, 2) - xf_r, ...
                      xf_l - paras.limits(1, 1), paras.limits(1, 2) - xf_l], [], 2);
        path_y = min([yr_r - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yr_r, ...
                      yr_l - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yr_l, ...
                      yf_r - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yf_r, ...
                      yf_l - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yf_l], [], 2);
    else
        path_x = min([xr_r.*(yr_r>paras.Parking_Y)-paras.limits(1, 1) + xr_r.*(yr_r<=paras.Parking_Y)-(-paras.Parking_X/2), ...
                      paras.limits(1, 2)-xr_r.*(yr_r>paras.Parking_Y) + (paras.Parking_X/2)-xr_r.*(yr_r<=paras.Parking_Y), ...
                      xr_l.*(yr_l>paras.Parking_Y)-paras.limits(1, 1) + xr_l.*(yr_l<=paras.Parking_Y)-(-paras.Parking_X/2), ...
                      paras.limits(1, 2)-xr_l.*(yr_l>paras.Parking_Y) + (paras.Parking_X/2)-xr_l.*(yr_l<=paras.Parking_Y), ...
                      xf_r.*(yf_r>paras.Parking_Y)-paras.limits(1, 1) + xf_r.*(yf_r<=paras.Parking_Y)-(-paras.Parking_X/2), ...
                      paras.limits(1, 2)-xf_r.*(yf_r>paras.Parking_Y) + (paras.Parking_X/2)-xf_r.*(yf_r<=paras.Parking_Y), ...
                      xf_l.*(yf_l>paras.Parking_Y)-paras.limits(1, 1) + xf_l.*(yf_l<=paras.Parking_Y)-(-paras.Parking_X/2), ...
                      paras.limits(1, 2)-xf_l.*(yf_l>paras.Parking_Y) + (paras.Parking_X/2)-xf_l.*(yf_l<=paras.Parking_Y)], [], 2);
        path_y = min([yr_r - paras.limits(2, 1), paras.limits(2, 2) - yr_r, ...
                      yr_l - paras.limits(2, 1), paras.limits(2, 2) - yr_l, ...
                      yf_r - paras.limits(2, 1), paras.limits(2, 2) - yf_r, ...
                      yf_l - paras.limits(2, 1), paras.limits(2, 2) - yf_l], [], 2);
    end

    dae = [xdot, ydot, thetadot, deltadot, vdot, path_x, path_y, thetadot];

end