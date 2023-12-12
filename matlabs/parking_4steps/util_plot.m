function util_plot()
    load("paras.mat")
    load("output.mat")
    
    figure
    for i = 1:4
        state = output.solution(i).state;
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

        if i <= 3
            path_x = min([xr_r - paras.limits(1, 1), paras.limits(1, 2) - xr_r, ...
                          xr_l - paras.limits(1, 1), paras.limits(1, 2) - xr_l, ...
                          xf_r - paras.limits(1, 1), paras.limits(1, 2) - xf_r, ...
                          xf_l - paras.limits(1, 1), paras.limits(1, 2) - xf_l], [], 2);
            path_y = min([yr_r - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yr_r, ...
                          yr_l - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yr_l, ...
                          yf_r - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yf_r, ...
                          yf_l - (paras.limits(2, 1) + paras.Parking_Y), paras.limits(2, 2) - yf_l], [], 2);
        else
            path_x = min([(xr_r-paras.limits(1, 1)).*(yr_r>paras.Parking_Y) + (xr_r-(-paras.Parking_X/2)).*(yr_r<=paras.Parking_Y), ...
                          (paras.limits(1, 2)-xr_r).*(yr_r>paras.Parking_Y) + ((paras.Parking_X/2)-xr_r).*(yr_r<=paras.Parking_Y), ...
                          (xr_l-paras.limits(1, 1)).*(yr_l>paras.Parking_Y) + (xr_l-(-paras.Parking_X/2)).*(yr_l<=paras.Parking_Y), ...
                          (paras.limits(1, 2)-xr_l).*(yr_l>paras.Parking_Y) + ((paras.Parking_X/2)-xr_l).*(yr_l<=paras.Parking_Y), ...
                          (xf_r-paras.limits(1, 1)).*(yf_r>paras.Parking_Y) + (xf_r-(-paras.Parking_X/2)).*(yf_r<=paras.Parking_Y), ...
                          (paras.limits(1, 2)-xf_r).*(yf_r>paras.Parking_Y) + ((paras.Parking_X/2)-xf_r).*(yf_r<=paras.Parking_Y), ...
                          (xf_l-paras.limits(1, 1)).*(yf_l>paras.Parking_Y) + (xf_l-(-paras.Parking_X/2)).*(yf_l<=paras.Parking_Y), ...
                          (paras.limits(1, 2)-xf_l).*(yf_l>paras.Parking_Y) + ((paras.Parking_X/2)-xf_l).*(yf_l<=paras.Parking_Y)], [], 2);
            path_y = min([yr_r - paras.limits(2, 1), paras.limits(2, 2) - yr_r, ...
                          yr_l - paras.limits(2, 1), paras.limits(2, 2) - yr_l, ...
                          yf_r - paras.limits(2, 1), paras.limits(2, 2) - yf_r, ...
                          yf_l - paras.limits(2, 1), paras.limits(2, 2) - yf_l], [], 2);
        end

        clf
        subplot(2, 2, 1)
        plot([paras.limits(1, 1), paras.limits(1, 2)], [paras.limits(2, 2), paras.limits(2, 2)], "k")
        hold on
        plot([paras.limits(1, 1), -paras.Parking_X/2], [paras.Parking_Y, paras.Parking_Y], "k")
        hold on
        plot([paras.Parking_X/2, paras.limits(1, 2)], [paras.Parking_Y, paras.Parking_Y], "k")
        hold on
        plot([-paras.Parking_X/2, -paras.Parking_X/2], [paras.limits(2, 1), paras.Parking_Y], "k")
        hold on
        plot([paras.Parking_X/2, paras.Parking_X/2], [paras.limits(2, 1), paras.Parking_Y], "k")
        hold on
        
        if i>1
            for k = 1:(i-1)
                subplot(2, 2, 1)
                plot(output.solution(k).state(:, 1), output.solution(k).state(:, 2), "k--")
                hold on
                subplot(2, 2, 2)
                plot(output.solution(k).time(:, 1), output.solution(k).state(:, 5), "r")
                hold on
                plot([output.solution(k).time(end, 1), output.solution(k).time(end, 1)], [paras.limits(5, 1), paras.limits(5, 2)], "r--")
                hold on
                subplot(2, 2, 4)
                plot(output.solution(k).time(:, 1), output.solution(k).control(:, 1), "b")
                hold on
                plot([output.solution(k).time(end, 1), output.solution(k).time(end, 1)], [paras.controls(1, 1), paras.controls(1, 2)], "b--")
                hold on
            end
        end

        for j = 1:5:size(state, 1)
            subplot(2, 2, 1)
            plot([xr_r(j), xr_l(j)], [yr_r(j), yr_l(j)], "b")
            hold on
            plot([xr_l(j), xf_l(j)], [yr_l(j), yf_l(j)], "b")
            hold on
            plot([xf_l(j), xf_r(j)], [yf_l(j), yf_r(j)], "b")
            hold on
            plot([xf_r(j), xr_r(j)], [yf_r(j), yr_r(j)], "b")
            hold on
            plot(state(1:j, 1), state(1:j, 2), "r--")
            hold on
            title("Trajectory")
            subplot(2, 2, 2)
            plot(output.solution(i).time(1:j, 1), state(1:j, 5), "r")
            title("V")
            subplot(2, 2, 3)
            plot([output.solution(i).time(1, 1), output.solution(i).time(end, 1)], [paras.paths(1, 1), paras.paths(1, 1)], "k--")
            hold on
            plot([output.solution(i).time(1, 1), output.solution(i).time(end, 1)], [10.0, 10.0], "k--")
            hold on
            plot(output.solution(i).time(1:j, 1), path_x(1:j), "g")
            hold on
            plot(output.solution(i).time(1:j, 1), path_y(1:j), "r")
            hold on
            title("SafeDis")
            subplot(2, 2, 4)
            plot(output.solution(i).time(1:j, 1), output.solution(i).control(1:j, 1), "b")
            title("前轮转角角速度")
            pause(0.05)
        end
    end
end