function util_plot(paras,output)
    figure
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
    for i = 1:3
            xmid_r = output.solution(i).state(:, 1);
            ymid_r = output.solution(i).state(:, 2);
            xmid_f = xmid_r + cos(output.solution(i).state(:, 3)) .* paras.Car_L;
            ymid_f = ymid_r + sin(output.solution(i).state(:, 3)) .* paras.Car_L;
            xmid_r = xmid_r - cos(output.solution(i).state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;
            ymid_r = ymid_r - sin(output.solution(i).state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;
            xmid_f = xmid_f + cos(output.solution(i).state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;
            ymid_f = ymid_f + sin(output.solution(i).state(:, 3)) .* (paras.Car_Length - paras.Car_L)/2;
        
            xr_r = xmid_r + cos(output.solution(i).state(:, 3) - pi/2) .* paras.Car_Width/2;
            yr_r = ymid_r + sin(output.solution(i).state(:, 3) - pi/2) .* paras.Car_Width/2;
            xr_l = xmid_r + cos(output.solution(i).state(:, 3) + pi/2) .* paras.Car_Width/2;
            yr_l = ymid_r + sin(output.solution(i).state(:, 3) + pi/2) .* paras.Car_Width/2;
            xf_r = xmid_f + cos(output.solution(i).state(:, 3) - pi/2) .* paras.Car_Width/2;
            yf_r = ymid_f + sin(output.solution(i).state(:, 3) - pi/2) .* paras.Car_Width/2;
            xf_l = xmid_f + cos(output.solution(i).state(:, 3) + pi/2) .* paras.Car_Width/2;
            yf_l = ymid_f + sin(output.solution(i).state(:, 3) + pi/2) .* paras.Car_Width/2;
        for j = 1:5:size(output.solution(i).state, 1)
            plot([xr_r(j), xr_l(j)], [yr_r(j), yr_l(j)], "b")
            plot([xr_l(j), xf_l(j)], [yr_l(j), yf_l(j)], "b")
            plot([xf_l(j), xf_r(j)], [yf_l(j), yf_r(j)], "b")
            plot([xf_r(j), xr_r(j)], [yf_r(j), yr_r(j)], "b")
            plot(output.solution(i).state(1:j, 1), output.solution(i).state(1:j, 2), "k--")
            pause(0.01)
        end
    end
    hold on
end