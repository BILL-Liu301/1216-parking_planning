function output = call_optim(init_x, init_y, init_theta)
%     addpath('../matlabs/parking_5steps/')
    try
        [~, parking_anchors, solution, SNOPT_info, Duration, Euclidean] = evalc('optim_main(init_x, init_y, init_theta);');
    end
    % [parking_anchors, solution, SNOPT_info, Duration, Euclidean] = optim_main(init_x, init_y, init_theta);
    output.parking_anchors = parking_anchors;
    output.solution = solution;
    output.SNOPT_info = SNOPT_info;
    output.Duration = Duration;
    output.Euclidean = Euclidean;
    % rmpath('../matlabs/parking_5steps/')
    
    % delete('./output.mat')
    % delete('./paras.mat')
    % delete('./snoptmain.out')
    % delete('./optim_parking.txt')

    % util_plot()
end
