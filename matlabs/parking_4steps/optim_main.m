function [parking_anchors, solution, SNOPT_info, Duration, Euclidean] = optim_main(init_x, init_y, init_theta)

    paras = optim_paras(init_x, init_y, init_theta);
    limits = optim_limits(paras);
    guess = optim_guess(paras, limits);
    linkages = optim_linkages();
    setup = optim_setup(limits, guess, linkages);
    save('paras.mat', 'paras')
    
    % figure()
    [output,~] = gpops(setup);
    save('output.mat', 'output')

    sizes = util_sizes(output);
    parking_anchors(1, :) = [output.solution(1).state(1, 1:3)];
    parking_anchors(2, :) = [output.solution(1).state(sizes(1), 1:3)];
    parking_anchors(3, :) = [output.solution(1).state(end, 1:3)];
    parking_anchors(4, :) = [output.solution(2).state(sizes(2), 1:3)];
    parking_anchors(5, :) = [output.solution(2).state(end, 1:3)];
    parking_anchors(6, :) = [output.solution(3).state(sizes(3), 1:3)];
    parking_anchors(7, :) = [output.solution(3).state(end, 1:3)];
    parking_anchors(8, :) = [output.solution(4).state(round(sizes(4)/2), 1:3)];
    parking_anchors(9, :) = [output.solution(4).state(sizes(4), 1:3)];
    % dis_ = abs(output.solution(4).state(:, 2) - paras.Parking_Y);
    % ind_ = find(dis_ == min(dis_));
    % parking_anchors(4, :) = [output.solution(4).state(ind_, 1), output.solution(4).state(ind_, 2), output.solution(4).state(ind_, 3)];
    solution = [ones(size(output.solution(1).time, 1), 1), output.solution(1).time, output.solution(1).state];
    solution = [solution; [ones(size(output.solution(2).time, 1), 1) * 2, output.solution(2).time, output.solution(2).state]];
    solution = [solution; [ones(size(output.solution(3).time, 1), 1) * 3, output.solution(3).time, output.solution(3).state]];
    solution = [solution; [ones(size(output.solution(4).time, 1), 1) * 4, output.solution(4).time, output.solution(4).state]];
    SNOPT_info = output.SNOPT_info;
    Duration = [output.solution(1).Mayer_cost, output.solution(2).Mayer_cost, output.solution(3).Mayer_cost, output.solution(4).Mayer_cost];
    Euclidean = [output.solution(1).Lagrange_cost, output.solution(2).Lagrange_cost, output.solution(3).Lagrange_cost, output.solution(4).Lagrange_cost];

end