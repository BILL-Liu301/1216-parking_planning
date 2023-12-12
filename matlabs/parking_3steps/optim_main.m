function [parking_anchors, solution, SNOPT_info, Duration] = optim_main(init_x, init_y, init_theta)

    paras = optim_paras(init_x, init_y, init_theta);
    limits = optim_limits(paras);
    guess = optim_guess(paras);
    linkages = optim_linkages();
    setup = optim_setup(limits, guess, linkages);
    save('paras.mat', 'paras')
    
    [output,~] = gpops(setup);
    save('output.mat', 'output')
    
    parking_anchors(1, :) = [output.solution(1).state(end, 1), output.solution(1).state(end, 2), output.solution(1).state(end, 3)];
    parking_anchors(2, :) = [output.solution(2).state(end, 1), output.solution(2).state(end, 2), output.solution(2).state(end, 3)];
    parking_anchors(3, :) = [output.solution(3).state(end, 1), output.solution(3).state(end, 2), output.solution(3).state(end, 3)];
    solution = [ones(size(output.solution(1).time, 1), 1), output.solution(1).time, output.solution(1).state];
    solution = [solution; [ones(size(output.solution(2).time, 1), 1) * 2, output.solution(2).time, output.solution(2).state]];
    solution = [solution; [ones(size(output.solution(3).time, 1), 1) * 3, output.solution(3).time, output.solution(3).state]];
    SNOPT_info = output.SNOPT_info;
    Duration = [output.solution(1).Mayer_cost, output.solution(2).Mayer_cost, output.solution(3).Mayer_cost];

end