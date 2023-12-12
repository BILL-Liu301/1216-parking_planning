function guess = optim_guess(paras, limits)

    iphase = 1;
    guess(iphase).time = [0.0; paras.tf * 1 / 4];
    guess(iphase).state(1,:) = paras.init;
    guess(iphase).state(2,:) = [(limits(iphase).state.min(1, end) + limits(iphase).state.max(1, end))/2,... 
                                (limits(iphase).state.min(2, end) + limits(iphase).state.max(2, end))/2, 0.0, 0.0, 0.0];
    guess(iphase).control = [0.0 0.0; 0.0 0.0];
    guess(iphase).parameter = [];

    iphase = 2;
    guess(iphase).time = [paras.tf * 1 / 4; paras.tf * 2 / 4];
    guess(iphase).state(1,:) = [(limits(iphase).state.min(1, 1) + limits(iphase).state.max(1, 1))/2,... 
                                (limits(iphase).state.min(2, 1) + limits(iphase).state.max(2, 1))/2, 0.0, 0.0, 0.0];
    guess(iphase).state(2,:) = [(limits(iphase).state.min(1, end) + limits(iphase).state.max(1, end))/2,... 
                                (limits(iphase).state.min(2, end) + limits(iphase).state.max(2, end))/2, 0.0, 0.0, 0.0];
    guess(iphase).control = [0.0 0.0; 0.0 0.0];
    guess(iphase).parameter = [];

    iphase = 3;
    guess(iphase).time = [paras.tf * 2 / 4; paras.tf * 3 / 4];
    guess(iphase).state(1,:) = [(limits(iphase).state.min(1, 1) + limits(iphase).state.max(1, 1))/2,... 
                                (limits(iphase).state.min(2, 1) + limits(iphase).state.max(2, 1))/2, 0.0, 0.0, 0.0];
    guess(iphase).state(2,:) = [(limits(iphase).state.min(1, end) + limits(iphase).state.max(1, end))/2,... 
                                (limits(iphase).state.min(2, end) + limits(iphase).state.max(2, end))/2, 0.0, 0.0, 0.0];
    guess(iphase).control = [0.0 0.0; 0.0 0.0];
    guess(iphase).parameter = [];

    iphase = 4;
    guess(iphase).time = [paras.tf * 3 / 4; paras.tf * 4 / 4];
    guess(iphase).state(1,:) = [(limits(iphase).state.min(1, 1) + limits(iphase).state.max(1, 1))/2,... 
                                (limits(iphase).state.min(2, 1) + limits(iphase).state.max(2, 1))/2, 0.0, 0.0, 0.0];
    guess(iphase).state(2,:) = paras.end;
    guess(iphase).control = [0.0 0.0; 0.0 0.0];
    guess(iphase).parameter = [];

end