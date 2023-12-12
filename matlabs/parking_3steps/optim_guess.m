function guess = optim_guess(paras) 

    iphase = 1;
    guess(iphase).time = [0.0; paras.tf * 1 / 3];
    guess(iphase).state(1,:) = paras.init;
    guess(iphase).state(2,:) = [2.0, 11.5, 0.0, 0.0, 0.0];
    guess(iphase).control = [0.0 0.0; 0.0 0.0];
    guess(iphase).parameter = [];

    iphase = 2;
    guess(iphase).time = [paras.tf * 1 / 3; paras.tf * 2 / 3];
    guess(iphase).state(1,:) = [2.0, 11.5, 0.0, 0.0, 0.0];
    guess(iphase).state(2,:) = [paras.end(1), paras.limits(2, 1) + paras.Parking_Y, paras.end(3), paras.end(4), paras.limits(5, 2)];
    guess(iphase).control = [0.0 0.0; 0.0 0.0];
    guess(iphase).parameter = [];
    
    iphase = 3;
    guess(iphase).time = [paras.tf * 2 / 3; paras.tf * 3 / 3];
    guess(iphase).state(1,:) = [paras.end(1), paras.limits(2, 1) + paras.Parking_Y, paras.end(3), paras.end(4), paras.limits(5, 2)];
    guess(iphase).state(2,:) = paras.end;
    guess(iphase).control = [0.0 0.0; 0.0 0.0];
    guess(iphase).parameter = [];

end