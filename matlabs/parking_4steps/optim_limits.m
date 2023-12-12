function limits = optim_limits(paras)

    iphase = 1;
    limits(iphase).time.min = [0.0, 0.1];
    limits(iphase).time.max = [0.0, paras.tf];
    limits(iphase).state.min(1,:) = [paras.init(1), paras.limits(1, 1), paras.end(1) + paras.Parking_X/2];
    limits(iphase).state.min(2,:) = [paras.init(2), paras.limits(2, 1) + paras.Parking_Y, paras.limits(2, 1) + paras.Parking_Y];
    limits(iphase).state.min(3,:) = [paras.init(3), paras.limits(3, 1), paras.limits(3, 1)];
    limits(iphase).state.min(4,:) = [paras.init(4), paras.limits(4, 1), paras.limits(4, 1)];
    limits(iphase).state.min(5,:) = [paras.init(5), 0.0, 0.0];
    limits(iphase).state.max(1,:) = [paras.init(1), paras.limits(1, 2), paras.limits(1, 2)];
    limits(iphase).state.max(2,:) = [paras.init(2), paras.limits(2, 2), paras.limits(2, 2)];
    limits(iphase).state.max(3,:) = [paras.init(3), paras.limits(3, 2), paras.limits(3, 2)];
    limits(iphase).state.max(4,:) = [paras.init(4), paras.limits(4, 2), paras.limits(4, 2)];
    limits(iphase).state.max(5,:) = [paras.init(5), paras.limits(5, 2), 0.0];
    limits(iphase).control.min = [paras.controls(1, 1); paras.controls(2, 1)];
    limits(iphase).control.max = [paras.controls(1, 2); paras.controls(2, 2)];
    limits(iphase).parameter.min = [];
    limits(iphase).parameter.max = [];
    limits(iphase).path.min = [paras.paths(1, 1); paras.paths(1, 1); paras.paths(2, 1)];
    limits(iphase).path.max = [paras.paths(1, 2); paras.paths(1, 2); paras.paths(2, 2)];
    limits(iphase).duration.min = 0.0;
    limits(iphase).duration.max = paras.tf;

    iphase = 2;
    limits(iphase).time.min = [0.1, 0.1];
    limits(iphase).time.max = [paras.tf, paras.tf];
    limits(iphase).state.min(1,:) = [limits(iphase-1).state.min(1,end), paras.limits(1, 1), paras.limits(1, 1)];
    limits(iphase).state.min(2,:) = [limits(iphase-1).state.min(2,end), paras.limits(2, 1) + paras.Parking_Y, paras.limits(2, 1) + paras.Parking_Y];
    limits(iphase).state.min(3,:) = [limits(iphase-1).state.min(3,end), paras.limits(3, 1), paras.limits(3, 1)];
    limits(iphase).state.min(4,:) = [limits(iphase-1).state.min(4,end), paras.limits(4, 1), paras.limits(4, 1)];
    limits(iphase).state.min(5,:) = [limits(iphase-1).state.min(5,end), paras.limits(5, 1), 0.0];
    limits(iphase).state.max(1,:) = [limits(iphase-1).state.max(1,end), paras.limits(1, 2), paras.end(1) - paras.Parking_X/2];
    limits(iphase).state.max(2,:) = [limits(iphase-1).state.max(2,end), paras.limits(2, 2), paras.limits(2, 1) + paras.Parking_Y + paras.Freespace_Y/2];
    limits(iphase).state.max(3,:) = [limits(iphase-1).state.max(3,end), paras.limits(3, 2), paras.limits(3, 2)];
    limits(iphase).state.max(4,:) = [limits(iphase-1).state.max(4,end), paras.limits(4, 2), paras.limits(4, 2)];
    limits(iphase).state.max(5,:) = [limits(iphase-1).state.max(5,end), 0.0, 0.0];
    limits(iphase).control.min = [paras.controls(1, 1); paras.controls(2, 1)];
    limits(iphase).control.max = [paras.controls(1, 2); paras.controls(2, 2)];
    limits(iphase).parameter.min = [];
    limits(iphase).parameter.max = [];
    limits(iphase).path.min = [paras.paths(1, 1); paras.paths(1, 1); paras.paths(2, 1)];
    limits(iphase).path.max = [paras.paths(1, 2); paras.paths(1, 2); paras.paths(2, 2)];
    limits(iphase).duration.min = 0.0;
    limits(iphase).duration.max = paras.tf;

    iphase = 3;
    limits(iphase).time.min = [0.1, 0.1];
    limits(iphase).time.max = [paras.tf, paras.tf];
    limits(iphase).state.min(1,:) = [limits(iphase-1).state.min(1,end), paras.limits(1, 1), paras.end(1) + paras.Parking_X/2];
    limits(iphase).state.min(2,:) = [limits(iphase-1).state.min(2,end), paras.limits(2, 1) + paras.Parking_Y, paras.limits(2, 1) + paras.Parking_Y + paras.Freespace_Y*1/2];
    limits(iphase).state.min(3,:) = [limits(iphase-1).state.min(3,end), paras.limits(3, 1), paras.limits(3, 1)];
    limits(iphase).state.min(4,:) = [limits(iphase-1).state.min(4,end), paras.limits(4, 1), paras.limits(4, 1)];
    limits(iphase).state.min(5,:) = [limits(iphase-1).state.min(5,end), 0.0, 0.0];
    limits(iphase).state.max(1,:) = [limits(iphase-1).state.max(1,end), paras.limits(1, 2), paras.limits(1, 2)];
    limits(iphase).state.max(2,:) = [limits(iphase-1).state.max(2,end), paras.limits(2, 2), paras.limits(2, 2)];
    limits(iphase).state.max(3,:) = [limits(iphase-1).state.max(3,end), paras.limits(3, 2), paras.limits(3, 2)];
    limits(iphase).state.max(4,:) = [limits(iphase-1).state.max(4,end), paras.limits(4, 2), paras.limits(4, 2)];
    limits(iphase).state.max(5,:) = [limits(iphase-1).state.max(5,end), paras.limits(5, 2), 0.0];
    limits(iphase).control.min = [paras.controls(1, 1); paras.controls(2, 1)];
    limits(iphase).control.max = [paras.controls(1, 2); paras.controls(2, 2)];
    limits(iphase).parameter.min = [];
    limits(iphase).parameter.max = [];
    limits(iphase).path.min = [paras.paths(1, 1); paras.paths(1, 1); paras.paths(2, 1)];
    limits(iphase).path.max = [paras.paths(1, 2); paras.paths(1, 2); paras.paths(2, 2)];
    limits(iphase).duration.min = 0.0;
    limits(iphase).duration.max = paras.tf;
    
    iphase = 4;
    limits(iphase).time.min = [0.1, 0.1];
    limits(iphase).time.max = [paras.tf, paras.tf];
    limits(iphase).state.min(1,:) = [limits(iphase-1).state.min(1,end), paras.limits(1, 1), paras.end(1)];
    limits(iphase).state.min(2,:) = [limits(iphase-1).state.min(2,end), paras.limits(2, 1), paras.end(2)];
    limits(iphase).state.min(3,:) = [limits(iphase-1).state.min(3,end), paras.limits(3, 1), paras.end(3)];
    limits(iphase).state.min(4,:) = [limits(iphase-1).state.min(4,end), paras.limits(4, 1), paras.end(4)];
    limits(iphase).state.min(5,:) = [limits(iphase-1).state.min(5,end), paras.limits(5, 1), paras.end(5)];
    limits(iphase).state.max(1,:) = [limits(iphase-1).state.max(1,end), paras.limits(1, 2), paras.end(1)];
    limits(iphase).state.max(2,:) = [limits(iphase-1).state.max(2,end), paras.limits(2, 2), paras.end(2)];
    limits(iphase).state.max(3,:) = [limits(iphase-1).state.max(3,end), paras.limits(3, 2), paras.end(3)];
    limits(iphase).state.max(4,:) = [limits(iphase-1).state.max(4,end), paras.limits(4, 2), paras.end(4)];
    limits(iphase).state.max(5,:) = [limits(iphase-1).state.max(5,end), 0.0, paras.end(5)];
    limits(iphase).control.min = [paras.controls(1, 1); paras.controls(2, 1)];
    limits(iphase).control.max = [paras.controls(1, 2); paras.controls(2, 2)];
    limits(iphase).parameter.min = [];
    limits(iphase).parameter.max = [];
    limits(iphase).path.min = [paras.paths(1, 1); paras.paths(1, 1); paras.paths(2, 1)];
    limits(iphase).path.max = [paras.paths(1, 2); paras.paths(1, 2); paras.paths(2, 2)];
    limits(iphase).duration.min = 0.0;
    limits(iphase).duration.max = paras.tf;

end