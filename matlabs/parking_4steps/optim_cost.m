function [Mayer,Lagrange] = optim_cost(sol)
    tf = sol.terminal.time;
    t = sol.time;

    Mayer = tf; 
    Lagrange = zeros(size(t));

    % Mayer = tf;
    % Lagrange = abs(sol.state(:, 5));
    
    % state = sol.state;
    % control = sol.control;
    % load('paras.mat')
    % plot([paras.limits(1, 1), paras.limits(-
    % 1, 2)], [paras.limits(2, 2), paras.limits(2, 2)])
    % hold on
    % plot([paras.limits(1, 1), -paras.Parking_X/2], [paras.Parking_Y, paras.Parking_Y])
    % hold on
    % plot([paras.Parking_X/2, paras.limits(1, 2)], [paras.Parking_Y, paras.Parking_Y])
    % hold on
    % plot([-paras.Parking_X/2, -paras.Parking_X/2], [paras.limits(2, 1), paras.Parking_Y])
    % hold on
    % plot([paras.Parking_X/2, paras.Parking_X/2], [paras.limits(2, 1), paras.Parking_Y])
    % hold on
    % plot(state(:, 1), state(:, 2))
    % hold on
    % pause(0.01)
    % if sol.phase == 5
    %     hold off
    % end

end