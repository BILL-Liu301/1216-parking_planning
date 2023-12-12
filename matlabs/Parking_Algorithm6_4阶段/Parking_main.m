%--------------------------------------------------------------------------
%----------------------4阶段泊车轨迹规划程序--------------------------------

function parking_anchor = Parking_main(init_x, init_y)
    global CONSTANTS
    CONSTANTS.Length_Range=20;
    CONSTANTS.Width_Range=6;
    CONSTANTS.Length_Car=4.588;
    CONSTANTS.Width_Car=1.868;
    CONSTANTS.Hr_Car=2.705;   %轴距取为2.8 m
    CONSTANTS.Hx_Car=0.974;   %后悬取为0.9m
    CONSTANTS.Length_Carparking=2.8;
    CONSTANTS.Width_Carparking=5;
    CONSTANTS.R_min=5;
    CONSTANTS.threshold=0.2;  %阈值为0.2m

    Length_Range=CONSTANTS.Length_Range;
    Width_Range=CONSTANTS.Width_Range;
    Length_Car=CONSTANTS.Length_Car;
    Width_Car=CONSTANTS.Width_Car;
    Hr_Car=CONSTANTS.Hr_Car;
    Hx_Car=CONSTANTS.Hx_Car;
    Qx_Car=CONSTANTS.Length_Car-CONSTANTS.Hx_Car;
    Length_Carparking=CONSTANTS.Length_Carparking;
    Width_Carparking=CONSTANTS.Width_Carparking;
    R_min=CONSTANTS.R_min;
    threshold=CONSTANTS.threshold;

    tic
    tf = 50;         % 结束时间

    xmin =threshold+Hx_Car;
    xmax =Length_Range-threshold-Hx_Car;
    ymin =0+threshold+Hx_Car;
    ymax =Width_Range+Width_Carparking-threshold-Hx_Car;
    fai_min = -2*pi;
    fai_max =2*pi;

%     x10=Length_Range/2;
%     y10=Length_Car-Hx_Car+threshold;
    x10 = init_x;
    y10 = init_y;
    fai10 =-pi/2; 


    y2f =1+Width_Carparking;         % 后轴中心点终值纵坐标
    fai2f =-pi;                          % 终值航向角


    u1min = 2/3.6;         % 车速
    u1max = 2/3.6;

    u2_max=atan(Hr_Car/R_min);

    u2min = -u2_max;     % 轮胎转角
    u2max = u2_max;

    iphase = 1;
    limits(iphase).time.min = [0 0.1];
    limits(iphase).time.max = [0 tf];
    limits(iphase).state.min(1,:) = [x10 xmin xmin];
    limits(iphase).state.max(1,:) = [x10 xmax xmax];
    limits(iphase).state.min(2,:) = [y10 ymin ymax-6];
    limits(iphase).state.max(2,:) = [y10 ymax ymax];
    limits(iphase).state.min(3,:) = [fai10 -pi -pi];
    limits(iphase).state.max(3,:) = [fai10 fai10 fai10];
    limits(iphase).control.min(1,:) = -u1max;
    limits(iphase).control.max(1,:) = -u1max;
    limits(iphase).control.min(2,:) = u2min;
    limits(iphase).control.max(2,:) = u2max;
    limits(iphase).parameter.min    = [];
    limits(iphase).parameter.max    = [];
    % limits(iphase).path.min    = [];
    % limits(iphase).path.max    = [];
    limits(iphase).path.min    = [Length_Range/2-Length_Carparking/2+threshold;0];
    limits(iphase).path.max    = [Length_Range;Width_Range+Width_Carparking-threshold];
    limits(iphase).duration.min = 0;
    limits(iphase).duration.max = tf;
    guess(iphase).time = [0; 5];
    guess(iphase).state(:,1) = [x10; x10];
    guess(iphase).state(:,2) = [y10; y10+6];
    guess(iphase).state(:,3) = [fai10; fai10];
    guess(iphase).control(:,1) = [-u1max; -u1max];
    guess(iphase).control(:,2) = [0; 0];
    guess(iphase).parameter    = [];

    iphase = 2;
    limits(iphase).time.min = [0.1 0.1];
    limits(iphase).time.max = [tf tf];
    limits(iphase).state.min(1,:) = [10 xmin xmin];
    limits(iphase).state.max(1,:) = [15 xmax xmax];
    limits(iphase).state.min(2,:) = [ymin ymin ymin];
    limits(iphase).state.max(2,:) = [ymax ymax ymax];
    limits(iphase).state.min(3,:) = [fai_min fai_min fai_min];
    limits(iphase).state.max(3,:) = [fai_max fai_max fai_max];
    limits(iphase).control.min(1,:) = u1max;
    limits(iphase).control.max(1,:) = u1max;
    limits(iphase).control.min(2,:) = u2min;
    limits(iphase).control.max(2,:) = u2max;
    limits(iphase).parameter.min    = [];
    limits(iphase).parameter.max    = [];
    limits(iphase).path.min    = [Width_Carparking+threshold;5];
    limits(iphase).path.max    = [Width_Range+Width_Carparking-threshold;Width_Range+Width_Carparking-threshold];
    % limits(iphase).path.min    = [];
    % limits(iphase).path.max    = [];
    limits(iphase).duration.min = 0;
    limits(iphase).duration.max = tf;
    guess(iphase).time = [5; 10];
    guess(iphase).state(:,1) = [10; 9];
    guess(iphase).state(:,2) = [8; 5.5];
    guess(iphase).state(:,3) = [-pi/2; -3/4*pi];
    guess(iphase).control(:,1) = [u1max; u1max];
    guess(iphase).control(:,2) = [0; 0];
    guess(iphase).parameter    = [];

    iphase = 3;
    limits(iphase).time.min = [0.1 0.1];
    limits(iphase).time.max = [tf tf];
    limits(iphase).state.min(1,:) = [10.1 xmin xmin];%10  12
    limits(iphase).state.max(1,:) = [xmax xmax xmax];%16
    limits(iphase).state.min(2,:) = [ymin ymin ymin];
    limits(iphase).state.max(2,:) = [ymax inf inf];
    limits(iphase).state.min(3,:) = [fai_min fai_min -pi]; % -pi
    limits(iphase).state.max(3,:) = [fai_max fai_max -pi]; % -pi
    limits(iphase).control.min(1,:) = -u1max;
    limits(iphase).control.max(1,:) = -u1max;
    limits(iphase).control.min(2,:) = u2min;
    limits(iphase).control.max(2,:) = u2max;
    limits(iphase).parameter.min    = [];
    limits(iphase).parameter.max    = [];
    limits(iphase).path.min    = [5.2;5;5];
    limits(iphase).path.max    = [10.8;10.8;10.8];
    limits(iphase).duration.min = 0;
    limits(iphase).duration.max = tf;
    guess(iphase).time = [10; 30];
    guess(iphase).state(:,1) = [8; 11];
    guess(iphase).state(:,2) = [5.5; 8];
    guess(iphase).state(:,3) = [-pi; -pi];
    guess(iphase).control(:,1) = [-u1max; -u1max];
    guess(iphase).control(:,2) = [0; 0];
    guess(iphase).parameter    = [];

    iphase = 4;
    limits(iphase).time.min = [0.1 0.1];
    limits(iphase).time.max = [tf tf];
    limits(iphase).state.min(1,:) = [xmin xmin 7.32]; % 11.32
    limits(iphase).state.max(1,:) = [xmax xmax 7.32]; % 11.32
    % limits(iphase).state.min(1,:) = [xmin xmin Length_Range/2+Length_Car/2-Hx_Car];
    % limits(iphase).state.max(1,:) = [xmax xmax Length_Range/2+Length_Car/2-Hx_Car];
    limits(iphase).state.min(2,:) = [ymin ymin Width_Carparking+Width_Car/2+1.2+0.2];
    limits(iphase).state.max(2,:) = [ymax ymax Width_Carparking+Width_Car/2+1.2+0.2];
    limits(iphase).state.min(3,:) = [fai_min fai_min -pi];
    limits(iphase).state.max(3,:) = [fai_max fai_max -pi];
    limits(iphase).control.min(1,:) = u1max;
    limits(iphase).control.max(1,:) = u1max;
    limits(iphase).control.min(2,:) = u2min;
    limits(iphase).control.max(2,:) = u2max;
    limits(iphase).parameter.min    = [];
    limits(iphase).parameter.max    = [];
    limits(iphase).path.min    = [];
    limits(iphase).path.max    = [];
    limits(iphase).duration.min = 0;
    limits(iphase).duration.max = tf;
    guess(iphase).time = [30; 40];
    guess(iphase).state(:,1) = [8; 5];
    guess(iphase).state(:,2) = [6.5; 6.5];
    guess(iphase).state(:,3) = [-3/4*pi; -pi];
    guess(iphase).control(:,1) = [u1max; u1max];
    guess(iphase).control(:,2) = [0; 0];
    guess(iphase).parameter    = [];

    ipair = 1;
    linkages(ipair).left.phase = 1;
    linkages(ipair).right.phase = 2;
    linkages(ipair).min = [0; 0; 0];
    linkages(ipair).max = [0; 0; 0];

    ipair = 2;
    linkages(ipair).left.phase = 2;
    linkages(ipair).right.phase = 3;
    linkages(ipair).min = [0; 0; 0];
    linkages(ipair).max = [0; 0; 0];

    ipair = 3;
    linkages(ipair).left.phase = 3;
    linkages(ipair).right.phase = 4;
    linkages(ipair).min = [0; 0; 0];
    linkages(ipair).max = [0; 0; 0];

    setup.name = 'Parking_main';
    setup.funcs.dae = 'Parking_dae';
    setup.funcs.cost = 'Parking_cost';
    setup.funcs.link = 'Parking_link';
    setup.derivatives = 'finite-difference';
    setup.checkDerivatives = 0;
    setup.limits = limits;
    setup.guess = guess;
    setup.linkages = linkages;
    setup.autoscale = 'off';
    setup.mesh.tolerance = 1e-6;
    setup.mesh.iteration = 7;
    setup.mesh.nodesPerInterval.min = 4;
    setup.mesh.nodesPerInterval.max = 8;

    [output,gpopsHistory] = gpops(setup);
    
    parking_anchor(1, :) = [output.solution(1).state(end, 1), output.solution(1).state(end, 2)];
    parking_anchor(2, :) = [output.solution(2).state(end, 1), output.solution(2).state(end, 2)];
    parking_anchor(3, :) = [output.solution(3).state(end, 1), output.solution(3).state(end, 2)];
    
    toc
    time=toc;
    solutionPlot= output.solutionPlot;
    solution = output.solution;
    output.SNOPT_info
    if output.SNOPT_info==1
        disp('Success!')
    else 
        disp('Failue!')
    end

    %----------------可行驶区域-------------------------------------------------
    figure
    LINE([0,Width_Range+Width_Carparking],[Length_Range,Width_Range+Width_Carparking],'b')
    LINE([0,Width_Range+Width_Carparking-threshold],[Length_Range,Width_Range+Width_Carparking-threshold],'k--')
    LINE([0,Width_Range+Width_Carparking],[0,Width_Carparking],'b')
    LINE([Length_Range,Width_Range+Width_Carparking],[Length_Range,Width_Carparking],'b')
    LINE([0,Width_Carparking],[Length_Range/2-Length_Carparking/2,Width_Carparking],'b')
    LINE([0,Width_Carparking+threshold],[Length_Range/2-Length_Carparking/2,Width_Carparking+threshold],'k--')
    LINE([Length_Range/2+Length_Carparking/2,Width_Carparking+threshold],[Length_Range,Width_Carparking+threshold],'k--')
    LINE([Length_Range/2+Length_Carparking/2,Width_Carparking],[Length_Range,Width_Carparking],'b')
    LINE([Length_Range/2-Length_Carparking/2,0],[Length_Range/2-Length_Carparking/2,Width_Carparking],'b')
    LINE([Length_Range/2-Length_Carparking/2+threshold,0],[Length_Range/2-Length_Carparking/2+threshold,Width_Carparking],'k--')
    LINE([Length_Range/2+Length_Carparking/2,0],[Length_Range/2+Length_Carparking/2,Width_Carparking],'b')
    LINE([Length_Range/2+Length_Carparking/2-threshold,0],[Length_Range/2+Length_Carparking/2-threshold,Width_Carparking],'k--')
    LINE([Length_Range/2-Length_Carparking/2,0],[Length_Range/2+Length_Carparking/2,0],'b')
    LINE([Length_Range/2-Length_Carparking/2,threshold],[Length_Range/2+Length_Carparking/2,threshold],'k--')


    %--------------------------------------------------------------------------

    %---------------------------车辆后轴中心坐标计算-----------------------------
    %--------------------------------第一阶段P1---------------------------------


    for i=1:5:length(solutionPlot(1).state(:,1))
    Car_plot(solutionPlot(1).state(i,1),solutionPlot(1).state(i,2),solutionPlot(1).state(i,3),'g')
    end
    plot(solutionPlot(1).state(:,1),solutionPlot(1).state(:,2),'g-.')
    
    
    for i=1:5:length(solutionPlot(2).state(:,1))
    Car_plot(solutionPlot(2).state(i,1),solutionPlot(2).state(i,2),solutionPlot(2).state(i,3),'r')
    end
    plot(solutionPlot(2).state(:,1),solutionPlot(2).state(:,2),'r-.')
    
    for i=1:5:length(solutionPlot(3).state(:,1))
    Car_plot(solutionPlot(3).state(i,1),solutionPlot(3).state(i,2),solutionPlot(3).state(i,3),'b')
    end
    Car_plot(solutionPlot(3).state(end,1),solutionPlot(3).state(end,2),solutionPlot(3).state(end,3),'b')
    
    plot(solutionPlot(3).state(:,1),solutionPlot(3).state(:,2),'b-.')
    
    for i=1:5:length(solutionPlot(4).state(:,1))
    Car_plot(solutionPlot(4).state(i,1),solutionPlot(4).state(i,2),solutionPlot(4).state(i,3),'k')
    end
    Car_plot(solutionPlot(4).state(end,1),solutionPlot(4).state(end,2),solutionPlot(4).state(end,3),'k')
    
    plot(solutionPlot(4).state(:,1),solutionPlot(4).state(:,2),'k-.')
    
    hold off
    % 
    % 
    % figure
    % plot(max(solutionPlot(4).time)-solutionPlot(1).time,solutionPlot(1).state(:,1),'g')
    % hold on
    % plot(max(solutionPlot(4).time)-solutionPlot(2).time,solutionPlot(2).state(:,1),'r')
    % plot(max(solutionPlot(4).time)-solutionPlot(3).time,solutionPlot(3).state(:,1),'b')
    % plot(max(solutionPlot(4).time)-solutionPlot(4).time,solutionPlot(4).state(:,1),'k')
    % xlabel('时间t/s')
    % ylabel('后轴中心横坐标x/m')
    % hold off
    % 
    % figure
    % plot(max(solutionPlot(4).time)-solutionPlot(1).time,solutionPlot(1).state(:,2),'g')
    % hold on
    % plot(max(solutionPlot(4).time)-solutionPlot(2).time,solutionPlot(2).state(:,2),'r')
    % plot(max(solutionPlot(4).time)-solutionPlot(3).time,solutionPlot(3).state(:,2),'b')
    % plot(max(solutionPlot(4).time)-solutionPlot(4).time,solutionPlot(4).state(:,2),'k')
    % xlabel('时间t/s')
    % ylabel('后轴中心纵坐标y/m')
    % hold off
    % 
    % figure
    % plot(max(solutionPlot(4).time)-solutionPlot(1).time,solutionPlot(1).state(:,3)*180/pi,'g')
    % hold on
    % plot(max(solutionPlot(4).time)-solutionPlot(2).time,solutionPlot(2).state(:,3)*180/pi,'r')
    % plot(max(solutionPlot(4).time)-solutionPlot(3).time,solutionPlot(3).state(:,3)*180/pi,'b')
    % plot(max(solutionPlot(4).time)-solutionPlot(4).time,solutionPlot(4).state(:,3)*180/pi,'k')
    % xlabel('时间t/s')
    % ylabel('航向角theta/°')
    % hold off
    % 
    % figure
    % plot(max(solutionPlot(4).time)-solutionPlot(1).time,-solutionPlot(1).control(:,1),'g')
    % hold on
    % plot(max(solutionPlot(4).time)-solutionPlot(2).time,-solutionPlot(2).control(:,1),'r')
    % plot(max(solutionPlot(4).time)-solutionPlot(3).time,-solutionPlot(3).control(:,1),'b')
    % plot(max(solutionPlot(4).time)-solutionPlot(4).time,-solutionPlot(4).control(:,1),'k')
    % xlabel('时间t/s')
    % ylabel('车速v/(m/s)')
    % hold off
    % 
    % figure
    % plot(max(solutionPlot(4).time)-solutionPlot(1).time,solutionPlot(1).control(:,2)*180/pi,'g')
    % hold on
    % plot(max(solutionPlot(4).time)-solutionPlot(2).time,solutionPlot(2).control(:,2)*180/pi,'r')
    % plot(max(solutionPlot(4).time)-solutionPlot(3).time,solutionPlot(3).control(:,2)*180/pi,'b')
    % plot(max(solutionPlot(4).time)-solutionPlot(4).time,solutionPlot(4).control(:,2)*180/pi,'k')
    % xlabel('时间t/s')
    % ylabel('轮胎转角delta/°')
    % hold off
end