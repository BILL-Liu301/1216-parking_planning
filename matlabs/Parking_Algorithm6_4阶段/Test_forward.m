close all

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
CONSTANTS.threshold= 0.2;  %阈值为0.2m

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


time=[solution(1).time;solution(2).time;solution(3).time;solution(4).time];
u1=[solution(1).control(:,1);solution(2).control(:,1);solution(3).control(:,1);solution(4).control(:,1)]; 
u2=[solution(1).control(:,2);solution(2).control(:,2);solution(3).control(:,2);solution(4).control(:,2)]; 
% x1(1)=solution(1).state(1,1);
% x2(1)=solution(1).state(1,2);
% x3(1)=solution(1).state(1,3);
% 
% for i=1:1:length(time)-1
%     x1(i+1)=x1(i)+u1(i)*cos(x3(i))*(time(i+1)-time(i));
%     x2(i+1)=x2(i)+u1(i)*sin(x3(i))*(time(i+1)-time(i));
%     x3(i+1)=x3(i)+u1(i)*tan(u2(i))/Hr_Car*(time(i+1)-time(i));   
% end

n=length(solution(4).state(:,1));
x1(1)=solution(4).state(n,1);
x2(1)=solution(4).state(n,2);
x3(1)=solution(4).state(n,3);
t(1)=time(length(time));
j=1;
for i=length(time):-1:2
    x3(j+1)=x3(j)-u1(i-1)*tan(u2(i-1))/Hr_Car*(time(i)-time(i-1));   
    x1(j+1)=x1(j)-u1(i-1)*cos(x3(j+1))*(time(i)-time(i-1));
    x2(j+1)=x2(j)-u1(i-1)*sin(x3(j+1))*(time(i)-time(i-1));
    t(j+1)=time(i-1);
    j=j+1;
end

figure
plot(max(solutionPlot(4).time)-solutionPlot(1).time,solutionPlot(1).state(:,1),'g')
hold on
plot(max(solutionPlot(4).time)-solutionPlot(2).time,solutionPlot(2).state(:,1),'r')
plot(max(solutionPlot(4).time)-solutionPlot(3).time,solutionPlot(3).state(:,1),'b')
plot(max(solutionPlot(4).time)-solutionPlot(4).time,solutionPlot(4).state(:,1),'k')
% plot(max(time)-time,x1,'k--')
plot(max(t)-t,x1,'k--')
xlabel('时间t/s')
ylabel('后轴中心横坐标x/m')
hold off

figure
plot(max(solutionPlot(4).time)-solutionPlot(1).time,solutionPlot(1).state(:,2),'g')
hold on
plot(max(solutionPlot(4).time)-solutionPlot(2).time,solutionPlot(2).state(:,2),'r')
plot(max(solutionPlot(4).time)-solutionPlot(3).time,solutionPlot(3).state(:,2),'b')
plot(max(solutionPlot(4).time)-solutionPlot(4).time,solutionPlot(4).state(:,2),'k')
% plot(max(time)-time,x2,'k--')
plot(max(t)-t,x2,'k--')
xlabel('时间t/s')
ylabel('后轴中心纵坐标y/m')
hold off

figure
plot(max(solutionPlot(4).time)-solutionPlot(1).time,solutionPlot(1).state(:,3)*180/pi,'g')
hold on
plot(max(solutionPlot(4).time)-solutionPlot(2).time,solutionPlot(2).state(:,3)*180/pi,'r')
plot(max(solutionPlot(4).time)-solutionPlot(3).time,solutionPlot(3).state(:,3)*180/pi,'b')
plot(max(solutionPlot(4).time)-solutionPlot(4).time,solutionPlot(4).state(:,3)*180/pi,'b')
% plot(max(time)-time,x3*180/pi,'k--')
plot(max(t)-t,x3*180/pi,'k--')
xlabel('时间t/s')
ylabel('航向角theta/°')
hold off


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

for i=1:1:length(time)
Car_plot(x1(i),x2(i),x3(i),'b')
end
hold off

