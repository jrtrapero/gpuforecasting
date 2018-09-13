%In this file we are going to analyze the results given by the m-file gpusimu2:
%We create 3 plots.

%Author: Juan R. Trapero - UCLM.
%Version: 1
%Date: 06/09/2018
%Comments: Just plot the results for publishing purposes

%% Load  the previous results
clearvars
load initialvar
load(sprintf('tiempocpu%d.mat',na))
load(sprintf('tiempogpu%d.mat',na))


%% Article figures
% Reduce in one dimension that is redundant and 
% calculate the sum of times for CPU and GPU
tiempocpu=squeeze(cputime);
tiempogpu=squeeze(naiveGPUTime);
sumcpu=sum(tiempocpu,2);
sumgpu=sum(tiempogpu,2);

%Figure 1
%computational time for cpu and gpu versus sample size
%Percentage that GPU is faster than CPU        
figure
plot(m,sumcpu,'--k',m,sumgpu,'-k'), 
xlabel('Length of time series (m)'), ylabel('Total computational time')
legend('CPU','GPU','location','NorthWest')
print -depsc simu2_na_100




%Figure 2
%Percentage that GPU is faster than CPU        
figure
subplot(3,2,1)
plot(m,tiempocpu(:,1)./tiempogpu(:,1),'-k'), title('Parallelization'), ylabel(' GPUx faster')
grid
subplot(3,2,3)
plot(m,tiempocpu(:,2)./tiempogpu(:,2),'-k'), title('Optimization'), ylabel(' GPUx faster')
grid
subplot(3,2,5)
plot(m,tiempocpu(:,3)./tiempogpu(:,3),'-k'), title('Forecasting'),
grid
xlabel('Length of time series (m)'), ylabel('GPUx faster')
% print -depsc gpufastercpuna50



%Analyze the evolution of the computational time versus sample size
%We can see what is the sample size where GPU is faster than CPU

% figure
subplot(3,2,2)
loglog(m,cputime(:,1,1),'--k',m,naiveGPUTime(:,1,1),'-k','linewidth',1.2)
grid
ylabel('Computational time'), 
title('Parallelization')
% legend('cpu','gpu','location','bestoutside'),

subplot(3,2,4)
loglog(m,cputime(:,1,2),'--k',m,naiveGPUTime(:,1,2),'-k','linewidth',1.2)
grid
ylabel('Computational time'), 
title('Optimization')
% legend('cpu','gpu','location','bestoutside'),

subplot(3,2,6)
loglog(m,cputime(:,1,3),'--k',m,naiveGPUTime(:,1,3),'-k','linewidth',1.2)
grid
ylabel('Computational time'), xlabel('Length of time series (m)')
title('Forecasting')
% legend('cpu','gpu','location','bestoutside'),
print -depsc simu22_na_100


% Figure 3
%Do it for a determined sample size as an example.
figure
b=bar([tiempocpu(12,:)' tiempogpu(12,:)'],'FaceColor','flat');
b(1).CData(1:3,:)=0.2*ones(3,3);
b(2).CData(1:3,:)=0.8*ones(3,3);
xticklabels({'Parallelization','Optimization','Forecasting'})
ylabel('Computational time')
legend('CPU','GPU')
str=sprintf('m=%d',m(12));
text(3,2.4,str)
print -depsc simu222_na_100




