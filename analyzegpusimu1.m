%In this file we are going to analyze the results given by the m-file gpusimu1:
%We create 3 plots.

%Author: Juan R. Trapero - UCLM.
%Version: 1
%Date: 13/09/2018
%Comments: Just plot the results for publishing purposes

%% Load  the previous results
clearvars
load initialvar
load(sprintf('simu1tiempocpu%d.mat',na))
load(sprintf('simu1tiempogpu%d.mat',na))


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
plot(n,sumcpu,'--k',n,sumgpu,'-k'), 
title('length of each time series (m=100)'), 
xlabel('Nº of time series (n)'), ylabel('Total computational time')
legend('CPU','GPU','location','NorthWest')
print -depsc simu1_na_100


%Figure 2
%Analyze the evolution of the computational time versus sample size
%We can see what is the sample size where GPU is faster than CPU

figure
subplot(3,1,1)
plot(n,tiempocpu(:,1),'--k',n,tiempogpu(:,1),'-k','linewidth',1.2)
ylabel('Computational time'), 
title('Parallelization')
% legend('cpu','gpu','location','bestoutside'),

subplot(3,1,2)
plot(n,tiempocpu(:,2),'--k',n,tiempogpu(:,2),'-k','linewidth',1.2)
ylabel('Computational time'), 
title('Optimization')
% legend('cpu','gpu','location','bestoutside'),

subplot(3,1,3)
plot(n,tiempocpu(:,3),'--k',n,tiempogpu(:,3),'-k','linewidth',1.2)
ylabel('Computational time'), xlabel('Nº of time series (n)')
title('Forecasting')
print -depsc simu11_na_100

%Figure 3
%Do it for a determined sample size as an example.
figure
b=bar([tiempocpu(9,:)' tiempogpu(9,:)'],'FaceColor','flat');
b(1).CData(1:3,:)=0.2*ones(3,3);
b(2).CData(1:3,:)=0.8*ones(3,3);
xticklabels({'Parallelization','Optimization','Forecasting'})
ylabel('Computational time')
legend('CPU','GPU')
str=sprintf('n=%d',n(9));
text(3,0.28,str)
print -depsc simu111_na_100




