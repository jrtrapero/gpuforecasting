%In this file we are going to analyze the results given by the m-file gpusimu2:
%We create 3 plots.

%Author: Juan R. Trapero - UCLM.
%Version: 1
%Date: 06/09/2018
%Comments: Just plot the results for publishing purposes

%% Load  the previous results
clearvars
%Establish the value na=???
na=100;
load(sprintf('tiempocpu%d.mat',na))
load(sprintf('tiempogpu%d.mat',na))


%% Article figures
%suma de tiempos
tiempocpu=squeeze(cputime);
tiempogpu=squeeze(naiveGPUTime);
%Figure 1
%Percentage that GPU is faster than CPU        
figure
subplot(3,1,1)
plot(m,tiempocpu(:,1)./tiempogpu(:,1),'-k'), title('Parallelization'), ylabel(' GPUx faster')
subplot(3,1,2)
plot(m,tiempocpu(:,2)./tiempogpu(:,2),'-k'), title('Optimization'), ylabel(' GPUx faster')
subplot(3,1,3)
plot(m,tiempocpu(:,3)./tiempogpu(:,3),'-k'), title('Forecasting'),
xlabel('sample size'), ylabel('GPUx faster')
print -depsc gpufastercpuna50


%Figure 2
%Analyze the evolution of the computational time versus sample size
%We can see what is the sample size where GPU is faster than CPU

figure
subplot(3,1,1)
loglog(m,cputime(:,1,1),'--k',m,naiveGPUTime(:,1,1),'-k','linewidth',1.2)
ylabel('Computational time'), 
title('Parallelization')
% legend('cpu','gpu','location','bestoutside'),

subplot(3,1,2)
loglog(m,cputime(:,1,2),'--k',m,naiveGPUTime(:,1,2),'-k','linewidth',1.2)
ylabel('Computational time'), 
title('Optimization')
% legend('cpu','gpu','location','bestoutside'),

subplot(3,1,3)
loglog(m,cputime(:,1,3),'--k',m,naiveGPUTime(:,1,3),'-k','linewidth',1.2)
ylabel('Computational time'), xlabel('Sample size')
title('Forecasting')
% legend('cpu','gpu','location','bestoutside'),


% Figure 3
%Do it for a determined sample size as an example.
figure
b=bar([tiempocpu(12,:)' tiempogpu(12,:)'],'FaceColor','flat');
b(1).CData(1:3,:)=0.2*ones(3,3);
b(2).CData(1:3,:)=0.8*ones(3,3);
xticklabels({'Parallelization','Optimization','Forecasting'})
ylabel('Computational time')
legend('CPU','GPU')
str=sprintf('Sample size=%d',m(12));
text(2.5,1,str)
%Sample size m(12)=4000 (length time series)

% str1=num2str(sum(tiempocpu(12,:)))
% str2=num2str(sum(tiempogpu(12,:)))
% text(2.5,0.8,str1)
% text(2.5,0.6,str2)




