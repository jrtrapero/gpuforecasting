%In this file we are going to explore the gpu performance for many time
%series. We simulate a retailer/manufacturer that has to forecast many skus
%that are assumed to be independent one of each other.
%exponential smoothing forecasting technique:
%Optimization grid and search
%First simulation in the working paper

%Author: Juan R. Trapero - UCLM.
%Version: 2
%Date: 19/09/2018
%Comments: parallelize forecasting
clearvars
%% Define variables
n=[1e2 1e3:1e3:1e4];  %Number of skus
m=1e2; %Number of observations per sku
na=100; %Number of steps regarding the grid and search optimization algorithm
h=1; %Forecasting horizon
save initialvar n m na h 
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CPU %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load initialvar
cputime=nan(length(m),length(n),3); %Variable that stores computational times
for l=1:length(m)
    for ll=1:length(n)
        %simulate gaussian random number, mean=50, std=1;
        demand=50+randn(m(l),n(ll),'single'); %Each column is a time series
        %% Parallelization (In CPU it's just prepare the data, nothing else!)
        t=tic();
        trainset=round(0.5*size(demand,1)); %50% for training set (hold-in sample)
        traindata=demand(1:trainset,1:n(ll));
        cputime(l,ll,1)=toc(t); %Necessary time for the "parallelization"
        %% Optimization (Serial CPU)   
        t=tic();
        alphamin = gridandsearch(traindata,na,0);
        cputime(l,ll,2)=toc(t); %Required time for the optimization
        %% Forecast (Serial CPU)
        t=tic();
        f=nan(size(demand,1),1); %Variable that stores forecasts
        k=1;
        for j=1:n(ll)
            b=alphamin(j);
            a=[1 -(1-alphamin(j))];
            f=filter(b,a,demand(trainset:end,j),demand(1,j),1); %Forecast
        end
        cputime(l,ll,3)=toc(t); %Required time for the forecasting
        sprintf( 'Experiment %d (percentage)  time CPU= %1.3fsecs',...
            (l*ll)/(length(m)*length(n))*100,cputime(l,ll,2)) %show times in the screen
        resultscpu=sprintf('simu1tiempocpu%d.mat',na);
        save(resultscpu,'cputime') 
    end
end
clearvars
%% 
load initialvar
alphav=linspace(0,1,na);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GPU %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
naiveGPUTime=nan(length(m),length(n),3);
for l=1:length(m)
    for ll=1:length(n)
        %Random number generation directly in GPU card
        demand=50+rand(m(l),n(ll),'single','gpuArray'); 
        %% Parallelization (GPGPU)
        t=tic();
        trainset=round(0.5*size(demand,1));
        traindata=demand(1:trainset,1:n(ll));
        naiveGPUTime(l,ll,1)=toc(t);
        %% Optimization (GPGPU)
        t=tic();
        mse=nan(length(alphav),size(demand,2),'single','gpuArray');
        for j=1:length(alphav)
            alpha=alphav(j);
            %Run exponential smoothing as a filter
            b=alpha;
            a=[1 -(1-alpha)];
            f=filter(b,a,traindata,traindata(1,:),1);
            %Calculate Mean Squared Error (MSE)
            mse(j,:)=mean((traindata(10:end,:)-f(10-h:trainset-h,:)).^2,1);
        end     
        [~,imin]=min(mse); %Find indices with minimum MSE
        alphamin=alphav(imin); %Find corresponding alpha for those indices
        naiveGPUTime(l,ll,2)=toc(t);       
        %% Forecast (GPGPU)
        t=tic();
        k=1;
        %This loop searches for the same alpha values to pass it to the
        %command filter at once
        for j=1:length(alphav)
            samea=find(alphamin==alphav(j));
            if ~isempty(samea)
            b=alphav(j);
            a=[1 -(1-alphav(j))];
            f=filter(b,a,demand(trainset:end-h,samea),demand(trainset,samea),1);
            pred=f(end,:)';
            end
        end
        forecast=gather(pred); %Send one-step-ahead forecasts from GPU to CPU
        naiveGPUTime(l,ll,3)=toc(t);
    end
end
resultsgpu=sprintf('simu1tiempogpu%d.mat',na);
save(resultsgpu,'naiveGPUTime') 


