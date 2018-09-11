%In this file we are going to explore the gpu performance for a single
%exponential smoothing forecasting technique:
%Optimization grid and search
%Divide the total sample in a windows size and optimize at every step

%Author: Juan R. Trapero - UCLM.
%Version: 2
%Date: 06/09/2018
%Comments: parallelize forecasting and sliding window approach
clearvars
%% Define variables
n=1; % number of time series
m=[(1e2:1e2:1e3), (2e3:2e3:1e4)]; % length of each time series
na=100; %Number of steps regarding the grid and search optimization algorithm
h=1; %Forecasting horizon
cputime=nan(length(m),length(n),3); %Variable that stores computational times

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CPU %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for l=1:length(m)
    for ll=1:length(n)
        %simulate gaussian random number, mean=50, std=1;
        demand=50+randn(m(l),n(ll),'single'); %Each column is a time series
        %% Parallelization (In CPU it's just prepare the data, nothing else!)
        t=tic();
        trainset=round(0.5*length(demand)); %50% for training set (hold-in sample)
%         traindata=nan(trainset,trainset-h); %Rest of data for test set (hold-out sample)
%         for i=1:trainset
%             traindata(:,i)=demand(i:trainset+i-1,1);
%         end
        traindata=hankel(demand(1:trainset),demand(trainset:end-h));
        cputime(l,ll,1)=toc(t); %Necessary time for the "parallelization"
        %% Optimization (Serial CPU)   
        t=tic();
        alphamin = gridandsearch(traindata,na,0);
        cputime(l,ll,2)=toc(t); %Required time for the optimization
        %% Forecast (Serial CPU)
        t=tic();
        f=nan(size(demand,1),1); %Variable that stores forecasts
        k=1;
        for j=trainset:size(demand,1)-h
            b=alphamin(k);
            a=[1 -(1-alphamin(k))];
            pred=filter(b,a,demand(j:end,1),demand(j,1),1);
            f(j+h,1)=pred(end);
            k=k+1;
        end
        cputime(l,ll,3)=toc(t); %Required time for the forecasting
        sprintf( 'Experimento %d (porcentaje)  tiempo CPU= %1.3fsecs',...
            l/length(m)*100,cputime(l,ll,2)) %show the times in the screen
        resultscpu=sprintf('tiempocpu%d.mat',na);
        save(resultscpu,'cputime','n','m','na') 
    end
end
clearvars
%% Define variables
%n=number of time series
%m= length of each time series
n=1;
m=[(1e2:1e2:1e3), (2e3:2e3:1e4)];
na=100; %Number of steps regarding the grid and search optimization algorithm
alphav=linspace(0,1,na);
h=1; %Forecasting horizon
naiveGPUTime=nan(length(m),length(n),3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GPU %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
for l=1:length(m)
    for ll=1:length(n)
        %Random number generation directly in GPU card
        demand=50+rand(m(l),n(ll),'single','gpuArray'); 
        %% Parallelization (GPGPU)
        t=tic();
        trainset=round(0.5*length(demand));
        traindata=hankel(demand(1:trainset),demand(trainset:end-h));
        naiveGPUTime(l,ll,1)=toc(t);
        %% Optimization (GPGPU)
        t=tic();
        mse=nan(length(alphav),length(demand(trainset:end-h)),'single','gpuArray');
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
            f=filter(b,a,traindata(:,samea),traindata(1,samea),1);
            pred=f(end,2:end)';
            end
        end
        forecast=gather(pred); %Send one-step-ahead forecasts from GPU to CPU
        naiveGPUTime(l,ll,3)=toc(t);
    end
end
resultsgpu=sprintf('tiempogpu%d.mat',na);
save(resultsgpu,'naiveGPUTime','n','m','na') 

%         sprintf( '%1.3fsecs (naive GPU) = %1.1fx faster',...
%             naiveGPUTime(l,ll,2), cputime(l,ll,2)/naiveGPUTime(l,ll,2) )
    %     figure
    %     plot([demand(2:end,3),forecast(1:end-1,3)])
    %     legend('real','forecast')
    %     title('GPU computing')

