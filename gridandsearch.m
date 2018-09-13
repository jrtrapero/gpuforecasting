function alphamin = gridandsearch(y,na)
%This function optimizes SES by using grid and search
% y: data
% na: number of steps
        dim=size(y,2);
        alphav=repmat(linspace(0,1,na)',1,dim);
        mse=nan(size(alphav,1),dim);
            for j=1:size(alphav,1)
                alpha=alphav(j);
                %Run exponential smoothing as a filter
                b=alpha;
                a=[1 -(1-alpha)];
                f=filter(b,a,y,y(1,:),1);
                %Compute forecast error
                mse(j,:)=mean((y(2:end,:)-f(1:end-1,:)).^2,1);              
            end
        %Select exponential smoothing constant that minimizes MSE
        [~,imin]=min(mse);
        alphamin=alphav(imin);
end

