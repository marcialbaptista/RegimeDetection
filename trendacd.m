function trend=trendacd(x0)
%TRENDACD estimates the monotonic component of the data.
%   TRENDACD(X0) estimates the monotonic component of the time series X0 as
%   a succession of straight segments using an algorithm presented in
%   Chapter 5 of the book [ATE] C. Vamos and M. Craciun, Automatic Trend
%   Estimation, Springer 2012.

N=length(x0);
Kf=round(0.1*N); if Kf<1 Kf=1; end %the maximum semilength of the averaging window (p. 68 [ATE])

% If the time series has less than 5 values, then the trend is replaced by the
% mean of the time series
if N<5
    trend=mean(x0)*ones(1,N); return; end

% We estimate the standard deviation of the noise supperposed over a
% monotonic trend using the algorithm presented in Appendix D of [ATE].
NN=floor(N/2); norm_diffx=zeros(1,NN+1);
r=1; d0=1;
norm_diffx(1)=sum((x0(2:N)-x0(1:N-1)).^2);
while r>0 && d0<NN
    norm_diffx(d0+1)=sum((x0(d0+2:N)-x0(1:N-d0-1)).^2);
    r=norm_diffx(d0+1)-norm_diffx(d0);
    d0=d0+1;
end
% d0 is determined from Eq. (D5) in Appendix D of [ATE].
if d0<NN
    sig_est=std(x0(d0:N)-x0(1:N-d0+1))/1.41;  % Eq. (D6)
else
    % when there is no d0 satisfying condition (D5)
    dx0=x0(2:N)-x0(1:N-1);
    r=1; d0=1;
    dnorm_diffx(1)=norm(dx0(2:N-1)-dx0(1:N-2),2)^2;
    while r>0 && d0<NN
        dnorm_diffx(d0+1)=sum((dx0(d0+2:N-1)-dx0(1:N-d0-2)).^2);
        r=dnorm_diffx(d0+1)-dnorm_diffx(d0);
        d0=d0+1;
    end
    sig_est=std(x0(d0:N)-x0(1:N-d0+1))/1.41;  % Eq. (D6)
end

% we determine the number of disjoint intervals of the time series values
Jmin=2; %the minimum number of intervals of time series values
Nmin=14; %the minimum number of values in an interval of time series values
Jmax=round(N/Nmin); if Jmax<Jmin Jmax=Jmin; end
J=round((max(x0)-min(x0))/sig_est); % Eq. (5.5) in [ATE]
if J<Jmin J=Jmin; end
if J>Jmax J=Jmax; end

%memory preallocation
Nj=zeros(1,J); % number N_j of time series values in the interval I_j
sumNj=zeros(1,J);
csi=zeros(1,J+1); % the boundaries of the intervals I_j
g=zeros(1,J); % average displacement on the interval I_j given by Eq. (5.1) in [ATE]
nrsum=zeros(1,J); % the numitor in Eq. (5.1) in [ATE]
gmin=zeros(1,J); % the minimum of the estimated trend slope
t_j=zeros(1,J+1); % the boundaries t_j of the time intervals ([ATE] pp. 62 and 63)
fest=zeros(1,J+1); % the estimated trend given by Eq. (5.2)
xrang=zeros(1,N);
yscal=zeros(1,N);


% the iterations of smoothings and trend extractions described in Sect. 5.1 of [ATE]
normmin=sig_est/sqrt(N); % the minimum quadratic norm of the residuals for
                         % which the iterations are stoped
normx_i(1)=std(x0); 
imed=0; % initialization of the number of smoothings
itrend=0; % initialization of the number of trend extractions
x_i=x0; % order zero residuals x^{(0)} of the iterations
trend=zeros(1,N); % order zero trend f^{(0)} of the iterations
flag=1; % the iterations go on until flag=0
while flag==1
    
% determination of the boundaries and the number of values of the intervals I_j
    [xord,kxord]=sort(x_i);
    xmin=xord(1); xmax=xord(N);
    ddx=(xmax-xmin)/N;
    r=N/J; sums=0;
    for j=1:J
        Nj(j)=round(j*r)-sums;
        sums=sums+Nj(j);
        sumNj(j)=sums; end
    csi(1)=xord(1)-ddx; csi(J+1)=xord(N)+ddx;
    csi(2:J)=(xord(sumNj(1:J-1))+xord(sumNj(1:J-1)+1))/2;
    index1=1; index2=0;
    for j=1:J
        index1=index2+1; index2=index2+Nj(j);
        xrang(kxord(index1:index2))=j;
    end
    % j=xrang(n) is the interval I_j which contains the value x_i(n)
    
% computation of the average slope g_j of the interval I_j given by Eq. (5.1) in [ATE]
    dx=x_i(2:N)-x_i(1:N-1);
    nrsum=2*Nj(1:J); % the numitor in Eq. (5.1) in [ATE]
    nrsum(xrang(1))=nrsum(xrang(1))-1; % x_i(1) enters only in a single term in the sum in Eq. (5.1)
    nrsum(xrang(N))=nrsum(xrang(N))-1; % x_i(N) enters only in a single term in the sum in Eq. (5.1)
    sumdx=zeros(max(Nj(1:J))+1,J); % the terms of the sum in Eq. (5.1)
    kN=find(kxord(:)==N);
    if kN>1 dx_reduced(1:kN-1)=dx(kxord(1:kN-1)); end
    if kN<N dx_reduced(kN:N-1)=dx(kxord(kN+1:N)); end
    index1=1; index2=0;
    for j=1:J
        index1=index2+1; index2=index2+Nj(j);
        if j==xrang(N) index2=index2-1; end
        sumdx(1:index2-index1+1,j)=dx_reduced(index1:index2)';
    end
    k1=find(kxord(:)==1);
    if k1>1 dx_reduced(1:k1-1)=dx(kxord(1:k1-1)-1); end
    if k1<N dx_reduced(k1:N-1)=dx(kxord(k1+1:N)-1); end
    index1=1; index2=0;
    for j=1:J
        index1=index2+1; index2=index2+Nj(j);
        if j==xrang(1) index2=index2-1; end
        sumdx(1:index2-index1+1,j)=sumdx(1:index2-index1+1,j)+dx_reduced(index1:index2)';
    end
    g=sum(sumdx)./nrsum; % Eq. (5.1) in [ATE]
    
    sumsign=sum(sign(g));
    if abs(sumsign)<J
% if the slopes g_j do not have the same signs, then the iteration applies the
% moving average given by Eqs. (4.2)-(4.4)
        imed=imed+1;
        K=imed; if imed>Kf K=Kf; end
        T=2*K+1;
        tablou=zeros(T,N); numitor=zeros(1,N);
        numitor(1:K)=[K+1:2*K];
        numitor(N-K+1:N)=[2*K:-1:K+1];
        numitor(K+1:N-K)=T*ones(1,N-2*K);
        for t=1:K+1 tablou(t,1:N+t-K-1)=x_i(K+2-t:N); end
        for t=1:K tablou(K+1+t,t+1:N)=x_i(1:N-t); end
        x_i=sum(tablou)./numitor;
    else
% if the slopes g_j have the same signs, then a trend component is removed
% in accordance with Eq. (5.3)
        sensmed=1; if sumsign<J sensmed=-1; end
        
        % the condition that g_j should not be smaller than gmin ([ATE] p. 64)
        for j=1:J
            indecs=find(xrang(:)==j);
            gmin(j)=sensmed*(csi(j+1)-csi(j))/(max(indecs)-min(indecs));
            if abs(g(j))<abs(gmin(j))
                g(j)=gmin(j); end
        end

        % determination of the moments limiting the straight segments of
        % the trend components used in Eq. (5.2)
        t_j(1)=0;
        if sensmed==1
            fest(1)=xmin; fest(2:J)=csi(2:J); fest(J+1)=xmax;
            for j=2:J+1
                t_j(j)=t_j(j-1)+(fest(j)-fest(j-1))/g(j-1); end
        elseif sensmed==-1
            fest(1)=xmax; fest(2:J)=csi(J:-1:2); fest(J+1)=xmin;
            for j=2:J+1
                t_j(j)=t_j(j-1)+(fest(j)-fest(j-1))/g(J-j+2); end
        end
        T=t_j(J+1);
        
        options = optimset('Display','off');
        if T>=N
            % optimum translation of the sampling moments of the time
            % series ([ATE] p. 63)
            taumin=fminsearch(@(tau)transl_f_i(tau,x_i,fest,t_j,J),(T-N+1)/2,options);
            [fmintrans,f_i]=transl_f_i(taumin,x_i,fest,t_j,J);
        else
            % optimum translation of the estimated trend  ([ATE] p. 63)
            taumin=fminsearch(@(tau)transl_x_i(tau,x_i,fest,t_j,J),(N-T+1)/2,options);
            [fmintrans,f_i]=transl_x_i(taumin,x_i,fest,t_j,J);
        end
        
        % scaling of the estimated trend ([ATE] p. 64)
        j=1;
        for n=1:N
            r=(n-1)*T/(N-1);
            while r>t_j(j+1) && j<J j=j+1; end
            yscal(n)=fest(j)+(r-t_j(j))*(fest(j+1)-fest(j))/(t_j(j+1)-t_j(j));
        end
        if norm(x_i(1:N)-yscal(1:N))<fmintrans f_i(1:N)=yscal(1:N); end

        trendtemp=trend+f_i;
        d2trend=(trendtemp(3:N)-trendtemp(2:N-1)).*(trendtemp(2:N-1)-trendtemp(1:N-2));
        if length(find(d2trend<0))>0
            flag=0; % the iteration is interrupted when the estimated trend 
                    % becomes nonmonotonic ([ATE] p. 65)
        else
            trend=trendtemp;
            x_i=x_i-f_i;
            itrend=itrend+1;
        end
    end

    rnorm=std(x_i);
    if rnorm<normmin || rnorm>normx_i(itrend+imed)
        if itrend==0 trend=mean(x0)*ones(1,N); end
        flag=0;% the iteration is interrupted when the standard deviation of the residuals
               % is smaller than normmin or it is larger than at the
               % previous iteration ([ATE] pp. 64 and 65)
    end
    normx_i(itrend+imed+1)=rnorm;
end

end

% quadratic norm of the difference between the estimated trend component
% f_i and the residuals x translated through the distance tau ([ATE] p. 63)
function [error,f_i] = transl_f_i(tau,x,fest,t_j,J)
    N=length(x); T=t_j(J+1);
    if tau>T-N+1 tau=T-N+1;
    elseif tau<0 tau=0;
    end
    j=1;
    for n=1:N
        while tau+n-1>t_j(j+1) && j<J
            j=j+1; end
        f_i(n)=fest(j)+(tau+n-1-t_j(j))*(fest(j+1)-fest(j))/(t_j(j+1)-t_j(j));
    end
    error=norm(x-f_i);
end

% quadratic norm of the difference between the estimated trend component
% f_i translated through the distance tau and the residuals x ([ATE] p. 63)
function [error,f_i] = transl_x_i(tau,x,fest,t_j,J)
    N=length(x); T=t_j(J+1);
    if tau>N-1-T tau=N-1-T;
    elseif tau<0 tau=0;
    end
    t1=ceil(tau)+1; t2=floor(tau+T)+1;
    j=1;
    for t=t1:t2
        while t-1-tau>t_j(j+1) && j<J
            j=j+1; end
        f_i(t)=fest(j)+(t-1-tau-t_j(j))*(fest(j+1)-fest(j))/(t_j(j+1)-t_j(j));
    end
    if t1>1 f_i(1:t1-1)=fest(1)*ones(1,t1-1); end
    if t2<N f_i(t2+1:N)=fest(J+1)*ones(1,N-t2); end
    error=norm(x-f_i);
end