

N = 100; 
lambda1 = 80*0.03;
lambda2=20;
mu1 = 0.03;
mu2=0.01;
H= 1000;
q1=zeros(N,N,H);
q2=zeros(N,N,H);
q1Lb=zeros(N,H);
q2Lb=zeros(N,H);
q1int=zeros(N,N);
q2int=zeros(N,N);
qtotint=zeros(N,N);



for s10=1:100
    for s20=1:N-s10
            for t=1:H
                q2Lb(s20,t) = s20*exp(-(mu2)*t);
                q1(s10,s20,t) = min(s10*exp(-(mu1)*t) + (lambda1/mu1)*(1-exp(-(mu1)*t)),N-q2Lb(s20,t));
                q2(s10,s20,t) = max(q2Lb(s20,t),min(s20*exp(-(mu2)*t) + (lambda2/mu2)*(1-exp(-(mu2)*t)),N-q1(s10,s20,t)));
            end
    end
end




%compute integral

for s10=1:N
    for s20=1:N-s10
            for t=1:H
                q1int(s10,s20) = q1int(s10,s20)+ q1(s10,s20,t)-lambda1/mu1;
               q2int(s10,s20) = q2int(s10,s20)+ q2(s10,s20,t)-min((N-lambda1/mu1),lambda2/mu2);
            end
             qtotint(s10,s20) = q1int(s10,s20)+q2int(s10,s20);
    end
end

q2int(80,20)

x = 1:N;
y = 1:N; 
[X, Y] = meshgrid(x, y); % Create the grid

% Plot the surface
surf(X, Y, qtotint);
hold off;



