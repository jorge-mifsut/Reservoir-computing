clear all

%%% CHAOTIC TIME SERIES PREDICTION %%


% Data
x_train = csvread('training-set.csv');
x_test = csvread('test-set-7.csv');
T_train = length(x_train(1,:));
T_test = length(x_test(1,:));
N = 500;
n = 3;
k = 0.01;

% Network
w_in = randn(N,n) .* sqrt(0.002);
w = randn(N,N) .* sqrt(2/N);
R = zeros(N,T_train);
r = zeros(N,1);


for t = 1:T_train
    
    R(:,t) = r;
    r = tanh(w*r + w_in*x_train(:,t));
 
end

w_out = x_train * R' * inv(R*R' + k*eye(N));


% Prediction

r2 = zeros(N,1);
R2 = zeros(N,T_test);

for t = 1:T_test
    
    R2(:,t) = r2;
    r2 = tanh(w*r2 + w_in*x_test(:,t));
 
end

r2 = R2(:,T_test);
output = zeros(N,1);
x = zeros(N,1);
z = zeros(N,1);
for t = 1:N
    
    out = w_out*r2;
    x(t) = out(1);
    output(t) = out(2);         % y component
    z(t) = out(3);
    r2 = tanh(w*r2 + w_in*out);
end

plot3(x,output,z,'k')


csvwrite('prediction.csv',output);




