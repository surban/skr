global kernel_l;

%% parameters
rate_fac = 1.1;
kernel_l = 2;

%% load data
load('boston.mat');
data = boston.data';
targets = boston.target';

% data=data(:,1:10);
% targets=targets(:,1:10);

% normalize data
maxdata = max(data,[],2);
maxdata(maxdata == 0) = 1;
data = data ./ repmat(maxdata, 1, size(data,2));

% add bias
data = [data; repmat(1, 1, size(data,2))];

%% initialize variables 
n_samples = size(data,2);
n_pivs = floor(n_samples / 5);
n_features = size(data,1);
n_targets = size(targets,1);
n_hidden = n_features;

%% initialize weights and pivots
W = rand(n_targets, n_pivs) - 0.5;
V = rand(n_hidden, n_features) - 0.5;
P = rand(n_hidden, n_pivs) - 0.5;

%% iterate
n_iters = 1000000;
last_obj = 100000000;
rate = 0.001;
rate_hold = 1;

fprintf('Start:\n');
for iter=1:n_iters

    last_W = W;
    last_V = V;
    last_P = P;
    
    %% calculate hiddens H
    H = sigmoid(V * data);   
    dsigmoid = (1-H) .* H;
    
    %% train W and P
    [dLdW, dLdP] = pivot_regressor_gradients(W, P, H, targets);      

    %continue;
    
    %% train V
    K = kernel_matrix(P, H);    
    dLdK = - 2 * W.' * (targets - W*K);
    dLdH = kernel_gradient_wrt_x(K, P, H, dLdK);    
    dLdV = (dLdH .* dsigmoid) * data.';

    %% update parameters
    W = W - (rate/n_samples) * dLdW;
    P = P - (rate/n_samples) * dLdP;
    V = V - (rate/n_samples) * dLdV;   
    
    %% calculate performance
    H = sigmoid(V * data);   
    K = kernel_matrix(P, H);       
    tp = W * K;
    td = tp - targets;
    tar_err = norm(td, 'fro');
    
    fprintf('iter=%05d rate=%.5g tar_err=%.5f\n', ...
        iter, rate, tar_err);
   
    %% adjust learning rate
    obj = tar_err;   
    if obj > last_obj
        rate = rate / rate_fac;
        rate_hold = 100;
        
        % revert
        W = last_W;
        P = last_P;
        V = last_V;
        obj = last_obj;
    elseif rate_hold < 0
        rate = rate * rate_fac;        
    end
    rate_hold = rate_hold - 1;    
    last_obj = obj;          
end

%% final error
fprintf('Resulting error with %d hidden units and %d pivots: %g\n', ...
    n_hidden, n_pivs, tar_err);


