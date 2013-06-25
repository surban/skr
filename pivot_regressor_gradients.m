function [dLdW, dLdP] = pivot_regressor_gradients(W, P, data, targets)

K = kernel_matrix(P, data);

%% optimize weights
dLdW = -(targets - W*K) * K.';

%% optimize pivot points
% calulate gradient wrt pivot points
%L = norm(targets - W * K, 'fro')^2;
dLdK = - 2 * W.' * (targets - W*K);
dLdP = kernel_gradient_wrt_p(K, P, data, dLdK);
    
end