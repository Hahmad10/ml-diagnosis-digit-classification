% Calculates the gradient of the regularized logistic regression cost function
% for the WDBC dataset.

function g = g_wdbc(w_hat, D, mu)
% Inputs:
%   w_hat: Augmented weight vector [w; b] ((N+1) x 1)
%   D:     Training data matrix [Xtr; ytr] ((N+1) x P)
%   mu:    Regularization parameter (scalar)
% Output:
%   g:     Gradient vector ((N+1) x 1)

    % Extract features (Xtr) and labels (ytr)
    Xtr = D(1:end-1, :); % N x P
    ytr = D(end, :);     % 1 x P

    [N, P] = size(Xtr); % N = features (30), P = samples (285)

    % Create augmented data matrix Xhat_tr ((N+1) x P)
    Xhat_tr = [Xtr; ones(1, P)];

    % Calculate the denominator term (1 x P)
    denominator = 1 + exp(ytr .* (w_hat' * Xhat_tr));

    % Calculate the numerator term, element-wise scaled by ytr ((N+1) x P)
    % MATLAB automatically handles broadcasting ytr (1xP) with Xhat_tr ((N+1)xP)
    numerator = ytr .* Xhat_tr;

    % Calculate the sum term ((N+1) x 1) by summing across samples (dim 2)
    sum_term = sum(numerator ./ denominator, 2);

    % Calculate the gradient
    g = mu * w_hat - (1/P) * sum_term;

end