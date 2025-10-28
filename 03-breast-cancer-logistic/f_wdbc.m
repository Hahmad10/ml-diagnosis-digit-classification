% Calculates the regularized logistic regression cost function 
% for the WDBC dataset.

function f = f_wdbc(w_hat, D, mu)
% Inputs:
%   w_hat: Augmented weight vector [w; b] ((N+1) x 1)
%   D:     Training data matrix [Xtr; ytr] ((N+1) x P)
%   mu:    Regularization parameter (scalar)
% Output:
%   f:     Value of the cost function (scalar)

    % Extract features (Xtr) and labels (ytr)
    Xtr = D(1:end-1, :); % N x P
    ytr = D(end, :);     % 1 x P

    [N, P] = size(Xtr); % N = features (30), P = samples (285)

    % Create augmented data matrix Xhat_tr ((N+1) x P)
    Xhat_tr = [Xtr; ones(1, P)];

    % Calculate the logistic loss term
    log_term = log(1 + exp(-ytr .* (w_hat' * Xhat_tr))); % 1 x P vector
    loss = (1/P) * sum(log_term);

    % Calculate the regularization term
    % Note: ||w_hat||_2^2 is w_hat' * w_hat
    regularization = (mu / 2) * (w_hat' * w_hat);

    % Total cost
    f = loss + regularization;

end