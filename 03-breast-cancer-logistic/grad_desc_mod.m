% Gradient descent that runs exactly K iterations with a backtracking line
% search, passing extra data arguments (D, mu) to the cost and gradient.

function [xs, fs, k] = grad_desc_mod(fname, gname, x0, epsi, D, mu, K)
% Inputs:
%   fname: Name of the cost function M-file (string)
%   gname: Name of the gradient function M-file (string)
%   x0:    Initial point (column vector)
%   epsi:  Tolerance (not used for termination, but may be needed by line search)
%   D:     Training data matrix [Xtr; ytr] for fname/gname
%   mu:    Regularization parameter for fname/gname
%   K:     Number of iterations to perform
% Outputs:
%   xs:    Solution point after K iterations
%   fs:    Objective function value at xs
%   k:     Number of iterations performed (=K)

    format compact;
    format long;

    xk = x0; % Current point

    for k = 1:K % Loop exactly K times
        gk = feval(gname, xk, D, mu); % Calculate gradient at current point
        dk = -gk; % Search direction

        % Perform line search (needs D, mu passed implicitly via function handles)
        ak = bt_lsearch2019(xk, dk, fname, gname, D, mu); % Pass D and mu

        % Update the point
        xk = xk + ak * dk;


    end % end loop

    xs = xk; % The solution is the point after K iterations

    disp('Solution after K iterations:')
    xs
    disp('Objective function at solution point:')
    fs = feval(fname, xs, D, mu) % Calculate final cost
    format short
    disp('Number of iterations performed:')
    k % This will be K

end