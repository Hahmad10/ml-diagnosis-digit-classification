% Driver script: train regularized logistic regression on WDBC with gradient
% descent and evaluate on held-out patients.

clear; clc; close all;
format compact;

logFileName = 'wdbc_output.txt';
if exist(logFileName, 'file')
    delete(logFileName);
end
diary(logFileName); % Start saving Command Window output to the specified file

%% Load data
disp('--- Loading Data ---');
load D_wdbc.mat; % Loads the D_wdbc matrix (31x569)

%% Prepare and normalize data
disp('--- Preparing & Normalizing Data ---');

% (i) Get training and test data sets
Dtr = D_wdbc(:, 1:285); % 31x285
Dte = D_wdbc(:, 286:569); % 31x284

% (ii) Normalize the Data Sets
Xtr_raw = Dtr(1:30, :); % Raw training features (30x285)
[N, P_tr] = size(Xtr_raw); % N=30 features, P_tr=285 training samples

Xtr = zeros(N, P_tr); % To store normalized training features
m = zeros(1, N);      % To store means
v = zeros(1, N);      % To store standard deviations (sqrt of variance)

for i = 1:N
    xi = Xtr_raw(i, :);
    m(i) = mean(xi);
    % Use std which calculates sqrt(var(xi, 0, 2)) using N-1 denominator
    % var uses N-1 by default, sqrt(var(xi)) is equivalent to std(xi).
    v(i) = std(xi);
    % Handle cases where standard deviation is zero (avoid division by zero)
    if v(i) == 0
        Xtr(i, :) = xi - m(i); % Just center if variance is zero
    else
        Xtr(i, :) = (xi - m(i)) / v(i);
    end
end

% Normalize test data using mean (m) and std dev (v) from training data
Xte_raw = Dte(1:30, :); % Raw test features (30x284)
[~, P_te] = size(Xte_raw); % P_te = 284 test samples
Xte = zeros(N, P_te);     % To store normalized test features

for i = 1:N
    xi = Xte_raw(i, :);
    if v(i) == 0
        Xte(i, :) = xi - m(i); % Just center
    else
        Xte(i, :) = (xi - m(i)) / v(i);
    end
end

% Extract labels
ytr = Dtr(31, :); % 1x285
yte = Dte(31, :); % 1x284

% Combined data matrix D, as expected by f_wdbc / g_wdbc
D_train = [Xtr; ytr]; % (N+1) x P_tr = 31x285

% Create augmented data matrices for calculations
Xhat_tr = [Xtr; ones(1, P_tr)]; % (N+1) x P_tr
Xhat_te = [Xte; ones(1, P_te)]; % (N+1) x P_te

disp('Data normalization complete.');
fprintf('Training set size: %d features, %d samples\n', N, P_tr);
fprintf('Test set size:     %d features, %d samples\n', N, P_te);

%% Required function files
% Make sure f_wdbc.m and g_wdbc.m are in the path or current directory.

%% Required gradient descent files
% Make sure grad_desc_mod.m and bt_lsearch2019.m are in the path or current directory.

%% Run gradient descent for different settings
disp('--- Running Gradient Descent ---');

% Settings to compare (mu, K)
settings = {
    struct('mu', 0,     'K', 10);
    struct('mu', 0.1,   'K', 10);
    struct('mu', 0,     'K', 30);
    struct('mu', 0.075, 'K', 30)
};

num_settings = length(settings);
w_hat_star_list = cell(num_settings, 1); % Store results

% Initial point
w0 = zeros(N + 1, 1);
epsi = 1e-9; % Tolerance (not used for termination, but maybe by line search)

for i = 1:num_settings
    mu = settings{i}.mu;
    K = settings{i}.K;
    fprintf('Running setting %d: mu = %.3f, K = %d\n', i, mu, K);

    [w_hat_star, ~, ~] = grad_desc_mod('f_wdbc', 'g_wdbc', w0, epsi, D_train, mu, K);
    w_hat_star_list{i} = w_hat_star;

    fprintf('Gradient descent finished for setting %d.\n', i);
end

%% Evaluate performance on test data
disp('--- Evaluating Performance on Test Data ---');

for i = 1:num_settings
    mu = settings{i}.mu;
    K = settings{i}.K;
    w_hat_star = w_hat_star_list{i};

    fprintf('\n--- Results for Setting %d (mu = %.3f, K = %d) ---\n', i, mu, K);

    % Predict on test data
    pred_vals_te = w_hat_star' * Xhat_te; % 1 x P_te
    pred_labels_te = sign(pred_vals_te); % Predicted labels (-1 or 1)

    % Calculate Confusion Matrix elements
    % A: True Positives (Actual=1, Predicted=1)
    A = sum(pred_labels_te == 1 & yte == 1);
    % B: False Positives (Actual=-1, Predicted=1)
    B = sum(pred_labels_te == 1 & yte == -1);
    % C: False Negatives (Actual=1, Predicted=-1)
    C = sum(pred_labels_te == -1 & yte == 1);
    % D: True Negatives (Actual=-1, Predicted=-1)
    D = sum(pred_labels_te == -1 & yte == -1);

    % Construct confusion matrix (Rows: Predicted, Cols: Actual)
    %      Actual P (+1)  Actual N (-1)
    % Pred P (+1)   A              B
    % Pred N (-1)   C              D
    conf_mat = [A, B; C, D];

    fprintf('Confusion Matrix (Test Set):\n');
    disp(conf_mat);

    % Calculate Accuracy
    accuracy = 100 * (A + D) / (A + B + C + D);
    fprintf('Accuracy: %.2f%%\n', accuracy);
    fprintf('Number of Misclassifications: %d\n', B + C);

end

disp('--- Run Complete ---');

diary off; % Stop saving the output to the file

%% Observations
% * mu = 0.075 with K = 30 gives the best test accuracy (98.59%, 4 errors)
%   and removes false positives entirely.
% * Without regularization, 30 iterations lower the training cost but do not
%   improve test accuracy (97.54% at both K = 10 and K = 30).