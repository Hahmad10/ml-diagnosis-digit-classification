%% Load data

load('D_mpg.mat');

%% Split features/labels and augment with a bias row

y = D_mpg(7, :).';         % 392×1
M = numel(y);              % 392
P = 314;                   % training samples
T = M - P;                 % test samples (should be 78)

X  = D_mpg(1:6, :);        % 6×392 (features)
Xh = [X; ones(1, M)];      % 7×392 (augment with ones)

Xh_tr = Xh(:, 1:P);        % 7×314
y_tr  = y(1:P);            % 314×1


%% Test split

Xh_te = Xh(:, P+1:M);      % 7×78
y_te  = y(P+1:M);          % 78×1


%% Least-squares weights (pseudo-inverse)

A_tr = Xh_tr.';            % 314×7 design matrix
w_star = pinv(A_tr) * y_tr;


%% Predict and compute RMSE

yhat_tr = Xh_tr.' * w_star;   % 314×1 (vectorized prediction, no loop)
yhat_te = Xh_te.' * w_star;   % 78×1  (vectorized prediction, no loop)  

RMSE_train = sqrt(mean((y_tr - yhat_tr).^2));
RMSE_test  = sqrt(mean((y_te - yhat_te).^2));

fprintf('RMSE_train = %.6f\n', RMSE_train);
fprintf('RMSE_test  = %.6f\n', RMSE_test);

%% Plot predictions vs ground truth

figure;
plot(y_te,    'b-', 'LineWidth', 1.5); hold on;   % Ground truth in BLUE
plot(yhat_te, 'r-', 'LineWidth', 1.5);            % Prediction in RED
grid on;
xlabel('Test sample index');
ylabel('Fuel Consumption (MPG)');
legend('Ground truth', 'Prediction');
title('Fuel Consumption Prediction on Test Data');
