% Condensed version of mnist_softmax_raw_vs_hog.m

% Load data sets
load X1600.mat; load Te28.mat; load Lte28.mat;

% Prepare raw-pixel data matrices
u = ones(1,1600);
ytr = [u 2*u 3*u 4*u 5*u 6*u 7*u 8*u 9*u 10*u]; % Labels 1-10
Dtr = [X1600; ytr];
Dte = [Te28; 1+Lte28(:)']; % Adjust labels to 1-based indexing

% Prepare HOG data matrices
d = 7; B = 9; % HOG parameters
H = [];
for i = 1:16000
    xi = X1600(:,i);
    mi = reshape(xi, 28, 28);
    hi = hog20(mi, d, B); % Extract HOG features
    H = [H hi];
end
Dhtr = [H; ytr];

Hte = [];
for i = 1:length(Lte28)
    xi = Te28(:,i);
    mi = reshape(xi, 28, 28);
    hi = hog20(mi, d, B);
    Hte = [Hte hi];
end
Dhte = [Hte; 1+Lte28(:)'];

% Train and evaluate on raw-pixel data
mu_orig = 0.002; iter_orig = 62; K = 10;
disp('Training Softmax Regression on ORIGINAL Data...');
tic;
[Ws_orig, f_orig] = SRMCC_bfgsML(Dtr, 'f_SRMCC', 'g_SRMCC', mu_orig, K, iter_orig);
t_orig = toc;

% Evaluation
ytest = Dte(end,:);
Dtest_orig = [Dte(1:end-1,:); ones(1, size(Dte,2))]; % Add bias row
tic;
[~, ind_pre_orig] = max((Dtest_orig' * Ws_orig)');
te_time_orig = toc;
efficiency_orig = length(ytest)/te_time_orig;

% Build Confusion Matrix (Original)
C_orig = zeros(K,K);
for j = 1:K
    ind_j = find(ytest == j);
    for i = 1:K
        ind_pre_i = find(ind_pre_orig == i);
        C_orig(i,j) = length(intersect(ind_j, ind_pre_i));
    end
end
acc_orig = trace(C_orig) / sum(C_orig(:)) * 100;

% Train and evaluate on HOG data
mu_hog = 0.001; iter_hog = 57;
disp('Training Softmax Regression on HOG Data...');
tic;
[Ws_hog, f_hog] = SRMCC_bfgsML(Dhtr, 'f_SRMCC', 'g_SRMCC', mu_hog, K, iter_hog);
t_hog = toc;

% Evaluation
Dtest_hog = [Dhte(1:end-1,:); ones(1, size(Dhte,2))];
tic;
[~, ind_pre_hog] = max((Dtest_hog' * Ws_hog)');
te_time_hog = toc;
efficiency_hog = length(ytest)/te_time_hog;

% Build Confusion Matrix (HOG)
C_hog = zeros(K,K);
for j = 1:K
    ind_j = find(ytest == j);
    for i = 1:K
        ind_pre_i = find(ind_pre_hog == i);
        C_hog(i,j) = length(intersect(ind_j, ind_pre_i));
    end
end
acc_hog = trace(C_hog) / sum(C_hog(:)) * 100;

% Display Results
disp('Original Data Results:');
disp(['Accuracy: ', num2str(acc_orig), '%']);
disp('Confusion Matrix (Original):'); disp(C_orig);
disp('HOG Data Results:');
disp(['Accuracy: ', num2str(acc_hog), '%']);
disp('Confusion Matrix (HOG):'); disp(C_hog);