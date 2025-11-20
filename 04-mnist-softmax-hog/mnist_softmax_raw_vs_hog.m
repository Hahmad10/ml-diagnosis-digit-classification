clear; clc; close all;

%% Load data sets
disp('Loading data files...');
load('X1600.mat');   % Original training data (1600 samples per digit)
load('Te28.mat');    % Test data (10,000 samples)
load('Lte28.mat');   % Test labels

%% Prepare raw-pixel data matrices {Dtr, Dte}
disp('Preparing original data sets...');

% Prepare Training Data (Dtr)
% X1600 contains 1600 samples for digit 0, then 1600 for digit 1, etc.
% We create labels 1 through 10 (Softmax requires 1-K, not 0-9).
u = ones(1, 1600);
ytr = [u, 2*u, 3*u, 4*u, 5*u, 6*u, 7*u, 8*u, 9*u, 10*u]; % 
Dtr = [X1600; ytr]; % Append labels to the last row

% Prepare Test Data (Dte)
% Labels in Lte28 are 0-9, so we add 1 to make them 1-10.
Dte = [Te28; 1 + Lte28(:)'];  

%% Prepare HOG data matrices {Dhtr, Dhte}
disp('Calculating HOG features (this may take a moment)...');

% Parameters for hog20 function 
d = 7; 
B = 9;

% 3a. HOG for Training Data
H = [];
num_train = size(X1600, 2); % 16000 samples
for i = 1:num_train
    xi = X1600(:, i);
    mi = reshape(xi, 28, 28); % Reshape vector to 28x28 image 
    hi = hog20(mi, d, B);     % Compute HOG features
    H = [H, hi];
end
Dhtr = [H; ytr];  

% 3b. HOG for Test Data
Hte = [];
num_test = size(Te28, 2); % 10000 samples
for i = 1:num_test
    xi = Te28(:, i);
    mi = reshape(xi, 28, 28);
    hi = hog20(mi, d, B);
    Hte = [Hte, hi];
end
Dhte = [Hte; 1 + Lte28(:)']; 

%% Train and test on raw-pixel data
disp('------------------------------------------------');
disp('Training Softmax Regression on ORIGINAL Data...');

% Hyperparameters for raw-pixel data
mu_orig = 0.002;
iter_orig = 62;
K = 10; % Number of classes

% Train the model 
% Cost and gradient functions are passed by name
t_train_orig = tic;
[Ws_orig, f_val_orig] = SRMCC_bfgsML(Dtr, 'f_SRMCC', 'g_SRMCC', mu_orig, K, iter_orig);
time_train_orig = toc(t_train_orig);
fprintf('Training Time (Original): %.4f seconds\n', time_train_orig);

% Performance Evaluation 
disp('Evaluating Original Data Classifier...');

% Prepare prediction matrix (replace labels with 1s for bias term)
Dtest_orig = Dte;
Dtest_orig(end, :) = 1; 

% Perform Classification and measure efficiency
t_test_orig = tic;
scores_orig = Dtest_orig' * Ws_orig;       % Calculate inner products 
[~, ind_pre_orig] = max(scores_orig');     % Find max score index 
time_test_orig = toc(t_test_orig);

% Calculate Confusion Matrix
ytest = 1 + Lte28(:)'; % Ground truth labels
C_orig = zeros(K, K);

for j = 1:K
    ind_j = find(ytest == j); % Indices of samples actually belonging to class j
    for i = 1:K
        ind_pre_i = find(ind_pre_orig == i); % Indices predicted as class i
        % Intersection gives count of class j classified as i
        C_orig(i, j) = length(intersect(ind_j, ind_pre_i));
    end
end

% Calculate Accuracy
acc_orig = 100 * trace(C_orig) / sum(C_orig(:));
digits_per_sec_orig = num_test / time_test_orig;

fprintf('Original Data Results:\n');
fprintf('Accuracy: %.2f%%\n', acc_orig);
fprintf('Testing Efficiency: %.2f digits/sec\n', digits_per_sec_orig);
disp('Confusion Matrix (Original):');
disp(C_orig);

%% Train and test on HOG data
disp('------------------------------------------------');
disp('Training Softmax Regression on HOG Data...');

% Hyperparameters for HOG data
mu_hog = 0.001;
iter_hog = 57;

% Train the model
t_train_hog = tic;
[Ws_hog, f_val_hog] = SRMCC_bfgsML(Dhtr, 'f_SRMCC', 'g_SRMCC', mu_hog, K, iter_hog);
time_train_hog = toc(t_train_hog);
fprintf('Training Time (HOG): %.4f seconds\n', time_train_hog);

% Performance Evaluation
disp('Evaluating HOG Data Classifier...');

% Prepare prediction matrix
Dtest_hog = Dhte;
Dtest_hog(end, :) = 1;

% Perform Classification and measure efficiency
t_test_hog = tic;
scores_hog = Dtest_hog' * Ws_hog;
[~, ind_pre_hog] = max(scores_hog');
time_test_hog = toc(t_test_hog);

% Calculate Confusion Matrix
C_hog = zeros(K, K);
for j = 1:K
    ind_j = find(ytest == j);
    for i = 1:K
        ind_pre_i = find(ind_pre_hog == i);
        C_hog(i, j) = length(intersect(ind_j, ind_pre_i));
    end
end

% Calculate Accuracy
acc_hog = 100 * trace(C_hog) / sum(C_hog(:));
digits_per_sec_hog = num_test / time_test_hog;

fprintf('HOG Data Results:\n');
fprintf('Accuracy: %.2f%%\n', acc_hog);
fprintf('Testing Efficiency: %.2f digits/sec\n', digits_per_sec_hog);
disp('Confusion Matrix (HOG):');
disp(C_hog);

%% Comparison summary
disp('------------------------------------------------');
disp('Comparison Summary:');
fprintf('Original Accuracy: %.2f%% | HOG Accuracy: %.2f%%\n', acc_orig, acc_hog);
fprintf('Original Speed: %.0f/s    | HOG Speed: %.0f/s\n', digits_per_sec_orig, digits_per_sec_hog);