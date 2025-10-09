clear; 
clc; 

%% Load and prepare data
% Load the dataset and split it into training and testing sets for each class.
% For each class of 50 samples, the first 35 are used for
% training and the remaining 15 are for testing. 

disp('Loading and preparing data...');
load('X_iris.mat'); % Loads the 4x150 X_iris matrix 

% Split data by class 
x1 = X_iris(:, 1:50);   % Class 1: Setosa
x2 = X_iris(:, 51:100);  % Class 2: Versicolor
x3 = X_iris(:, 101:150); % Class 3: Virginica

% Create training and test sets for each class
xtr1 = x1(:, 1:35);  xte1 = x1(:, 36:50); % Setosa train/test split
xtr2 = x2(:, 1:35);  xte2 = x2(:, 36:50); % Versicolor train/test split
xtr3 = x3(:, 1:35);  xte3 = x3(:, 36:50); % Virginica train/test split

disp('Data preparation complete.');
disp('------------------------------------');

%% Train, classify, and evaluate
% Loop through K=[1, 3, 5] to compare performance.

K_values = [1, 3, 5];

for k_idx = 1:length(K_values)
    K = K_values(k_idx);
    fprintf('PERFORMING CLASSIFICATION FOR K = %d NEWTON ITERATIONS\n\n', K);

    % Step 1: Train three binary classifiers
    % For each classifier, one class is positive (P) and the other two are
    % negative (N).
    
    % Classifier 1: Class 1 vs. All
    X_train1 = [xtr1, xtr2, xtr3];
    y_train1 = [ones(1, 35), -ones(1, 70)]; % Class 1 is P, 2&3 are N
    ws1 = LRBC_newton(X_train1, y_train1, K);

    % Classifier 2: Class 2 vs. All
    X_train2 = [xtr2, xtr1, xtr3];
    y_train2 = [ones(1, 35), -ones(1, 70)]; % Class 2 is P, 1&3 are N
    ws2 = LRBC_newton(X_train2, y_train2, K);

    % Classifier 3: Class 3 vs. All
    X_train3 = [xtr3, xtr1, xtr2];
    y_train3 = [ones(1, 35), -ones(1, 70)]; % Class 3 is P, 1&2 are N
    ws3 = LRBC_newton(X_train3, y_train3, K);

    % Step 2: Normalize the parameter vectors
    % Normalize w and b for each classifier so they are comparable.
    ws_hat1 = normalize_params(ws1);
    ws_hat2 = normalize_params(ws2);
    ws_hat3 = normalize_params(ws3);
    
    % Combine into a single weight matrix for easy computation
    Ws_hat = [ws_hat1, ws_hat2, ws_hat3];

    % Evaluation on training data
    fprintf('--- Training Data Results (K=%d) ---\n', K);
    X_train = [xtr1, xtr2, xtr3];
    true_labels_train = [ones(35, 1); 2*ones(35, 1); 3*ones(35, 1)];
    
    % Augment data matrix with a row of ones for the bias term
    Xh_train = [X_train; ones(1, size(X_train, 2))];
    
    % Calculate scores for each class
    scores_train = Xh_train' * Ws_hat;
    
    % Find the index of the max score for each sample to get the prediction
    [~, predicted_labels_train] = max(scores_train, [], 2);
    
    % Build Confusion Matrix
    % Convention used here: C(i,j) = # samples from actual class 'j'
    % classified as predicted class 'i'.
    % MATLAB's confusionmat(TRUE, PRED) returns C(i,j) as: # samples of 
    % true class 'i' predicted to be class 'j'. We need the transpose.
    C_train = confusionmat(true_labels_train, predicted_labels_train)';
    
    % Calculate Accuracy
    accuracy_train = 100 * sum(diag(C_train)) / sum(C_train(:));
    
    disp('Confusion Matrix (Training):');
    disp(C_train);
    fprintf('Classification Accuracy (Training): %.2f%%\n\n', accuracy_train);

    % --- Evaluation on test data ---
    fprintf('--- Test Data Results (K=%d) ---\n', K);
    X_test = [xte1, xte2, xte3];
    true_labels_test = [ones(15, 1); 2*ones(15, 1); 3*ones(15, 1)];

    % Augment test data
    Xh_test = [X_test; ones(1, size(X_test, 2))];

    % Calculate scores for each class
    scores_test = Xh_test' * Ws_hat;
    
    % Predict labels
    [~, predicted_labels_test] = max(scores_test, [], 2);
    
    % Build Confusion Matrix
    C_test = confusionmat(true_labels_test, predicted_labels_test)';
    
    % Calculate Accuracy
    accuracy_test = 100 * sum(diag(C_test)) / sum(C_test(:));
    
    disp('Confusion Matrix (Test):');
    disp(C_test);
    fprintf('Classification Accuracy (Test): %.2f%%\n', accuracy_test);
    disp('------------------------------------');
end

%% Helper Function for Normalization
function ws_hat = normalize_params(ws)
    % Extract w and b, then divide both by ||w||
    % The input vector ws is [w; b]
    N = length(ws) - 1; % Number of features
    w = ws(1:N);
    b = ws(N+1);
    
    norm_w = norm(w);
    
    % Normalize b first, then w, using the original norm of w
    b_norm = b / norm_w;
    w_norm = w / norm_w;
    
    ws_hat = [w_norm; b_norm];
end