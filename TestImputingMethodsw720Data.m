%% Walk through: testing various imputation algoorithms on small subset of buoy data

% This code compares three different imputation methods for missing buoy
% data collected in Narragansset Bay. The methods we test are Beckers
% Rixen, Random Forest, and K-Nearest-Neighbors. Though the data itself has
% missing values, in this code, we test the accuracy of each method to
% "fill in" the holes by manufacturing synthetic ones. The subset of data
% we use has 277 observations of 7 variables. We create holes in the last
% 103 observations, and each method -- KNN, BR, and RF -- predicts the
% values at those holes. In this code, we also run the algorithms on an
% augmented dataset that has Chebyshev Polynomials up to the 5th power of
% each variable (not including the variable that is being estimated).
% Running this code will produce quantile-quantile plots of the performance
% accuracy of each method and its Chebyshev analog, along with producing
% various measures of skill in the form of RMSE, Skew Error, Kurtosis Error
% and KL Divergence for each method in each of the variables. Note that if
% using a matrix of a different size, some of the indexing will need to
% change to fit your data. 

rng(10);
data = readmatrix('./EOF_720_log10.csv');

subset = data([1,4:9],:); %Subset to only 7 variables, the rest are NaNs
subset2 = (rmmissing(subset'))'; %Remove all rows with NaNs
subset2_rand = subset2(:, randperm(size(subset2, 2))); %Shuffle the data (we want to create holes in random places)

M0 = subset2_rand; 
M0(1,175:end) = nan; %Manufacture "holes"

%Initialize training and testing sets for RF algorithm
M0trainX = M0(2:7,1:174); 
M0trainY = M0(1,1:174);
M0predX = M0(2:7,175:end);
rng('default');
tmp = templateTree('Surrogate','on','Reproducible',true);


%% Run each imputation algorithm on each variable in the subset matrix

%Initialize diagnostics tables
rmses = zeros(size(subset2_rand,1),6);
skews = zeros(size(subset2_rand,1),6);
kurts = zeros(size(subset2_rand,1),6);
kl_divs = zeros(size(subset2_rand,1),6);

for i = 1:7 %for each variable (row) in the data subset
    
    trainsubset = setdiff(subset2_rand, subset2_rand(i,:), 'rows'); %trainsubset is all the rows in the matrix except the one we are trying to estimate
    
    %Prepare M0 for input to KNN
    M0 = subset2_rand;
    M0(i,175:end) = nan; %KNN needs NaNs to what to impute

    %Prepare M0 for KNN with Chebyshev Polynomials
    n = size(subset2_rand,1)-1;
    M0Cheby = subset2_rand;
    M0Cheby(i,175:end) = nan;
    M0Cheby(8:13,:) = chebyshevT(2,trainsubset(:,:));
    M0Cheby(14:19,:) = chebyshevT(3,trainsubset(:,:));
    M0Cheby(20:25,:) = chebyshevT(4,trainsubset(:,:));
    M0Cheby(26:31,:) = chebyshevT(5,trainsubset(:,:));
    
    %Prepare M0 for Beckers Rixen
    M02 = subset2_rand;
    M02(i,175:end) = 0; %Beckers Rixen needs 0s to know what to impute

    %Prepare M0 for Beckers Rixen with Chebyshev Polynomials
    M02Cheby = M0Cheby;
    M02Cheby(i,175:end) = 0;
    
    %Prepare M0 for RF
    M0trainX = trainsubset(:,1:174);
    M0trainY = M0(i,1:174);
    M0predX = trainsubset(:,175:end);
    
    %RF w/ Chebyshev Polynomials
    M0trainXCheby = trainsubset(:,1:174);
    M0trainXCheby(8:13,:) = chebyshevT(2,trainsubset(:,1:174));
    M0trainXCheby(14:19,:)= chebyshevT(3,trainsubset(:,1:174));
    M0trainXCheby(20:25,:)= chebyshevT(4,trainsubset(:,1:174));
    M0trainXCheby(26:31,:)= chebyshevT(5,trainsubset(:,1:174));
    M0trainYCheby = M0(i,1:174);
    M0predXCheby = trainsubset(:,175:end);
    M0predXCheby(8:13,:) = chebyshevT(2,trainsubset(:,175:end));
    M0predXCheby(14:19,:)= chebyshevT(3,trainsubset(:,175:end));
    M0predXCheby(20:25,:)= chebyshevT(4,trainsubset(:,175:end));
    M0predXCheby(26:31,:)= chebyshevT(5,trainsubset(:,175:end));

    %Run the algorithms
    tic
    Ma=knnimpute(M0,4);
    toc

    tic
    MaCheby=knnimpute(M0Cheby,4);
    toc
    
    tic
    Ma2 = BeckersRixen(M02);
    toc

    tic
    Ma2Cheby = BeckersRixen(M02Cheby);
    toc
    
    tic
    rfMdl = fitrensemble(M0trainX',M0trainY);
    rfPredict = predict(rfMdl, M0predX');
    toc
    
    tic
    rfMdlCheby = fitrensemble(M0trainXCheby',M0trainYCheby);
    rfPredictCheby = predict(rfMdlCheby, M0predXCheby');
    toc

    %Run diagnostics
    target_vals = subset2_rand(i,175:end);
    
    err_knn = target_vals - Ma(i,175:end);
    err_knncheby = target_vals - MaCheby(i,175:end);
    err_br = target_vals - Ma2(i,175:end);
    err_brcheby = target_vals - Ma2Cheby(i,175:end);
    err_rf = target_vals - rfPredict';
    err_rfcheby = target_vals - rfPredictCheby';
    
    %Populate tables for RMSE, Skew Error, and Kurtosis Error
    rmses(i,:) = [sqrt(sum(err_knn.^2)/length(err_knn)), sqrt(sum(err_knncheby.^2)/length(err_knncheby)), sqrt(sum(err_br.^2)/length(err_br)), sqrt(sum(err_brcheby.^2)/length(err_brcheby)), sqrt(sum(err_rf.^2)/length(err_rf)), sqrt(sum(err_rfcheby.^2)/length(err_rfcheby))];
    skews(i,:) = [(sum(err_knn.^3)/length(err_knn))^-3, (sum(err_knncheby.^3)/length(err_knncheby))^-3, (sum(err_br.^3)/length(err_br))^-3, (sum(err_brcheby.^3)/length(err_brcheby))^-3, (sum(err_rf.^3)/length(err_rf))^-3, (sum(err_rfcheby.^3)/length(err_rfcheby))^-3];
    kurts(i,:) = [(sum(err_knn.^4)/length(err_knn))^-4, (sum(err_knncheby.^4)/length(err_knncheby))^-4, (sum(err_br.^4)/length(err_br))^-4, (sum(err_brcheby.^4)/length(err_brcheby))^-4,(sum(err_rf.^4)/length(err_rf))^-4, (sum(err_rfcheby.^4)/length(err_rfcheby))^-4];

    %Populate table for KL Divergence
    xKL_knn = [Ma(i,175:end)'; target_vals'];
    xKL_knncheby = [MaCheby(i,175:end)'; target_vals'];
    xKL_br = [Ma2(i,175:end)'; target_vals'];
    xKL_brcheby = [Ma2Cheby(i,175:end)'; target_vals'];
    xKL_rf = [rfPredict; target_vals'];
    xKL_rfcheby = [rfPredictCheby; target_vals'];
    iKL = logical([zeros(103,1); ones(103,1)]);

    kl_divs(i,:) = [relativeEntropy(xKL_knn, iKL), relativeEntropy(xKL_knncheby, iKL), relativeEntropy(xKL_br, iKL), relativeEntropy(xKL_brcheby, iKL), relativeEntropy(xKL_rf, iKL), relativeEntropy(xKL_rfcheby, iKL)];
    
    % QQ Plots for each method
    figure(i)
    subplot(2,3,1)
    qqplot(target_vals,Ma(i,175:end))
    title('Imputing with KNN (k=4)');
   
    subplot(2,3,2)
    qqplot(target_vals,Ma2(i,175:end))
    title('Imputing with BeckersRixen');
    
    subplot(2,3,3)
    qqplot(target_vals,rfPredict)
    title('Imputing with Random Forest');
   
    subplot(2,3,4)
    qqplot(target_vals,MaCheby(i,175:end))
    title('KNN (k=4) & Chebyshev');
    
    subplot(2,3,5)
    qqplot(target_vals,Ma2Cheby(i,175:end))
    title('BeckersRixen & Chebyshev');
    
    subplot(2,3,6)
    qqplot(target_vals,rfPredictCheby)
    title('RF  w/ Chebyshev');
end

% Summary Statistics
rmse_tbl = array2table(rmses);
rmse_tbl.Properties.VariableNames(1:6) = {'RMSE: KNN','RMSE: KNN Cheby','RMSE: BR','RMSE: BR Cheby','RMSE: RF', 'RMSE: RF Cheby'};
rmse_vars = ["RMSE: KNN","RMSE: KNN Cheby","RMSE: BR","RMSE: BR Cheby","RMSE: RF", "RMSE: RF Cheby"];
rmse_means = mean(rmse_tbl{:,rmse_vars});

skew_tbl = array2table(skews);
skew_tbl.Properties.VariableNames(1:6) = {'Skew Err: KNN','Skew Err: KNN Cheby','Skew Err: BR','Skew Err: BR Cheby','Skew Err: RF', 'Skew Err: RF Cheby'};
skew_vars = ["Skew Err: KNN","Skew Err: KNN Cheby","Skew Err: BR","Skew Err: BR Cheby","Skew Err: RF", "Skew Err: RF Cheby"];
skew_means = mean(skew_tbl{:,skew_vars});

kurt_tbl = array2table(kurts);
kurt_tbl.Properties.VariableNames(1:6) = {'Kurt Err: KNN','Kurt Err: KNN Cheby','Kurt Err: BR','Kurt Err: BR Cheby','Kurt Err: RF', 'Kurt Err: RF Cheby'};
kurt_vars = ["Kurt Err: KNN","Kurt Err: KNN Cheby","Kurt Err: BR","Kurt Err: BR Cheby","Kurt Err: RF", "Kurt Err: RF Cheby"];
kurt_means = mean(kurt_tbl{:,kurt_vars});

kl_tbl = array2table(kl_divs);
kl_tbl.Properties.VariableNames(1:6) = {'KL Div: KNN','KL Div: KNN Cheby','KL Div: BR','KL Div: BR Cheby','KL Div: RF', 'KL Div: RF Cheby'};
kl_vars = ["KL Div: KNN","KL Div: KNN Cheby","KL Div: BR","KL Div: BR Cheby","KL Div: RF", "KL Div: RF Cheby"];
kl_means = mean(kl_tbl{:,kl_vars});


