clc;
clear;
close all;

load("FallAllD.mat");
waistData = strcmp({FallAllD.Device},'Waist');
waistData = waistData';
waistIndex = find(waistData==1);

for i = 1:length(waistIndex)
    
    % FT
    eval(['waistDataAcc_', num2str(i), '=FallAllD(waistIndex(i,1)).Acc;']);
    eval(['waistDataGyr_', num2str(i), '=FallAllD(waistIndex(i,1)).Gyr;']);
    eval(['waistData_', num2str(i), '=[waistDataAcc_', num2str(i),',waistDataGyr_', num2str(i),'];']);
    
    % GT
    waistDataActivityGT(i, 1) = FallAllD(waistIndex(i,1)).AtivityID;
    if waistDataActivityGT(i, 1) < 100
        waistDataActivityGT(i, 2) = 0;
    else 
        waistDataActivityGT(i, 2) = 1;
    end
    waistDataActivityGT = double(waistDataActivityGT);
end

% FT
for s = 1:length(waistIndex)
    for i = 0:5
        eval(['FT_waist(s, 1 + i*8) = mean(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 2 + i*8) = std(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 3 + i*8) = var(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 4 + i*8) = max(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 5 + i*8) = min(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 6 + i*8) = range(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 7 + i*8) = kurtosis(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 8 + i*8) = skewness(waistData_', num2str(s), '(:, i+1));'])
    end
end

% combine FT & GT
FT_GT_waist = [FT_waist, waistDataActivityGT(:,2)];

foldIndex_waist = randi([1, 10], length(waistIndex),1);
for waistkFold = 1:10
    test = (foldIndex_waist == waistkFold);
    train = ~test;
    FT_GT_train_waist = FT_GT_waist(train, :);
    FT_GT_test_waist = FT_GT_waist(test, :);
    
    % naive bayes model setup
    modelnb = fitcnb(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49));
    predictmodel = predict(modelnb, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 1) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 1) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 1) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);
    


    % decision tree model setup
    modeltree = fitctree(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49));
    predictmodel = predict(modeltree, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 2) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 2) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 2) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);



    % KNN
    modelknn = fitcknn(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49), 'NumNeighbors', 3);
    predictmodel = predict(modelknn, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 3) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 3) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 3) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);

end

% TotalMatrix spec
% TotalMatrix(kfold, Acc/Sen/Pre, NB/Tree/KNN)

for i = 1:3 % 1=SVM, 2=NB, 3=Decision Tree
    FinalMatrix_waist(i ,1) = mean(waist_Matrix(:, 1, i)); % Acc
    FinalMatrix_waist(i ,2) = mean(waist_Matrix(:, 2, i)); % Sen
    FinalMatrix_waist(i ,3) = mean(waist_Matrix(:, 3, i)); % Pre
end






waistData = strcmp({FallAllD.Device},'Neck');
waistData = waistData';
waistIndex = find(waistData==1);

for i = 1:length(waistIndex)
    
    % FT
    eval(['waistDataAcc_', num2str(i), '=FallAllD(waistIndex(i,1)).Acc;']);
    eval(['waistDataGyr_', num2str(i), '=FallAllD(waistIndex(i,1)).Gyr;']);
    eval(['waistData_', num2str(i), '=[waistDataAcc_', num2str(i),',waistDataGyr_', num2str(i),'];']);
    
    % GT
    waistDataActivityGT(i, 1) = FallAllD(waistIndex(i,1)).AtivityID;
    if waistDataActivityGT(i, 1) < 100
        waistDataActivityGT(i, 2) = 0;
    else 
        waistDataActivityGT(i, 2) = 1;
    end
    waistDataActivityGT = double(waistDataActivityGT);
end

% FT
for s = 1:length(waistIndex)
    for i = 0:5
        eval(['FT_waist(s, 1 + i*8) = mean(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 2 + i*8) = std(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 3 + i*8) = var(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 4 + i*8) = max(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 5 + i*8) = min(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 6 + i*8) = range(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 7 + i*8) = kurtosis(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 8 + i*8) = skewness(waistData_', num2str(s), '(:, i+1));'])
    end
end

% combine FT & GT
FT_GT_waist = [FT_waist, waistDataActivityGT(:,2)];

foldIndex_waist = randi([1, 10], length(waistIndex),1);
for waistkFold = 1:10
    test = (foldIndex_waist == waistkFold);
    train = ~test;
    FT_GT_train_waist = FT_GT_waist(train, :);
    FT_GT_test_waist = FT_GT_waist(test, :);
    
    % naive bayes model setup
    modelnb = fitcnb(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49));
    predictmodel = predict(modelnb, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 1) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 1) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 1) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);
    


    % decision tree model setup
    modeltree = fitctree(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49));
    predictmodel = predict(modeltree, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 2) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 2) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 2) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);



    % KNN
    modelknn = fitcknn(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49), 'NumNeighbors', 3);
    predictmodel = predict(modelknn, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 3) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 3) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 3) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);

end

% TotalMatrix spec
% TotalMatrix(kfold, Acc/Sen/Pre, NB/Tree/KNN)

for i = 1:3 % 1=SVM, 2=NB, 3=Decision Tree
    FinalMatrix_wrist(i ,1) = mean(waist_Matrix(:, 1, i)); % Acc
    FinalMatrix_wrist(i ,2) = mean(waist_Matrix(:, 2, i)); % Sen
    FinalMatrix_wrist(i ,3) = mean(waist_Matrix(:, 3, i)); % Pre
end

waistData = strcmp({FallAllD.Device},'Wrist');
waistData = waistData';
waistIndex = find(waistData==1);

for i = 1:length(waistIndex)
    
    % FT
    eval(['waistDataAcc_', num2str(i), '=FallAllD(waistIndex(i,1)).Acc;']);
    eval(['waistDataGyr_', num2str(i), '=FallAllD(waistIndex(i,1)).Gyr;']);
    eval(['waistData_', num2str(i), '=[waistDataAcc_', num2str(i),',waistDataGyr_', num2str(i),'];']);
    
    % GT
    waistDataActivityGT(i, 1) = FallAllD(waistIndex(i,1)).AtivityID;
    if waistDataActivityGT(i, 1) < 100
        waistDataActivityGT(i, 2) = 0;
    else 
        waistDataActivityGT(i, 2) = 1;
    end
    waistDataActivityGT = double(waistDataActivityGT);
end

% FT
for s = 1:length(waistIndex)
    for i = 0:5
        eval(['FT_waist(s, 1 + i*8) = mean(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 2 + i*8) = std(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 3 + i*8) = var(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 4 + i*8) = max(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 5 + i*8) = min(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 6 + i*8) = range(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 7 + i*8) = kurtosis(waistData_', num2str(s), '(:, i+1));'])
        eval(['FT_waist(s, 8 + i*8) = skewness(waistData_', num2str(s), '(:, i+1));'])
    end
end

% combine FT & GT
FT_GT_waist = [FT_waist, waistDataActivityGT(:,2)];

foldIndex_waist = randi([1, 10], length(waistIndex),1);
for waistkFold = 1:10
    test = (foldIndex_waist == waistkFold);
    train = ~test;
    FT_GT_train_waist = FT_GT_waist(train, :);
    FT_GT_test_waist = FT_GT_waist(test, :);
    
    % naive bayes model setup
    modelnb = fitcnb(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49));
    predictmodel = predict(modelnb, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 1) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 1) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 1) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);
    


    % decision tree model setup
    modeltree = fitctree(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49));
    predictmodel = predict(modeltree, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 2) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 2) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 2) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);



    % KNN
    modelknn = fitcknn(FT_GT_train_waist(:,1:48), FT_GT_train_waist(:, 49), 'NumNeighbors', 3);
    predictmodel = predict(modelknn, FT_GT_test_waist(:, 1: 48));

    % confusion matrix
    eval(['waist_confusionMatrix_', num2str(waistkFold), '=confusionmat(FT_GT_test_waist(:,49), predictmodel);']);
    % Acc
    eval(['waist_Matrix(waistkFold, 1, 3) = sum(diag(waist_confusionMatrix_', num2str(waistkFold), '))/sum(sum(waist_confusionMatrix_', num2str(waistkFold), '));']);
    % Sen
    eval(['waist_Matrix(waistkFold, 2, 3) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(2,:));']);
    % Pre
    eval(['waist_Matrix(waistkFold, 3, 3) = waist_confusionMatrix_', num2str(waistkFold), '(2,2)/sum(waist_confusionMatrix_', num2str(waistkFold), '(:,2));']);

end

% TotalMatrix spec
% TotalMatrix(kfold, Acc/Sen/Pre, NB/Tree/KNN)

for i = 1:3 % 1=SVM, 2=NB, 3=Decision Tree
    FinalMatrix_neck(i ,1) = mean(waist_Matrix(:, 1, i)); % Acc
    FinalMatrix_neck(i ,2) = mean(waist_Matrix(:, 2, i)); % Sen
    FinalMatrix_neck(i ,3) = mean(waist_Matrix(:, 3, i)); % Pre
end

disp(FinalMatrix_waist);
disp(FinalMatrix_wrist);
disp(FinalMatrix_neck);


