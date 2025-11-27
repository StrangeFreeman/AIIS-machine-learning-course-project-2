# AIIS-machine-learning-course-project-2
this is a AIIS machine learning course project 2 using matlab

## pre requirements
- dataset
    - FallALLD.mat
        - author: Chia-Yeh Hsieh

- matlab add-on
    - statistics and machine learning toolbox

## machine learning steps
- Before machine learning model, I create three big feature table(waist, wrist, neck)
    - feature table includes
        - data
            - Accelerometer data
            - Gyroscope data
        - statistical data
            - mean
            - standard deviation
            - variance
            - maximum
            - minimum
            - range
            - kurtosis
            - skewness      
    - ground truth includes
        - activity ID
            - < 100 = not fall
            - else fall

- After I created the big feature table, I setup random index for k-fold cross validation.

- Then inside k-fold cross validation, is creating machine learning model and create predictmodel using
    - naive bayes
    ```
    modelnb = fitcnb(FT_GT_train(:,1:561), FT_GT_train(:, 562));
    predictmodel = predict(modelnb, FT_GT_test(:, 1: 561));
    ```

    - decision tree
    ```
    modeltree = fitctree(FT_GT_train(:,1:561), FT_GT_train(:, 562));
    predictmodel = predict(modeltree, FT_GT_test(:, 1: 561));
    ```

    - kNN
    ```
    modelknn = fitcknn(FT_GT_train(:,1:561), FT_GT_train(:, 562), 'NumNeighbors', 3);
    predictmodel = predict(modelknn, FT_GT_test(:, 1: 561));
    ```

- After test models using GT dataset, I create confusion matrix on every machine learning model

- And at the end I caculate Accuracy, Sensitivity, Precision persentages average, and made it a 3x3x3 matrix name "FinalMatrix"

## final output description

three 3x3 matrix

- waist data:

| Model | Accuracy | Sensitivity | Precision |
|-------|---------|---------|---------|
| NB    | 0.9028  | 0.8277  | 0.8057  |
| DT    | 0.9608  | 0.9145  | 0.9376  |
| KNN   | 0.8815  | 0.7731  | 0.7768  |

- wrist data:

| Model | Accuracy | Sensitivity | Precision |
|-------|---------|---------|---------|
| NB    | 0.9020  | 0.8574  | 0.8391  |
| DT    | 0.9506  | 0.9188  | 0.9239  |
| KNN   | 0.8649  | 0.8075  | 0.7798  |

- neck data:

| Model | Accuracy | Sensitivity | Precision |
|-------|---------|---------|---------|
| NB    | 0.8859  | 0.7891  | 0.7926  |
| DT    | 0.9225  | 0.7918  | 0.8272  |
| KNN   | 0.8537  | 0.6095  | 0.6611  |


### program output

|     |     |     |
|-----|-----|-----|
| 0.9028  | 0.8277  | 0.8057  |
| 0.9608  | 0.9145  | 0.9376  |
| 0.8815  | 0.7731  | 0.7768  |

|     |     |     |
|-----|-----|-----|
| 0.9020  | 0.8574  | 0.8391  |
| 0.9506  | 0.9188  | 0.9239  |
| 0.8649  | 0.8075  | 0.7798  |


|     |     |     |
|-----|-----|-----|
| 0.8859  | 0.7891  | 0.7926  |
| 0.9225  | 0.7918  | 0.8272  |
| 0.8537  | 0.6095  | 0.6611  |
