# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.

    This challenge is on using various techniques to train and evaluate models with imbalanced classes. The purpose of this analysis is to use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

* Explain what financial information the data was on, and what you needed to predict.

    This financial data was on historical lending activity from a peer-to-peer lending service company. The data contains various parameter like loan_size, interest_rate, debt values and other. A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.
    
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

    `value_counts`: Function to count the number of values presented in each label data.
    
    `train_test_split`: Function from the scikit-learn library to automatically split the data into training and testing data.
    
    `balanced_accuracy_score`: Function helps getting the accuracy score for the test results (one from test data spliting and prediction of data)
    
    `confusion_matrix`: Module from scikit-learn accepts the predicted values and calculates the confusion matrix True Positive(TP), True Negative(TN), False Positive(FP), FN(False Negative).
    
    `classification_report_imbalanced`: Function gives us the precision value(Percentage of prediction were correct), Recall value (Fraction of             positives that were correctly identified), F1 Score (weighted harmonic mean of precisions and recall such that the best score is 1 and worst is 0.0)
    
    `RandomOverSampler`: Function is used for oversample the data to have a balanced class and prediction can be done without biased.
    
    `fit_resample`: Function is used to fit the sampled data into the model.
    
* Describe the stages of the machine learning process you went through as part of this analysis.

    Split the Data into Training and Testing Sets

    Create a Logistic Regression Model with the Original Data

    Predict a Logistic Regression Model with Resampled Training Data

    Write a Credit Risk Analysis Report
    
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

LogisticRegression classifier and the resampled data to fit the model and make predictions.
RandomOverSampler module from the imbalanced-learn library used to resample the data. 


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
Accuracy score for logistic regression model is 0.95 appr with average precisiom value and recall value 0.99.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores. 
  
The logistic regression model for fitting with oversampled data is better than the orginal data prediction.
The accuracy score for oversampled data is 0.99. Model used for resampled data better detected true positives/true negatives.
The oversampled data has value of 0.99 which is much better than original data. The Model using resampled data was much better at detecting risky loan and healthy loan generated using the original, imbalanced dataset.
  
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

In our analysis of prediction on original data and resampling data the precision and recall value holds 0.99 which same for both the cases. However, accuracy after resampling of data is aroung 0.99 comparing which is superior to the original data prediction of 0.95.
Performance depends on the predicted value of loan status where the 1's are high risk loans and 0's are healthy loans. In the original data set 0 75036 1 2500. Compared to oversampled data 0 56271 1 56271.
As we can see in the original data the difference betweeen 0's and 1's is large, therefor the spliting data and then traing and testing seems bias. But  after resampling data into equal values the model gives better result without get biased.

Recommendation would be the oversampled model which performed better comparing to the original model.


