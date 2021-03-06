# XYZ Corp
Classification, Regression, Clustering and Natural Language Processing (NLP) models for credit risk dataset.

## Description
explore.ipynb - A notebook showing the steps taken to wrangle the data and some investigation on the output parameters of the models.

data_wrangling.py - A Python file that contains the code that wrangles the data and creates a .csv file with the updated data.

problem1.py - This model assumes the following problem. A credit risk analyst is using previous loan data to predict if some current loans are going to be default or not and if they are, calculate the recoveries based on them.
It includes two classification models, one predicting if the loan will be default ('default_ind'), and one for recoveries being zero or non-zero, and a regression model predicting the value of the non-zero recoveries.

problem2.py - This model assumes the following problem. A credit risk analyst is using previous loan data to predict if a new loan from a new or existing customer will go default, calculate its recoveries if default and calculate an interest rate for the loan. The main difference between this and model1 is that model2 does not include certain inputs for its prediction and also predicts intrest rate. It includes two classification models, one predicting if the loan will be default ('default_ind'), and one for recoveries being zero or non-zero, one regression model predicting the value of the non-zero recoveries and one regression model predicting interest rate.

clust_model.py - This model attempts to predict the grade of the loan for Problem 2 using KNN.

unsupervised.py - This model attempts to predict the grade of the loan for Problem 2 using unsupervised learning and K-Means-Square.

nlp_model.py - This model attempts to predict the purpose of the loan based on the description given by its borrower.

app.py - A Python file containing the steps to deploy Problem 2 models to API to be used by front-end.

A respective jupyter notebook is included for each of the models as well.

## Before running any file
Run the data_wrangling notebook to get the 'wrang_xyz_data.csv'.
Before running app.py you should have run all the models for problem2_model and uncommented the last line of code using picke.dump.

## Additional instructions
The models include certain parameters that can be changed for different reasons. These are all explained in the comments of the models' code.

## Where to find the database and problem
https://www.kaggle.com/datasets/sonujha090/xyzcorp-lendingdata