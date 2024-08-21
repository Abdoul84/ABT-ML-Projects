### Childhood Respiratory Disease Analysis

* In this analysis, I compared several linear models to predict the impact of smoking on childhood respiratory diseases. The analysis required preprocessing techniques to convert categorical data into numerical values and to apply scaling and normalization to the data.

## Steps

* I used pandas's `get_dummies` function to convert the categorical data into binary values.

* I fitted a standard scaler model to the training data.

* I applied the scaler transformation to both the training and testing data.

* I compared the performance of the following models: `LinearRegression`, `Lasso`, `Ridge`, and `ElasticNet`.
  
  * For each model, I computed the Mean Squared Error (MSE) and R2 score for the test data.

### Additional Task

* I plotted the residuals for both the training and testing data.

### Guidance

* I referred to Scikit-learn's [documentation](http://scikit-learn.org/stable/modules/linear_model.html) for instructions on how to use `Lasso`, `Ridge`, and `ElasticNet`. Each of these models followed the familiar pattern of `model->fit->predict`.
- - -

© 2021 Trilogy Education Services, LLC, a 2U, Inc. brand. Confidential and Proprietary. All Rights Reserved.
