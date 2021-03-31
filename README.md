# #FIFA_players
# Data Analysis
This is a dataset containing 19402 observations of FIFA soccer players with 16 features with 0 null value. Out of the 16 features, there are 2 categorical variables (Player and Preferred Foot), and 14 numerical variables. Preferred.Foot is set as a dummy variable (“right” as 1, “left” as 0).
There are three distributions that appear to be abnormal: Market Value, Weekly Salary and Goalkeeping. Therefore, we applied log transformation on these three variables.

We also explore the correlations between each variable; Since the goal is to predict the future market values (var: Market Value) of the football players, we looked at
variables correlated with Market Value with a correlation score higher than 0.3;

It's also worth mentioning that some of them correlate with each other with a correlation score over 0.75. For example, weight and height are positively correlated, ball skills are positively correlated with the following variables: shooting, goalkeeping, physical, passing, mental. Interestingly, mental is strongly correlated with passing and shooting. Passing is also highly positively correlated with physical and shooting

# Feature Selection
First, we built a full model, regressing the Market Value on all 14 other variables (except player name) in the soccer player data using a linear model and found the R square of the full model to be 0.9657.
Then, we randomly split the data into training sets (70% of the data) and testing sets (30% of the data), and scaled x_train data. After that, we conducted a backward stepwise feature selection. Below is its ANOVA table and coefficients

Based on the backward stepwise selection, we decided to drop Passing, Mental, Ball.Skills, Potential.Score, and Weight. Therefore, our reduced model includes 9 variables: Overall.Score, Weekly.Salary, Height, Age, Defence, Physical, Shooting, Goalkeeping and Right.foot.dummy.

# Model Selection
To compare the L1 and L2 Norm regularization techniques, different penalty terms and model transformations, we used our training data to build and compare eight models. Four of them are Lasso regression models and four of them are Ridge regression models.

1. Ridge Regression
We first used information criteria AIC to select lambda in Ridge Regression and we plotted the AIC value against different lambda values. According to AIC, we found the optimal lambda value to be 0.3511192. Next, we used Cross Validation to find the optimal lambda value and found it to be 0.1320045
We fit model 1 and model 3 on the training data, and the regression coefficients are shown as below. Note that by default, the function glmnet() standardized variables so that their scales are comparable. However, the coefficients are always returned on the original scale.
According to the coefficients in model 1, we decided to discard Height, Defence, Physical and Shooting since the absolute values of their coefficients are less than 0.01. The adjusted model is named model1_2. According to the coefficients in model 3, we decided to discard Height, Defence, Shooting and Right.foot.dummy since the absolute values of their coefficients are less than 0.01. The adjusted model is named model3_2.

2. Lasso Regression
We first used information criteria AIC to select lambda in Lasso regression and we plotted the AIC against different lambda values. According to AIC, we found the optimal lambda to be 0.001. Next, we used Cross Validation to find the optimal lambda value and found it to be 0.002361099
We then fit model 2 and model 4 on the training data, and the regression coefficients are shown as below.
According to the coefficients in model 2, we decided to discard Height, Defence, Physical, Shooting and Right.foot.dummy since their coefficients are less than 0.01. The adjusted model is named model 2_2. According to the coefficients in model 4, we decided to discard Height, Defence, Physical and Shooting since the absolute values of their coefficients are less than 0.01. The adjusted model is named model 4_2.

# Model Evaluation
We compare the out of sample R square and RMSE of ridge regression and lasso regression with lambda values generated from AIC and cross validation. We found that the linear regression from the reduced model, with an R square of 0.97 and RMSE of 0.2393 performs the best. Among the rest ridge/lasso regression, we concluded that model_4_2 performs the best with highest R square and lowest RMSE.
We plot the residuals values and the fitted values and find that the most of the residuals are close to 0, but there are some outliers that can be handled for future model improvement.

# Conclusion
According to out of sample R-squared, linear regression seems to have the best performance with an R square of 0.97. When comparing Ridge regression and Lasso regression, the latter with lambda selected by AIC as 0.001 has best performance (model 4_2).
It indicates that: 1. AIC can be more accurate than Cross validation when calculating lambda values. 2. Lasso and Ridge regression are not always the best, and solutions should be tailored to different problems.
