FIFA = read.csv("FIFA_Player_List.csv")
set.seed(123)

# Use install.packages("~/Downloads/faux_0.0.1.6.tgz", repos = NULL, type = .Platform$pkgType) if can't dl
library(dplyr)
library(faux)
library(corrplot)
library(RColorBrewer)
library(caret)
library(randomForest)
library(stats)
library(psych)
library(glmnet)

##### Data Analysis #####

### Check if there's any NULL values in the dataset, dimensions, types of attributes, and statistical analysis
is.null(FIFA)
dim(FIFA)
sapply(FIFA, class)
summary(FIFA)


#### Preferred.Foot(Column 9) is categorical data; Set dummy variables
hist(FIFA$Market.Value)
hist(FIFA$Weekly.Salary)
hist(FIFA$Goalkeeping)

Player = FIFA$Player
Overall.Score = FIFA$Overall.Score
Potential.Score = FIFA$Potential.Score
Market.Value = log(FIFA$Market.Value)
Weekly.Salary = log(FIFA$Weekly.Salary)
Height = FIFA$Height
Weight = FIFA$Weight
Age = FIFA$Age
Right.foot.dummy = ifelse(FIFA$Preferred.Foot == "Right", 1, 0)
Ball.Skills = FIFA$Ball.Skills
Defence = FIFA$Defence
Mental = FIFA$Mental
Passing = FIFA$Passing
Physical = FIFA$Physical
Shooting = FIFA$Shooting
Goalkeeping = log(FIFA$Goalkeeping)

data = as.data.frame(cbind(Market.Value, Overall.Score, Potential.Score,
             Weekly.Salary, Height, Weight, Age, Right.foot.dummy,
             Ball.Skills, Defence, Mental, Passing, Physical, Shooting,
             Goalkeeping))
head(data,6)


### Find correlations
cor_matrix = round(cor(data),2)
corrplot(cor_matrix, type="upper", order="hclust", col=brewer.pal(n=8, name="RdYlBu"))


### Overall.Score, Potential.Score, Weekly.Salary, Ball.Skills, Mental, Passing, Physical, Shooting have a correlation greater than 0.3 with Market Value
plot(Overall.Score, Market.Value, xlab = "Overall Score", ylab = "Market Value")
plot(Potential.Score, Market.Value, xlab = "Potential Score", ylab = "Market Value")
plot(Weekly.Salary, Market.Value, xlab = "Weekly Salary", ylab = "Market Value")
plot(Ball.Skills, Market.Value, xlab = "Ball.Skills", ylab = "Market Value")
plot(Mental, Market.Value, xlab = "Mental", ylab = "Market Value")
plot(Physical, Market.Value, xlab = "Physical", ylab = "Market Value")
plot(Passing, Market.Value, xlab = "Passing", ylab = "Market Value")
plot(Shooting, Market.Value, xlab = "Shooting", ylab = "Market Value")


### It's also worth mentioning that some of them correlate with each other with a correlation score over 0.75

plot(Weight, Height, xlab = "Weight", ylab = "Height")

plot(Ball.Skills, Shooting, xlab = "Ball.Skills", ylab = "Shooting")
plot(Ball.Skills, Goalkeeping, xlab = "Ball.Skills", ylab = "Goalkeeping")
plot(Ball.Skills, Physical, xlab = "Ball.Skills", ylab = "Physical")
plot(Ball.Skills, Passing, xlab = "Ball.Skills", ylab = "Passing")
plot(Ball.Skills, Mental, xlab = "Ball.Skills", ylab = "Mental")

plot(Mental, Passing, xlab = "Mental", ylab = "Passing")
plot(Mental, Shooting, xlab = "Mental", ylab = "Shooting")

plot(Passing, Physical, xlab = "Passing", ylab = "Physical")
plot(Passing, Shooting, xlab = "Passing", ylab = "Shooting")



##### Feature Selection #####

# R^2 for the full model is 0.9657
full_model = lm(data$Market.Value ~ ., data = data)
summary(full_model)

# Training: 70%; Test: 30%
# Features
x <- data %>%
  select(Overall.Score, Potential.Score, Weekly.Salary, Height, Weight, Age, Ball.Skills, 
         Defence, Mental, Passing, Physical, Shooting, Goalkeeping, Right.foot.dummy) %>% as.data.frame()

# Target variable
y <- data$Market.Value

inTrain <- createDataPartition(y, p = .70, list = FALSE)[,1]

x_train <- x[ inTrain, ]
x_test  <- x[-inTrain, ]

y_train <- y[ inTrain]
y_test  <- y[-inTrain]

data_train = cbind(x_train,y_train)
intercept_only = lm(y_train ~ 1, data = data_train) #This is the mean
all = lm(y_train ~ . , data = data_train)
backward <- step(all, direction='backward', scope=formula(all), trace=0)
backward$anova
backward$coefficients


### Based on the backward stepwise selection, we decided to drop Passing, Ball.Skills, Potential.Score, Mental and Weight
### And keep Overall.Score, Weekly.Salary, Height, Age, Defence, Physical, Shooting,
### Goalkeeping and Right.foot.dummy 

### Model Selection
x_train = as.matrix(x_train[,c(1,3,4,6,8,11,12,13,14)])
y_train = as.vector(y_train)

x_test = as.matrix(x_test[,c(1,3,4,6,8,11,12,13,14)])
y_test = as.vector(y_test)
### Ridge Regression
# Use information criteria to select lambda 
lambdas_to_try <- 10^seq(-3, 9, length.out = 100)
X_scaled <- scale(x_train)
aic <- c()
for (lambda in seq(lambdas_to_try)) {
  # Run model
  model <- glmnet(x_train, y_train, alpha = 0, lambda = lambdas_to_try[lambda], standardize = TRUE)
  # Extract coefficients and residuals (remove first row for the intercept)
  betas <- as.vector((as.matrix(coef(model))[-1, ]))
  resid <- y_train - (X_scaled %*% betas)
  # Compute hat-matrix and degrees of freedom
  ld <- lambdas_to_try[lambda] * diag(ncol(X_scaled))
  H <- X_scaled %*% solve(t(X_scaled) %*% X_scaled + ld) %*% t(X_scaled)
  df <- tr(H)
  # Compute information criteria
  aic[lambda] <- nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df
}

log_lambdas_to_try <- log(lambdas_to_try)
# Plot information criteria against tried values of lambdas
plot(log_lambdas_to_try, aic, col = "orange", type = "l",
     ylab = "Information Criterion", main ="Ridge Regression")

# Optimal lambdas according to AIC criteria
lambda_aic <- lambdas_to_try[which.min(aic)]
lambda_aic

# Find the best lambda using cross-validation
cv1 <- cv.glmnet(x_train, y_train, alpha = 0, standardize = TRUE)
# Display the best lambda value
cv1$lambda.min
plot(cv1)
# Fit the final model on the training data
model1 <- glmnet(x_train, y_train, alpha = 0, lambda = cv1$lambda.min)
model3 <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_aic)
# Display regression coefficients
coef(model1)
model1_2 <- glmnet(as.matrix(x_train[,c(-3,-5,-6,-7)]), y_train, alpha = 0, lambda = cv1$lambda.min)
## According to the coefficients in model 1, we decided to discard Height,
## Defence, Physical and Shooting since their coefficients are less than 0.01
## and we got model1_2.
coef(model3)
model3_2 <- glmnet(as.matrix(x_train[,c(-3,-5,-7,-9)]), y_train, alpha = 0, lambda = lambda_aic)
## According to the coefficients in model 3, we decided to discard Height,
## Defence, Shooting and Right.foot.dummy since their coefficients are
## less than 0.01 and we got model3_2.

# Note that by default, the function glmnet() standardizes variables so that 
## their scales are comparable. However, the coefficients are always returned 
## on the original scale.

# Make predictions on the test data
predictions1 <- model1_2 %>% predict(as.matrix(x_test[,c(-3,-5,-6,-7)])) %>% as.vector()
predictions3 <- model3_2 %>% predict(as.matrix(x_test[,c(-3,-5,-7,-9)])) %>% as.vector()
# Model performance metrics
Rsquare_ridge_min = R2(predictions1, y_test)
Rsquare_ridge_AIC = R2(predictions3, y_test)


res <- glmnet(x_train, y_train, alpha = 0, lambda = lambdas_to_try, standardize = FALSE)
plot(res, xvar = "lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .5)


### Lasso Regression
aic2 <- c()
for (lambda in seq(lambdas_to_try)) {
  # Run model
  model <- glmnet(x_train, y_train, alpha = 1, lambda = lambdas_to_try[lambda], standardize = TRUE)
  # Extract coefficients and residuals (remove first row for the intercept)
  betas <- as.vector((as.matrix(coef(model))[-1, ]))
  resid <- y_train - (X_scaled %*% betas)
  # Compute hat-matrix and degrees of freedom
  ld <- lambdas_to_try[lambda] * diag(ncol(X_scaled))
  H <- X_scaled %*% solve(t(X_scaled) %*% X_scaled + ld) %*% t(X_scaled)
  df <- tr(H)
  # Compute information criteria
  aic2[lambda] <- nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df
}

# Plot information criteria against tried values of lambdas
plot(log_lambdas_to_try, aic2, col = "orange", type = "l",
     ylab = "Information Criterion", main ="Lasso Regression")
# Optimal lambdas according to AIC criteria
lambda_aic2 <- lambdas_to_try[which.min(aic2)]
lambda_aic2

# Find the best lambda using cross-validation
cv2 <- cv.glmnet(x_train, y_train, alpha = 1)
# Display the best lambda value
cv2$lambda.min
plot(cv2)
# Fit the final model on the training data
model2 <- glmnet(x_train, y_train, alpha = 1, lambda = cv2$lambda.min)
model4 <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_aic2)
# Dsiplay regression coefficients
coef(model2)
model2_2 <- glmnet(as.matrix(x_train[,c(-3,-5,-6,-7,-9)]), y_train, alpha = 1, lambda = cv2$lambda.min)
## According to the coefficients in model 2, we decided to discard Height,
## Defence, Physical, Shooting and Right.foot.dummy since their coefficients are less than 0.01
## and we got model2_2.
coef(model4)
model4_2 <- glmnet(as.matrix(x_train[,c(-3,-5,-6,-7)]), y_train, alpha = 1, lambda = lambda_aic2)
## According to the coefficients in model 4, we decided to discard Height,
## Defence, Physical and Shooting since their coefficients are less than 0.01
## and we got model4_2.
# Make predictions on the test data
predictions2 <- model2_2 %>% predict(x_test[,c(-3,-5,-6,-7,-9)]) %>% as.vector()
predictions4 <- model4_2 %>% predict(as.matrix(x_test[,c(-3,-5,-6,-7)])) %>% as.vector()
# Model performance metrics
Rsquare_lasso_min = R2(predictions2, y_test)
Rsquare_lasso_AIC = R2(predictions4, y_test)

res <- glmnet(x_train, y_train, alpha = 1, lambda = lambdas_to_try, standardize = FALSE)
plot(res, xvar = "lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .5)



## Summary
# Make predictions on the test data for the reduced model
data_train_2 = as.data.frame(cbind(y_train,x_train))
all_2 = lm(y_train ~ . , data = data_train_2)
predictions0 <- all_2 %>% predict(as.data.frame(x_test)) %>% as.vector()
Rsquare_linear_reduced = R2(predictions0, y_test)

RMSE_linear_reduced=RMSE(predictions0, y_test)
RMSE_ridge_min=RMSE(predictions1, y_test)
RMSE_ridge_AIC=RMSE(predictions3, y_test)
RMSE_lasso_min=RMSE(predictions2, y_test)
RMSE_lasso_AIC=RMSE(predictions4, y_test)

data.frame(Rsquare_linear_reduced, Rsquare_ridge_min, Rsquare_ridge_AIC, Rsquare_lasso_min, Rsquare_lasso_AIC)
data.frame(RMSE_linear_reduced, RMSE_ridge_min, RMSE_ridge_AIC, RMSE_lasso_min, RMSE_lasso_AIC)
## According to out of sample R-squared, linear regression seems to have the 
## best performance. But comparing Ridge and Lasso, Lasso regression with lambda
## selected by AIC as 0.001 has best performance (model 4_2).


## Visualize best model(model 4_2)
x_new <- data %>%
  select(Overall.Score, Weekly.Salary, Age, Goalkeeping, Right.foot.dummy) %>% as.matrix()
y <- data$Market.Value
res <- glmnet(x_new, y, alpha = 1, lambda = lambdas_to_try, standardize = FALSE)
plot(res, xvar = "lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_new), cex = .5)


## Evaluate model 4_2
x_test_new <- x_test[,c("Overall.Score", "Weekly.Salary", "Age", "Goalkeeping", "Right.foot.dummy")] 
Predicted <- exp(predict(model4_2, x_test_new))
Y_original <- exp(y_test)
Residual <- Y_original-Predicted
plot(Predicted, Residual, ylab="Residuals", xlab="Fitted Value") 
abline(0, 0)   


