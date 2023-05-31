#ENSEMBLE
load("~/Desktop/DATA MINING/train_test.RData")

require(caret)
library(xgboost)

tic()
set.seed(123)
folds <- createFolds(1:nrow(train), k = 5, list = TRUE, returnTrain = FALSE)


#fit del random forest
load("~/Desktop/DATA MINING/train_test.RData")
y_true <- test_y$y
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
test_y <- test_y %>% mutate(across(starts_with("q"), .fns = log1p))

prev2 <- lapply(folds, function(x, train) {
  fit <- ranger(y ~., data = train[-x,], mtry = 20, num.trees = 1000)
  yhat <- predict(fit, data = train[x,-ncol(train)])$predictions
  return(yhat)
}, train = train)


#Fit boosting
load("~/Desktop/DATA MINING/train_test.RData")
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
y_train <- train$y
train <- model.matrix(y ~., train)[,-1]



prev3 <- lapply(folds, function(x, train, y_train) {
  fit <- xgboost(data = train[-x,], label = y_train[-x],
                 params = list(max_depth = 6, eta = 0.005, gamma = 0,
                               colsample_bytree = 0.5,
                               min_child_weight = 1,
                               subsample = 1), 
                 nrounds = 4000)
  yhat <- predict(fit, newdata = train[x,])
  return(yhat)
}, train = train, y_train = y_train)

#Ensemble
load("~/Desktop/DATA MINING/train_test.RData")
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
test_y <- test_y %>% mutate(across(starts_with("q"), .fns = log1p)) 

z2 <- unlist(prev2)
z3 <- unlist(prev3)

meta_train <- data.frame(z2 = z2, z3 = z3, id = unlist(folds))
meta_train <- meta_train %>% arrange(id) %>% dplyr::select(-id) %>% bind_cols(y = train$y)
fit_meta_lm = lm(y ~ 0 + z2 + z3, meta_train)



load("~/Desktop/DATA MINING/train_test.RData")
y_true <- test_y$y
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
test_y <- test_y %>% mutate(across(starts_with("q"), .fns = log1p))
fit2 <- ranger(y ~., data = train, mtry = 20, num.trees = 1000)
yhat2 <- predict(fit2, test_y[,-ncol(test_y)])$predictions

load("~/Desktop/DATA MINING/train_test.RData")
y_true <- test_y$y
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
test_y <- test_y %>% mutate(across(starts_with("q"), .fns = log1p))
y_train <- train$y
train <- model.matrix(y ~., train)[,-1]
test_y <- model.matrix(y ~., test_y)[,-1]

fit3 <- xgboost(data = train, label = y_train,
                params = list(max_depth = 6, eta = 0.005, gamma = 0,
                              colsample_bytree = 0.5,
                              min_child_weight = 1,
                              subsample = 1), 
                nrounds = 4000)
yhat3 <- predict(fit3, newdata = test_y)

yhat <- predict(fit_meta_lm, newdata = data.frame(z2 = yhat2, z3 = yhat3))
yhat[yhat<0] <- 0
yhat <- expm1(yhat)
err_meta_lm <- sum((log1p(y_true) - log1p(yhat))^2)
toc()


#60250




