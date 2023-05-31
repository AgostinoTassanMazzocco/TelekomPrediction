load("~/Desktop/DATA MINING/train_test.RData")
load("~/Desktop/DATA MINING/vars_logit_selected.RData")
load("~/Desktop/DATA MINING/vars_selected.RData")


y_true <- test_y$y
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
y_train <- train$y



#CLASSIFICAZIONE
tic()
errors <- lapply(folds, function(x, train, y_train, vars_logit_selected, vars_selected) {
  train <- train[-x,]
  val <- train[x,]
  train_bin <- train %>% mutate(y = case_when(y > 0 ~ 1, y == 0 ~ 0)) 
  train_bin$y <- as.factor(train_bin$y)
  
  #Classificazione:
  fit <- glm(y ~ ., data = train_bin[,c(vars_logit_selected, "y")], family = binomial(link = "logit"))
  val_fitted <- predict(fit, newdata = val[,vars_logit_selected], type = "response")
  tr_fitted <- predict(fit, type = "response")
  
  id_val_0 <- which(val_fitted <= 0.6)
  id_val_1 <- which(val_fitted > 0.6)
  
  id_train_0 <- which(tr_fitted <= 0.6)
  id_train_1 <- which(tr_fitted > 0.6)
  
  
  folds_0 <- createFolds(1:length(id_train_0), k = 10, list = TRUE, returnTrain = FALSE)
  folds_1 <- createFolds(1:length(id_train_1), k = 10, list = TRUE, returnTrain = FALSE)
  
  res_0 <- lapply(folds_0, function(j, train, vars_selected, y_train) {
    fit_lm_0 <- lm(y ~., data = train[id_train_0,][-j,c(vars_selected, "y")])
    yhat_1 <- predict(fit_lm_0, newdata = train[id_train_0,][j,vars_selected])
    
    fit_rf_0 <- ranger(y ~., data = train[id_train_0,][-j,], mtry = 20, num.trees = 1000)
    yhat_2 <- predict(fit_rf_0, data = train[id_train_0,][j,])$predictions
    
    fit_boost_0 <- xgboost(y ~., data = model.matrix(y ~., train[id_train_0,][-j,])[,-1], label = y_train[id_train_0][-j], 
                           params = list(max_depth = 6, eta = 0.005, gamma = 0,
                                         colsample_bytree = 0.6,
                                         min_child_weight = 1,
                                         subsample = 1), 
                           nrounds = 3000)
    yhat_3 <- predict(fit_boost_0, newdata = model.matrix(y ~., train[id_train_0,][j,])[,-1])
    
    return(list(prevs = tibble(lm = yhat_1, rf = yhat_2, boost = yhat_3)))
  }, train = train, vars_selected = vars_selected, y_train = y_train)
  
  
  
  res_1 <- lapply(folds_1, function(j, train, vars_selected, y_train) {
    fit_lm_1 <- lm(y ~., data = train[id_train_1,][-j,c(vars_selected, "y")])
    yhat_1 <- predict(fit_lm_1, newdata = train[id_train_1,][j,vars_selected])
    
    fit_rf_1 <- ranger(y ~., data = train[id_train_1,][-j,], mtry = 20, num.trees = 1000)
    yhat_2 <- predict(fit_rf_1, data = train[id_train_1,][j,])$predictions
    
    fit_boost_1 <- xgboost(y ~., data = model.matrix(y ~., train[id_train_1,][-j,])[,-1], label = y_train[id_train_1][-j], 
                           params = list(max_depth = 6, eta = 0.005, gamma = 0,
                                         colsample_bytree = 0.6,
                                         min_child_weight = 1,
                                         subsample = 1), 
                           nrounds = 3000)
    yhat_3 <- predict(fit_boost_1, newdata = model.matrix(y ~., train[id_train_1,][j,])[,-1])
    
    return(list(prevs = tibble(lm = yhat_1, rf = yhat_2, boost = yhat_3)))
  }, train = train, vars_selected = vars_selected, y_train = y_train)
  
  #Meta train 0
  meta_train_0 <- res_0[[1]][[1]]
  for (j in 2:10) {
    meta_train_0 <- bind_rows(meta_train_0, res_0[[j]][[1]])
  }
  meta_train_0 <- meta_train_0 %>% bind_cols(tibble(id = unlist(folds_0))) %>% arrange(id) %>% 
    dplyr::select(-id) %>% bind_cols(y = y_train[x][id_train_0])
  colnames(meta_train_0) <- c("lm", "rf", "boost", "y")
  meta_train_0 <- meta_train_0 %>% mutate(y = log1p(y))
  meta_train_0[meta_train_0$lm < 0, "lm"] <- 0
  fit_0 = lm(y ~ 0 + lm + rf + boost, meta_train_0) 
  
  #Meta_train_1
  meta_train_1 <- res_1[[1]][[1]]
  for (j in 2:10) {
    meta_train_1 <- bind_rows(meta_train_1, res_1[[j]][[1]])
  }
  meta_train_1 <- meta_train_1 %>% bind_cols(tibble(id = unlist(folds_1))) %>% arrange(id) %>% 
    dplyr::select(-id) %>% bind_cols(y = y_train[x][id_train_1])
  colnames(meta_train_1) <- c("lm", "rf", "boost", "y")
  meta_train_1 <- meta_train_1 %>% mutate(y = log1p(y))
  meta_train_1[meta_train_1$lm < 0, "lm"] <- 0
  fit_1 = lm(y ~ 0 + lm + rf + boost, meta_train_1) 
  
  #Fitting modelli e previsioni sul validation (0)
  fit_lm_0 <- lm(y ~., data = train[id_train_0,c(vars_selected, "y")])
  yhat_0_1 <- predict(fit_lm_0, newdata = val[id_val_0,vars_selected])
  fit_rf_0 <- ranger(y ~., data = train[id_train_0,], mtry = 20, num.trees = 1000)
  yhat_0_2 <- predict(fit_rf_0, data = val[id_val_0,])$predictions
  fit_boost_0 <- xgboost(y ~., data = model.matrix(y ~., train[id_train_0,])[,-1], label = y_train[id_train_0], 
                         params = list(max_depth = 6, eta = 0.005, gamma = 0,
                                       colsample_bytree = 0.6,
                                       min_child_weight = 1,
                                       subsample = 1), 
                         nrounds = 3000)
  yhat_0_3 <- predict(fit_boost_0, newdata = model.matrix(y ~., val[id_val_0,])[,-1])
  
  prev_0 <- expm1(predict(fit_0, newdata = data.frame(lm = yhat_0_1, rf = yhat_0_2, boost = yhat_0_3)))
  
  #Fitting modelli e previsioni sul validation (1)
  fit_lm_1 <- lm(y ~., data = train[id_train_1,c(vars_selected, "y")])
  yhat_1_1 <- predict(fit_lm_1, newdata = val[id_val_1,vars_selected])
  fit_rf_1 <- ranger(y ~., data = train[id_train_1,], mtry = 20, num.trees = 1000)
  yhat_1_2 <- predict(fit_rf_1, data = val[id_val_1,])$predictions
  fit_boost_1 <- xgboost(y ~., data = model.matrix(y ~., train[id_train_1,])[,-1], label = y_train[id_train_1], 
                         params = list(max_depth = 6, eta = 0.005, gamma = 0,
                                       colsample_bytree = 0.6,
                                       min_child_weight = 1,
                                       subsample = 1), 
                         nrounds = 3000)
  yhat_1_3 <- predict(fit_boost_1, newdata = model.matrix(y ~., val[id_val_1,])[,-1])
  
  prev_1 <- expm1(predict(fit_1, newdata = data.frame(lm = yhat_1_1, rf = yhat_1_2, boost = yhat_1_3)))
  
  prev_fin <- vector()
  prev_fin[id_val_0] <- prev_0
  prev_fin[id_val_1] <- prev_1
  
  err <- sum((y_train[x] - log1p(prev_fin))^2)
  return(err)
  
}, train = train, y_train = y_train, vars_logit_selected = vars_logit_selected, vars_selected = vars_selected)
toc()
mean(unlist(errors))





