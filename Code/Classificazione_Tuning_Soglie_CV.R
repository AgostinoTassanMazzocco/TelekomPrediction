load("~/Desktop/DATA MINING/train_test.RData")
load("~/Desktop/DATA MINING/vars_selected.RData")
load("~/Desktop/DATA MINING/vars_logit_selected.RData")

#Classificazione con tuning sulle soglie 
require(caret)
set.seed(123)
folds <- createFolds(1:nrow(train), k = 5, list = TRUE, returnTrain = FALSE)


sills <- c(0.2,0.3,0.4,0.5,0.6,0.7,0.8)

two_step_sill_tuning <- function(folds, sills, train, vars_logit_selected, vars_selected, test_y) {
  res <- tibble(sill = sills, val_error = NA, val_mean_error = NA, tr_error = NA, tr_mean_error = NA)
  for (i in 1:length(sills)) {
    #Classificazione
    res1 <- lapply(folds, function(x, train, sills, vars_logit_selected, vars_selected) {
      train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
      train_bin <- train %>% mutate(y = case_when(y > 0 ~ 1, y == 0 ~ 0)) 
      train_bin$y <- as.factor(train_bin$y)
      fit <- glm(y ~ ., data = train_bin[-x, c(vars_logit_selected, "y")], family = binomial(link = "logit"))
      val_fitted <- predict(fit, newdata = train[x, vars_logit_selected], type = "response")
      tr_fitted <- predict(fit, type = "response")
      
      id_val_0 <- which(val_fitted <= sills[i])
      id_val_1 <- which(val_fitted > sills[i])
      
      id_tr_0 <- which(tr_fitted <= sills[i])
      id_tr_1 <- which(tr_fitted > sills[i])
      
      #Regressione 1
      fit_lm_0 <- lm(y ~., data = train[-x, c(vars_selected, "y")][id_tr_0,])
      fit_lm_1 <- lm(y ~., data = train[-x, c(vars_selected, "y")][id_tr_1,])
      #previsioni
      val_fitted_lm_0 <- predict(fit_lm_0, newdata = train[x, vars_selected][id_val_0,]) 
      val_fitted_lm_1 <- predict(fit_lm_1, newdata = train[x, vars_selected][id_val_1,])
      
      tr_fitted_lm_0 <- predict(fit_lm_0) 
      tr_fitted_lm_1 <- predict(fit_lm_1)
      
      prev_lm <- vector()
      prev_lm[id_val_0] <- val_fitted_lm_0
      prev_lm[id_val_1] <- val_fitted_lm_1
      
      prev_tr_lm <- vector()
      prev_tr_lm[id_tr_0] <- tr_fitted_lm_0
      prev_tr_lm[id_tr_1] <- tr_fitted_lm_1
      
      
      #Regressione 2
      fit_rf_0 <- ranger(y ~., data = train[-x,][id_tr_0,])
      fit_rf_1 <- ranger(y ~., data = train[-x,][id_tr_1,])
      #previsioni
      val_fitted_rf_0 <- predict(fit_rf_0, data = train[x,-ncol(train)][id_val_0,])$predictions 
      val_fitted_rf_1 <- predict(fit_rf_1, data = train[x,-ncol(train)][id_val_1,])$predictions
      
      tr_fitted_rf_0 <- predict(fit_rf_0, data = train[-x,-ncol(train)][id_tr_0,])$predictions
      tr_fitted_rf_1 <- predict(fit_rf_1, data = train[-x,-ncol(train)][id_tr_1,])$predictions
      
      prev_rf <- vector()
      prev_rf[id_val_0] <- val_fitted_rf_0
      prev_rf[id_val_1] <- val_fitted_rf_1
      
      prev_tr_rf <- vector()
      prev_tr_rf[id_tr_0] <- tr_fitted_rf_0
      prev_tr_rf[id_tr_1] <- tr_fitted_rf_1
      
      return(list(prev_validation = tibble(lm = prev_lm, rf = prev_rf))) 
      }, train = train, sills = sills, vars_logit_selected = vars_logit_selected, vars_selected = vars_selected)
      
      meta_train <- res1[[1]][[1]]
      for (j in 2:5) {
        meta_train <- bind_rows(meta_train, res1[[j]][[1]])
      }
      meta_train <- meta_train %>% bind_cols(tibble(id = unlist(folds))) %>% arrange(id) %>% 
        dplyr::select(-id) %>% bind_cols(y = train$y)
      colnames(meta_train) <- c("lm", "rf", "y")
      meta_train <- meta_train %>% mutate(y = log1p(y))
      meta_train[meta_train$lm < 0, "lm"] <- 0
      fit = lm(y ~ 0 + lm + rf, meta_train)
      
      #Verifica sul test set
      #Classificazione:
      
      y_true = test_y$y
      y_train = train$y
      train2 <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
      test_y2 <- test_y %>% mutate(across(starts_with("q"), .fns = log1p))
      train_bin <- train %>% mutate(y = case_when(y > 0 ~ 1, y == 0 ~ 0)) 
      train_bin$y <- as.factor(train_bin$y)
      test_y_bin <- test_y %>% mutate(y = case_when(y > 0 ~ 1, y == 0 ~ 0)) 
      test_y_bin$y <- as.factor(test_y_bin$y)
      
      fit_class <- glm(y ~ ., data = train_bin[,c(vars_logit_selected, "y")], family = binomial(link = "logit"))
      yhat1 <- predict(fit_class, newdata = test_y2[,-ncol(test_y2)][,vars_logit_selected], type = "response")
      tr_fitted <- predict(fit_class, type = "response")
      
      id_0 <- which(yhat1 <= sills[i])
      id_1 <- which(yhat1 > sills[i])
      
      id_tr_0 <- which(tr_fitted <= sills[i])
      id_tr_1 <- which(tr_fitted > sills[i])
      
      #Regressione 1
      fit_lm_0 <- lm(y ~., data = train2[, c(vars_selected, "y")][id_tr_0,])
      fit_lm_1 <- lm(y ~., data = train2[, c(vars_selected, "y")][id_tr_1,])
      #previsioni
      yhat_lm_0 <- predict(fit_lm_0, newdata = test_y2[,-ncol(test_y)][,vars_selected][id_0,]) 
      yhat_lm_1 <- predict(fit_lm_1, newdata = test_y2[,-ncol(test_y)][,vars_selected][id_1,])
      
      tr_fitted_lm_0 <- predict(fit_lm_0) 
      tr_fitted_lm_1 <- predict(fit_lm_1)
      
      prev_lm <- vector()
      prev_lm[id_0] <- yhat_lm_0
      prev_lm[id_1] <- yhat_lm_1
      prev_lm[prev_lm < 0] <- 0
      
      prev_tr_lm <- vector()
      prev_tr_lm[id_tr_0] <- tr_fitted_lm_0
      prev_tr_lm[id_tr_1] <- tr_fitted_lm_1
      prev_tr_lm[prev_tr_lm < 0] <- 0
      
      #Regressione 2
      fit_rf_0 <- ranger(y ~., data = train2[id_tr_0,])
      fit_rf_1 <- ranger(y ~., data = train2[id_tr_1,])
      #previsioni
      yhat_rf_0 <- predict(fit_rf_0, data = test_y2[id_0,-ncol(test_y)])$predictions 
      yhat_rf_1 <- predict(fit_rf_1, data = test_y2[id_1,-ncol(test_y)])$predictions
      
      tr_fitted_rf_0 <- predict(fit_rf_0, data = train2[id_tr_0,-ncol(train)])$predictions
      tr_fitted_rf_1 <- predict(fit_rf_1, data = train2[id_tr_1,-ncol(train)])$predictions
      
      prev_rf <- vector()
      prev_rf[id_0] <- yhat_rf_0
      prev_rf[id_1] <- yhat_rf_1
      
      prev_tr_rf <- vector()
      prev_tr_rf[id_tr_0] <- tr_fitted_rf_0
      prev_tr_rf[id_tr_1] <- tr_fitted_rf_1
      
      yhat_fin <- expm1(predict(fit, newdata = data.frame(lm = prev_lm, rf = prev_rf)))
      yhat_train <- expm1(predict(fit, newdata = data.frame(lm = prev_tr_lm, rf = prev_tr_rf)))
      
      res[i,2] <- sum((log1p(y_true) - log1p(yhat_fin))^2) 
      res[i,3] <- mean((log1p(y_true) - log1p(yhat_fin))^2) 
      res[i,4] <- sum((log1p(y_train) - log1p(yhat_train))^2) 
      res[i,5] <- mean((log1p(y_train) - log1p(yhat_train))^2) 
      print(sills[i])
  }
  return(res)
}


tic()
result <- two_step_sill_tuning(folds = folds, sills = sills, train = train, 
                               vars_logit_selected = vars_logit_selected, 
                               vars_selected = vars_selected, test_y = test_y)
toc()




load("~/Desktop/DATA MINING/train_test.RData")
load("~/Desktop/DATA MINING/vars_selected.RData")
load("~/Desktop/DATA MINING/vars_logit_selected.RData")

y_true <- test_y$y
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
test_y <- test_y %>% mutate(across(starts_with("q"), .fns = log1p)) 
train_bin <- train %>% mutate(y = case_when(y > 0 ~ 1, y == 0 ~ 0)) 
train_bin$y <- as.factor(train_bin$y)
test_y_bin <- test_y %>% mutate(y = case_when(y > 0 ~ 1, y == 0 ~ 0)) 
test_y_bin$y <- as.factor(test_y_bin$y)


fit1 <- glm(y ~., data = train_bin[,c(vars_logit_selected, "y")])
probs <- predict(fit1, newdata = test_y[,vars_logit_selected], type = "response")
id_0 <- which(probs <= 0.2)
id_1 <- which(probs > 0.2)

id_tr_0 <- which()

lapply(folds, function(x, train, test_y) {
  fit_lm_0 <- lm(y ~., data = train[])
}) 