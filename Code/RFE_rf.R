#RFE ALOGRYTHM 

load("~/Desktop/train_test.RData")
y_true <- train$y

require(caret)
set.seed(123)
folds <- createFolds(1:nrow(train), k = 10, list = TRUE, returnTrain = FALSE)


my_rfe <- function(train, y_true, folds) {
  res <- list(vars = list(), perf = tibble(n.vars = 99:1, val_mean_err = NA, tr_mean_err = NA, val_err = NA, tr_err = NA))
  train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
  res_full <- lapply(folds, function(x, train, y_true) {
    fit_rf_full <- ranger(y ~ ., data = train[-x,], importance = "impurity")
    var_imp <- fit_rf_full$variable.importance
    fitted_val <- expm1(predict(fit_rf_full, data = train[x,-ncol(train)])$predictions) 
    fitted_tr <- expm1(predict(fit_rf_full, data = train[-x, -ncol(train)])$predictions)
    val_mean_err = mean( ( log1p(y_true[x]) - log1p(fitted_val))^2 )
    tr_mean_err = mean( ( log1p(y_true[-x]) - log1p(fitted_tr))^2 )
    val_err = sum( ( log1p(y_true[x]) - log1p(fitted_val))^2 )
    tr_err = sum( ( log1p(y_true[-x]) - log1p(fitted_tr))^2 )
    return(list(err = c(val_err, tr_err, val_mean_err, tr_mean_err), var_imp <- var_imp))
    
  }, train = train, y_true = y_true)
  vars_im <- bind_rows(res_full$Fold01[[2]], res_full$Fold02[[2]], res_full$Fold03[[2]], res_full$Fold04[[2]],
                       res_full$Fold05[[2]], res_full$Fold06[[2]], res_full$Fold07[[2]], res_full$Fold08[[2]],
                       res_full$Fold09[[2]], res_full$Fold10[[2]]) %>% colMeans() %>% sort(decreasing = T) %>% names()
  res$vars[[99]] <- vars_im
  vars <- vars_im[-99]
  errs <- res_full[[1]][[1]]
  for (i in 2:10) {
    errs <- c(errs, res_full[[i]][[1]])
  }
  res$perf[res$perf$n.vars == 99,2] <- round(mean(errs[seq(3, 40, by = 4)]), 3)
  res$perf[res$perf$n.vars == 99,3] <- round(mean(errs[seq(4, 40, by = 4)]), 3)
  res$perf[res$perf$n.vars == 99,4] <- round(mean(errs[seq(1, 40, by = 4)]), 3)
  res$perf[res$perf$n.vars == 99,5] <- round(mean(errs[seq(2, 40, by = 4)]), 3)
  for (i in 98:1) {
    res$vars[[i]] <- vars
    train <- train[,c(vars,"y")]
    res_par <- lapply(folds, function(x, train, y_true) {
      fit_rf <- ranger(y ~ ., data = train[-x,], importance = "impurity")
      var_imp <- fit_rf$variable.importance
      fitted_val <- expm1(predict(fit_rf, data = train[x,-ncol(train)])$predictions) 
      fitted_tr <- expm1(predict(fit_rf, data = train[-x, -ncol(train)])$predictions)
      val_mean_err = mean( ( log1p(y_true[x]) - log1p(fitted_val))^2 )
      tr_mean_err = mean( ( log1p(y_true[-x]) - log1p(fitted_tr))^2 )
      val_err = sum( ( log1p(y_true[x]) - log1p(fitted_val))^2 )
      tr_err = sum( ( log1p(y_true[-x]) - log1p(fitted_tr))^2 )
      return(list(err = c(val_err, tr_err, val_mean_err, tr_mean_err), var_imp <- var_imp))
    }, train = train, y_true = y_true)
    vars_im <- bind_rows(res_par$Fold01[[2]], res_par$Fold02[[2]], res_par$Fold03[[2]], res_par$Fold04[[2]],
                         res_par$Fold05[[2]], res_par$Fold06[[2]], res_par$Fold07[[2]], res_par$Fold08[[2]],
                         res_par$Fold09[[2]], res_par$Fold10[[2]]) %>% colMeans() %>% sort(decreasing = T) %>% names()
    vars <- vars_im[-i]
    errs <- res_par[[1]][[1]]
    for (j in 2:10) {
      errs <- c(errs, res_par[[j]][[1]])
    }
    res$perf[res$perf$n.vars == i,2] <- round(mean(errs[seq(3, 40, by = 4)]), 3)
    res$perf[res$perf$n.vars == i,3] <- round(mean(errs[seq(4, 40, by = 4)]), 3)
    res$perf[res$perf$n.vars == i,4] <- round(mean(errs[seq(1, 40, by = 4)]), 3)
    res$perf[res$perf$n.vars == i,5] <- round(mean(errs[seq(2, 40, by = 4)]), 3)
    cat(paste0(i, "\b"))
  }
  return(res)
}


tic()
rfe_rf <- my_rfe(train = train, y_true = y_true, folds = folds)
toc()


best_subset <- rfe_rf$perf %>% arrange(val_mean_err) %>% pull(n.vars)
best_subset <- best_subset[1]
best_subset <- rfe_rf$vars[[best_subset]]

#GBM
tic()
load("~/Desktop/train_test.RData")
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
train <- train[,c(best_subset, "y")]
fit_gbm <- gbm::gbm(y ~ ., data = train, distribution = "gaussian", n.trees = 5000, shrinkage = 0.1, 
                      interaction.depth = 3, n.minobsinnode = 10, cv.folds = 10)
best <- which.min(fit_gbm$cv.error)
fit_gbm$cv.error[best]
best_iter <- gbm.perf(fit_gbm, method = "cv")
toc()


#TUNING LEARNING RATE

load("~/Desktop/train_test.RData")
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
train <- train[,c(best_subset, "y")]

hyper_grid <- expand.grid(
  learning_rate = c(0.1, 0.05, 0.01, 0.005, 0.001),
  MSE = NA,
  trees = NA,
  time = NA
)

tic()
for(i in 1:nrow(hyper_grid)) {
  
  # fit gbm
  set.seed(123)  # for reproducibility
  train_time <- system.time({
    m <- gbm::gbm(
      formula = y ~ .,
      data = train,
      distribution = "gaussian",
      n.trees = 5000, 
      shrinkage = hyper_grid$learning_rate[i], 
      interaction.depth = 3, 
      n.minobsinnode = 10,
      cv.folds = 10 
    )
  })
  
  # add SSE, trees, and training time to results
  hyper_grid$MSE[i]  <- min(m$cv.error)
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
}
toc()

# results

best_LR <- arrange(hyper_grid, MSE)[1,1]


#Valutazione su vero test set

load("~/Desktop/train_test.RData")
train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
train <- train[,c(best_subset, "y")]
test_y <- test_y %>% mutate(across(starts_with("q"), .fns = log1p)) 
test_y <- test_y[,c(best_subset, "y")]
tic()
gbm_2 <- gbm::gbm(y ~ ., data = train, distribution = "gaussian", shrinkage = best_LR, 
                  n.trees = 6000)
toc()
gbm_2$var.levels
yhat <- expm1(predict(gbm_2, newdata = test_y[,-ncol(test_y)]))
sum((log1p(test_y$y) - log1p(yhat))^2)





