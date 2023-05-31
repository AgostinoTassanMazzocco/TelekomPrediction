#RFE ROBUSTO RANDOM FOREST


load("~/Desktop/DATA MINING/train_test.RData")
require(caret)
library(tictoc)

set.seed(123)
folds <- createFolds(1:nrow(train), k = 5, list = TRUE, returnTrain = FALSE)


r_forest_grid <- function(grid, train, folds, vars) {
  y_true <- train$y
  train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
  train <- train[,c(vars,"y")]
  
  eval <- bind_cols(grid, tibble(val_err = NA, tr_err = NA))
  for (i in 1:nrow(grid)) {
    mtry <- grid[i,1]
    n_tree <- grid[i,2]
    errors <- lapply(folds, function(train, x) {
      fit_rf <- ranger(y ~ ., data = train[-x,], mtry = mtry, num.trees = n_tree)
      fitted_val <- expm1(predict(fit_rf, data = train[x,-ncol(train)])$predictions) 
      fitted_tr <- expm1(predict(fit_rf, data = train[-x, -ncol(train)])$predictions)
      val_err = sum( ( log1p(y_true[x]) - log1p(fitted_val))^2 )
      tr_err = sum( ( log1p(y_true[-x]) - log1p(fitted_tr))^2 )
      cat(paste0(i, " out of ", nrow(grid), "\n"))
      return(list(val_err = val_err, tr_err = tr_err))
    }
    ,train = train)
    
    val_errs <- vector()
    tr_errs <- vector()
    for (j in 1:5) {
      val_errs[j] <- errors[[j]][[1]]
      tr_errs[j] <- errors[[j]][[2]]
    }
    eval[i,3] <- mean(val_errs)
    eval[i,4] <- mean(tr_errs)
  }
  return(eval)
}



#Feature selection based on the result of the best parameters

step_rf_var_imp <- function(folds, train, vars, mtry, num.trees) {
  y_true <- train$y
  train <- train %>% mutate(across(starts_with("q"), .fns = log1p)) %>% mutate(y = log1p(y))
  train <- train[,c(vars,"y")]
  
  res <- lapply(folds, function(x, train, y_true, mtry, num.trees) {
    rf_fit <- ranger(y ~., data = train[-x,], mtry = mtry, num.trees = num.trees, importance = "impurity")
    yhat <- expm1(predict(rf_fit, data = train[x, -ncol(train)])$predictions)
    yhat_tr <- expm1(predict(rf_fit, data = train[-x, -ncol(train)])$predictions)
    vars_imp <- rf_fit$variable.importance
    val_err <- sum((log1p(y_true[x]) - log1p(yhat))^2)
    tr_err <- sum((log1p(y_true[-x]) - log1p(yhat_tr))^2)
    return(list(val_err = val_err, tr_err = tr_err, vars_imp = vars_imp))
  }, train = train, y_true = y_true, mtry = mtry, num.trees = num.trees)
  
  vars_imp <- bind_rows(res$Fold1[[3]], res$Fold2[[3]], res$Fold3[[3]], res$Fold4[[3]],
                        res$Fold5[[3]]) %>% colMeans() 
  rank <- vars_imp %>% sort(decreasing = T) %>% names()
  vars_sel <- rank[1:(length(rank)-round(0.1*length(rank)))]
  
  val_errs <- vector()
  tr_errs <- vector()
  for (i in 1:5) {
    val_errs[i] <- res[[i]][[1]]
    tr_errs[i] <- res[[i]][[2]]
  }
  return(list(vars = vars, vars_sel = vars_sel, assess = tibble(num_var = length(vars), val_error = mean(val_errs), tr_error = mean(tr_errs))))
}


tic()
#STEP1 
#TUNING 99 VARIABILI 
vars <- colnames(train[,-ncol(train)])
mtry <- c(10,15,20)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)
grid_results_step1 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)
best_comb <- arrange(grid_results_step1, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees
step1 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step1$vars
step1$vars_sel
step1$assess


#STEP2
#TUNING 89 VARIABILI:
vars <- step1$vars_sel
mtry <- c(10,15,20)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step2 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step2, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step2 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step2$vars
step2$vars_sel 
step2$assess

#STEP3
#TUNING 80 VARIABILI:

vars <- step2$vars_sel
mtry <- c(10,15,20)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step3 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step3, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step3 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step3$vars
step3$vars_sel
step3$assess


#STEP4
#TUNING 72 VARIABILI:
vars <- step3$vars_sel
mtry <- c(10,15,20)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step4 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step4, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step4 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step4$vars
step4$vars_sel
step4$assess



#STEP5
#TUNING 65 VARIABILI:
vars <- step4$vars_sel
mtry <- c(10,15,20)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step5 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step5, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step5 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step5$vars
step5$vars_sel
step5$assess



#STEP6
#TUNING 59 VARIABILI:
vars <- step5$vars_sel
mtry <- c(10,15,20)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step6 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step6, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step6 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step6$vars
step6$vars_sel
step6$assess


#STEP7
#TUNING 53 VARIABILI:
vars <- step6$vars_sel
mtry <- c(10,15,20)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step7 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step7, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step7 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step7$vars
step7$vars_sel
step7$assess




#STEP8
#TUNING 48 VARIABILI:
vars <- step7$vars_sel
mtry <- c(5,10,15)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step8 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step8, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step8 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)

step8$vars
step8$vars_sel
step8$assess



#STEP9
#TUNING 43 VARIABILI:
vars <- step8$vars_sel
mtry <- c(5,10,15)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step9 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step9, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step9 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)

step9$vars
step9$vars_sel
step9$assess

#STEP10
#TUNING 39 VARIABILI:
vars <- step9$vars_sel
mtry <- c(5,10,15)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step10 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step10, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step10 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)

step10$vars
step10$vars_sel
step10$assess


#STEP11
#TUNING 35 VARIABILI:
vars <- step10$vars_sel
mtry <- c(5,10,15)
n_trees <- c(1000, 1500, 2000)
grid <- expand.grid(mtry = mtry, n_trees = n_trees)

grid_results_step11 <- r_forest_grid(grid = grid, train = train, folds = folds, vars = vars)

best_comb <- arrange(grid_results_step11, val_err)[1,]
mtry_best <- best_comb$mtry
n_trees_best <- best_comb$n_trees

step11 <- step_rf_var_imp(folds = folds, train = train, vars = vars, mtry = mtry_best, num.trees = n_trees_best)
step11$vars
step11$vars_sel
step11$assess

toc()



res_fin <- bind_rows(step1$assess, step2$assess, step3$assess, step4$assess, step5$assess, step6$assess,
                     step7$assess, step8$assess, step9$assess, step10$assess, step11$assess)


rfe_res_fin <- list(step1 = step1, step2 = step2, step3 = step3, step4 = step4, step5 = step5, step6 = step6, step7 = step7,
                    step8 = step8, step9 = step9)


save(rfe_res_fin, file = "~/Desktop/DATA MINING/rfe_res_fin.RData")
