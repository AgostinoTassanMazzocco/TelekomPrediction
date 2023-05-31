library(dplyr)
library(ggplot2)

PATH <- "https://raw.githubusercontent.com/aldosolari/DM/master/docs/DATA/"
train <- read.csv(paste0(PATH,"train.csv"), sep="")
test <- read.csv(paste0(PATH,"test.csv"), sep="")
#fit <- lm(y ~ 1, train)
#yhat <- predict(fit, newdata = test)
#write.table(file="2575_previsione.txt", yhat, row.names = FALSE, col.names = FALSE)



#DATASET COMBINATO 
y <- train$y
n <- nrow(train)
m <- nrow(test)
combi <- bind_rows(train[,-ncol(train)], test)
combi <- combi %>% as_tibble() %>% mutate_at(.vars = c("vas1", "vas2", "payment.method", "gender", "status", "tariff.plan", "activ.area", "activ.chan"), factor) 



#ANALISI ESPLORATIVA
levels(combi$activ.area)
table(combi$activ.area)
#activ.area ha un livello (=0) con una sola osservazione. In teoria (dall'infofile) ha 
#solo 4 livelli, invece che 5
#Imputazione di quel valore con il livello più frequente all'interno della variabile, quindi
#con activ.area = 0
combi$activ.area[which(combi$activ.area==0)] <- 1
combi$activ.area <-  factor(combi$activ.area, levels = levels(combi$activ.area)[-1])

#Ci sono alcuni valori negativi nelle variabili numeriche che devono essere positive
#in q03.in.dur.tot e q09.out.dur.peak
#q03.in.dur.tot
subs <- combi %>% filter(q03.in.dur.tot < 0) %>% select(ends_with("in.dur.tot"), -q03.in.dur.tot) %>% rowMeans() 
combi[which(combi$q03.in.dur.tot < 0), "q03.in.dur.tot"] <- as.integer(subs)
#q09.out.dur.peak
subs <- combi %>% filter(q09.out.dur.peak < 0) %>% select(ends_with("out.dur.peak"), -q09.out.dur.peak) %>% rowMeans() 
combi[which(combi$q09.out.dur.peak < 0), "q09.out.dur.peak"] <- as.integer(subs)



#ESTRAGGO LE RISPOSTE DEL TEST SET 

#Lettura dei veri dati
file <- "~/Desktop/telekom-www/telekom.training.dat"
train_true <- read.table(file) %>% as_tibble()
file <- "~/Desktop/telekom-www/telekom.test.dat"
test_true <- read.table(file) %>% as_tibble()
combi_true <- bind_rows(train_true, test_true)
#Aggiungo la variabile risposta y
combi_true <- combi_true %>% mutate(y = q10.out.dur.peak + q10.out.dur.offpeak, .keep = "unused")
#Elimino le colonne in più
diff <- setdiff(colnames(combi_true), colnames(train))
combi_true <- combi_true %>% select(-all_of(diff))
#Trasoformo le variabili in factor
combi_true <- combi_true %>% as_tibble() %>% mutate_at(.vars = c("vas1", "vas2", "payment.method", "gender", "status", "tariff.plan", "activ.area", "activ.chan"), factor) 
combi_true$activ.area[which(combi_true$activ.area==0)] <- 1
combi_true$activ.area <-  factor(combi_true$activ.area, levels = levels(combi_true$activ.area)[-1])
subs <- combi_true %>% filter(q03.in.dur.tot < 0) %>% select(ends_with("in.dur.tot"), -q03.in.dur.tot) %>% rowMeans() 
combi_true[which(combi_true$q03.in.dur.tot < 0), "q03.in.dur.tot"] <- as.integer(subs)
subs <- combi_true %>% filter(q09.out.dur.peak < 0) %>% select(ends_with("out.dur.peak"), -q09.out.dur.peak) %>% rowMeans() 
combi_true[which(combi_true$q09.out.dur.peak < 0), "q09.out.dur.peak"] <- as.integer(subs)



train <- combi[1:n,]
train <- bind_cols(train, y)
colnames(train) <- c(colnames(train)[1:ncol(train)-1], "y")
test <- combi[(n+1):(n+m),]
#Estraggo la risposta di tutte le osservazioni
y_true <- combi_true$y
#Estraggo da combi_true le osservazioni che appartengono al test set creato dal prof
test_y <- left_join(test, distinct(combi_true), by = colnames(test))
test_y <- test_y[sample(1:15374, 15309, F),]


save(test_y, train, test, file = "train_test.RData")
load("~/Desktop/train_test.RData")
