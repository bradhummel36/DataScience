options(warn = -1)

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")

library(caret)
library(dplyr)
library(tidyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(lattice)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(RColorBrewer)


#################################################
# Load data and look at it                      #
#################################################

data <- read.csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
  header=FALSE
)

glimpse(data)


#################################################
# Cleanse data                                  #
#################################################

# Add column headers
names(data) <- c("age", "sex", "cp", "trestbps", "choi", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thai", "num")
head(data)

# Set any nums > 1 to 1, check class
data$num[data$num > 1] <- 1
summary(data)
sapply(data, class)

# Coerce attributes to appropriate type
data <- transform(
  data,
  age=as.integer(age),
  sex=as.factor(sex),
  cp=as.factor(cp),
  trestbps=as.integer(trestbps),
  choi=as.integer(choi),
  fbs=as.factor(fbs),
  restecg=as.factor(restecg),
  thalach=as.integer(thalach),
  exang=as.factor(exang),
  oldpeak=as.numeric(oldpeak),
  slope=as.factor(slope),
  ca=as.factor(ca),
  thai=as.factor(thai),
  num=as.factor(num)
)

# Check set again
sapply(data, class)

summary(data)

# Change any data that is a "?" to NA
data[ data == "?"] <- NA
colSums(is.na(data))

summary(data)

# Replace NAs for a certain predictors, sum NAs
data$thai[which(is.na(data$thai))] <- as.factor("3.0")
data <- data[!(data$ca %in% c(NA)),]
colSums(is.na(data))

summary(data)

# Coerce a couple of predictors to factor to remove rows
data$ca <- factor(data$ca)
data$thai <- factor(data$thai)
summary(data)

# Data Exploration

# Plot proportion of disease present vs disease not present
data %>%
  ggplot(aes(x = num)) + 
  geom_histogram(stat = 'count', fill = "steelblue") +
  theme_bw()

# Determine present vs not present
data_nrows <- nrow(data)
data_nrows

data_present <- sum(as.numeric(data$num) == 2)
data_present

data_not_present <- data_nrows - data_present
data_not_present

data_present_percent <- data_present / data_nrows
data_present_percent

data_not_present_percent <- 1 - data_present_percent
data_not_present_percent

# Plot a few predictors
data %>%
  gather(-sex, -cp, -fbs, -restecg, -exang, -slope, -ca, -thai, -num, key = "var", value = "value") %>%
  ggplot(aes(x = value, y = ..count.. , colour = num)) +
  scale_color_manual(values=c("#0080FF", "#FF0000"))+
  geom_density() +
  facet_wrap(~var, scales = "free",  nrow = 2) +
  theme_bw()



#################################################
# Partition data                                #
#################################################

set.seed(123, sample.kind="Rounding")

# Create train and test sets
test_index <- createDataPartition(y = data$num, times = 1, p = 0.25, list = FALSE)

train_set <- data[-test_index,]
test_set <- data[test_index,]

dt_train_set <- data[-test_index,]
dt_test_set <- data[test_index,]



#################################################
# Decision Tree                                 #
#################################################

# Create a full Design Tree
dt_full <- rpart(num ~ . , data = dt_train_set, method = "class", cp = 0)

# Print the complexity parameter table to help select the decision
# tree that minimizes misclassification error
printcp(dt_full)

# Plot the cp
plotcp(dt_full, lty = 3, col = 2, upper = "splits")

# Find the best cp to prune based on xerror
bestcp <- dt_full$cptable[which.min(dt_full$cptable[,"xerror"]),"CP"]
bestcp

# Prune the bestcp
dt_pruned <- prune(dt_full, cp = bestcp)

summary(dt_pruned)

# Predict based on the pruned decision tree
dt_predict <- predict(dt_pruned, dt_test_set, type = "class")

# Calculate confusion matrix and plot it
table <- data.frame(confusionMatrix(dt_predict, test_set$num)$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "green", bad = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))

# Calculate the classification error, subtract from 1 to get accuracy
# Sum up the false positives and false negatives and divide by the total of the confusion
# matrix.
dt_accuracy <- 1 - round((table$Freq[2] + table$Freq[3]) / (table$Freq[1] + table$Freq[2] + table$Freq[3] + table$Freq[4]),3)
dt_accuray



#################################################
# Random Forest                                 #
#################################################

# Train using random forest default
#   mtry = floor(sqrt(13)) = 3
#   ntree = 500
#   nodesize = 1

set.seed(123, sample.kind="Rounding")

# Set train control for tuning parameters
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")

rf_default <- train(num ~ .,
                    data = train_set,
                    method = "rf",
                    metric = "Accuracy",
                    trControl = trControl)

rf_default

# Do prediction with the default fit on test set
pred_default <-predict(rf_default, test_set)

# Calculate confusion matrix and plot it
table <- data.frame(confusionMatrix(pred_default, test_set$num)$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "green", bad = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))

# Calculate the classification error, subtract from 1 to get accuracy
# Sum up the false positives and false negatives and divide by the total of the confusion
# matrix.
rf_accuracy <- 1 - round((table$Freq[2] + table$Freq[3]) / (table$Freq[1] + table$Freq[2] + table$Freq[3] + table$Freq[4]),3)
rf_accuracy

# Calculate Important predictors and plot it.
varImp(rf_default)

ggplot2::ggplot(varImp(rf_default))
                
                
# Find best mtry

#set.seed(123, sample.kind="Rounding")

# Set range of mtry to try and train over that range
tuneMtry <- expand.grid(.mtry = c(1: 10))

rf_mtry <- train(num ~ .,
                 data = train_set,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneMtry,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 300)

rf_mtry

# Get the best mtry
best_mtry <- rf_mtry$bestTune$mtry
best_mtry


# Find best maxnodes

# Creat list of maxnodes list and get the best mtry
maxNodeList <- list()
tuneMtry <- expand.grid(.mtry = best_mtry)

# Train with a list of maxnodes
for (mn in c(5: 15)) {
  set.seed(123, sample.kind="Rounding")
  rf_maxnode <- train(num ~ .,
                      data = train_set,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneMtry,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = mn,
                      ntree = 300)
  current_iteration <- toString(mn)
  maxNodeList[[current_iteration]] <- rf_maxnode
}

# Save resulting maxnodes and show results
results_maxnodes <- resamples(maxNodeList)
summary(results_maxnodes)

# Do a boxplot of the maxnodes
boxplot(results_maxnodes$values[1:10,2:23], col = "red")


# Find best maxtrees

# Creat list of maxtrees list and get the best mtry
maxTreeList <- list()
tuneMtry <- expand.grid(.mtry = best_mtry)

# Train with a list of maxtrees
for (nt in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(123)
  rf_maxtrees <- train(num ~ .,
                       data = train_set,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneMtry,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 10,
                       ntree = nt)
  key <- toString(nt)
  maxTreeList[[key]] <- rf_maxtrees
}

# Save resulting maxtrees and show results
results_tree <- resamples(maxTreeList)
summary(results_tree)

# Do a boxplot of the maxtrees
boxplot(results_tree$values[1:10,2:23], col = "green")


# Now use the optimized parameters for the final model

set.seed(123)

# Get the best mtry
tuneMtry <- expand.grid(.mtry = best_mtry)

# Train with the best tuning parameters
rf_fit <-train(
  num ~ ., 
  data = train_set,
  method = "rf",
  metric = "Accuracy",
  trControl = trControl,
  tuneGrid = tuneMtry,
  importance = TRUE,
  nodesize = 14,
  maxnodes = 10,
  ntree = 800)

rf_fit

# Do prediction with the optimized fit on test set
prediction <-predict(rf_fit, test_set)

# Calculate confusion matrix and plot it
table <- data.frame(confusionMatrix(prediction, test_set$num)$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "green", bad = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))

# Calculate the classification error, subtract from 1 to get accuracy
# Sum up the false positives and false negatives and divide by the total
# of the confusion matrix.
rf_accuracy <- 1 - round((table$Freq[2] + table$Freq[3]) / (table$Freq[1] + table$Freq[2] + table$Freq[3] + table$Freq[4]),3)
rf_accuracy

# Determine variable importance and plot.
varImp(rf_fit)
ggplot2::ggplot(varImp(rf_fit))


