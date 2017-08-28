setwd("C:\\Users\\SROY\\Documents\\CodeBase\\Datasets\\HiggsBoson")
#load("HiggsBoson.RData")
#save.image("HiggsBoson.RData")

test <- read.csv2("test.csv", header = TRUE, sep = ',')
train <- read.csv2("training.csv", header = TRUE, sep = ',')

# test and train columns does not match
testcols <- names(test)
traincols <- names(train)
setdiff(traincols, testcols)

# Weight and label is the diff. Why is weight not in test?
head(train)

# Check structure
str(train)
# Too many factors. EventID is int which is fine. PRI_jet_num is int which is fine
# Convert eveything to float except EventID, PRI_jet_num and Label
excludeList <- c("EventId","PRI_jet_num","Label")
includeList <- names(train)[!names(train) %in% c(excludeList)]

for (i in includeList){
  train[,i] <- as.numeric(as.character(train[,i]))
}

# Find missing values
apply(train[,includeList], 2, function(x) length(which(x == "" | 
                                                         is.na(x) | x == "NA" |  x == "-")))
# Doesn't have common abnormal values

# Meaningless values
# Check -0.0
apply(train[,includeList], 2, function(x) length(which(x == -0.0)) * 100/length(x))
# -0.0 is automatically converted

# Check -999.0
Meaningless <- apply(train[,includeList], 2, function(x) length(which(x == -999.0)) * 100/length(x))
# Many columns have so many values for -999.0 ???????
# Excluding columns having more than 30% values as -999
perColExclude = 30
excludeList <- names(Meaningless)[Meaningless > perColExclude]
# Add eventID and Label
tempExcludeList <- c("EventId", "Label")
includeList <- names(train)[!names(train) %in% c(excludeList, tempExcludeList)]

########################### Re-run section ##############################
# This section comes later after the full model has been run.
# This is to ensure we are playing right with the required columns.
# Columns PRI_jet_leading_pt, PRI_jet_leading_eta, PRI_jet_leading_phi have 39% meaningless values
# We can run significance test to verify before removing it.
yVar <- 'Label'
xVars <- names(train)[!names(train) %in% c(tempExcludeList, 'Weight')]

createModelFormula <- function(yVar, xVars, includeIntercept = TRUE){
  if(includeIntercept){
    modelForm <- as.formula(paste(yVar, "~", paste(xVars, collapse = '+ ')))
  } else {
    modelForm <- as.formula(paste(yVar, "~", paste(xVars, collapse = '+ '), -1))
  }
  return(modelForm)
}

modelForm <- createModelFormula(yVar = yVar, xVars = xVars, includeIntercept = FALSE)
model <- lm(modelForm, data = train)
summary(model)

# Interestingly PRI_jet_leading_pt, though has 40% meaningless values shows high significance.
# Let's check if this improves the previous results of 83.48 Accuracy.
# Final note. PRI_jet_leading_pt decreased the performance slightly

########################### End of Re-run section ##############################

# library(ggplot2)
# barplot(Meaningless[Meaningless > 0])
# 
# meaninglessDf <- data.frame(Meaningless[Meaningless > 0])
# colnames(meaninglessDf) <- c("Property", "Perc Meaningless values")
# ggplot(Meaningless[Meaningless > 0]) + geom_bar()

#summary
summary(train[,includeList])

# Check duplicate eventid
stopifnot(length(unique(train$EventId)) == nrow(train))
# Looks fine

# Since DER is derived from PRI, Let's check correlation
corMatrix <- cor(train[,includeList])
corMatrix[lower.tri(corMatrix,diag=TRUE)]=NA
corMatrix=as.data.frame(as.table(corMatrix))
corMatrix=na.omit(corMatrix)
corMatrix=corMatrix[order(-abs(corMatrix$Freq)),]
corMatrix[1:10,]
# some very high correlation observed
# looks like DER_sum_pt is derived from two PRI. Let's remove it.
# also "Weight" is not in test.
excludeList <- c(excludeList, "DER_sum_pt", "Weight")
includeList <- names(train)[!names(train) %in% c(excludeList, tempExcludeList)]
#includeList <- c(includeList, 'PRI_jet_leading_pt') # Re-run section
xVars <- includeList
yVar <- "Label"

# the labels need to converted to binary
train$Label <- trimws(train$Label)
train[train$Label == 'b',"Label"] <- 0
train[train$Label == 's',"Label"] <- 1
train$Label <- as.numeric(train$Label)


# stratefied sampling
library(caret)
seedVal = 17869
set.seed(seedVal)
trainPct <- .8
testPct <- 1 - trainPct
inTrain <- createDataPartition(y = train$Label, p = trainPct, list = FALSE)
traindata <- train[inTrain,]
testdata <- train[-inTrain,]
stopifnot(nrow(traindata) + nrow(testdata) == nrow(train))

##################### saved till here ###############
#save.image("HiggsBoson.RData")

# xgboost
library(xgboost)

# Prepare matrix
mtrain <- model.matrix(~.+0,data = traindata[,xVars]) 
mtest <- model.matrix(~.+0,data = testdata[,xVars])
dtrain <- xgb.DMatrix(data = mtrain,label = traindata[,yVar]) 
dtest <- xgb.DMatrix(data = mtest,label=testdata[,yVar])

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1,
               seed = 17869, nthread = 4,
               subsample=0.8, colsample_bytree=1)

# Calculate the best round
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, 
                 early_stopping_rounds = 20, maximize = F)

# Train model with the best round
xgbmodel <- xgb.train (params = params, data = dtrain, nrounds = xgbcv$best_iteration, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 10, early_stopping_rounds = 10, 
                   maximize = F , eval_metric = "logloss")

summary(xgbmodel)

# Compute feature importance matrix
impFeatures <- xgb.importance(xVars, model = xgbmodel)
# Nice graph
xgb.plot.importance(impFeatures[1:20,])

# Predict new instances
pred <- predict(xgbmodel, dtest)
y_pred <- ifelse(pred > 0.5, 1, 0)

# Measure performance
confusion <- confusionMatrix(y_pred, testdata[,yVar])
confusion$overall[1]*100

######################## Grid Tune #########################

library(mlr)
tm <- proc.time()

# Since we have best rounds using default params, we can use grid tune to improve performance
# convert characters to factors
traindata$Label <- as.factor(traindata$Label)
testdata$Label <- as.factor(testdata$Label)

# create tasks
traintask <- makeClassifTask (data = traindata,target = "Label")
testtask <- makeClassifTask (data = testdata,target = "Label")

# One hot encoding
traintask <- createDummyFeatures (obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="logloss", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",
                                          values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 20L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 20L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 20L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
tuned <- tuneParams(learner = lrn, task = traintask, 
                     resampling = rdesc, measures = acc, 
                     par.set = params, control = ctrl, show.info = T)
print(tuned)
elapsedtm <- proc.time() - tm
print(elapsedtm[3])

######################## Re run using tuned results ########

params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.1, gamma=0, max_depth=tuned$x[2][[1]], min_child_weight=tuned$x[3][[1]],
               seed = 17869, nthread = 4,
               subsample=tuned$x[4][[1]], colsample_bytree=tuned$x[5][[1]])

# Calculate the best round
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, 
                 early_stopping_rounds = 20, maximize = F)

# Train model with the best round
xgbmodel <- xgb.train (params = params, data = dtrain, nrounds = xgbcv$best_iteration, 
                       watchlist = list(val=dtest,train=dtrain), 
                       print_every_n = 10, early_stopping_rounds = 10, 
                       maximize = F , eval_metric = "logloss")

summary(xgbmodel)

# Compute feature importance matrix
impFeatures <- xgb.importance(xVars, model = xgbmodel)
# Nice graph
xgb.plot.importance(impFeatures[1:20,])

# Predict new instances
pred <- predict(xgbmodel, dtest)
y_pred <- ifelse(pred > 0.5, 1, 0)

# Measure performance
confusion <- confusionMatrix(y_pred, testdata[,yVar])
confusion$overall[1]*100

######################## Test Data #########################

# Clean true test data
for (i in includeList){
  test[,i] <- as.numeric(as.character(test[,i]))
}

# Check duplicate event id
stopifnot(length(unique(test$EventId)) == nrow(test))

# Create model matrix
mtestNew <- model.matrix(~.+0,data = test[,xVars])
dtestNew <- xgb.DMatrix(data = mtestNew)

# Predict new instances
newTestPredProb <- predict(xgbmodel, dtestNew)
newTestPredClass <- ifelse(pred > 0.5, 's', 'b')

# Append values to test
submission <- test[,'EventId']
submission <- data.frame(cbind(submission, newTestPredProb, newTestPredClass))
names(submission) <- c("EventId","Prob","Class")
submission$RankOrder <- as.integer(rank(submission$Prob, ties.method= "first"))

# Export to file
expData <- submission[,c("EventId","RankOrder","Class")]
write.table(expData, file = "subOutput.csv", quote = FALSE, row.names=FALSE, sep=",")
