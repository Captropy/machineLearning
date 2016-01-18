# This is my new script for the machine learning course
# Inspiration
# http://rstudio-pubs-static.s3.amazonaws.com/19452_2fd823509cb64054813867d90c02b34c.html
# http://amunategui.github.io/binary-outcome-modeling/


rm(list = ls())
#++++++++++++++++++++
# Settings
#++++++++++++++++++++

reloadData <- 0

#++++++++++++++++++++
# Load packages
#++++++++++++++++++++
library(ggplot2)
library(caret)
library(Amelia)
library(rattle)
library(rpart.plot)
library(rpart)

#++++++++++++++++++++
# Read data
#++++++++++++++++++++

if(reloadData == 1){
  build <- read.csv("pml-training.csv")
  test <- read.csv("pml-testing.csv")
  
  save(build, file = "pml-training.Rdata")
  save(test, file = "pml-testing.Rdata")
}else{
  load("pml-training.Rdata")
  load("pml-testing.Rdata")
}


build[,7:159] <- sapply(build[,7:159], as.numeric)
test[,7:159] <- sapply(test[,7:159], as.numeric)

# Check for missing values
missmap(test, main = "Missingness Map Test")

# select the activity features only
build <- build[8:160]
test <- test[8:160]

# Find columns with colsums NA
#nas <- is.na(apply(test,2,sum))
nas <- is.na(sapply(test,sum))

test <- test[,!nas]
dim(test)
build <- build[,!nas]

#++++++++++++++++++++
# Split training in test and cross-validation
#++++++++++++++++++++

## set the seed to make your partition reproductible
set.seed(123)
# create validation data set using Train 
inTrain <- createDataPartition(y=build$classe, p=0.75, list=FALSE)
train <- build[inTrain,]
val <- build[-inTrain,]
rm(inTrain,nas,build)

#++++++++++++++++++++
# PreProcess data
#++++++++++++++++++++

lastFeatureIndex <- (ncol(train)-1)

# Investigate PCA 
preProc <- preProcess(train[,1:lastFeatureIndex], method = "pca", thresh = 0.9)
preProc$rotation

# Look for correlated predictors
descrCor <-  cor(train[,1:lastFeatureIndex])
summary(descrCor[upper.tri(descrCor)])
# Count number of variables with correlation higher than...
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .99)

# Remove correlated variables
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
filteredDescr <- filteredDescr[,-highlyCorDescr]
descrCor2 <- cor(filteredDescr)
summary(descrCor2[upper.tri(descrCor2)])

# Center and scale the data
preProcVal <- preProcess(train[,1:lastFeatureIndex], method = c("center", "scale"))


#++++++++++++++++++++
# Model 1 - Regression Tree
#++++++++++++++++++++

##  regression tree model
set.seed(123)
#Mod1 <- train(train$classe ~ .,data=train[,1:lastFeatureIndex], method="rpart")
#save(Mod1,file="Mod1.RData")
load("Mod1.Rdata")
#Mod1.sc <- train(train$classe ~ .,data=train[,1:lastFeatureIndex], method="rpart", preProc = c("center", "scale"))
#save(Mod1.sc,file="Mod1.sc.RData")
load("Mod1.sc.Rdata")
#Mod1.pca <- train(train$classe ~ .,data=train[,1:lastFeatureIndex], method="rpart", preProcess = "pca", trControl = trainControl(preProcOptions = list(thresh = 0.9)))
#save(Mod1.pca,file="Mod1.pca.RData")
load("Mod1.pca.Rdata") 

confusion1 <- confusionMatrix(val$classe, predict(Mod1,val))
print(confusion1)
confusion1.sc <- confusionMatrix(val$classe, predict(Mod1.sc,val))
print(confusion1.sc)
confusion1.pca <- confusionMatrix(val$classe, predict(Mod1.pca,val))
print(confusion1.pca)

# save(Mod0,file="Mod0.RData")
