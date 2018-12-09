Synopsis
========

Using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways, we will predict the manner("Classe") in which they did the exercise.

Exploratory Data Analysis
=========================

The training data for this project are available here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv> The test data are available here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.4.4

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
train <- read.csv("training.csv")
test <- read.csv("testing.csv")

#Exploring the data
dim(train)
```

    ## [1] 19622   160

``` r
dim(test)
```

    ## [1]  20 160

``` r
differing <- match(FALSE,(names(test) == names(train)))
names(train)[differing]
```

    ## [1] "classe"

``` r
names(test)[differing]
```

    ## [1] "problem_id"

``` r
#Plot the distribution of Excercise
plot(train$classe, xlab="Activity class", ylab="count", main="Distribution of Exercise Method",
col="red")
```

![](Practical_Machine_Learning_Project_files/figure-markdown_github/unnamed-chunk-2-1.png)

In the plot, we can see that most activities are classified in class "A", which is performing the activity exactly as specified.

Cleaning the data
=================

``` r
# Here we create a partition of the traning data set 
set.seed(37)
intrain1 <- createDataPartition(train$classe, p=0.7, list=FALSE)
train1 <- train[intrain1,]
test1 <- train[-intrain1,]
dim(train1)
```

    ## [1] 13737   160

If there are columns of completely NA values in any of the data sets, we will remove them from both.

``` r
# remove variables with nearly zero variance
nzv <- nearZeroVar(train1)
train1 <- train1[, -nzv]
test <- test[, -nzv]
test1 <- test1[, -nzv]

# remove variables that are almost always NA
mostlyNA <- sapply(train1, function(x) mean(is.na(x))) > 0.95
train1 <- train1[, mostlyNA==F]
test <- test[, mostlyNA==F]
test1 <- test1[, mostlyNA==F]

# remove variables that don't make sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp)
train1 <- train1[, -(1:5)]
test <- test[, -(1:5)]
test1 <- test1[, -(1:5)]

dim(train1)
```

    ## [1] 13737    54

``` r
dim(test1)
```

    ## [1] 5885   54

``` r
dim(test)
```

    ## [1] 20 54

Model Building
==============

Different models: classification tree, random forest, gradient boosting method 5 foldscross-validation is used to avoid overfitting and improve the efficicency of the model

``` r
# Classification tree model
library(rattle)
```

    ## Warning: package 'rattle' was built under R version 3.4.4

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
trControl <- trainControl(method="cv", number=5)
model_CT <- train(classe~., data=train1, method="rpart", trControl=trControl)
fancyRpartPlot(model_CT$finalModel)
```

![](Practical_Machine_Learning_Project_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
trainpred <- predict(model_CT,newdata=test1)
confMatCT <- confusionMatrix(test1$classe,trainpred)
confMatCT$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1518   32  122    0    2
    ##          B  477  379  283    0    0
    ##          C  457   38  531    0    0
    ##          D  415  173  376    0    0
    ##          E  136  167  282    0  497

``` r
confMatCT$overall[1]
```

    ##  Accuracy 
    ## 0.4970263

The accuracy of this first model is very low (about 50%). This means that the prediction is not very well.

``` r
# Random Forest model
#trControl <- trainControl(method="cv", number=5)
model_RF <- train(classe ~ ., data=train1, method="rf", trControl=trControl)
```

``` r
trainpred <- predict(model_RF,newdata=test1)
confMatRF <- confusionMatrix(test1$classe,trainpred)
# display confusion matrix and model accuracy
confMatRF$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    0    0    0    1
    ##          B    4 1132    3    0    0
    ##          C    0    3 1023    0    0
    ##          D    0    0    8  956    0
    ##          E    0    0    0    6 1076

``` r
confMatRF$overall[1]
```

    ##  Accuracy 
    ## 0.9957519

With random forest, we reach an accuracy of 99.6% using cross-validation with 5 steps. This is pretty good. But let's see what we can expect with Gradient boosting.

``` r
# Gradient Boosting model
model_GBM <- train(classe~., data=train1, method="gbm", trControl=trControl, verbose=FALSE)
```

``` r
trainpred <- predict(model_GBM,newdata=test1)
confMatGBM <- confusionMatrix(test1$classe,trainpred)
confMatGBM$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    2    0    0    0
    ##          B    9 1112   13    5    0
    ##          C    0    5 1015    4    2
    ##          D    0    1   19  944    0
    ##          E    0    9    1    8 1064

``` r
confMatGBM$overall[1]
```

    ## Accuracy 
    ## 0.986746

Precision with 5 folds is 98.4%.

Conclusion
==========

This shows that the random forest model is the best one with the out of sample error of 0.4%. We will then use it to predict the values of classe for the test data set.

``` r
FinalTestPred <- predict(model_RF,newdata=test)
FinalTestPred
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
