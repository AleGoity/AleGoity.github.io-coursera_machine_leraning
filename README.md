# coursera_machine_leraning
---
title: "Machine Learning project"
author: "Ale Goity"
date: "15/3/2021"
output:
  pdf_document: default
  html_document: default
---
# Before start  

To make it easier the visualization of the project a pdf version was uploaded. In the pdf version you can see the plots related to the project.



#Summary
We will analyzed the data using decision trees. I generated three models. Two models based in  decision trees, the first using all the variables and a second model with only some variables. And a third model using parallel random forest (parRF).
I alsp predicted the classe of the testing data using all the models. I compared the difference between the results obtain with the decision trees models and finally the prediction with the random forest model that predicted correctly all the testing predictions.

## Read and cleaning the data

```{r}
setwd("/Users/alegoity/Dropbox/Cursos/Coursera/R/Course8 (Machine Learning)/project/")
training<-read.csv("pml-training.csv")
testing<-read.csv("pml-testing.csv")
```


Remove columns with no information for the analysis. No variation in the column or NA values. Also, the information from columns 1 to 5. Doing this, we reduced variables to 54.
```{r}
library(caret)
a<-nearZeroVar(training)
training<-training[,-a] #eliminate columns with no variation
training<-training[ , -(1:5)]
training<- training[,colSums(is.na(training))<nrow(training)-1000] #to eliminate columns with high number of NA

dim(training)

```

Use createDataPartition to generate training_data (70%) and testing_data (30%) group

```{r}
set.seed(222)
inTrain<-createDataPartition(training$classe, p=0.7, list=FALSE)
training_data <- training[inTrain,]
testing_data <- training[-inTrain,]
```

# Modelling

## Create a decision tree model using all the variables
```{r fig.height=10, fig.width=10}
library(rpart)
library(rpart.plot)
set.seed(222)
model_DT1 <- rpart(classe ~., data = training_data, method = "class")

# Plot the trees
rpart.plot(model_DT1)

```

```{r}
testing_data$classe<-as.factor(testing_data$classe)
predict_tree <- predict(model_DT1, newdata= testing_data, type="class")
conMatrixtree <- confusionMatrix(testing_data$classe, predict_tree)
conMatrixtree
```

The accuracy of model_DT1 is 73.1%




## Create a decision tree using cross validation to predict the accuracy of the model.
trControl is set to 10-fold cross validation and tuneLenght to 10
```{r}
set.seed(222)
model_DT2<- train(classe~., data = training_data, method = "rpart", trControl = trainControl("cv", number = 10), tuneLength = 10, na.action = na.pass)

plot(model_DT2)
```
To determing the cp at which is obtained the best model accuracy 
```{r}
model_DT2$bestTune
```
plot the best decision tree obtained

```{r fig.height=10, fig.width=10}
rpart.plot(model_DT2$finalModel)

```

Decisions rules the model
```{r}
model_DT2$finalModel
```

Make predictions on the test data
```{r}
testing_data$classe<-as.factor(testing_data$classe)
predicted_classe <-predict(model_DT2, testing_data)
```


Compute model accuracy rate on test data
```{r}
mean(predicted_classe == testing_data$classe)
```

The accuracy of model_DT2 is 68.8%

# Predict testing cases
Based on the accuracy of the models we will use model_DT1 (use all variables to make the prediction).

```{r}
testing_prediction1<-predict(model_DT1, newdata = testing, type="class")
testing_prediction1
```


# EXTRA
We will compare the predictions obtain using model_DT1 and model_DT2.

Prediction using model_DT2
```{r}
testing_prediction2<-predict(model_DT2, newdata = testing)
testing_prediction2
```

Compare how many equal predictions we obtain using model_DT1 or model_DT2
```{r}
sum(testing_prediction1 == testing_prediction2) 
```

We obtain equal results in 18 of 20 cases using any of the two models. Observing differences in two cases were model_DT1 should have a better performance.

# Model random forest

## Create a random forest model with cross validation to predict the accuracy of the model.
To diminish calculation time it was set to 3-fold cross validation and 100 trees. And parallele random forest (parRF) was used. 

```{r}
library(doParallel)
model_RF<- train(classe~., data = training_data, method = "parRF",  trControl = trainControl("cv", number = 3, allowParallel = TRUE), na.action = na.pass, ntrees=100)


plot(model_RF) 
```

To evaluate the values of the model_RF

```{r}
model_RF$finalModel
```
Error rate is 0.25%

Make predictions on the test data
```{r}
testing_data$classe<-as.factor(testing_data$classe)
predicted_classe <-predict(model_RF, testing_data)

```

Compute model accuracy rate on test data
```{r}
mean(predicted_classe == testing_data$classe)
```

The accuracy of model_DT2 is 99.8%

# Predict testing cases
Based on the accuracy of the models we will use model_DT1 (use all variables to make the prediction).

```{r}
testing_prediction_RF<-predict(model_RF, newdata = testing)
testing_prediction_RF
```

## Using random forest we were able to predict correctly a 100% of the predictions
