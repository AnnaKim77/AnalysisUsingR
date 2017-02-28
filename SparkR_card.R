############################################################################# Data Handling

##===================================================
## Step1. Getting input
##---------------------------------------------------
card<-read.csv("/home/anna/kh/FEA/ML/Dataset/creditcard.csv", header=T) #dim(284807,31)
str(card)
card$Class<-as.factor(card$Class)
summary(card)
##===================================================


##===================================================
## Step2. Data Sampling
##---------------------------------------------------
class.0<-card[card$Class==0,]
class.1<-card[card$Class==1,]
class.only.0<-class.0[sample(c(0:nrow(class.0)),(nrow(class.1)*4)),]
new.card<-rbind(class.only.0,class.1)
summary(new.card)
##===================================================


##===================================================
## Step3. Save new dataset
##---------------------------------------------------
write.csv(new.card,"/home/anna/kh/FEA/ML/Dataset/creditcard_1vs4.csv",row.names = F)
##===================================================



############################################################################# Make SparkDataFrame
############################################################################# Logistic regression


##===================================================
## Step1. Prerequisite
##---------------------------------------------------
# install.packages("devtools")
# devtools::install_github('apache/spark@v2.1.0', subdir='R/pkg')
# install.packages("caret")
##===================================================


##===================================================
## Step2.Load Libraries to Start
##---------------------------------------------------
Sys.getenv()
Sys.setenv(SPARK_HOME="/usr/local/src/spark-2.1.0-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
library(caret)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g"))
##===================================================


##===================================================
## Step3. Getting Input
##---------------------------------------------------
card<-read.df("/home/anna/kh/FEA/ML/Dataset/creditcard_1vs4.csv","csv",header = "true")
head(card)
printSchema(card)
card<-drop(card,card$ID)
dim(card) 
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - Split data 70 : 30
##---------------------------------------------------
card.list<-randomSplit(card,weights=c(8,2), seed=0)
training<-card.list[[1]]
testing<-card.list[[2]]
head(summarize(groupBy(training,training$Class), count=n(training$Class)))
head(summarize(groupBy(testing,testing$Class), count=n(training$Class)))
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - Logistic Regression 
##---------------------------------------------------
card.logit<-spark.glm(training[,-32], Class~., family="binomial")
summary(card.logit)
##===================================================


##===================================================
## Step5. Check output
##---------------------------------------------------
predictions<-predict(card.logit,newData=testing)
summary(predictions)
##===================================================



############################################################################# Make RData
############################################################################# Logistic regression


##===================================================
## Step1. Prerequisite
##---------------------------------------------------
# install.packages("devtools")
# devtools::install_github('apache/spark@v2.1.0', subdir='R/pkg')
# install.packages("caret")
##===================================================


##===================================================
## Step2.Load Libraries to Start
##---------------------------------------------------
Sys.getenv()
Sys.setenv(SPARK_HOME="/usr/local/src/spark-2.1.0-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
library(caret)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g"))
##===================================================


##===================================================
## Step3. Getting Input
##---------------------------------------------------
card<-read.csv("/home/anna/kh/FEA/ML/Dataset/creditcard_1vs4.csv", header = T)
head(card)
str(card)
card<-card[,-32]
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - Split data 70 : 30
##---------------------------------------------------
set.seed(20170223)
inTrain<-createDataPartition(y=card$Class, p=.7, list=F)
train_card<-card[inTrain,] 
test_card<-card[-inTrain,] 
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - Convert Rdata to SparkDataFrame
##---------------------------------------------------
train<-createDataFrame(train_card)
test<-createDataFrame(test_card)
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - Logistic Regression 
##---------------------------------------------------
card.logit<-spark.glm(train, Class~., family="binomial")
summary(card.logit)
##===================================================


##===================================================
## Step5. Check output
##---------------------------------------------------
predictions.logit<-SparkR::predict(card.logit,newData=test)
pred.logit<-collect(predictions.logit)
head(pred.logit)
pred.logit$pred_Class<-base::ifelse(pred.logit$prediction>0.5,1,0)

confusionMatrix(pred.logit$Class,pred.logit$pred_Class)
##===================================================


############################################################################# Make RData
############################################################################# RandomForest


##===================================================
## Step1. Prerequisite
##---------------------------------------------------
# install.packages("devtools")
# devtools::install_github('apache/spark@v2.1.0', subdir='R/pkg')
# install.packages("caret")
##===================================================


##===================================================
## Step2.Load Libraries to Start
##---------------------------------------------------
Sys.getenv()
Sys.setenv(SPARK_HOME="/usr/local/src/spark-2.1.0-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
library(caret)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g"))
##===================================================


##===================================================
## Step3. Getting Input
##---------------------------------------------------
card<-read.csv("/home/anna/kh/FEA/ML/Dataset/creditcard_1vs4.csv", header = T)
head(card)
str(card)
card<-card[,-32]
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - Split data 70 : 30
##---------------------------------------------------
set.seed(20170223)
inTrain<-createDataPartition(y=card$Class, p=.7, list=F)
train_card<-card[inTrain,] 
test_card<-card[-inTrain,] 
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - Convert Rdata to SparkDataFrame
##---------------------------------------------------
train<-createDataFrame(train_card)
test<-createDataFrame(test_card)
##===================================================


##===================================================
## Step4. Predictive Modeling
##       - RandomForest
##---------------------------------------------------
card.rf<-spark.randomForest(data = train, formula = Class~.)
summary(card.rf)
##===================================================


##===================================================
## Step5. Check output
##---------------------------------------------------
predictions.rf<-SparkR::predict(card.logit,newData=test)
pred.rf<-collect(predictions.rf)
head(pred.rf)
pred.rf$pred_Class<-base::ifelse(pred.rf$prediction>0.5,1,0)

confusionMatrix(pred.rf$Class,pred.rf$pred_Class)
##===================================================




## path <- "/home/anna/kh/FEA/ML/model"
## write.ml(card.model, path)
## savedModel <- read.ml(path)
## summary(savedModel)

