
############################################################################# iris examples
# Load Library
Sys.getenv()
Sys.setenv(SPARK_HOME="/usr/local/src/spark-2.1.0-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
library(caret)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g"))

# Data read
iris$ID<-c(1:150)
irisdf<-createDataFrame(iris)
printSchema(irisdf)

# Data Modeling -kmeans
model<- spark.kmeans(data = irisdf, formula = Sepal_Length ~ Petal_Width , k = 3)
summary(model)

# Prediction -kmeans
pred<-predict(model,irisdf)
head(summarize(groupBy(pred,pred$prediction), count=n(pred$prediction)))
pred_kmeans<-select(pred,"ID","prediction")

j_iris<-join(irisdf,pred_kmeans,irisdf$ID ==pred_kmeans$ID)
showDF(j_iris)
dim(j_iris)
crosstab(j_iris,"Species","prediction")


conv_iris<-collect(j_iris)
library(caret)
head(conv_iris)
conv_iris<-conv_iris[,-c(6,7)]
conv_iris$Species<-as.factor(conv_iris$Species)
conv_iris$prediction<-as.factor(conv_iris$prediction)
summary(conv_iris)
confusionMatrix(conv_iris$prediction,conv_iris$Species)
labels(conv_iris$Species)
