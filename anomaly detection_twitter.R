#Using Twitter's Anomaly detection algorithms

devtools::install_github("twitter/AnomalyDetection")
library(AnomalyDetection)

yahoo1<-read.csv("/home/ubuntu/kh/FEA/ML/Anomaly Detect/labeled data/A1Benchmark/real_4.csv",header=TRUE)
yahoo2<-na.omit(yahoo1)

ts.plot(yahoo2$value)

# period : seasonal decomposition에 사용되며, 한 기간에 몇개가 관측 되었는지 정의함
# max_anoms = 0.1 : 최종결과에 이상치가 최대 data의 10% 존재
model1<-AnomalyDetectionVec(yahoo2$value, max_anoms = 0.1, direction="both", period=24, plot=TRUE)
anoms<-model1$anoms
yahoo2$test_anomaly<-rep(0,nrow(yahoo2))
yahoo2$test_anomaly<-ifelse(yahoo2$timestamp %in% anoms$index,1,0)


# Visualization
library(ggplot2)
graph<-ggplot(data=yahoo2, aes(timestamp,value)) +
       geom_line()+
       geom_point(data=yahoo2[which(yahoo2$is_anomaly==1),], aes(timestamp,value),color="red",size=3,shape=11)+
       geom_point(data=yahoo2[which(yahoo2$test_anomaly==1),], aes(timestamp,value),color="blue",size=2,shape=20)+
       
graph

# Evaluation
# ref : https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values
# Pos Pred Value = TP/(TP+FP), Neg Pred Value = TN/(FN+TN)
library(caret)
confusionMatrix(yahoo2$test_anomaly,yahoo2$is_anomaly)


