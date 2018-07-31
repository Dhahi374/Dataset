
#################################################
#Neural Network Modeling of Computer Data Part II
#################################################

##############
#Load Packages
##############

#Load libraries
library(plyr)
library(ggplot2)
library(gridExtra)
library(neuralnet)
library(caret)
#install.packages("caret")

#############
#Read in Data
#############

#Read in [newdata.csv]
data1 = read.csv(file.choose(),header=T)
data1$app = factor(data1$app,levels=c("MPI_job","web_server","cassandra_stress","spark_word_count"))
data1$app_num = as.numeric(data1$app)
data1$machine_num = as.numeric(data1$machine)
data1$nodes_factor = as.factor(data1$nodes)

#############################
#Prepare Data for ANN Model 1
#############################

#Save activation functions
sigmoid = function(x){1/(1+exp(-x))}
hypertan = function(x){tanh(x)}

#Retain only numerically coded data
data3 = data1[,c(3:10)]

#Scale instructions 0-1
data3$instructions = scale(data3$instructions,center=min(data3$instructions),scale=max(data3$instructions)-min(data3$instructions))
#Scale time 0-1
data3$time = scale(data3$time,center=min(data3$time),scale=max(data3$time)-min(data3$time))
#Scale time 0-1
data3$data_trans = scale(data3$data_trans,center=min(data3$data_trans),scale=max(data3$data_trans)-min(data3$data_trans))
#Scale cpu 0-1
data3$cpu = scale(data3$cpu,center=min(data3$cpu),scale=max(data3$cpu)-min(data3$cpu))
#Scale mips 0-1
data3$mips = scale(data3$mips,center=min(data3$mips),scale=max(data3$mips)-min(data3$mips))

#For loop scaling machine_num -1,1 and app_num -1,1 and nodes -1,1
data3$n01 = rep(-1,nrow(data3))
data3$n02 = rep(-1,nrow(data3))
data3$n03 = rep(-1,nrow(data3))
data3$n04 = rep(-1,nrow(data3))
data3$n05 = rep(-1,nrow(data3))
data3$n06 = rep(-1,nrow(data3))
data3$n07 = rep(-1,nrow(data3))
data3$n08 = rep(-1,nrow(data3))
data3$n09 = rep(-1,nrow(data3))
data3$n10 = rep(-1,nrow(data3))
data3$n11 = rep(-1,nrow(data3))
data3$n12 = rep(-1,nrow(data3))
data3$n13 = rep(-1,nrow(data3))
data3$n14 = rep(-1,nrow(data3))
data3$n15 = rep(-1,nrow(data3))
data3$machinerpi2_coded = rep(-1,nrow(data3))
data3$machinerpi3_coded = rep(-1,nrow(data3))
data3$appa_coded = rep(-1,nrow(data3))
data3$appb_coded = rep(-1,nrow(data3))
data3$appc_coded = rep(-1,nrow(data3))
data3$appd_coded = rep(-1,nrow(data3))
for (i in 1:nrow(data3)){
  if (data3$machine_num[i]==1){
    data3$machinerpi2_coded[i] = 1
  }
  if (data3$machine_num[i]==2){
    data3$machinerpi3_coded[i] = 1
  }
  if (data3$app_num[i]==1){
    data3$appa_coded[i] = 1
  }
  if (data3$app_num[i]==2){
    data3$appb_coded[i] = 1
  }
  if (data3$app_num[i]==3){
    data3$appc_coded[i] = 1
  }
  if (data3$app_num[i]==4){
    data3$appd_coded[i] = 1
  }
  if (data3$nodes[i]==1){
    data3$n01[i] = 1
  }
  if (data3$nodes[i]==2){
    data3$n02[i] = 1
  }
  if (data3$nodes[i]==3){
    data3$n03[i] = 1
  }
  if (data3$nodes[i]==4){
    data3$n04[i] = 1
  }
  if (data3$nodes[i]==5){
    data3$n05[i] = 1
  }
  if (data3$nodes[i]==6){
    data3$n06[i] = 1
  }
  if (data3$nodes[i]==7){
    data3$n07[i] = 1
  }
  if (data3$nodes[i]==8){
    data3$n08[i] = 1
  }
  if (data3$nodes[i]==9){
    data3$n09[i] = 1
  }
  if (data3$nodes[i]==10){
    data3$n10[i] = 1
  }
  if (data3$nodes[i]==11){
    data3$n11[i] = 1
  }
  if (data3$nodes[i]==12){
    data3$n12[i] = 1
  }
  if (data3$nodes[i]==13){
    data3$n13[i] = 1
  }
  if (data3$nodes[i]==14){
    data3$n14[i] = 1
  }
  if (data3$nodes[i]==15){
    data3$n15[i] = 1
  }
}
#Reduce data3 to only coded columns
data3 = data3[,c(1:2,4:6,9:29)]

################
#Fit ANN Model 1
################

#Split the data into a test and training set
index = sample(1:nrow(data3),round(0.80*nrow(data3)))
train_data = as.data.frame(data3[index,])
test_data = as.data.frame(data3[-index,])
#Train the model
model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
#Predict test data from trained nn
pred_nn3_scaled = compute(model_nn3,test_data[,c(1:4,6:26)])
#Calculate max and min time for rescaling
min_time = min(data1$time)
max_time = max(data1$time)
#Rescale test prediction
pred_test_data_time = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
#Rescale test actual
test_data_time = test_data$time*(max_time-min_time)+min_time
#Rescale train prediction
pred_train_data_time = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
#Rescale train actual
train_data_time = train_data$time*(max_time-min_time)+min_time
#Combine into data frames
a07 = cbind.data.frame(test_data_time,pred_test_data_time)
colnames(a07) = c("testdata","testpred")
a08 = cbind.data.frame(train_data_time,pred_train_data_time)
colnames(a08) = c("traindata","trainpred")
p01 = ggplot(a07,aes(x=testdata,y=testpred))+
  geom_point()+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p02 = ggplot(a08,aes(x=traindata,y=trainpred))+
  geom_point()+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
grid.arrange(p01,p02,nrow=1)
plot(model_nn3)

###################################################################################################################
#ANN Model 1 All 8 Machine x App Exclusions One-by-One This Did Not Work As MACHINE Was Still Included in the Model
###################################################################################################################

#Allocate excluded points to the test data
a1 = c(1:450)
b1 = c(451:3600)
a2 = c(451:900)
b2 = c(1:450,901:3600)
a3 = c(901:1350)
b3 = c(1:900,1351:3600)
a4 = c(1351:1800)
b4 = c(1:1350,1801:3600)
a5 = c(1801:2250)
b5 = c(1:1800,2251:3600)
a6 = c(2251:2700)
b6 = c(1:2250,2701:3600)
a7 = c(2701:3150)
b7 = c(1:2700,3151:3600)
a8 = c(3151:3600)
b8 = c(1:3150)
#Split the data into a test and training set
test_data1 = as.data.frame(data3[a1,])
train_data1 = as.data.frame(data3[b1,])
test_data2 = as.data.frame(data3[a2,])
train_data2 = as.data.frame(data3[b2,])
test_data3 = as.data.frame(data3[a3,])
train_data3 = as.data.frame(data3[b3,])
test_data4 = as.data.frame(data3[a4,])
train_data4 = as.data.frame(data3[b4,])
test_data5 = as.data.frame(data3[a5,])
train_data5 = as.data.frame(data3[b5,])
test_data6 = as.data.frame(data3[a6,])
train_data6 = as.data.frame(data3[b6,])
test_data7 = as.data.frame(data3[a7,])
train_data7 = as.data.frame(data3[b7,])
test_data8 = as.data.frame(data3[a8,])
train_data8 = as.data.frame(data3[b8,])
#Train the model
model_nn1 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data1,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn2 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data2,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data3,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn4 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data4,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn5 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data5,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn6 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data6,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn7 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data7,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn8 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+machinerpi2_coded+machinerpi3_coded+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data8,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
#Predict test data from trained nn
pred_nn1_scaled = compute(model_nn1,test_data1[,c(1:4,6:26)])
pred_nn2_scaled = compute(model_nn2,test_data2[,c(1:4,6:26)])
pred_nn3_scaled = compute(model_nn3,test_data3[,c(1:4,6:26)])
pred_nn4_scaled = compute(model_nn4,test_data4[,c(1:4,6:26)])
pred_nn5_scaled = compute(model_nn5,test_data5[,c(1:4,6:26)])
pred_nn6_scaled = compute(model_nn6,test_data6[,c(1:4,6:26)])
pred_nn7_scaled = compute(model_nn7,test_data7[,c(1:4,6:26)])
pred_nn8_scaled = compute(model_nn8,test_data8[,c(1:4,6:26)])
#Rescale test prediction
pred_test_data_time1 = pred_nn1_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time2 = pred_nn2_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time3 = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time4 = pred_nn4_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time5 = pred_nn5_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time6 = pred_nn6_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time7 = pred_nn7_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time8 = pred_nn8_scaled$net.result*(max_time-min_time)+min_time
#Rescale test actual
test_data_time1 = test_data1$time*(max_time-min_time)+min_time
test_data_time2 = test_data2$time*(max_time-min_time)+min_time
test_data_time3 = test_data3$time*(max_time-min_time)+min_time
test_data_time4 = test_data4$time*(max_time-min_time)+min_time
test_data_time5 = test_data5$time*(max_time-min_time)+min_time
test_data_time6 = test_data6$time*(max_time-min_time)+min_time
test_data_time7 = test_data7$time*(max_time-min_time)+min_time
test_data_time8 = test_data8$time*(max_time-min_time)+min_time
#Rescale train prediction
pred_train_data_time1 = model_nn1$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time2 = model_nn2$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time3 = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time4 = model_nn4$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time5 = model_nn5$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time6 = model_nn6$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time7 = model_nn7$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time8 = model_nn8$net.result[[1]][,1]*(max_time-min_time)+min_time
#Rescale train actual
train_data_time1 = train_data1$time*(max_time-min_time)+min_time
train_data_time2 = train_data2$time*(max_time-min_time)+min_time
train_data_time3 = train_data3$time*(max_time-min_time)+min_time
train_data_time4 = train_data4$time*(max_time-min_time)+min_time
train_data_time5 = train_data5$time*(max_time-min_time)+min_time
train_data_time6 = train_data6$time*(max_time-min_time)+min_time
train_data_time7 = train_data7$time*(max_time-min_time)+min_time
train_data_time8 = train_data8$time*(max_time-min_time)+min_time
#Combine into data frames
a1_1 = cbind.data.frame(test_data_time1,pred_test_data_time1,data1$app[a1])
a2_1 = cbind.data.frame(test_data_time2,pred_test_data_time2,data1$app[a2])
a3_1 = cbind.data.frame(test_data_time3,pred_test_data_time3,data1$app[a3])
a4_1 = cbind.data.frame(test_data_time4,pred_test_data_time4,data1$app[a4])
a5_1 = cbind.data.frame(test_data_time5,pred_test_data_time5,data1$app[a5])
a6_1 = cbind.data.frame(test_data_time6,pred_test_data_time6,data1$app[a6])
a7_1 = cbind.data.frame(test_data_time7,pred_test_data_time7,data1$app[a7])
a8_1 = cbind.data.frame(test_data_time8,pred_test_data_time8,data1$app[a8])
colnames(a1_1) = c("testdata","testpred","app")
colnames(a2_1) = c("testdata","testpred","app")
colnames(a3_1) = c("testdata","testpred","app")
colnames(a4_1) = c("testdata","testpred","app")
colnames(a5_1) = c("testdata","testpred","app")
colnames(a6_1) = c("testdata","testpred","app")
colnames(a7_1) = c("testdata","testpred","app")
colnames(a8_1) = c("testdata","testpred","app")
a1_2 = cbind.data.frame(train_data_time1,pred_train_data_time1,data1$app[b1])
a2_2 = cbind.data.frame(train_data_time2,pred_train_data_time2,data1$app[b2])
a3_2 = cbind.data.frame(train_data_time3,pred_train_data_time3,data1$app[b3])
a4_2 = cbind.data.frame(train_data_time4,pred_train_data_time4,data1$app[b4])
a5_2 = cbind.data.frame(train_data_time5,pred_train_data_time5,data1$app[b5])
a6_2 = cbind.data.frame(train_data_time6,pred_train_data_time6,data1$app[b6])
a7_2 = cbind.data.frame(train_data_time7,pred_train_data_time7,data1$app[b7])
a8_2 = cbind.data.frame(train_data_time8,pred_train_data_time8,data1$app[b8])
colnames(a1_2) = c("traindata","trainpred","app")
colnames(a2_2) = c("traindata","trainpred","app")
colnames(a3_2) = c("traindata","trainpred","app")
colnames(a4_2) = c("traindata","trainpred","app")
colnames(a5_2) = c("traindata","trainpred","app")
colnames(a6_2) = c("traindata","trainpred","app")
colnames(a7_2) = c("traindata","trainpred","app")
colnames(a8_2) = c("traindata","trainpred","app")
p1_1 = ggplot(a1_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p1_2 = ggplot(a1_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_1 = ggplot(a2_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(-1000,1000,100),limits=c(-1000,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_2 = ggplot(a2_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(-1000,1000,100),limits=c(-1000,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_1 = ggplot(a3_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_2 = ggplot(a3_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_1 = ggplot(a4_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_2 = ggplot(a4_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_1 = ggplot(a5_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(-1000,1000,100),limits=c(-1000,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_2 = ggplot(a5_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(-1000,1000,100),limits=c(-1000,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_1 = ggplot(a6_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_2 = ggplot(a6_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_1 = ggplot(a7_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_2 = ggplot(a7_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_1 = ggplot(a8_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_2 = ggplot(a8_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))

grid.arrange(p1_1,p1_2,nrow=1)
grid.arrange(p2_1,p2_2,nrow=1)
grid.arrange(p3_1,p3_2,nrow=1)
grid.arrange(p4_1,p4_2,nrow=1)
grid.arrange(p5_1,p5_2,nrow=1)
grid.arrange(p6_1,p6_2,nrow=1)
grid.arrange(p7_1,p7_2,nrow=1)
grid.arrange(p8_1,p8_2,nrow=1)

#Enter filepath here
write.csv(a1_1,"C:/Users/51264/Desktop/test1.csv")
write.csv(a1_2,"C:/Users/51264/Desktop/train1.csv")
write.csv(a2_1,"C:/Users/51264/Desktop/test2.csv")
write.csv(a2_2,"C:/Users/51264/Desktop/train2.csv")
write.csv(a3_1,"C:/Users/51264/Desktop/test3.csv")
write.csv(a3_2,"C:/Users/51264/Desktop/train3.csv")
write.csv(a4_1,"C:/Users/51264/Desktop/test4.csv")
write.csv(a4_2,"C:/Users/51264/Desktop/train4.csv")
write.csv(a5_1,"C:/Users/51264/Desktop/test5.csv")
write.csv(a5_2,"C:/Users/51264/Desktop/train5.csv")
write.csv(a6_1,"C:/Users/51264/Desktop/test6.csv")
write.csv(a6_2,"C:/Users/51264/Desktop/train6.csv")
write.csv(a7_1,"C:/Users/51264/Desktop/test7.csv")
write.csv(a7_2,"C:/Users/51264/Desktop/train7.csv")
write.csv(a8_1,"C:/Users/51264/Desktop/test8.csv")
write.csv(a8_2,"C:/Users/51264/Desktop/train8.csv")

#############################################################################################################
#Prepare Data for ANN Model 2 This Time Remove Machine Entirely While Leaving Machine Attributes CPU and MIPS
#############################################################################################################

#Save activation functions
sigmoid = function(x){1/(1+exp(-x))}
hypertan = function(x){tanh(x)}

#Retain only numerically coded data
data4 = data1[,c(3:9)]

#Scale instructions 0-1
data4$instructions = scale(data4$instructions,center=min(data4$instructions),scale=max(data4$instructions)-min(data4$instructions))
#Scale time 0-1
data4$time = scale(data4$time,center=min(data4$time),scale=max(data4$time)-min(data4$time))
#Scale time 0-1
data4$data_trans = scale(data4$data_trans,center=min(data4$data_trans),scale=max(data4$data_trans)-min(data4$data_trans))
#Scale cpu 0-1
data4$cpu = scale(data4$cpu,center=min(data4$cpu),scale=max(data4$cpu)-min(data4$cpu))
#Scale mips 0-1
data4$mips = scale(data4$mips,center=min(data4$mips),scale=max(data4$mips)-min(data4$mips))

#For loop scaling machine_num -1,1 and app_num -1,1 and nodes -1,1
data4$n01 = rep(-1,nrow(data4))
data4$n02 = rep(-1,nrow(data4))
data4$n03 = rep(-1,nrow(data4))
data4$n04 = rep(-1,nrow(data4))
data4$n05 = rep(-1,nrow(data4))
data4$n06 = rep(-1,nrow(data4))
data4$n07 = rep(-1,nrow(data4))
data4$n08 = rep(-1,nrow(data4))
data4$n09 = rep(-1,nrow(data4))
data4$n10 = rep(-1,nrow(data4))
data4$n11 = rep(-1,nrow(data4))
data4$n12 = rep(-1,nrow(data4))
data4$n13 = rep(-1,nrow(data4))
data4$n14 = rep(-1,nrow(data4))
data4$n15 = rep(-1,nrow(data4))
data4$appa_coded = rep(-1,nrow(data4))
data4$appb_coded = rep(-1,nrow(data4))
data4$appc_coded = rep(-1,nrow(data4))
data4$appd_coded = rep(-1,nrow(data4))
for (i in 1:nrow(data4)){
  if (data4$app_num[i]==1){
    data4$appa_coded[i] = 1
  }
  if (data4$app_num[i]==2){
    data4$appb_coded[i] = 1
  }
  if (data4$app_num[i]==3){
    data4$appc_coded[i] = 1
  }
  if (data4$app_num[i]==4){
    data4$appd_coded[i] = 1
  }
  if (data4$nodes[i]==1){
    data4$n01[i] = 1
  }
  if (data4$nodes[i]==2){
    data4$n02[i] = 1
  }
  if (data4$nodes[i]==3){
    data4$n03[i] = 1
  }
  if (data4$nodes[i]==4){
    data4$n04[i] = 1
  }
  if (data4$nodes[i]==5){
    data4$n05[i] = 1
  }
  if (data4$nodes[i]==6){
    data4$n06[i] = 1
  }
  if (data4$nodes[i]==7){
    data4$n07[i] = 1
  }
  if (data4$nodes[i]==8){
    data4$n08[i] = 1
  }
  if (data4$nodes[i]==9){
    data4$n09[i] = 1
  }
  if (data4$nodes[i]==10){
    data4$n10[i] = 1
  }
  if (data4$nodes[i]==11){
    data4$n11[i] = 1
  }
  if (data4$nodes[i]==12){
    data4$n12[i] = 1
  }
  if (data4$nodes[i]==13){
    data4$n13[i] = 1
  }
  if (data4$nodes[i]==14){
    data4$n14[i] = 1
  }
  if (data4$nodes[i]==15){
    data4$n15[i] = 1
  }
}
#Reduce data4 to only coded columns
data4 = data4[,c(1:2,4:6,8:26)]

###########################
#Fit Neural Network Model 2
###########################

#Split the data into a test and training set
index = sample(1:nrow(data4),round(0.80*nrow(data4)))
train_data = as.data.frame(data4[index,])
test_data = as.data.frame(data4[-index,])
#Train the model
model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
#Predict test data from trained nn
pred_nn3_scaled = compute(model_nn3,test_data[,c(1:4,6:24)])
#Calculate max and min time for rescaling
min_time = min(data1$time)
max_time = max(data1$time)
#Rescale test prediction
pred_test_data_time = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
#Rescale test actual
test_data_time = test_data$time*(max_time-min_time)+min_time
#Rescale train prediction
pred_train_data_time = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
#Rescale train actual
train_data_time = train_data$time*(max_time-min_time)+min_time
#Combine into data frames
a07 = cbind.data.frame(test_data_time,pred_test_data_time,data1$app[-index])
colnames(a07) = c("testdata","testpred","app")
a08 = cbind.data.frame(train_data_time,pred_train_data_time,data1$app[index])
colnames(a08) = c("traindata","trainpred","app")
p01 = ggplot(a07,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p02 = ggplot(a08,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
grid.arrange(p01,p02,nrow=1)
plot(model_nn3)

######################################################
#ANN Model 2 All 8 Machine x App Exclusions One-by-One
######################################################

#Allocate excluded points to the test data
a1 = c(1:450)
b1 = c(451:3600)
a2 = c(451:900)
b2 = c(1:450,901:3600)
a3 = c(901:1350)
b3 = c(1:900,1351:3600)
a4 = c(1351:1800)
b4 = c(1:1350,1801:3600)
a5 = c(1801:2250)
b5 = c(1:1800,2251:3600)
a6 = c(2251:2700)
b6 = c(1:2250,2701:3600)
a7 = c(2701:3150)
b7 = c(1:2700,3151:3600)
a8 = c(3151:3600)
b8 = c(1:3150)
#Split the data into a test and training set
test_data1 = as.data.frame(data4[a1,])
train_data1 = as.data.frame(data4[b1,])
test_data2 = as.data.frame(data4[a2,])
train_data2 = as.data.frame(data4[b2,])
test_data3 = as.data.frame(data4[a3,])
train_data3 = as.data.frame(data4[b3,])
test_data4 = as.data.frame(data4[a4,])
train_data4 = as.data.frame(data4[b4,])
test_data5 = as.data.frame(data4[a5,])
train_data5 = as.data.frame(data4[b5,])
test_data6 = as.data.frame(data4[a6,])
train_data6 = as.data.frame(data4[b6,])
test_data7 = as.data.frame(data4[a7,])
train_data7 = as.data.frame(data4[b7,])
test_data8 = as.data.frame(data4[a8,])
train_data8 = as.data.frame(data4[b8,])
#Train the model
model_nn1 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data1,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn2 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data2,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data4,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn4 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data4,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn5 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data5,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn6 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data6,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn7 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data7,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn8 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data8,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
#Predict test data from trained nn
pred_nn1_scaled = compute(model_nn1,test_data1[,c(1:4,6:24)])
pred_nn2_scaled = compute(model_nn2,test_data2[,c(1:4,6:24)])
pred_nn3_scaled = compute(model_nn3,test_data4[,c(1:4,6:24)])
pred_nn4_scaled = compute(model_nn4,test_data4[,c(1:4,6:24)])
pred_nn5_scaled = compute(model_nn5,test_data5[,c(1:4,6:24)])
pred_nn6_scaled = compute(model_nn6,test_data6[,c(1:4,6:24)])
pred_nn7_scaled = compute(model_nn7,test_data7[,c(1:4,6:24)])
pred_nn8_scaled = compute(model_nn8,test_data8[,c(1:4,6:24)])
#Rescale test prediction
pred_test_data_time1 = pred_nn1_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time2 = pred_nn2_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time3 = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time4 = pred_nn4_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time5 = pred_nn5_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time6 = pred_nn6_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time7 = pred_nn7_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time8 = pred_nn8_scaled$net.result*(max_time-min_time)+min_time
#Rescale test actual
test_data_time1 = test_data1$time*(max_time-min_time)+min_time
test_data_time2 = test_data2$time*(max_time-min_time)+min_time
test_data_time3 = test_data4$time*(max_time-min_time)+min_time
test_data_time4 = test_data4$time*(max_time-min_time)+min_time
test_data_time5 = test_data5$time*(max_time-min_time)+min_time
test_data_time6 = test_data6$time*(max_time-min_time)+min_time
test_data_time7 = test_data7$time*(max_time-min_time)+min_time
test_data_time8 = test_data8$time*(max_time-min_time)+min_time
#Rescale train prediction
pred_train_data_time1 = model_nn1$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time2 = model_nn2$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time3 = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time4 = model_nn4$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time5 = model_nn5$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time6 = model_nn6$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time7 = model_nn7$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time8 = model_nn8$net.result[[1]][,1]*(max_time-min_time)+min_time
#Rescale train actual
train_data_time1 = train_data1$time*(max_time-min_time)+min_time
train_data_time2 = train_data2$time*(max_time-min_time)+min_time
train_data_time3 = train_data4$time*(max_time-min_time)+min_time
train_data_time4 = train_data4$time*(max_time-min_time)+min_time
train_data_time5 = train_data5$time*(max_time-min_time)+min_time
train_data_time6 = train_data6$time*(max_time-min_time)+min_time
train_data_time7 = train_data7$time*(max_time-min_time)+min_time
train_data_time8 = train_data8$time*(max_time-min_time)+min_time
#Combine into data frames
a1_1 = cbind.data.frame(test_data_time1,pred_test_data_time1,data1$app[a1])
a2_1 = cbind.data.frame(test_data_time2,pred_test_data_time2,data1$app[a2])
a3_1 = cbind.data.frame(test_data_time3,pred_test_data_time3,data1$app[a3])
a4_1 = cbind.data.frame(test_data_time4,pred_test_data_time4,data1$app[a4])
a5_1 = cbind.data.frame(test_data_time5,pred_test_data_time5,data1$app[a5])
a6_1 = cbind.data.frame(test_data_time6,pred_test_data_time6,data1$app[a6])
a7_1 = cbind.data.frame(test_data_time7,pred_test_data_time7,data1$app[a7])
a8_1 = cbind.data.frame(test_data_time8,pred_test_data_time8,data1$app[a8])
colnames(a1_1) = c("testdata","testpred","app")
colnames(a2_1) = c("testdata","testpred","app")
colnames(a3_1) = c("testdata","testpred","app")
colnames(a4_1) = c("testdata","testpred","app")
colnames(a5_1) = c("testdata","testpred","app")
colnames(a6_1) = c("testdata","testpred","app")
colnames(a7_1) = c("testdata","testpred","app")
colnames(a8_1) = c("testdata","testpred","app")
a1_2 = cbind.data.frame(train_data_time1,pred_train_data_time1,data1$app[b1])
a2_2 = cbind.data.frame(train_data_time2,pred_train_data_time2,data1$app[b2])
a3_2 = cbind.data.frame(train_data_time3,pred_train_data_time3,data1$app[b3])
a4_2 = cbind.data.frame(train_data_time4,pred_train_data_time4,data1$app[b4])
a5_2 = cbind.data.frame(train_data_time5,pred_train_data_time5,data1$app[b5])
a6_2 = cbind.data.frame(train_data_time6,pred_train_data_time6,data1$app[b6])
a7_2 = cbind.data.frame(train_data_time7,pred_train_data_time7,data1$app[b7])
a8_2 = cbind.data.frame(train_data_time8,pred_train_data_time8,data1$app[b8])
colnames(a1_2) = c("traindata","trainpred","app")
colnames(a2_2) = c("traindata","trainpred","app")
colnames(a3_2) = c("traindata","trainpred","app")
colnames(a4_2) = c("traindata","trainpred","app")
colnames(a5_2) = c("traindata","trainpred","app")
colnames(a6_2) = c("traindata","trainpred","app")
colnames(a7_2) = c("traindata","trainpred","app")
colnames(a8_2) = c("traindata","trainpred","app")
p1_1 = ggplot(a1_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p1_2 = ggplot(a1_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_1 = ggplot(a2_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_2 = ggplot(a2_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_1 = ggplot(a3_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_2 = ggplot(a3_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_1 = ggplot(a4_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_2 = ggplot(a4_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_1 = ggplot(a5_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_2 = ggplot(a5_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_1 = ggplot(a6_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_2 = ggplot(a6_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_1 = ggplot(a7_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_2 = ggplot(a7_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_1 = ggplot(a8_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_2 = ggplot(a8_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))

grid.arrange(p1_1,p1_2,nrow=1)
grid.arrange(p2_1,p2_2,nrow=1)
grid.arrange(p3_1,p3_2,nrow=1)
grid.arrange(p4_1,p4_2,nrow=1)
grid.arrange(p5_1,p5_2,nrow=1)
grid.arrange(p6_1,p6_2,nrow=1)
grid.arrange(p7_1,p7_2,nrow=1)
grid.arrange(p8_1,p8_2,nrow=1)

#Enter filepath here
write.csv(a1_1,"C:/Users/51264/Desktop/test1.csv")
write.csv(a1_2,"C:/Users/51264/Desktop/train1.csv")
write.csv(a2_1,"C:/Users/51264/Desktop/test2.csv")
write.csv(a2_2,"C:/Users/51264/Desktop/train2.csv")
write.csv(a3_1,"C:/Users/51264/Desktop/test3.csv")
write.csv(a3_2,"C:/Users/51264/Desktop/train3.csv")
write.csv(a4_1,"C:/Users/51264/Desktop/test4.csv")
write.csv(a4_2,"C:/Users/51264/Desktop/train4.csv")
write.csv(a5_1,"C:/Users/51264/Desktop/test5.csv")
write.csv(a5_2,"C:/Users/51264/Desktop/train5.csv")
write.csv(a6_1,"C:/Users/51264/Desktop/test6.csv")
write.csv(a6_2,"C:/Users/51264/Desktop/train6.csv")
write.csv(a7_1,"C:/Users/51264/Desktop/test7.csv")
write.csv(a7_2,"C:/Users/51264/Desktop/train7.csv")
write.csv(a8_1,"C:/Users/51264/Desktop/test8.csv")
write.csv(a8_2,"C:/Users/51264/Desktop/train8.csv")


###################
#Fit Linear Model 1
###################

#Split the data into a test and training set
index = sample(1:nrow(data1),round(0.80*nrow(data1)))
train_data = as.data.frame(data1[index,])
test_data = as.data.frame(data1[-index,])
#Train the model
model_lin = lm(time~app*machine*nodes*instructions*data_trans,data=train_data)
#Predicted values train
ptrain = model_lin$fitted.values
#Predicted values test
ptest = predict(model_lin,test_data)
#Combine into data frames
a07 = cbind.data.frame(test_data$time,ptest,data1$app[-index])
colnames(a07) = c("testdata","testpred","app")
a08 = cbind.data.frame(train_data$time,ptrain,data1$app[index])
colnames(a08) = c("traindata","trainpred","app")
p01 = ggplot(a07,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p02 = ggplot(a08,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
grid.arrange(p01,p02,nrow=1)

#######################################################################################################################
#Linear Model 1 All 8 Machine x App Exclusions One-by-One This Didn't Work Because APP*MACHINE Interaction Was Included
#######################################################################################################################

#Allocate excluded points to the test data
a1 = c(1:450)
b1 = c(451:3600)
a2 = c(451:900)
b2 = c(1:450,901:3600)
a3 = c(901:1350)
b3 = c(1:900,1351:3600)
a4 = c(1351:1800)
b4 = c(1:1350,1801:3600)
a5 = c(1801:2250)
b5 = c(1:1800,2251:3600)
a6 = c(2251:2700)
b6 = c(1:2250,2701:3600)
a7 = c(2701:3150)
b7 = c(1:2700,3151:3600)
a8 = c(3151:3600)
b8 = c(1:3150)
#Split the data into a test and training set
test_data1 = as.data.frame(data1[a1,])
train_data1 = as.data.frame(data1[b1,])
test_data2 = as.data.frame(data1[a2,])
train_data2 = as.data.frame(data1[b2,])
test_data3 = as.data.frame(data1[a3,])
train_data3 = as.data.frame(data1[b3,])
test_data4 = as.data.frame(data1[a4,])
train_data4 = as.data.frame(data1[b4,])
test_data5 = as.data.frame(data1[a5,])
train_data5 = as.data.frame(data1[b5,])
test_data6 = as.data.frame(data1[a6,])
train_data6 = as.data.frame(data1[b6,])
test_data7 = as.data.frame(data1[a7,])
train_data7 = as.data.frame(data1[b7,])
test_data8 = as.data.frame(data1[a8,])
train_data8 = as.data.frame(data1[b8,])
#Train the model
model_lin1 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data1)
model_lin2 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data2)
model_lin3 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data3)
model_lin4 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data4)
model_lin5 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data5)
model_lin6 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data6)
model_lin7 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data7)
model_lin8 = lm(time~app*machine*nodes*instructions*data_trans,data=train_data8)
#Predicted values train
ptrain1 = model_lin1$fitted.values
ptrain2 = model_lin2$fitted.values
ptrain3 = model_lin3$fitted.values
ptrain4 = model_lin4$fitted.values
ptrain5 = model_lin5$fitted.values
ptrain6 = model_lin6$fitted.values
ptrain7 = model_lin7$fitted.values
ptrain8 = model_lin8$fitted.values
#Predicted values test
ptest1 = predict(model_lin1,test_data1)
ptest2 = predict(model_lin2,test_data2)
ptest3 = predict(model_lin3,test_data3)
ptest4 = predict(model_lin4,test_data4)
ptest5 = predict(model_lin5,test_data5)
ptest6 = predict(model_lin6,test_data6)
ptest7 = predict(model_lin7,test_data7)
ptest8 = predict(model_lin8,test_data8)
#Combine into data frames
a1_1 = cbind.data.frame(test_data1$time,ptest1,data1$app[a1])
a2_1 = cbind.data.frame(test_data2$time,ptest2,data1$app[a2])
a3_1 = cbind.data.frame(test_data3$time,ptest3,data1$app[a3])
a4_1 = cbind.data.frame(test_data4$time,ptest4,data1$app[a4])
a5_1 = cbind.data.frame(test_data5$time,ptest5,data1$app[a5])
a6_1 = cbind.data.frame(test_data6$time,ptest6,data1$app[a6])
a7_1 = cbind.data.frame(test_data7$time,ptest7,data1$app[a7])
a8_1 = cbind.data.frame(test_data8$time,ptest8,data1$app[a8])
colnames(a1_1) = c("testdata","testpred","app")
colnames(a2_1) = c("testdata","testpred","app")
colnames(a3_1) = c("testdata","testpred","app")
colnames(a4_1) = c("testdata","testpred","app")
colnames(a5_1) = c("testdata","testpred","app")
colnames(a6_1) = c("testdata","testpred","app")
colnames(a7_1) = c("testdata","testpred","app")
colnames(a8_1) = c("testdata","testpred","app")
a1_2 = cbind.data.frame(train_data1$time,ptrain1,data1$app[b1])
a2_2 = cbind.data.frame(train_data2$time,ptrain2,data1$app[b2])
a3_2 = cbind.data.frame(train_data3$time,ptrain3,data1$app[b3])
a4_2 = cbind.data.frame(train_data4$time,ptrain4,data1$app[b4])
a5_2 = cbind.data.frame(train_data5$time,ptrain5,data1$app[b5])
a6_2 = cbind.data.frame(train_data6$time,ptrain6,data1$app[b6])
a7_2 = cbind.data.frame(train_data7$time,ptrain7,data1$app[b7])
a8_2 = cbind.data.frame(train_data8$time,ptrain8,data1$app[b8])
colnames(a1_2) = c("traindata","trainpred","app")
colnames(a2_2) = c("traindata","trainpred","app")
colnames(a3_2) = c("traindata","trainpred","app")
colnames(a4_2) = c("traindata","trainpred","app")
colnames(a5_2) = c("traindata","trainpred","app")
colnames(a6_2) = c("traindata","trainpred","app")
colnames(a7_2) = c("traindata","trainpred","app")
colnames(a8_2) = c("traindata","trainpred","app")
p1_1 = ggplot(a1_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p1_2 = ggplot(a1_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_1 = ggplot(a2_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_2 = ggplot(a2_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_1 = ggplot(a3_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_2 = ggplot(a3_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_1 = ggplot(a4_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_2 = ggplot(a4_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_1 = ggplot(a5_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_2 = ggplot(a5_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_1 = ggplot(a6_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_2 = ggplot(a6_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_1 = ggplot(a7_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_2 = ggplot(a7_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_1 = ggplot(a8_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_2 = ggplot(a8_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))

grid.arrange(p1_1,p1_2,nrow=1)
grid.arrange(p2_1,p2_2,nrow=1)
grid.arrange(p3_1,p3_2,nrow=1)
grid.arrange(p4_1,p4_2,nrow=1)
grid.arrange(p5_1,p5_2,nrow=1)
grid.arrange(p6_1,p6_2,nrow=1)
grid.arrange(p7_1,p7_2,nrow=1)
grid.arrange(p8_1,p8_2,nrow=1)

#########################################################
#Linear Model 2 All 8 Machine x App Exclusions One-by-One
#########################################################

#Allocate excluded points to the test data
a1 = c(1:450)
b1 = c(451:3600)
a2 = c(451:900)
b2 = c(1:450,901:3600)
a3 = c(901:1350)
b3 = c(1:900,1351:3600)
a4 = c(1351:1800)
b4 = c(1:1350,1801:3600)
a5 = c(1801:2250)
b5 = c(1:1800,2251:3600)
a6 = c(2251:2700)
b6 = c(1:2250,2701:3600)
a7 = c(2701:3150)
b7 = c(1:2700,3151:3600)
a8 = c(3151:3600)
b8 = c(1:3150)
#Split the data into a test and training set
test_data1 = as.data.frame(data1[a1,])
train_data1 = as.data.frame(data1[b1,])
test_data2 = as.data.frame(data1[a2,])
train_data2 = as.data.frame(data1[b2,])
test_data3 = as.data.frame(data1[a3,])
train_data3 = as.data.frame(data1[b3,])
test_data4 = as.data.frame(data1[a4,])
train_data4 = as.data.frame(data1[b4,])
test_data5 = as.data.frame(data1[a5,])
train_data5 = as.data.frame(data1[b5,])
test_data6 = as.data.frame(data1[a6,])
train_data6 = as.data.frame(data1[b6,])
test_data7 = as.data.frame(data1[a7,])
train_data7 = as.data.frame(data1[b7,])
test_data8 = as.data.frame(data1[a8,])
train_data8 = as.data.frame(data1[b8,])
#Train the model
model_lin1 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data1)
model_lin2 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data2)
model_lin3 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data3)
model_lin4 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data4)
model_lin5 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data5)
model_lin6 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data6)
model_lin7 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data7)
model_lin8 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data8)
#Predicted values train
ptrain1 = model_lin1$fitted.values
ptrain2 = model_lin2$fitted.values
ptrain3 = model_lin3$fitted.values
ptrain4 = model_lin4$fitted.values
ptrain5 = model_lin5$fitted.values
ptrain6 = model_lin6$fitted.values
ptrain7 = model_lin7$fitted.values
ptrain8 = model_lin8$fitted.values
#Predicted values test
ptest1 = predict(model_lin1,test_data1)
ptest2 = predict(model_lin2,test_data2)
ptest3 = predict(model_lin3,test_data3)
ptest4 = predict(model_lin4,test_data4)
ptest5 = predict(model_lin5,test_data5)
ptest6 = predict(model_lin6,test_data6)
ptest7 = predict(model_lin7,test_data7)
ptest8 = predict(model_lin8,test_data8)
#Combine into data frames
a1_1 = cbind.data.frame(test_data1$time,ptest1,data1$app[a1])
a2_1 = cbind.data.frame(test_data2$time,ptest2,data1$app[a2])
a3_1 = cbind.data.frame(test_data3$time,ptest3,data1$app[a3])
a4_1 = cbind.data.frame(test_data4$time,ptest4,data1$app[a4])
a5_1 = cbind.data.frame(test_data5$time,ptest5,data1$app[a5])
a6_1 = cbind.data.frame(test_data6$time,ptest6,data1$app[a6])
a7_1 = cbind.data.frame(test_data7$time,ptest7,data1$app[a7])
a8_1 = cbind.data.frame(test_data8$time,ptest8,data1$app[a8])
colnames(a1_1) = c("testdata","testpred","app")
colnames(a2_1) = c("testdata","testpred","app")
colnames(a3_1) = c("testdata","testpred","app")
colnames(a4_1) = c("testdata","testpred","app")
colnames(a5_1) = c("testdata","testpred","app")
colnames(a6_1) = c("testdata","testpred","app")
colnames(a7_1) = c("testdata","testpred","app")
colnames(a8_1) = c("testdata","testpred","app")
a1_2 = cbind.data.frame(train_data1$time,ptrain1,data1$app[b1])
a2_2 = cbind.data.frame(train_data2$time,ptrain2,data1$app[b2])
a3_2 = cbind.data.frame(train_data3$time,ptrain3,data1$app[b3])
a4_2 = cbind.data.frame(train_data4$time,ptrain4,data1$app[b4])
a5_2 = cbind.data.frame(train_data5$time,ptrain5,data1$app[b5])
a6_2 = cbind.data.frame(train_data6$time,ptrain6,data1$app[b6])
a7_2 = cbind.data.frame(train_data7$time,ptrain7,data1$app[b7])
a8_2 = cbind.data.frame(train_data8$time,ptrain8,data1$app[b8])
colnames(a1_2) = c("traindata","trainpred","app")
colnames(a2_2) = c("traindata","trainpred","app")
colnames(a3_2) = c("traindata","trainpred","app")
colnames(a4_2) = c("traindata","trainpred","app")
colnames(a5_2) = c("traindata","trainpred","app")
colnames(a6_2) = c("traindata","trainpred","app")
colnames(a7_2) = c("traindata","trainpred","app")
colnames(a8_2) = c("traindata","trainpred","app")
p1_1 = ggplot(a1_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p1_2 = ggplot(a1_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_1 = ggplot(a2_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_2 = ggplot(a2_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_1 = ggplot(a3_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_2 = ggplot(a3_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_1 = ggplot(a4_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_2 = ggplot(a4_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_1 = ggplot(a5_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_2 = ggplot(a5_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_1 = ggplot(a6_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_2 = ggplot(a6_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_1 = ggplot(a7_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_2 = ggplot(a7_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_1 = ggplot(a8_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_2 = ggplot(a8_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))

grid.arrange(p1_1,p1_2,p2_1,p2_2,nrow=2)
grid.arrange(p3_1,p3_2,p4_1,p4_2,nrow=2)
grid.arrange(p5_1,p5_2,p6_1,p6_2,nrow=2)
grid.arrange(p7_1,p7_2,p8_1,p8_2,nrow=2)

test = cbind(a1_1,a2_1,a3_1,a4_1,a5_1,a6_1,a7_1,a8_1)
train = cbind(a1_2,a2_2,a3_2,a4_2,a5_2,a6_2,a7_2,a8_2)

#Enter filepath here
write.csv(test,"C:/Users/CJ/Desktop/test.csv")
write.csv(train,"C:/Users/CJ/Desktop/train.csv")

########################################################################################################################
########################################################################################################################
#Neural Network Modeling of Computer Data Part III
########################################################################################################################
########################################################################################################################

##############
#Load Packages
##############

#Load libraries
library(plyr)
library(ggplot2)
library(gridExtra)
library(neuralnet)
library(reshape2)

#############
#Read in Data
#############

#Read in [newdata4.csv]
data1 = read.csv(file.choose(),header=T)
data1$app = factor(data1$app,levels=c("MPI_job","web_server","cassandra_stress","spark_word_count"))
data1$app_num = as.numeric(data1$app)
data1$machine_num = as.numeric(data1$machine)
data1$nodes_factor = as.factor(data1$nodes)

#############################################################################################################
#Prepare Data for ANN Model 2 This Time Remove Machine Entirely While Leaving Machine Attributes CPU and MIPS
#############################################################################################################

#Save activation functions
sigmoid = function(x){1/(1+exp(-x))}
hypertan = function(x){tanh(x)}

#Retain only numerically coded data
data4 = data1[,c(3:9)]

#Scale instructions 0-1
data4$instructions = scale(data4$instructions,center=min(data4$instructions),scale=max(data4$instructions)-min(data4$instructions))
#Scale time 0-1
data4$time = scale(data4$time,center=min(data4$time),scale=max(data4$time)-min(data4$time))
#Scale time 0-1
data4$data_trans = scale(data4$data_trans,center=min(data4$data_trans),scale=max(data4$data_trans)-min(data4$data_trans))
#Scale cpu 0-1
data4$cpu = scale(data4$cpu,center=min(data4$cpu),scale=max(data4$cpu)-min(data4$cpu))
#Scale mips 0-1
data4$mips = scale(data4$mips,center=min(data4$mips),scale=max(data4$mips)-min(data4$mips))

#For loop scaling machine_num -1,1 and app_num -1,1 and nodes -1,1
data4$n01 = rep(-1,nrow(data4))
data4$n02 = rep(-1,nrow(data4))
data4$n03 = rep(-1,nrow(data4))
data4$n04 = rep(-1,nrow(data4))
data4$n05 = rep(-1,nrow(data4))
data4$n06 = rep(-1,nrow(data4))
data4$n07 = rep(-1,nrow(data4))
data4$n08 = rep(-1,nrow(data4))
data4$n09 = rep(-1,nrow(data4))
data4$n10 = rep(-1,nrow(data4))
data4$n11 = rep(-1,nrow(data4))
data4$n12 = rep(-1,nrow(data4))
data4$n13 = rep(-1,nrow(data4))
data4$n14 = rep(-1,nrow(data4))
data4$n15 = rep(-1,nrow(data4))
data4$appa_coded = rep(-1,nrow(data4))
data4$appb_coded = rep(-1,nrow(data4))
data4$appc_coded = rep(-1,nrow(data4))
data4$appd_coded = rep(-1,nrow(data4))
for (i in 1:nrow(data4)){
  if (data4$app_num[i]==1){
    data4$appa_coded[i] = 1
  }
  if (data4$app_num[i]==2){
    data4$appb_coded[i] = 1
  }
  if (data4$app_num[i]==3){
    data4$appc_coded[i] = 1
  }
  if (data4$app_num[i]==4){
    data4$appd_coded[i] = 1
  }
  if (data4$nodes[i]==1){
    data4$n01[i] = 1
  }
  if (data4$nodes[i]==2){
    data4$n02[i] = 1
  }
  if (data4$nodes[i]==3){
    data4$n03[i] = 1
  }
  if (data4$nodes[i]==4){
    data4$n04[i] = 1
  }
  if (data4$nodes[i]==5){
    data4$n05[i] = 1
  }
  if (data4$nodes[i]==6){
    data4$n06[i] = 1
  }
  if (data4$nodes[i]==7){
    data4$n07[i] = 1
  }
  if (data4$nodes[i]==8){
    data4$n08[i] = 1
  }
  if (data4$nodes[i]==9){
    data4$n09[i] = 1
  }
  if (data4$nodes[i]==10){
    data4$n10[i] = 1
  }
  if (data4$nodes[i]==11){
    data4$n11[i] = 1
  }
  if (data4$nodes[i]==12){
    data4$n12[i] = 1
  }
  if (data4$nodes[i]==13){
    data4$n13[i] = 1
  }
  if (data4$nodes[i]==14){
    data4$n14[i] = 1
  }
  if (data4$nodes[i]==15){
    data4$n15[i] = 1
  }
}
#Reduce data4 to only coded columns
data4 = data4[,c(1:2,4:6,8:26)]

###########################
#Fit Neural Network Model 2
###########################

#Split the data into a test and training set
index = sample(1:nrow(data4),round(0.80*nrow(data4)))
train_data = as.data.frame(data4[index,])
test_data = as.data.frame(data4[-index,])
#Train the model
model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
#Predict test data from trained nn
pred_nn3_scaled = compute(model_nn3,test_data[,c(1:4,6:24)])
#Calculate max and min time for rescaling
min_time = min(data1$time)
max_time = max(data1$time)
#Rescale test prediction
pred_test_data_time = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
#Rescale test actual
test_data_time = test_data$time*(max_time-min_time)+min_time
#Rescale train prediction
pred_train_data_time = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
#Rescale train actual
train_data_time = train_data$time*(max_time-min_time)+min_time
#Combine into data frames
a07 = cbind.data.frame(test_data_time,pred_test_data_time,data1$app[-index])
colnames(a07) = c("testdata","testpred","app")
a08 = cbind.data.frame(train_data_time,pred_train_data_time,data1$app[index])
colnames(a08) = c("traindata","trainpred","app")
p01 = ggplot(a07,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p02 = ggplot(a08,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
grid.arrange(p01,p02,nrow=1)

MSE.test = mean((a07$testdata-a07$testpred)^2)
MSE.train = mean((a08$traindata-a08$trainpred)^2)
MSE.test
MSE.train

MSE.testp = mean(sqrt((a07$testdata-a07$testpred)^2)/a07$testdata)*100
MSE.trainp = mean(sqrt((a08$traindata-a08$trainpred)^2)/a08$traindata)*100
MSE.testp
MSE.trainp

#plot(model_nn3)

######################################################
#ANN Model 2 All 8 Machine x App Exclusions One-by-One
######################################################

#Allocate excluded points to the test data
a1 = c(1:450)
b1 = c(451:3600)
a2 = c(451:900)
b2 = c(1:450,901:3600)
a3 = c(901:1350)
b3 = c(1:900,1351:3600)
a4 = c(1351:1800)
b4 = c(1:1350,1801:3600)
a5 = c(1801:2250)
b5 = c(1:1800,2251:3600)
a6 = c(2251:2700)
b6 = c(1:2250,2701:3600)
a7 = c(2701:3150)
b7 = c(1:2700,3151:3600)
a8 = c(3151:3600)
b8 = c(1:3150)
#Split the data into a test and training set
test_data1 = as.data.frame(data4[a1,])
train_data1 = as.data.frame(data4[b1,])
test_data2 = as.data.frame(data4[a2,])
train_data2 = as.data.frame(data4[b2,])
test_data3 = as.data.frame(data4[a3,])
train_data3 = as.data.frame(data4[b3,])
test_data4 = as.data.frame(data4[a4,])
train_data4 = as.data.frame(data4[b4,])
test_data5 = as.data.frame(data4[a5,])
train_data5 = as.data.frame(data4[b5,])
test_data6 = as.data.frame(data4[a6,])
train_data6 = as.data.frame(data4[b6,])
test_data7 = as.data.frame(data4[a7,])
train_data7 = as.data.frame(data4[b7,])
test_data8 = as.data.frame(data4[a8,])
train_data8 = as.data.frame(data4[b8,])
#Train the model
model_nn1 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data1,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn2 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data2,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data4,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn4 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data4,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn5 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data5,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn6 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data6,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn7 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data7,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
model_nn8 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data8,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
#Predict test data from trained nn
pred_nn1_scaled = compute(model_nn1,test_data1[,c(1:4,6:24)])
pred_nn2_scaled = compute(model_nn2,test_data2[,c(1:4,6:24)])
pred_nn3_scaled = compute(model_nn3,test_data4[,c(1:4,6:24)])
pred_nn4_scaled = compute(model_nn4,test_data4[,c(1:4,6:24)])
pred_nn5_scaled = compute(model_nn5,test_data5[,c(1:4,6:24)])
pred_nn6_scaled = compute(model_nn6,test_data6[,c(1:4,6:24)])
pred_nn7_scaled = compute(model_nn7,test_data7[,c(1:4,6:24)])
pred_nn8_scaled = compute(model_nn8,test_data8[,c(1:4,6:24)])
#Rescale test prediction
pred_test_data_time1 = pred_nn1_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time2 = pred_nn2_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time3 = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time4 = pred_nn4_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time5 = pred_nn5_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time6 = pred_nn6_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time7 = pred_nn7_scaled$net.result*(max_time-min_time)+min_time
pred_test_data_time8 = pred_nn8_scaled$net.result*(max_time-min_time)+min_time
#Rescale test actual
test_data_time1 = test_data1$time*(max_time-min_time)+min_time
test_data_time2 = test_data2$time*(max_time-min_time)+min_time
test_data_time3 = test_data4$time*(max_time-min_time)+min_time
test_data_time4 = test_data4$time*(max_time-min_time)+min_time
test_data_time5 = test_data5$time*(max_time-min_time)+min_time
test_data_time6 = test_data6$time*(max_time-min_time)+min_time
test_data_time7 = test_data7$time*(max_time-min_time)+min_time
test_data_time8 = test_data8$time*(max_time-min_time)+min_time
#Rescale train prediction
pred_train_data_time1 = model_nn1$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time2 = model_nn2$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time3 = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time4 = model_nn4$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time5 = model_nn5$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time6 = model_nn6$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time7 = model_nn7$net.result[[1]][,1]*(max_time-min_time)+min_time
pred_train_data_time8 = model_nn8$net.result[[1]][,1]*(max_time-min_time)+min_time
#Rescale train actual
train_data_time1 = train_data1$time*(max_time-min_time)+min_time
train_data_time2 = train_data2$time*(max_time-min_time)+min_time
train_data_time3 = train_data4$time*(max_time-min_time)+min_time
train_data_time4 = train_data4$time*(max_time-min_time)+min_time
train_data_time5 = train_data5$time*(max_time-min_time)+min_time
train_data_time6 = train_data6$time*(max_time-min_time)+min_time
train_data_time7 = train_data7$time*(max_time-min_time)+min_time
train_data_time8 = train_data8$time*(max_time-min_time)+min_time
#Combine into data frames
a1_1 = cbind.data.frame(test_data_time1,pred_test_data_time1,data1$app[a1])
a2_1 = cbind.data.frame(test_data_time2,pred_test_data_time2,data1$app[a2])
a3_1 = cbind.data.frame(test_data_time3,pred_test_data_time3,data1$app[a3])
a4_1 = cbind.data.frame(test_data_time4,pred_test_data_time4,data1$app[a4])
a5_1 = cbind.data.frame(test_data_time5,pred_test_data_time5,data1$app[a5])
a6_1 = cbind.data.frame(test_data_time6,pred_test_data_time6,data1$app[a6])
a7_1 = cbind.data.frame(test_data_time7,pred_test_data_time7,data1$app[a7])
a8_1 = cbind.data.frame(test_data_time8,pred_test_data_time8,data1$app[a8])
colnames(a1_1) = c("testdata","testpred","app")
colnames(a2_1) = c("testdata","testpred","app")
colnames(a3_1) = c("testdata","testpred","app")
colnames(a4_1) = c("testdata","testpred","app")
colnames(a5_1) = c("testdata","testpred","app")
colnames(a6_1) = c("testdata","testpred","app")
colnames(a7_1) = c("testdata","testpred","app")
colnames(a8_1) = c("testdata","testpred","app")
a1_2 = cbind.data.frame(train_data_time1,pred_train_data_time1,data1$app[b1])
a2_2 = cbind.data.frame(train_data_time2,pred_train_data_time2,data1$app[b2])
a3_2 = cbind.data.frame(train_data_time3,pred_train_data_time3,data1$app[b3])
a4_2 = cbind.data.frame(train_data_time4,pred_train_data_time4,data1$app[b4])
a5_2 = cbind.data.frame(train_data_time5,pred_train_data_time5,data1$app[b5])
a6_2 = cbind.data.frame(train_data_time6,pred_train_data_time6,data1$app[b6])
a7_2 = cbind.data.frame(train_data_time7,pred_train_data_time7,data1$app[b7])
a8_2 = cbind.data.frame(train_data_time8,pred_train_data_time8,data1$app[b8])
colnames(a1_2) = c("traindata","trainpred","app")
colnames(a2_2) = c("traindata","trainpred","app")
colnames(a3_2) = c("traindata","trainpred","app")
colnames(a4_2) = c("traindata","trainpred","app")
colnames(a5_2) = c("traindata","trainpred","app")
colnames(a6_2) = c("traindata","trainpred","app")
colnames(a7_2) = c("traindata","trainpred","app")
colnames(a8_2) = c("traindata","trainpred","app")
p1_1 = ggplot(a1_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p1_2 = ggplot(a1_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_1 = ggplot(a2_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_2 = ggplot(a2_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_1 = ggplot(a3_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_2 = ggplot(a3_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_1 = ggplot(a4_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_2 = ggplot(a4_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_1 = ggplot(a5_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_2 = ggplot(a5_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_1 = ggplot(a6_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_2 = ggplot(a6_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_1 = ggplot(a7_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi2 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_2 = ggplot(a7_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_1 = ggplot(a8_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TEST DATA: Rpi3 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_2 = ggplot(a8_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Neural Net Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))

grid.arrange(p1_1,p1_2,nrow=1)
grid.arrange(p2_1,p2_2,nrow=1)
grid.arrange(p3_1,p3_2,nrow=1)
grid.arrange(p4_1,p4_2,nrow=1)
grid.arrange(p5_1,p5_2,nrow=1)
grid.arrange(p6_1,p6_2,nrow=1)
grid.arrange(p7_1,p7_2,nrow=1)
grid.arrange(p8_1,p8_2,nrow=1)

MSE.test1 = mean((a1_1$testdata-a1_1$testpred)^2)
MSE.train1 = mean((a1_2$traindata-a1_2$trainpred)^2)
MSE.test2 = mean((a2_1$testdata-a2_1$testpred)^2)
MSE.train2 = mean((a2_2$traindata-a2_2$trainpred)^2)
MSE.test3 = mean((a3_1$testdata-a3_1$testpred)^2)
MSE.train3 = mean((a3_2$traindata-a3_2$trainpred)^2)
MSE.test4 = mean((a4_1$testdata-a4_1$testpred)^2)
MSE.train4 = mean((a4_2$traindata-a4_2$trainpred)^2)
MSE.test5 = mean((a5_1$testdata-a5_1$testpred)^2)
MSE.train5 = mean((a5_2$traindata-a5_2$trainpred)^2)
MSE.test6 = mean((a6_1$testdata-a6_1$testpred)^2)
MSE.train6 = mean((a6_2$traindata-a6_2$trainpred)^2)
MSE.test7 = mean((a7_1$testdata-a7_1$testpred)^2)
MSE.train7 = mean((a7_2$traindata-a7_2$trainpred)^2)
MSE.test8 = mean((a8_1$testdata-a8_1$testpred)^2)
MSE.train8 = mean((a8_2$traindata-a8_2$trainpred)^2)
MSE.test1
MSE.train1
MSE.test2
MSE.train2
MSE.test3
MSE.train3
MSE.test4
MSE.train4
MSE.test5
MSE.train5
MSE.test6
MSE.train6
MSE.test7
MSE.train7
MSE.test8
MSE.train8

MSE.test1p = mean(sqrt((a1_1$testdata-a1_1$testpred)^2)/a1_1$testdata)*100
MSE.train1p = mean(sqrt((a1_2$traindata-a1_2$trainpred)^2)/a1_2$traindata)*100
MSE.test2p = mean(sqrt((a2_1$testdata-a2_1$testpred)^2)/a2_1$testdata)*100
MSE.train2p = mean(sqrt((a2_2$traindata-a2_2$trainpred)^2)/a2_2$traindata)*100
MSE.test3p = mean(sqrt((a3_1$testdata-a3_1$testpred)^2)/a3_1$testdata)*100
MSE.train3p = mean(sqrt((a3_2$traindata-a3_2$trainpred)^2)/a3_2$traindata)*100
MSE.test4p = mean(sqrt((a4_1$testdata-a4_1$testpred)^2)/a4_1$testdata)*100
MSE.train4p = mean(sqrt((a4_2$traindata-a4_2$trainpred)^2)/a4_2$traindata)*100
MSE.test5p = mean(sqrt((a5_1$testdata-a5_1$testpred)^2)/a5_1$testdata)*100
MSE.train5p = mean(sqrt((a5_2$traindata-a5_2$trainpred)^2)/a5_2$traindata)*100
MSE.test6p = mean(sqrt((a6_1$testdata-a6_1$testpred)^2)/a6_1$testdata)*100
MSE.train6p = mean(sqrt((a6_2$traindata-a6_2$trainpred)^2)/a6_2$traindata)*100
MSE.test7p = mean(sqrt((a7_1$testdata-a7_1$testpred)^2)/a7_1$testdata)*100
MSE.train7p = mean(sqrt((a7_2$traindata-a7_2$trainpred)^2)/a7_2$traindata)*100
MSE.test8p = mean(sqrt((a8_1$testdata-a8_1$testpred)^2)/a8_1$testdata)*100
MSE.train8p = mean(sqrt((a8_2$traindata-a8_2$trainpred)^2)/a8_2$traindata)*100
MSE.test1p
MSE.train1p
MSE.test2p
MSE.train2p
MSE.test3p
MSE.train3p
MSE.test4p
MSE.train4p
MSE.test5p
MSE.train5p
MSE.test6p
MSE.train6p
MSE.test7p
MSE.train7p
MSE.test8p
MSE.train8p

#Enter filepath here
#write.csv(a1_1,"C:/Users/51264/Desktop/test1.csv")
#write.csv(a1_2,"C:/Users/51264/Desktop/train1.csv")
#write.csv(a2_1,"C:/Users/51264/Desktop/test2.csv")
#write.csv(a2_2,"C:/Users/51264/Desktop/train2.csv")
#write.csv(a3_1,"C:/Users/51264/Desktop/test3.csv")
#write.csv(a3_2,"C:/Users/51264/Desktop/train3.csv")
#write.csv(a4_1,"C:/Users/51264/Desktop/test4.csv")
#write.csv(a4_2,"C:/Users/51264/Desktop/train4.csv")
#write.csv(a5_1,"C:/Users/51264/Desktop/test5.csv")
#write.csv(a5_2,"C:/Users/51264/Desktop/train5.csv")
#write.csv(a6_1,"C:/Users/51264/Desktop/test6.csv")
#write.csv(a6_2,"C:/Users/51264/Desktop/train6.csv")
#write.csv(a7_1,"C:/Users/51264/Desktop/test7.csv")
#write.csv(a7_2,"C:/Users/51264/Desktop/train7.csv")
#write.csv(a8_1,"C:/Users/51264/Desktop/test8.csv")
#write.csv(a8_2,"C:/Users/51264/Desktop/train8.csv")


###################
#Fit Linear Model 1
###################

#Split the data into a test and training set
index = sample(1:nrow(data1),round(0.80*nrow(data1)))
train_data = as.data.frame(data1[index,])
test_data = as.data.frame(data1[-index,])
#Train the model
model_lin = lm(time~app*machine*nodes*instructions*data_trans,data=train_data)
#Predicted values train
ptrain = model_lin$fitted.values
#Predicted values test
ptest = predict(model_lin,test_data)
#Combine into data frames
a07 = cbind.data.frame(test_data$time,ptest,data1$app[-index])
colnames(a07) = c("testdata","testpred","app")
a08 = cbind.data.frame(train_data$time,ptrain,data1$app[index])
colnames(a08) = c("traindata","trainpred","app")
p01 = ggplot(a07,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p02 = ggplot(a08,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
grid.arrange(p01,p02,nrow=1)

MSE.test = mean((a07$testdata-a07$testpred)^2)
MSE.train = mean((a08$traindata-a08$trainpred)^2)
MSE.test
MSE.train

MSE.testp = mean(sqrt((a07$testdata-a07$testpred)^2)/a07$testdata)*100
MSE.trainp = mean(sqrt((a08$traindata-a08$trainpred)^2)/a08$traindata)*100
MSE.testp
MSE.trainp

#########################################################
#Linear Model 2 All 8 Machine x App Exclusions One-by-One
#########################################################

#Allocate excluded points to the test data
a1 = c(1:450)
b1 = c(451:3600)
a2 = c(451:900)
b2 = c(1:450,901:3600)
a3 = c(901:1350)
b3 = c(1:900,1351:3600)
a4 = c(1351:1800)
b4 = c(1:1350,1801:3600)
a5 = c(1801:2250)
b5 = c(1:1800,2251:3600)
a6 = c(2251:2700)
b6 = c(1:2250,2701:3600)
a7 = c(2701:3150)
b7 = c(1:2700,3151:3600)
a8 = c(3151:3600)
b8 = c(1:3150)
#Split the data into a test and training set
test_data1 = as.data.frame(data1[a1,])
train_data1 = as.data.frame(data1[b1,])
test_data2 = as.data.frame(data1[a2,])
train_data2 = as.data.frame(data1[b2,])
test_data3 = as.data.frame(data1[a3,])
train_data3 = as.data.frame(data1[b3,])
test_data4 = as.data.frame(data1[a4,])
train_data4 = as.data.frame(data1[b4,])
test_data5 = as.data.frame(data1[a5,])
train_data5 = as.data.frame(data1[b5,])
test_data6 = as.data.frame(data1[a6,])
train_data6 = as.data.frame(data1[b6,])
test_data7 = as.data.frame(data1[a7,])
train_data7 = as.data.frame(data1[b7,])
test_data8 = as.data.frame(data1[a8,])
train_data8 = as.data.frame(data1[b8,])
#Train the model
model_lin1 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data1)
model_lin2 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data2)
model_lin3 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data3)
model_lin4 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data4)
model_lin5 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data5)
model_lin6 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data6)
model_lin7 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data7)
model_lin8 = lm(time~app+machine+nodes*instructions*data_trans,data=train_data8)
#Predicted values train
ptrain1 = model_lin1$fitted.values
ptrain2 = model_lin2$fitted.values
ptrain3 = model_lin3$fitted.values
ptrain4 = model_lin4$fitted.values
ptrain5 = model_lin5$fitted.values
ptrain6 = model_lin6$fitted.values
ptrain7 = model_lin7$fitted.values
ptrain8 = model_lin8$fitted.values
#Predicted values test
ptest1 = predict(model_lin1,test_data1)
ptest2 = predict(model_lin2,test_data2)
ptest3 = predict(model_lin3,test_data3)
ptest4 = predict(model_lin4,test_data4)
ptest5 = predict(model_lin5,test_data5)
ptest6 = predict(model_lin6,test_data6)
ptest7 = predict(model_lin7,test_data7)
ptest8 = predict(model_lin8,test_data8)
#Combine into data frames
a1_1 = cbind.data.frame(test_data1$time,ptest1,data1$app[a1])
a2_1 = cbind.data.frame(test_data2$time,ptest2,data1$app[a2])
a3_1 = cbind.data.frame(test_data3$time,ptest3,data1$app[a3])
a4_1 = cbind.data.frame(test_data4$time,ptest4,data1$app[a4])
a5_1 = cbind.data.frame(test_data5$time,ptest5,data1$app[a5])
a6_1 = cbind.data.frame(test_data6$time,ptest6,data1$app[a6])
a7_1 = cbind.data.frame(test_data7$time,ptest7,data1$app[a7])
a8_1 = cbind.data.frame(test_data8$time,ptest8,data1$app[a8])
colnames(a1_1) = c("testdata","testpred","app")
colnames(a2_1) = c("testdata","testpred","app")
colnames(a3_1) = c("testdata","testpred","app")
colnames(a4_1) = c("testdata","testpred","app")
colnames(a5_1) = c("testdata","testpred","app")
colnames(a6_1) = c("testdata","testpred","app")
colnames(a7_1) = c("testdata","testpred","app")
colnames(a8_1) = c("testdata","testpred","app")
a1_2 = cbind.data.frame(train_data1$time,ptrain1,data1$app[b1])
a2_2 = cbind.data.frame(train_data2$time,ptrain2,data1$app[b2])
a3_2 = cbind.data.frame(train_data3$time,ptrain3,data1$app[b3])
a4_2 = cbind.data.frame(train_data4$time,ptrain4,data1$app[b4])
a5_2 = cbind.data.frame(train_data5$time,ptrain5,data1$app[b5])
a6_2 = cbind.data.frame(train_data6$time,ptrain6,data1$app[b6])
a7_2 = cbind.data.frame(train_data7$time,ptrain7,data1$app[b7])
a8_2 = cbind.data.frame(train_data8$time,ptrain8,data1$app[b8])
colnames(a1_2) = c("traindata","trainpred","app")
colnames(a2_2) = c("traindata","trainpred","app")
colnames(a3_2) = c("traindata","trainpred","app")
colnames(a4_2) = c("traindata","trainpred","app")
colnames(a5_2) = c("traindata","trainpred","app")
colnames(a6_2) = c("traindata","trainpred","app")
colnames(a7_2) = c("traindata","trainpred","app")
colnames(a8_2) = c("traindata","trainpred","app")
p1_1 = ggplot(a1_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p1_2 = ggplot(a1_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_1 = ggplot(a2_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 MPI_job):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p2_2 = ggplot(a2_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_1 = ggplot(a3_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p3_2 = ggplot(a3_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_1 = ggplot(a4_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 web_server):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p4_2 = ggplot(a4_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_1 = ggplot(a5_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p5_2 = ggplot(a5_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_1 = ggplot(a6_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 cassandra_stress):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p6_2 = ggplot(a6_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_1 = ggplot(a7_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi2 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p7_2 = ggplot(a7_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_1 = ggplot(a8_1,aes(x=testdata,y=testpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TEST DATA: Rpi3 spark_word_count):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))
p8_2 = ggplot(a8_2,aes(x=traindata,y=trainpred))+
  geom_point(aes(color=app))+
  scale_y_continuous("Predicted Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  scale_x_continuous("Actual Execution Time [s]",breaks=seq(0,1000,100),limits=c(0,1000))+
  ggtitle("Linear Fit (TRAIN DATA):\nActual vs. Predicted Execution Time")+
  geom_abline(intercept=0,slope=1)+
  theme_light()+
  theme(text=element_text(size=14))

grid.arrange(p1_1,p1_2,p2_1,p2_2,nrow=2)
grid.arrange(p3_1,p3_2,p4_1,p4_2,nrow=2)
grid.arrange(p5_1,p5_2,p6_1,p6_2,nrow=2)
grid.arrange(p7_1,p7_2,p8_1,p8_2,nrow=2)

MSE.test1 = mean((a1_1$testdata-a1_1$testpred)^2)
MSE.train1 = mean((a1_2$traindata-a1_2$trainpred)^2)
MSE.test2 = mean((a2_1$testdata-a2_1$testpred)^2)
MSE.train2 = mean((a2_2$traindata-a2_2$trainpred)^2)
MSE.test3 = mean((a3_1$testdata-a3_1$testpred)^2)
MSE.train3 = mean((a3_2$traindata-a3_2$trainpred)^2)
MSE.test4 = mean((a4_1$testdata-a4_1$testpred)^2)
MSE.train4 = mean((a4_2$traindata-a4_2$trainpred)^2)
MSE.test5 = mean((a5_1$testdata-a5_1$testpred)^2)
MSE.train5 = mean((a5_2$traindata-a5_2$trainpred)^2)
MSE.test6 = mean((a6_1$testdata-a6_1$testpred)^2)
MSE.train6 = mean((a6_2$traindata-a6_2$trainpred)^2)
MSE.test7 = mean((a7_1$testdata-a7_1$testpred)^2)
MSE.train7 = mean((a7_2$traindata-a7_2$trainpred)^2)
MSE.test8 = mean((a8_1$testdata-a8_1$testpred)^2)
MSE.train8 = mean((a8_2$traindata-a8_2$trainpred)^2)
MSE.test1
MSE.train1
MSE.test2
MSE.train2
MSE.test3
MSE.train3
MSE.test4
MSE.train4
MSE.test5
MSE.train5
MSE.test6
MSE.train6
MSE.test7
MSE.train7
MSE.test8
MSE.train8

MSE.test1p = mean(sqrt((a1_1$testdata-a1_1$testpred)^2)/a1_1$testdata)*100
MSE.train1p = mean(sqrt((a1_2$traindata-a1_2$trainpred)^2)/a1_2$traindata)*100
MSE.test2p = mean(sqrt((a2_1$testdata-a2_1$testpred)^2)/a2_1$testdata)*100
MSE.train2p = mean(sqrt((a2_2$traindata-a2_2$trainpred)^2)/a2_2$traindata)*100
MSE.test3p = mean(sqrt((a3_1$testdata-a3_1$testpred)^2)/a3_1$testdata)*100
MSE.train3p = mean(sqrt((a3_2$traindata-a3_2$trainpred)^2)/a3_2$traindata)*100
MSE.test4p = mean(sqrt((a4_1$testdata-a4_1$testpred)^2)/a4_1$testdata)*100
MSE.train4p = mean(sqrt((a4_2$traindata-a4_2$trainpred)^2)/a4_2$traindata)*100
MSE.test5p = mean(sqrt((a5_1$testdata-a5_1$testpred)^2)/a5_1$testdata)*100
MSE.train5p = mean(sqrt((a5_2$traindata-a5_2$trainpred)^2)/a5_2$traindata)*100
MSE.test6p = mean(sqrt((a6_1$testdata-a6_1$testpred)^2)/a6_1$testdata)*100
MSE.train6p = mean(sqrt((a6_2$traindata-a6_2$trainpred)^2)/a6_2$traindata)*100
MSE.test7p = mean(sqrt((a7_1$testdata-a7_1$testpred)^2)/a7_1$testdata)*100
MSE.train7p = mean(sqrt((a7_2$traindata-a7_2$trainpred)^2)/a7_2$traindata)*100
MSE.test8p = mean(sqrt((a8_1$testdata-a8_1$testpred)^2)/a8_1$testdata)*100
MSE.train8p = mean(sqrt((a8_2$traindata-a8_2$trainpred)^2)/a8_2$traindata)*100
MSE.test1p
MSE.train1p
MSE.test2p
MSE.train2p
MSE.test3p
MSE.train3p
MSE.test4p
MSE.train4p
MSE.test5p
MSE.train5p
MSE.test6p
MSE.train6p
MSE.test7p
MSE.train7p
MSE.test8p
MSE.train8p

#test = cbind(a1_1,a2_1,a3_1,a4_1,a5_1,a6_1,a7_1,a8_1)
#train = cbind(a1_2,a2_2,a3_2,a4_2,a5_2,a6_2,a7_2,a8_2)

#Enter filepath here
#write.csv(test,"C:/Users/CJ/Desktop/test.csv")
#write.csv(train,"C:/Users/CJ/Desktop/train.csv")

#######################################
#K-FOLD CV Prepare Data for ANN Model 2
#######################################

#Save activation functions
sigmoid = function(x){1/(1+exp(-x))}
hypertan = function(x){tanh(x)}

#Retain only numerically coded data
data4 = data1[,c(3:9)]

#Scale instructions 0-1
data4$instructions = scale(data4$instructions,center=min(data4$instructions),scale=max(data4$instructions)-min(data4$instructions))
#Scale time 0-1
data4$time = scale(data4$time,center=min(data4$time),scale=max(data4$time)-min(data4$time))
#Scale time 0-1
data4$data_trans = scale(data4$data_trans,center=min(data4$data_trans),scale=max(data4$data_trans)-min(data4$data_trans))
#Scale cpu 0-1
data4$cpu = scale(data4$cpu,center=min(data4$cpu),scale=max(data4$cpu)-min(data4$cpu))
#Scale mips 0-1
data4$mips = scale(data4$mips,center=min(data4$mips),scale=max(data4$mips)-min(data4$mips))

#For loop scaling machine_num -1,1 and app_num -1,1 and nodes -1,1
data4$n01 = rep(-1,nrow(data4))
data4$n02 = rep(-1,nrow(data4))
data4$n03 = rep(-1,nrow(data4))
data4$n04 = rep(-1,nrow(data4))
data4$n05 = rep(-1,nrow(data4))
data4$n06 = rep(-1,nrow(data4))
data4$n07 = rep(-1,nrow(data4))
data4$n08 = rep(-1,nrow(data4))
data4$n09 = rep(-1,nrow(data4))
data4$n10 = rep(-1,nrow(data4))
data4$n11 = rep(-1,nrow(data4))
data4$n12 = rep(-1,nrow(data4))
data4$n13 = rep(-1,nrow(data4))
data4$n14 = rep(-1,nrow(data4))
data4$n15 = rep(-1,nrow(data4))
data4$appa_coded = rep(-1,nrow(data4))
data4$appb_coded = rep(-1,nrow(data4))
data4$appc_coded = rep(-1,nrow(data4))
data4$appd_coded = rep(-1,nrow(data4))
for (i in 1:nrow(data4)){
  if (data4$app_num[i]==1){
    data4$appa_coded[i] = 1
  }
  if (data4$app_num[i]==2){
    data4$appb_coded[i] = 1
  }
  if (data4$app_num[i]==3){
    data4$appc_coded[i] = 1
  }
  if (data4$app_num[i]==4){
    data4$appd_coded[i] = 1
  }
  if (data4$nodes[i]==1){
    data4$n01[i] = 1
  }
  if (data4$nodes[i]==2){
    data4$n02[i] = 1
  }
  if (data4$nodes[i]==3){
    data4$n03[i] = 1
  }
  if (data4$nodes[i]==4){
    data4$n04[i] = 1
  }
  if (data4$nodes[i]==5){
    data4$n05[i] = 1
  }
  if (data4$nodes[i]==6){
    data4$n06[i] = 1
  }
  if (data4$nodes[i]==7){
    data4$n07[i] = 1
  }
  if (data4$nodes[i]==8){
    data4$n08[i] = 1
  }
  if (data4$nodes[i]==9){
    data4$n09[i] = 1
  }
  if (data4$nodes[i]==10){
    data4$n10[i] = 1
  }
  if (data4$nodes[i]==11){
    data4$n11[i] = 1
  }
  if (data4$nodes[i]==12){
    data4$n12[i] = 1
  }
  if (data4$nodes[i]==13){
    data4$n13[i] = 1
  }
  if (data4$nodes[i]==14){
    data4$n14[i] = 1
  }
  if (data4$nodes[i]==15){
    data4$n15[i] = 1
  }
}
#Reduce data4 to only coded columns
data4 = data4[,c(1:2,4:6,8:26)]

#####################################
#K-FOLD CV Fit Neural Network Model 2
#####################################

#Perform n fold cross validation
n = 10
data5 = data4
#Randomly shuffle the data
data5 = data5[sample(nrow(data5)),]
#Create n equally sized folds
folds = cut(seq(1,nrow(data5)),breaks=10,labels=FALSE)

#Start FOR LOOP
summary_nn = data.frame(matrix(rep(0,n*6),ncol=6))
colnames(summary_nn) = c("test_mean","train_mean","test_mse","train_mse","test_percent_error","train_percent_error")
for(i in 1:10){
  print(paste(i,Sys.time()))
  #Segment data by fold using the which() function 
  index = which(folds==i,arr.ind=TRUE)
  test_data = data5[index, ]
  train_data = data5[-index, ]
  #Use the test and train data partitions
  #Train the model
  model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
  #Predict test data from trained nn
  pred_nn3_scaled = compute(model_nn3,test_data[,c(1:4,6:24)])
  #Calculate max and min time for rescaling
  min_time = min(data1$time)
  max_time = max(data1$time)
  #Rescale test prediction
  pred_test_data_time = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
  #Rescale test actual
  test_data_time = test_data$time*(max_time-min_time)+min_time
  #Rescale train prediction
  pred_train_data_time = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
  #Rescale train actual
  train_data_time = train_data$time*(max_time-min_time)+min_time
  #Combine into data frames
  test_compare = cbind.data.frame(test_data_time,pred_test_data_time,data1$app[index])
  colnames(test_compare) = c("testdata","testpred","app")
  train_compare = cbind.data.frame(train_data_time,pred_train_data_time,data1$app[-index])
  colnames(train_compare) = c("traindata","trainpred","app")
  summary_nn$test_mean[i] = mean(test_compare$testdata)
  summary_nn$train_mean[i] = mean(train_compare$traindata)
  summary_nn$test_mse[i] = mean((test_compare$testpred-test_compare$testdata)^2)
  summary_nn$train_mse[i] = mean((train_compare$trainpred-train_compare$traindata)^2)
  summary_nn$test_percent_error[i] = mean(100*abs(test_compare$testpred-test_compare$testdata)/test_compare$testdata)
  summary_nn$train_percent_error[i] = mean(100*abs(train_compare$trainpred-train_compare$traindata)/train_compare$traindata)
  print(summary_nn)
  flush.console()
}

#############################################
#K-FOLD CV Fit Neural Network Model 2 5 TIMES
#############################################

m = 5
n = 5
summary_nn = data.frame(matrix(rep(0,n*m*7),ncol=7))
colnames(summary_nn) = c("iteration","test_mean","train_mean","test_mse","train_mse","test_percent_error","train_percent_error")
#Perform n fold cross validation
for (j in 1:m){
data5 = data4
#Randomly shuffle the data
data5 = data5[sample(nrow(data5)),]
#Create n equally sized folds
folds = cut(seq(1,nrow(data5)),breaks=10,labels=FALSE)
for(i in 1:n){
  print(paste(i,Sys.time()))
  #Segment data by fold using the which() function 
  index = which(folds==i,arr.ind=TRUE)
  test_data = data5[index, ]
  train_data = data5[-index, ]
  #Train the model
  model_nn3 = neuralnet(formula=time~cpu+mips+instructions+data_trans+n01+n02+n03+n04+n05+n06+n07+n08+n09+n10+n11+n12+n13+n14+n15+appa_coded+appb_coded+appc_coded+appd_coded,data=train_data,hidden=15,stepmax=1000000,threshold=0.01,act.fct=hypertan)
  #Predict test data from trained nn
  pred_nn3_scaled = compute(model_nn3,test_data[,c(1:4,6:24)])
  #Calculate max and min time for rescaling
  min_time = min(data1$time)
  max_time = max(data1$time)
  #Rescale test prediction
  pred_test_data_time = pred_nn3_scaled$net.result*(max_time-min_time)+min_time
  #Rescale test actual
  test_data_time = test_data$time*(max_time-min_time)+min_time
  #Rescale train prediction
  pred_train_data_time = model_nn3$net.result[[1]][,1]*(max_time-min_time)+min_time
  #Rescale train actual
  train_data_time = train_data$time*(max_time-min_time)+min_time
  #Combine into data frames
  test_compare = cbind.data.frame(test_data_time,pred_test_data_time,data1$app[index])
  colnames(test_compare) = c("testdata","testpred","app")
  train_compare = cbind.data.frame(train_data_time,pred_train_data_time,data1$app[-index])
  colnames(train_compare) = c("traindata","trainpred","app")
  summary_nn$iteration[i+(j-1)*m] = j
  summary_nn$test_mean[i+(j-1)*m] = mean(test_compare$testdata)
  summary_nn$train_mean[i+(j-1)*m] = mean(train_compare$traindata)
  summary_nn$test_mse[i+(j-1)*m] = mean((test_compare$testpred-test_compare$testdata)^2)
  summary_nn$train_mse[i+(j-1)*m] = mean((train_compare$trainpred-train_compare$traindata)^2)
  summary_nn$test_percent_error[i+(j-1)*m] = mean(100*abs(test_compare$testpred-test_compare$testdata)/test_compare$testdata)
  summary_nn$train_percent_error[i+(j-1)*m] = mean(100*abs(train_compare$trainpred-train_compare$traindata)/train_compare$traindata)
  print(summary_nn)
  flush.console()
}
}

##############
#DENSITY PLOTS
##############

head(data1)
data1$machine_app = paste(data1$machine,data1$app)

p1 = ggplot(data1,aes(machine_app,log10(time)))+
  geom_boxplot(alpha=0.5,lwd=1.2,color="springgreen3",fill="blue",outlier.color="cyan")+
  scale_y_continuous(breaks=seq(0,10,1))+
  xlab("Machine x Application Combination")+
  ylab("Log10 Execution Time [seconds]")+
  ggtitle("Log Execution Time Boxplots")+
  theme_dark()+
  theme(text=element_text(size=12))

p2 = ggplot(data1,aes(machine_app,time))+
  geom_boxplot(alpha=0.5,lwd=1.2,color="springgreen3",fill="blue",outlier.color="cyan")+
  xlab("Machine x Application Combination")+
  ylab("Execution Time [seconds]")+
  ggtitle("Execution Time Boxplots")+
  theme_dark()+
  theme(text=element_text(size=12))

grid.arrange(p1,p2,nrow=2)
