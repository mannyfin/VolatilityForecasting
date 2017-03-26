# FIN580 Ordered LASSO


# n = 50
# b = c(7,3,1,0)
# p = length(b)
# x = matrix(rnorm(n*p),nrow = n)
# sigma = 4
# y = x %*% b + sigma * rnorm(n, 0, 1)
# # method = "Solve.QP" by default
# result2 = orderedLasso(x,y, lambda = 1, intercept = TRUE, standardize =TRUE,
# method = "Solve.QP", strongly.ordered = TRUE)
# #http://finzi.psych.upenn.edu/library/orderedLasso/html/predict.orderedLasso.html
# https://www.researchgate.net/figure/281273999_fig2_Figure-2-Lambda-vs-MSE-for-Lasso-fit-Figure-3-Lambda-vs-MSE-for-Elastic-Net-fit

library(Metrics)
library(orderedLasso)
library(penalized)

setwd("C:/Users/YYJ/Desktop/FIN580/Homework1/VolatilityForecasting/src")
xmat_p_1 <- read.csv("xmat_p_1.csv", header = TRUE, sep = ",")
xmat_p_2 <- read.csv("xmat_p_2.csv", header = TRUE, sep = ",")
xmat_p_3 <- read.csv("xmat_p_3.csv", header = TRUE, sep = ",")
ymat_p_1 <- read.csv("ymat_p_1.csv", header = TRUE, sep = ",")
ymat_p_2 <- read.csv("ymat_p_2.csv", header = TRUE, sep = ",")
ymat_p_3 <- read.csv("ymat_p_3.csv", header = TRUE, sep = ",")

last_in_triaing_p_1 = which(xmat_p_1[,"Dates"]=="10/31/2011")
last_in_triaing_p_2 = which(xmat_p_2[,"Dates"]=="10/31/2011")
last_in_triaing_p_3 = which(xmat_p_3[,"Dates"]=="10/31/2011")

# predicted_y_p_1 <- data.frame()
# predicted_y_p_2 <- data.frame()
# predicted_y_p_3 <- data.frame()
observed_y <- ymat_p_1[(last_in_triaing_p_1:nrow(ymat_p_1)),][,c(-1,-2)]
# observed_y_p_2 <- ymat_p_2[(last_in_triaing_p_2:nrow(ymat_p_2)),][,c(-1,-2)]
# observed_y_p_3 <- ymat_p_3[(last_in_triaing_p_3:nrow(ymat_p_3)),][,c(-1,-2)]
rownames(observed_y) <- 1:nrow(observed_y)
# rownames(observed_y_p_2) <- 1:nrow(observed_y_p_2)
# rownames(observed_y_p_3) <- 1:nrow(observed_y_p_3)
observed_y <- exp(observed_y[complete.cases(observed_y),])
# observed_y_p_2 <- observed_y_p_2[complete.cases(observed_y_p_2),]
# observed_y_p_3 <- observed_y_p_3[complete.cases(observed_y_p_3),]
fileNames  = c('AUDUSD','CADUSD','CHFUSD', 'EURUSD', 'GBPUSD', 'JPYUSD', 'NOKUSD', 'NZDUSD', 'SEKUSD')
Dates <- as.Date(xmat_p_1[,2], format = "%m/%d/%Y")
Dates_test <- Dates[which(Dates =="2011-11-01"):length(Dates)]

# Ordered LASSO_training sample
prediction_y_training<-function(lambda,p){
  if(p==1){
    last_in_triaing = last_in_triaing_p_1
    xmat = xmat_p_1
    ymat = ymat_p_1
  }  else if(p==2){
    last_in_triaing = last_in_triaing_p_2
    xmat = xmat_p_2
    ymat = ymat_p_2
  }  else if(p==3){
    last_in_triaing = last_in_triaing_p_3
    xmat = xmat_p_3
    ymat = ymat_p_3
  }
  predicted_y = vector()
  for (k in c(1:9)){
    i = last_in_triaing-1
    x = xmat[(1:i),][,c(-1,-2)]
    y = ymat[(1:i),][,c(-1,-2)]
    x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)])
    result = orderedLasso(as.matrix(x),as.numeric(y[,k]), lambda, intercept = TRUE, standardize =TRUE,method = "Solve.QP", strongly.ordered = TRUE)
    predicted_y <- c(predicted_y,exp(predict(result, x_for_predict)$yhat.ordered))
  }
  return(predicted_y)
}

# LASSO_training sample

MSE <- function(observed, prediction){
  MSE <- mse(observed, prediction)
  print(MSE)
  return (MSE)
}

lambda_MSE_df <-function(fileNames,fileIndex,last_observed_y_in_training,p,lambdaSeq){

  lambda_MSE = data.frame()
  true_y = last_observed_y_in_training[fileIndex]
  
  for (i in c(1:length(lambdaSeq))){
    predict_y = prediction_y_training(lambda =lambdaSeq[i],p)[fileIndex]
    lambda_MSE[i,1] <- lambdaSeq[i]
    lambda_MSE[i,2] <- MSE(true_y,predict_y)
  }
  colnames(lambda_MSE) <- c("lambda", "MSE")
  
  lambda_output <- lambda_MSE[,1]
  MSE_output <- lambda_MSE[,2]
  
  jpeg(paste("lambda_MSE_p_",p,"_",fileNames[fileIndex],".jpg", sep=""))
  plot(lambda_output,MSE_output,pch=16,xlim=range(lambda_output), ylim=range(MSE_output),
       xlab = "Lambda",ylab = "MSE",main = paste("Relationship between Lambda and MSE_p_",p,"_",fileNames[fileIndex]))
  lines(lambda_output[order(lambda_output)], MSE_output[order(lambda_output)], 
        xlim=range(lambda_output), ylim=range(MSE_output), pch=16,col="blue")
  dev.off()
  
  return(lambda_MSE)
}

Optimal_lambda_df<-function(p,lambdaSeq,fileNames){

  if(p==1){
    last_in_triaing = last_in_triaing_p_1
    ymat = ymat_p_1
  }
  else if(p==2){
    last_in_triaing = last_in_triaing_p_2
    ymat = ymat_p_2
  }
  else if(p==3){
    last_in_triaing = last_in_triaing_p_3
    ymat = ymat_p_3
  }
  last_observed_y_in_training = as.numeric(observed_y[1,])
  optimal_lambdas = vector()
  for (fileIndex in c(1:9)){
    lambda_MSE_df_output <- lambda_MSE_df(fileNames,fileIndex,last_observed_y_in_training,p,lambdaSeq)
    index_min_MSE <- which.min(lambda_MSE_df_output$"MSE")
    optimal_lambda <- lambda_MSE_df_output$"lambda"[index_min_MSE]
    optimal_lambdas <- c(optimal_lambdas,optimal_lambda)
    write.csv(lambda_MSE_df_output, file = paste("lambda_MSE_df_output_p_",p,"_",fileNames[fileIndex],".csv",sep = ""))
  }
  return (optimal_lambdas)
}  


Optimal_lambdas_p_1 = Optimal_lambda_df(p=1,lambdaSeq=c(seq(0.01,2,0.01),seq(3,400,1)),fileNames)
# Optimal_lambdas_p_1:  0.01   1.11   0.07   1.38 132.00 400.00  20.00   0.01  23.00
Optimal_lambdas_p_2 = Optimal_lambda_df(p=2,lambdaSeq=c(seq(0.01,2,0.01),seq(3,400,1)),fileNames)
# Optimal_lambdas_p_2: 0.01 206.00   0.48   0.92   0.03   0.08   0.03   0.01   0.01
Optimal_lambdas_p_3 = Optimal_lambda_df(p=3,lambdaSeq=c(seq(0.01,2,0.01),seq(3,400,1)),fileNames)
# Optimal_lambdas_p_3


"""
According to the plots, for some combinations of lag and currenccy pair, 
increasing the value of lambda when lambda is large will decrease MSE for a very small amount, such as 10e-5.
Thus, we decided to round the values of MSE to 6 digits (i.e., 6 digits after the decimal point).

"""
# fileNames  = c('AUDUSD','CADUSD','CHFUSD', 'EURUSD', 'GBPUSD', 'JPYUSD', 'NOKUSD', 'NZDUSD', 'SEKUSD')

Optimal_lambdas_p_1_new = vector()
for(i in c(1:9)){
  lambda_MSE_df_output_p_1 <- read.csv( paste("lambda_MSE_df_output_p_1_",fileNames[i],".csv",sep = ""), header = TRUE, sep = ",")
  lambda_MSE_df_output_p_1$MSE <- round( lambda_MSE_df_output_p_1$MSE,6)
  Optimal_lambda <- lambda_MSE_df_output_p_1$lambda [which.min(lambda_MSE_df_output_p_1$MSE)]
  Optimal_lambdas_p_1_new <- c(Optimal_lambdas_p_1_new,Optimal_lambda)
}

Optimal_lambdas_p_2_new = vector()
for(i in c(1:9)){
  lambda_MSE_df_output_p_2 <- read.csv( paste("lambda_MSE_df_output_p_2_",fileNames[i],".csv",sep = ""), header = TRUE, sep = ",")
  lambda_MSE_df_output_p_2$MSE <- round( lambda_MSE_df_output_p_2$MSE,6)
  Optimal_lambda <- lambda_MSE_df_output_p_2$lambda [which.min(lambda_MSE_df_output_p_2$MSE)]
  Optimal_lambdas_p_2_new <- c(Optimal_lambdas_p_2_new,Optimal_lambda)
}

Optimal_lambdas_p_3_new = vector()
for(i in c(1:9)){
  lambda_MSE_df_output_p_3 <- read.csv( paste("lambda_MSE_df_output_p_3_",fileNames[i],".csv",sep = ""), header = TRUE, sep = ",")
  lambda_MSE_df_output_p_3$MSE <- round( lambda_MSE_df_output_p_3$MSE,6)
  Optimal_lambda <- lambda_MSE_df_output_p_3$lambda [which.min(lambda_MSE_df_output_p_3$MSE)]
  Optimal_lambdas_p_3_new <- c(Optimal_lambdas_p_3_new,Optimal_lambda)
}

# Optimal_lambdas_p_1_new: 0.01   1.11   0.01   1.38 127.00 279.00  17.00   0.01  21.00
# Optimal_lambdas_p_2_new: 
# Optimal_lambdas_p_3_new: 


prediction_y_test<-function(lambda,p){
  if(p==1){
    last_in_triaing = last_in_triaing_p_1
    xmat = xmat_p_1
    ymat = ymat_p_1
  }  else if(p==2){
    last_in_triaing = last_in_triaing_p_2
    xmat = xmat_p_2
    ymat = ymat_p_2
  }  else if(p==3){
    last_in_triaing = last_in_triaing_p_3
    xmat = xmat_p_3
    ymat = ymat_p_3
  }
  
  predicted_y = data.frame()
  for (k in c(1:9)){
    for (i in c(last_in_triaing:(nrow(xmat)-1))){
      x = xmat[(1:i),][,c(-1,-2)]
      y = ymat[(1:i),][,c(-1,-2)]
      x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)])
      result = orderedLasso(as.matrix(x),as.numeric(y[,k]), lambda[k], intercept = TRUE, standardize =TRUE,method = "Solve.QP", strongly.ordered = TRUE)
      predicted_y[i-last_in_triaing+1,k] <- exp(predict(result, x_for_predict)$yhat.ordered)
    }
  }
  return(predicted_y)
}


quasi_likelihood <- function(observed, prediction){
  value <-  observed/prediction
  QL = (1 /length(observed))*(sum(value - log(value) - 1))
  print(QL)
  return (QL)
}

Test_Performance <- function(p,observed_y, dates,fileNames){
  if(p==1){
    Optimal_lambda = Optimal_lambdas_p_1_new
  }  else if(p==2){
    Optimal_lambda = Optimal_lambdas_p_2_new
  }  else if(p==3){
    Optimal_lambda = Optimal_lambdas_p_3_new
  }
  predicted_y_test<-prediction_y_test(Optimal_lambda,p)
    
  
  MSEs = vector()
  QLs = vector()
  for (k in c(1:9)){
    MSEs <- c(MSEs, MSE(observed_y[,k],predicted_y_test[,k]))
    QLs <- c(QLs, quasi_likelihood(observed_y[,k],predicted_y_test[,k]))
    SE <- (observed_y[,k] - predicted_y_test[,k])^2
    jpeg(paste("QL_p_",p,"_",fileNames[k],".jpg", sep=""))
    plot(dates,SE,type="l",xlim=range(dates), ylim=range(SE),
         xlab = "Time",ylab = "Squared Error",main = paste("Squared Errors_Test Sample_p_",p,"_",fileNames[k]))
    dev.off()
  }
  
  MSE_QL_df = data.frame("MSE" = MSEs, "QL"=QLs)
  row.names(MSE_QL_df) = fileNames
  write.csv(MSE_QL_df, file = paste("MSE_QL_p_",p,".csv",sep = ""))
  return(MSE_QL_df)
}

Test_Performance_p_1 <- Test_Performance(p=1,observed_y[-1,], Dates_test,fileNames)
Test_Performance_p_2 <- Test_Performance(p=2,observed_y[-1,], Dates_test,fileNames)
Test_Performance_p_3 <- Test_Performance(p=3,observed_y[-1,], Dates_test,fileNames)



