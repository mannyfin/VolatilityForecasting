# FIN580 Ordered LASSO

library(Metrics)
library(orderedLasso)

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


observed_y <- ymat_p_1[(last_in_triaing_p_1:nrow(ymat_p_1)),][,c(-1,-2)]
rownames(observed_y) <- 1:nrow(observed_y)
observed_y <- exp(observed_y[complete.cases(observed_y),])

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
    i = last_in_triaing-1 # i = T1-1
    x = xmat[(1:i),][,c(-1,-2)] # from t=1 to t = T1-1, used to fit the model in the training sample
    y = ymat[(1:i),][,c(-1,-2)] # from t=1 to t = T1-1, used to fit the model in the training sample
    x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)]) # X used to make prediction at T1
    result = orderedLasso(as.matrix(x),as.numeric(y[,k]), lambda, intercept = TRUE, standardize =TRUE,method = "Solve.QP", strongly.ordered = TRUE)
    predicted_y <- c(predicted_y,exp(predict(result, x_for_predict)$yhat.ordered))
  }
  return(predicted_y)
}

# ordered LASSO_training sample

MSE <- function(observed, prediction){
  MSE <- mse(observed, prediction)
  print(MSE)
  return (MSE)
}

lambda_MSE_df <-function(fileNames,fileIndex,last_observed_y_in_training,p,lambdaSeq){

  # try different values of lambda from lambdaSeq
  # produce and output a csv file to record the relationship between lambda and MSE 
  # plot MSE against lambda and save the output
  
  lambda_MSE = data.frame()
  true_y = last_observed_y_in_training[fileIndex]
  
  for (i in c(1:length(lambdaSeq))){
    predict_y = prediction_y_training(lambda =lambdaSeq[i],p)[fileIndex]
    lambda_MSE[i,1] <- lambdaSeq[i]
    lambda_MSE[i,2] <- MSE(true_y,predict_y)
  }
  colnames(lambda_MSE) <- c("lambda", "MSE")
  
  lambda_output <- log(lambda_MSE[,1]) # scale lambda values in the plot below
  MSE_output <- lambda_MSE[,2]
  
  jpeg(paste("lambda_MSE_p_",p,"_",fileNames[fileIndex],".jpg", sep=""))
  plot(lambda_output,MSE_output,pch=16,xlim=range(lambda_output), ylim=range(MSE_output),
       xlab = "log(Lambda)",ylab = "MSE",main = paste("Ordered LASSO_MSE against Lambda_p_",p,"_",fileNames[fileIndex],sep = ""))
  lines(lambda_output[order(lambda_output)], MSE_output[order(lambda_output)], 
        xlim=range(lambda_output), ylim=range(MSE_output), pch=16,col="blue")
  dev.off()
  
  return(lambda_MSE)
}


Optimal_lambda_df<-function(p,lambdaSeq,fileNames){
  # pick the optimal value for lambda

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
  optimal_lambdas = vector() # this vector saves all optimal lambda's of the 9 files
  for (fileIndex in c(1:9)){
    lambda_MSE_df_output <- lambda_MSE_df(fileNames,fileIndex,last_observed_y_in_training,p,lambdaSeq)
    index_min_MSE <- which.min(lambda_MSE_df_output$"MSE") 
    optimal_lambda <- lambda_MSE_df_output$"lambda"[index_min_MSE]
    optimal_lambdas <- c(optimal_lambdas,optimal_lambda)
    write.csv(lambda_MSE_df_output, file = paste("Ordered LASSO lambda_MSE_df_output_p_",p,"_",fileNames[fileIndex],".csv",sep = ""))
  }
  return (optimal_lambdas)
}  


Optimal_lambdas_p_1 = Optimal_lambda_df(p=1,lambdaSeq= c(seq(0.01,2,0.01),c(3,400,1)),fileNames)
Optimal_lambdas_p_2 = Optimal_lambda_df(p=2,lambdaSeq= c(seq(0.01,2,0.01),c(3,400,1)),fileNames)
Optimal_lambdas_p_3 = Optimal_lambda_df(p=3,lambdaSeq= c(seq(0.01,2,0.01),c(3,400,1)),fileNames)

# lambdaSeq = c(seq(0.01,2,0.01),c(3,400,1))
# Optimal_lambdas_p_1:  0.01   1.11   0.07   1.38 132.00 400.00  20.00   0.01  23.00
# Optimal_lambdas_p_2: 0.01 206.00   0.48   0.92   0.03   0.08   0.03   0.01   0.01
# Optimal_lambdas_p_3: 0.01   0.03   0.03   0.04 178.00   0.43   0.08   0.01   0.01


Optimal_lambdas_p_1_ext = Optimal_lambda_df(p=1,lambdaSeq=c(exp(seq(-10,0,0.05)),seq(1.01,2,0.01),seq(3,400,1)),fileNames)
Optimal_lambdas_p_2_ext = Optimal_lambda_df(p=2,lambdaSeq=c(exp(seq(-10,0,0.05)),seq(1.01,2,0.01),seq(3,400,1)),fileNames)
Optimal_lambdas_p_3_ext = Optimal_lambda_df(p=3,lambdaSeq=c(exp(seq(-10,0,0.05)),seq(1.01,2,0.01),seq(3,400,1)),fileNames)


# lambdaSeq=c(exp(seq(-10,0,0.05)),seq(1.01,2,0.01),seq(3,400,1))
options("scipen"=999, "digits"=4)
Optimal_lambdas_p_1_ext
# Optimal_lambdas_p_1_ext: 0.0000454   1.1100000   0.0672055   0.0000454 132.0000000 400.0000000  20.0000000   0.0000454  23.0000000
Optimal_lambdas_p_2_ext
# Optimal_lambdas_p_2_ext: 0.0000454 0.0223708 0.0000454 0.9512294 0.0287246 0.0780817 0.0301974 0.0000454 0.0000454
Optimal_lambdas_p_3_ext
# Optimal_lambdas_p_3_ext: 0.006097 0.023518 0.022371 0.035084 0.014996 0.070651 0.074274 0.006097 0.005517



# According to the plots, for some combinations of lag and currenccy pair, 
# increasing the value of lambda when lambda is large will decrease MSE for a very small amount, such as 10e-5.
# Thus, we decided to round the values of MSE to 6 digits (i.e., 6 digits after the decimal point).


Optimal_lambdas_p_1_new = vector()
for(i in c(1:9)){
  lambda_MSE_df_output_p_1 <- read.csv( paste("lambda_MSE_df_output_p_1_",fileNames[i],".csv",sep = ""), header = TRUE, sep = ",")
  lambda_MSE_df_output_p_1$MSE <- round( lambda_MSE_df_output_p_1$MSE,6) # set digits of MSE's to 6 digits
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


# for the extended series of lambda values
Optimal_lambdas_p_1_ext_new = vector()
for(i in c(1:9)){
  lambda_MSE_df_output_p_1_ext <- read.csv( paste("Ordered LASSO lambda_MSE_df_output_p_1_",fileNames[i],".csv",sep = ""), header = TRUE, sep = ",")
  lambda_MSE_df_output_p_1_ext$MSE <- round( lambda_MSE_df_output_p_1_ext$MSE,6) # set digits of MSE's to 6 digits
  Optimal_lambda <- lambda_MSE_df_output_p_1_ext$lambda [which.min(lambda_MSE_df_output_p_1_ext$MSE)]
  Optimal_lambdas_p_1_ext_new <- c(Optimal_lambdas_p_1_ext_new,Optimal_lambda)
}

Optimal_lambdas_p_2_ext_new = vector()
for(i in c(1:9)){
  lambda_MSE_df_output_p_2_ext <- read.csv( paste("Ordered LASSO lambda_MSE_df_output_p_2_",fileNames[i],".csv",sep = ""), header = TRUE, sep = ",")
  lambda_MSE_df_output_p_2_ext$MSE <- round( lambda_MSE_df_output_p_2_ext$MSE,6)
  Optimal_lambda <- lambda_MSE_df_output_p_2_ext$lambda [which.min(lambda_MSE_df_output_p_2_ext$MSE)]
  Optimal_lambdas_p_2_ext_new <- c(Optimal_lambdas_p_2_ext_new,Optimal_lambda)
}

Optimal_lambdas_p_3_ext_new = vector()
for(i in c(1:9)){
  lambda_MSE_df_output_p_3_ext <- read.csv( paste("Ordered LASSO lambda_MSE_df_output_p_3_",fileNames[i],".csv",sep = ""), header = TRUE, sep = ",")
  lambda_MSE_df_output_p_3_ext$MSE <- round( lambda_MSE_df_output_p_3_ext$MSE,6)
  Optimal_lambda <- lambda_MSE_df_output_p_3_ext$lambda [which.min(lambda_MSE_df_output_p_3_ext$MSE)]
  Optimal_lambdas_p_3_ext_new <- c(Optimal_lambdas_p_3_ext_new,Optimal_lambda)
}

# previous results
# lambdaSeq = c(seq(0.01,2,0.01),c(3,400,1))
# Optimal_lambdas_p_1: 0.01   1.11   0.07   1.38 132.00 400.00  20.00   0.01  23.00
# Optimal_lambdas_p_2: 0.01 206.00   0.48   0.92   0.03   0.08   0.03   0.01   0.01
# Optimal_lambdas_p_3: 0.01   0.03   0.03   0.04 178.00   0.43   0.08   0.01   0.01

# lambdaSeq=c(exp(seq(-10,0,0.05)),seq(1.01,2,0.01),seq(3,400,1))
# Optimal_lambdas_p_1_ext: 0.0000454   1.1100000   0.0672055   0.0000454 132.0000000 400.0000000  20.0000000   0.0000454  23.0000000
# Optimal_lambdas_p_2_ext: 0.0000454 0.0223708 0.0000454 0.9512294 0.0287246 0.0780817 0.0301974 0.0000454 0.0000454
# Optimal_lambdas_p_3_ext: 0.006097 0.023518 0.022371 0.035084 0.014996 0.070651 0.074274 0.006097 0.005517


# new results
# lambdaSeq = c(seq(0.01,2,0.01),c(3,400,1))
# Optimal_lambdas_p_1_new: 0.01   1.11   0.01   1.38 127.00 279.00  17.00   0.01  21.00
# Optimal_lambdas_p_2_new: 0.01 206.00   0.48   0.92   0.03   0.08   0.03   0.01   0.01
# Optimal_lambdas_p_3_new:  0.01   0.03   0.03   0.04 178.00   0.43   0.08   0.01   0.01

# lambdaSeq=c(exp(seq(-10,0,0.05)),seq(1.01,2,0.01),seq(3,400,1))
# Optimal_lambdas_p_1_ext_new: 0.0000454   1.1100000   0.0000454   0.0000454 127.0000000 279.0000000  17.0000000   0.0000454  21.0000000
# Optimal_lambdas_p_2_ext_new: 0.0000454 0.0223708 0.0000454 0.9512294 0.0287246 0.0780817 0.0301974 0.0000454 0.0000454
# Optimal_lambdas_p_3_ext_new: 0.006097 0.023518 0.022371 0.035084 0.012277 0.070651 0.074274 0.006097 0.005517



prediction_y_test<-function(lambda,p){
  # make predictions in the test sample
  # lambda in the argument is the optimal lambda's for 9 files
  
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

# using the optiaml lambdas of Optimal_lambdas_p_1_new, Optimal_lambdas_p_2_new and Optimal_lambdas_p_3_new
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
  SEs = list()
  for (k in c(1:9)){
    MSEs <- c(MSEs, MSE(observed_y[,k],predicted_y_test[,k]))
    QLs <- c(QLs, quasi_likelihood(observed_y[,k],predicted_y_test[,k]))
    SEs[[k]] <- (observed_y[,k] - predicted_y_test[,k])^2
    # jpeg(paste("QL_p_",p,"_",fileNames[k],".jpg", sep=""))
    # plot(dates,SE,type="l",xlim=range(dates), ylim=range(SE),
    #      xlab = "Time",ylab = "Squared Error",main = paste("Ordered LASSO_Squared Errors_Test Sample_p_",p,"_",fileNames[k]))
    # dev.off()
  }
  
  jpeg(paste("SE_p_",p,".jpg", sep=""))
  plot(dates,log(SEs[[1]]),col=1,type="l", ylim=c(-25,13),
       xlab = "Time",ylab = "Squared Error",main = paste("Ordered LASSO_Log Squared Errors_Test Sample_p_",p))
  for (k in c(1:9)){
    lines(dates,log(SEs[[k]]),col=k+1,type="l")
  }  
  legend("topleft",legend = fileNames,col=seq(1,9,1),lty =rep(1,9))
  dev.off()
  
  MSE_QL_df = data.frame("MSE" = MSEs, "QL"=QLs)
  row.names(MSE_QL_df) = fileNames
  write.csv(MSE_QL_df, file = paste("OrderedLASSO_MSE_QL_p_",p,".csv",sep = ""))
  return(MSE_QL_df)
}

Test_Performance_p_1 <- Test_Performance(p=1,observed_y[-1,], Dates_test,fileNames)
Test_Performance_p_2 <- Test_Performance(p=2,observed_y[-1,], Dates_test,fileNames)
Test_Performance_p_3 <- Test_Performance(p=3,observed_y[-1,], Dates_test,fileNames)


# using the optiaml lambdas of Optimal_lambdas_p_1_ext_new, Optimal_lambdas_p_2_ext_new and Optimal_lambdas_p_3_ext_new

Test_Performance_ext <- function(p,observed_y, dates,fileNames){
  if(p==1){
    Optimal_lambda = Optimal_lambdas_p_1_ext_new
  }  else if(p==2){
    Optimal_lambda = Optimal_lambdas_p_2_ext_new
  }  else if(p==3){
    Optimal_lambda = Optimal_lambdas_p_3_ext_new
  }
  predicted_y_test<-prediction_y_test(Optimal_lambda,p)
  
  
  MSEs = vector()
  QLs = vector()
  SEs = list()
  for (k in c(1:9)){
    MSEs <- c(MSEs, MSE(observed_y[,k],predicted_y_test[,k]))
    QLs <- c(QLs, quasi_likelihood(observed_y[,k],predicted_y_test[,k]))
    SEs[[k]] <- (observed_y[,k] - predicted_y_test[,k])^2
    # jpeg(paste("QL_p_",p,"_",fileNames[k],".jpg", sep=""))
    # plot(dates,SE,type="l",xlim=range(dates), ylim=range(SE),
    #      xlab = "Time",ylab = "Squared Error",main = paste("Ordered LASSO_Squared Errors_Test Sample_p_",p,"_",fileNames[k]))
    # dev.off()
  }
  
  jpeg(paste("SE_ext_p_",p,".jpg", sep=""))
  plot(dates,log(SEs[[1]]),col=1,type="l", ylim=c(-25,13),
       xlab = "Time",ylab = "Squared Error",main = paste("Ordered LASSO(ext)_Log Squared Errors_Test Sample_p_",p))
  for (k in c(1:9)){
    lines(dates,log(SEs[[k]]),col=k+1,type="l")
  }  
  legend("topleft",legend = fileNames,col=seq(1,9,1),lty =rep(1,9))
  dev.off()
  
  MSE_QL_df = data.frame("MSE" = MSEs, "QL"=QLs)
  row.names(MSE_QL_df) = fileNames
  write.csv(MSE_QL_df, file = paste("OrderedLASSO_ext_MSE_QL_p_",p,".csv",sep = ""))
  return(MSE_QL_df)
}

Test_Performance_p_1_ext <- Test_Performance_ext(p=1,observed_y[-1,], Dates_test,fileNames)
Test_Performance_p_2_ext <- Test_Performance_ext(p=2,observed_y[-1,], Dates_test,fileNames)
Test_Performance_p_3_ext <- Test_Performance_ext(p=3,observed_y[-1,], Dates_test,fileNames)

