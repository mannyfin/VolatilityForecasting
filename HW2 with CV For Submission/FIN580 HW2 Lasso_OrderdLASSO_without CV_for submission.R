# FIN580 LASSO/Orderd LASSO without Cross Validation

library(Metrics)
# library(penalized)
library(orderedLasso)
library(glmnet)

setwd("C:/Users/YYJ/Desktop/FIN580/Homework1/VolatilityForecasting/src/LASSO_OrderedLASSO")
xmat_p_1 <- read.csv("xmat_p_1.csv", header = TRUE, sep = ",")
xmat_p_2 <- read.csv("xmat_p_2.csv", header = TRUE, sep = ",")
xmat_p_3 <- read.csv("xmat_p_3.csv", header = TRUE, sep = ",")
xmat_p_5 <- read.csv("xmat_p_5.csv", header = TRUE, sep = ",")

ymat_p_1 <- read.csv("ymat_p_1.csv", header = TRUE, sep = ",")
ymat_p_2 <- read.csv("ymat_p_2.csv", header = TRUE, sep = ",")
ymat_p_3 <- read.csv("ymat_p_3.csv", header = TRUE, sep = ",")
ymat_p_5 <- read.csv("ymat_p_5.csv", header = TRUE, sep = ",")

last_in_triaing_p_1 = which(xmat_p_1[,"Dates"]=="10/31/2011")
last_in_triaing_p_2 = which(xmat_p_2[,"Dates"]=="10/31/2011")
last_in_triaing_p_3 = which(xmat_p_3[,"Dates"]=="10/31/2011")
last_in_triaing_p_5 = which(xmat_p_5[,"Dates"]=="10/31/2011")


observed_y <- ymat_p_1[(last_in_triaing_p_1:nrow(ymat_p_1)),][,c(-1,-2)]
rownames(observed_y) <- 1:nrow(observed_y)
observed_y <- exp(observed_y[complete.cases(observed_y),])

fileNames  = c('AUDUSD','CADUSD','CHFUSD', 'EURUSD', 'GBPUSD', 'NOKUSD', 'NZDUSD')
Dates <- as.Date(xmat_p_1[,2], format = "%m/%d/%Y")
Dates_test <- Dates[which(Dates =="2011-11-01"):length(Dates)]

# training sample
prediction_y_training<-function(lambda,p, model){
  # model can be "OrderedLASSO" or "LASSO"
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
  }  else if(p==5){
    last_in_triaing = last_in_triaing_p_5
    xmat = xmat_p_5
    ymat = ymat_p_5
  }
  predicted_y = vector()
  for (k in c(1:length(fileNames))){
    # fit the models and make predictions in the training sample
    i = last_in_triaing-1 # i = T1-1
    x = xmat[(1:i),][,c(-1,-2)] 
    y = ymat[(1:i),][,c(-1,-2)]
    if(model=="OrderedLASSO"){
      x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)]) # X used to make prediction at T1
      result = orderedLasso(as.matrix(x),as.numeric(y[,k]), lambda, intercept = TRUE, standardize =TRUE,method = "Solve.QP", strongly.ordered = TRUE)
      predicted_y <- c(predicted_y,exp(predict(result, x_for_predict)$yhat))
    } else if(model=="LASSO"){
      x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)]) # X used to make prediction at T1
      # result = penalized(as.numeric(y[,k]), as.matrix(x), lambda1=lambda)
      # predicted_y <- c(predicted_y,exp(predict(result, t(x_for_predict))[[1]]))
      result = glmnet(as.matrix(x), as.numeric(y[,k]), family="gaussian",alpha=1, lambda=lambda) # alpha=1 for LASSO
      predicted_y <- c( predicted_y,exp( predict(result, t(as.matrix(x_for_predict)))[1] ) )
    }
  }
  return(predicted_y)
}

#  LASSO_training sample

MSE <- function(observed, prediction){
  MSE <- mse(observed, prediction)
  print(MSE)
  return (MSE)
}

lambda_MSE_df <-function(fileNames,fileIndex,last_observed_y_in_training,p,lambdaSeq,model){
  
  # try different values of lambda from lambdaSeq
  # produce and output a csv file to record the relationship between lambda and MSE 
  # plot MSE against lambda and save the output
  
  lambda_MSE = data.frame()
  true_y = last_observed_y_in_training[fileIndex]
  
  for (i in c(1:length(lambdaSeq))){
    predict_y = prediction_y_training(lambda =lambdaSeq[i],p,model)[fileIndex]
    lambda_MSE[i,1] <- lambdaSeq[i]
    lambda_MSE[i,2] <- MSE(true_y,predict_y)
  }
  colnames(lambda_MSE) <- c("lambda", "MSE")
  
  lambda_output <- log(lambda_MSE[,1]) # scale lambda values in the plot below
  MSE_output <- lambda_MSE[,2]
  
  jpeg(paste(model,"_lambda_MSE_p_",p,"_",fileNames[fileIndex],".jpg", sep=""))
  plot(lambda_output,MSE_output,type='l',xlim=range(lambda_output), ylim=range(MSE_output),
       xlab = "log(Lambda)",ylab = "MSE",main = paste(model,"_MSE against Lambda_p_",p,"_",fileNames[fileIndex],sep = ""))
  # lines(lambda_output[order(lambda_output)], MSE_output[order(lambda_output)], 
  #       xlim=range(lambda_output), ylim=range(MSE_output), pch=16,col="blue")
  dev.off()
  
  return(lambda_MSE)
}


Optimal_lambda_df<-function(p,lambdaSeq,fileNames,model){
  # pick the optimal value for lambda
  
  if(p==1){
    last_in_triaing = last_in_triaing_p_1
    ymat = ymat_p_1
  }  else if(p==2){
    last_in_triaing = last_in_triaing_p_2
    ymat = ymat_p_2
  }  else if(p==3){
    last_in_triaing = last_in_triaing_p_3
    ymat = ymat_p_3
  } else if(p==5){
    last_in_triaing = last_in_triaing_p_5
    ymat = ymat_p_5
  }
  last_observed_y_in_training = as.numeric(observed_y[1,])
  optimal_lambdas = vector() # this vector saves all optimal lambda's of all files
  for (fileIndex in c(1:length(fileNames))){
    lambda_MSE_df_output <- lambda_MSE_df(fileNames,fileIndex,last_observed_y_in_training,p,lambdaSeq,model)
    index_min_MSE <- which.min(lambda_MSE_df_output$"MSE") 
    optimal_lambda <- lambda_MSE_df_output$"lambda"[index_min_MSE]
    optimal_lambdas <- c(optimal_lambdas,optimal_lambda)
    write.csv(lambda_MSE_df_output, file = paste(model," lambda_MSE_df_output_p_",p,"_",fileNames[fileIndex],".csv",sep = ""))
  }
  return (optimal_lambdas)
}  


Optimal_lambdas_p_1_LASSO = Optimal_lambda_df(p=1,lambdaSeq= c(exp(seq(-12,1,0.05))),fileNames,"LASSO")
Optimal_lambdas_p_2_LASSO = Optimal_lambda_df(p=2,lambdaSeq= c(exp(seq(-12,1,0.05))),fileNames,"LASSO")
Optimal_lambdas_p_3_LASSO = Optimal_lambda_df(p=3,lambdaSeq= c(exp(seq(-12,1,0.05))),fileNames,"LASSO")

Optimal_lambdas_p_1_OrderedLASSO = Optimal_lambda_df(p=1,lambdaSeq= c(exp(seq(-12,5.5,0.05))),fileNames,"OrderedLASSO")
Optimal_lambdas_p_2_OrderedLASSO = Optimal_lambda_df(p=2,lambdaSeq= c(exp(seq(-12,5.5,0.05))),fileNames,"OrderedLASSO")
Optimal_lambdas_p_3_OrderedLASSO = Optimal_lambda_df(p=3,lambdaSeq= c(exp(seq(-12,5.5,0.05))),fileNames,"OrderedLASSO")
Optimal_lambdas_p_5_OrderedLASSO = Optimal_lambda_df(p=5,lambdaSeq= c(exp(seq(-12,5.5,0.05))),fileNames,"OrderedLASSO")

# options(scipen=999)
# Optimal_lambdas_p_1_LASSO:   0.01290681258 0.01576441648 0.00002254494 0.00429630469 0.01742237464 0.00499159391 0.01110899654
# Optimal_lambdas_p_2_LASSO: 0.246596963942 0.182683524053 0.000006144212 0.000071201263 0.201896517995 0.192049908621 0.286504796860
# Optimal_lambdas_p_3_LASSO:   0.246596963942 0.182683524053 0.000086965419 0.000006144212 0.201896517995 0.192049908621 0.286504796860

# Optimal_lambdas_p_1_OrderedLASSO: 3.490342957462  3.857425530697  0.000006144212  0.000006144212 94.632408314924  0.000006144212  3.490342957462
# Optimal_lambdas_p_2_OrderedLASSO: 221.40641620 200.33680997   0.03337327   0.01831564 181.27224188 156.02246449 164.02190730
# Optimal_lambdas_p_3_OrderedLASSO:  221.406416204188 181.272241875151   0.074273578214   0.031745636378 181.272241875151   0.000006144212 156.022464486395
# Optimal_lambdas_p_5_OrderedLASSO: 221.40641620   2.11700002   0.04504920   0.01227734  66.68633104   2.11700002 156.02246449

prediction_y_test<-function(lambda,p,model){
  # make predictions in the test sample
  # lambda in the argument is the optimal lambda's for all files
  
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
  } else if(p==5){
    last_in_triaing = last_in_triaing_p_5
    xmat = xmat_p_5
    ymat = ymat_p_5
  }
  
  predicted_y = data.frame()
  for (k in c(1:length(fileNames))){
    for (i in c(last_in_triaing:(nrow(xmat)-1))){
      x = xmat[(1:i),][,c(-1,-2)]
      y = ymat[(1:i),][,c(-1,-2)]
      if(model=="OrderedLASSO"){
        x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)])
        result = orderedLasso(as.matrix(x),as.numeric(y[,k]), lambda[k], intercept = TRUE, standardize =TRUE,method = "Solve.QP", strongly.ordered = TRUE)
        predicted_y[i-last_in_triaing+1,k] <- exp(predict(result, x_for_predict)$yhat)
      } else if(model=="LASSO"){
        x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)])
        # result = penalized(as.numeric(y[,k]), as.matrix(x),lambda1=lambda[k])
        # predicted_y[i-last_in_triaing+1,k] <- exp(predict(result, t(x_for_predict))[[1]])
        result = glmnet(as.matrix(x), as.numeric(y[,k]), family="gaussian",alpha=1, lambda=lambda) # alpha=1 for LASSO
        predicted_y[i-last_in_triaing+1,k] <- exp( predict(result, t(as.matrix(x_for_predict)))[1] ) 
      }
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


# Using the optiaml lambdas of Optimal_lambdas_p_1_LASSO, 
# Optimal_lambdas_p_2_LASSO and Optimal_lambdas_p_3_LASSO for LASSO model
# USing the optiaml lambdas of Optimal_lambdas_p_1_OrderedLASSO,Optimal_lambdas_p_2_OrderedLASSO, 
# Optimal_lambdas_p_3_OrderedLASSO and Optimal_lambdas_p_5_OrderedLASSO for ordered LASSO model

Test_Performance <- function(p,observed_y, dates,fileNames, model){
  if(p==1){
    if(model=="LASSO"){
      Optimal_lambda = Optimal_lambdas_p_1_LASSO
    } else if(model=="OrderedLASSO"){
      Optimal_lambda = Optimal_lambdas_p_1_OrderedLASSO
    }
  }  else if(p==2){
    if(model=="LASSO"){
      Optimal_lambda = Optimal_lambdas_p_2_LASSO
    } else if(model=="OrderedLASSO"){
      Optimal_lambda = Optimal_lambdas_p_2_OrderedLASSO
    }
  }  else if(p==3){
    if(model=="LASSO"){
      Optimal_lambda = Optimal_lambdas_p_3_LASSO
    } else if(model=="OrderedLASSO"){
      Optimal_lambda = Optimal_lambdas_p_3_OrderedLASSO
    }
  }  else if(p==5){
    if(model=="LASSO"){
      Optimal_lambda = Optimal_lambdas_p_5_LASSO
    } else if(model=="OrderedLASSO"){
      Optimal_lambda = Optimal_lambdas_p_5_OrderedLASSO
    }
  }    
  predicted_y_test<-prediction_y_test(Optimal_lambda,p,model)
  
  
  MSEs = vector()
  QLs = vector()
  SEs = list()
  for (k in c(1:length(fileNames))){
    MSEs <- c(MSEs, MSE(observed_y[,k],predicted_y_test[,k]))
    QLs <- c(QLs, quasi_likelihood(observed_y[,k],predicted_y_test[,k]))
    SEs[[k]] <- (observed_y[,k] - predicted_y_test[,k])^2
  }
  
  jpeg(paste(model,"_SE_p_",p,".jpg", sep=""))
  plot(dates,log(SEs[[1]]),col=1,type="l", ylim=c(-25,13),
       xlab = "Time",ylab = "Squared Error",main = paste(model,"_Log Squared Errors_Test Sample_p_",p))
  for (k in c(1:length(fileNames))){
    lines(dates,log(SEs[[k]]),col=k+1,type="l")
  }  
  legend("topleft",legend = fileNames,col=seq(1,length(fileNames),1),lty =rep(1,length(fileNames)))
  dev.off()
  
  MSE_QL_df = data.frame("MSE" = MSEs, "QL"=QLs)
  row.names(MSE_QL_df) = fileNames
  write.csv(MSE_QL_df, file = paste(model,"_MSE_QL_p_",p,".csv",sep = ""))
  return(c(MSE_QL_df, SEs))
}

Test_Performance_p_1_without_CV <- Test_Performance(p=1,observed_y[-1,], Dates_test,fileNames,"LASSO")
Test_Performance_p_2_without_CV <- Test_Performance(p=2,observed_y[-1,], Dates_test,fileNames,"LASSO")
Test_Performance_p_3_without_CV <- Test_Performance(p=3,observed_y[-1,], Dates_test,fileNames,"LASSO")

Test_Performance_p_1_ordered_without_CV <- Test_Performance(p=1,observed_y[-1,], Dates_test,fileNames,"OrderedLASSO")
Test_Performance_p_2_ordered_without_CV <- Test_Performance(p=2,observed_y[-1,], Dates_test,fileNames,"OrderedLASSO")
Test_Performance_p_3_ordered_without_CV <- Test_Performance(p=3,observed_y[-1,], Dates_test,fileNames,"OrderedLASSO")
Test_Performance_p_5_ordered_without_CV <- Test_Performance(p=5,observed_y[-1,], Dates_test,fileNames,"OrderedLASSO")


# comparing prediction performance of different models for each currency pair

Model_Names = c("LASSO_p_1","LASSO_p_2","LASSO_p_3","OrderedLASSO_p_1","OrderedLASSO_p_2",
                "OrderedLASSO_p_3","OrderedLASSO_p_5")
for (i in 1:length(fileNames)){
  jpeg(paste(fileNames[i]," SE Plots without Cross Validation.jpg", sep=""))
  plot(Dates_test,log(Test_Performance_p_1_without_CV[[i+2]]),col=1,type="l", ylim=c(-25,13),
       xlab = "Time",ylab = "ln(Squared Error)",main = paste(fileNames[i],"_Log Squared Errors_Test Sample_without Cross Validation"))  
  lines(Dates_test,log(Test_Performance_p_2_without_CV[[i+2]]),col=2,type="l")
  lines(Dates_test,log(Test_Performance_p_3_without_CV[[i+2]]),col=3,type="l")
  lines(Dates_test,log(Test_Performance_p_1_ordered_without_CV[[i+2]]),col=4,type="l")
  lines(Dates_test,log(Test_Performance_p_2_ordered_without_CV[[i+2]]),col=5,type="l")
  lines(Dates_test,log(Test_Performance_p_3_ordered_without_CV[[i+2]]),col=6,type="l")
  lines(Dates_test,log(Test_Performance_p_5_ordered_without_CV[[i+2]]),col=7,type="l")
  legend("topleft",legend = Model_Names,col=seq(1,length(Model_Names),1),lty =rep(1,length(Model_Names)))
  dev.off()
}


