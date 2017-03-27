#https://cran.r-project.org/web/packages/penalized/penalized.pdf
## A single lasso fit using the clinical risk factors


library(penalized)
i = last_in_triaing-1
x = xmat[(1:i),c(-1,-2)]
y = ymat[(1:i),c(-1,-2)]
k=1
lambda = 1
lambda1 = 1
x_for_predict =  x_for_predict = as.numeric(xmat[i+1,][,c(-1,-2)])

result = orderedLasso(as.matrix(x),as.numeric(y[,k]), lambda, intercept = TRUE, standardize =TRUE,method = "Solve.QP", strongly.ordered = TRUE)
predicted_y <- exp(predict(result, x_for_predict)$yhat.ordered)

response = as.numeric(y[,k])
penalized = as.matrix(x)
fit <- penalized (response, x , unpenalized = ~0,lambda1, standardize=TRUE)
predicted_y_LASSO <- exp(predict(fit,xmat[last_in_triaing,c(-1,-2)])[[1]])
