# VolatilityForecasting
This package has been created to forecast currency volatility using a variety of models:

1. Shifting 1 element (Past as Present)
2. Linear regression
3. ARCH
4. GARCH(p,q)
5. VAR
6. K-Nearest Neighbors (KNN)
7. LASSO (in R)
8. Ordered LASSO (in R)
git 

For the final project Make sure you go to:

src-> ridge -> main.py or main_combined.py

The zip file in the same folder titled "FinalprojectCode.zip" contains the final code

Alternatively, you can go to the folder titled "Final Submission to find the code and a PowerPoint presentation of the results"

In the final project, I explore different kinds of Ridge Regression towards predicting volatility including:

 0. Linear Regression (base case)
 1. Ridge Regression
 2. Keneral Ridge Regression (KRR)
 3. Bayesian Ridge Regression
 
We perform this analysis in two ways:
 1. By using each currency pair individually (main)
 2. By combining all the currency pairs together (main_combined)

Notes:
1. Some of the code runs in parallel...See "main_parallel.py" in the HW3 Branch.

2. Sometimes I output an iteration number just to see that things are working appropriately. 
If you dont like to see this (...100, 101, 102, 103...) as the code iterates, then feel free to comment it out.

We briefly explored using neural nets, but did not pursue this route due to time constraints.
I will keep those components of the project here for future reference but they are not used at all.
