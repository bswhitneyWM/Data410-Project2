# Project 2

This page explores the similarities and differences of Locally Weighted Linear Regression and Random Forests for Project 2 in Data 410: Advanced Applied Machine Learning. All analysis was performed by Bryce Whitney. 

## Theoretical  Discussion

### Locally Weighted Regression

### Random Forest

## Cars Dataset

## Data Preprocessing
Data was split into training and testing sets where the training data consisted of 75% of the observations, and the other 25% were in the test set. The data was then normalized using the training data. Scaling the data is only necessary for the Locally Weighted Regression, because Random Forest performs the same on normalized and unnormalized data as it doesn't calculate the distance between observations. The code for this is shown below. 

```python
# Train-Test Split
Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=13)

# Scale the data
scale = StandardScaler()
Xtrain_ss = scale.fit_transform(Xtrain.reshape(-1, 1))
Xtest_ss = scale.transform(Xtest.reshape(-1, 1))
```
For the Locally Weighted Regression the two hyperparameters I scanned for were the choice of kernel and tau value. There were three different kernel functions considered (Tricubic, Epanechnikov, Quartic) and tau values scanned were every tenth between 0.1 and 1. For each combination, the mean squared error on the scaled test set was calculated, and the parameters that produced the lowest mean squared error were saved to be used later. I found **tau = 0.1** with an **Epanechnikov kernel** were the best paramaters. The code is shown below. 

```python
taus = np.arange(0.1, 1.1, 0.1)
kernels = [tricubic, Epanechnikov, Quartic]
best_mse_lr = 10**10
best_params_lr = tuple()

for tau in taus:
    for kern in kernels:
        y_pred = lowess_reg(Xtrain_ss.reshape(Xtrain_ss.shape[0]), ytrain, Xtest_ss.reshape(Xtest_ss.shape[0]), kern, tau)
        mse = MSE(ytest, y_pred)
        
        if(mse < best_mse_lr):
            best_mse_lr = mse
            best_params_lr = (tau, kern)

print("Best MSE: ", best_mse_lr)
print("Best Parameters:", best_params_lr)
```
The same parameter selection process was used for the Random Forest. The two hyperparameters I scanned for were the number of estimators and max depth of the trees. The number of estimators to scan were chosen discretely between 50 and 1000, and the max depth scanned every integer from 2 to 10. I found a model with **50 trees** with a **max depth of 2** produced the lowest mean squared error.

```python
n_estimators = [50, 100, 200, 500, 1000]
max_depth = np.arange(2, 11, 1)

best_mse_rf = 10**10
best_params_rf = tuple()

for n in n_estimators:
    for d in max_depth:
        model = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=13)
        model.fit(Xtrain.reshape(-1, 1), ytrain)
        mse = MSE(ytest, model.predict(Xtest.reshape(-1,1)))

        if(mse < best_mse_rf):
            best_mse_rf = mse
            best_params_rf = (n, d)

print("Best MSE: ", best_mse_rf)
print("Best Parameters:", best_params_rf)
```

This process of splitting the data into trainging and testing sets and scanning for optimal hyperparameters was exectued for both the cars dataset and the Boston housing dataset. The optimal parameters were then used when calculating the crossvalidated mean squared error for each model. For both models I performed a 5-fold crossvalidation to obtain the crossvalidated mean squared error. I chose to use 5 folds because there were 392 observations in the car data, leaving roughly 78 observations for each fold. Anything less than this would have been too small a validation sample in my opinion. Random states were utilized to ensure results are reproducible. The code for obtaining each crossvalidated mean squared error are shown in their respective sections below. 

### Locally Weighted Regression
When calculating the crossvalidated mse, the data was always normalized before being passed to the 'lowess_reg' function. The model with **tau = 0.1** and the **Epanechnikov kernel** identified earlier was used with every fold. The Locally Weighted Regression produced a **crossvalidated MSE =  18.004**. The code used to obtain this is shown below. 

```python
kf = KFold(n_splits=5, random_state=13, shuffle=True)

scores = []

for train_idx, test_idx in kf.split(x):
    Xtrain, Xtest = x[train_idx], x[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    scale = StandardScaler()
    Xtrain_ss = scale.fit_transform(Xtrain.reshape(-1, 1))
    Xtest_ss = scale.transform(Xtest.reshape(-1, 1))
    
    y_pred = lowess_reg(Xtrain_ss.reshape(Xtrain_ss.shape[0]), ytrain, Xtest_ss.reshape(Xtest_ss.shape[0]), 
                                          best_params[1]_lr, best_params[0]_lr)
    mse = MSE(ytest, y_pred)
    
    scores.append(mse)
    
print("Locally Weighted Regression Crossvalidated MSE: ", np.average(scores))
```

### Random Forest
Unnormalized data was used for Random Forest calculations. This is because scaling the data doesn't make a difference, so it is more efficient to use the unnormalized data. The model with **n_estimators = 50** and  **max_depth = 2** identified earlier was used with every fold. The Random Forest Regression produced a **crossvalidated MSE =  18.228**. The code used to obtain this is shown below. 

```python
kf = KFold(n_splits=5, random_state=13, shuffle=True)

scores = []

for train_idx, test_idx in kf.split(x):
    Xtrain, Xtest = x[train_idx], x[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    model = RandomForestRegressor(n_estimators=best_params_rf[0], max_depth=best_params_rf[1], random_state=13)
    model.fit(Xtrain.reshape(-1, 1), ytrain)
    mse = MSE(ytest, model.predict(Xtest.reshape(-1, 1)))
    
    scores.append(mse)
    
print("Random Forest Crossvalidated MSE: ", np.average(scores))
```
### Conclusion
As a recap, the Locally Weighted Regression has a crossvalidated MSE = **18.004** while the Random Forest had a crossvalidated MSE = **18.228**. This indicates the **Locally Weighted Regression may be slightly more reliable when using the cars dataset**. This intuition holds when looking at the predictions made by the Locally Weighted regression and Random Forest in the figure below. The Locally Weighted Regression appears less sporadic than Random Forest highlighting that it may be slightly less overfit and better explain the relationship between a cars weight the miles per gallon it achieves. 

![](CarsComparison.png)

## Boston Housing Dataset

The same exact data preprocessing methods that were used for the cars dataset were used for this Boston Housing dataset. If you need a detailed descripton, please see the `Data Preprocessing` section. For the Locally Weighted Regression, the best model had **tau = 0.3** and a **Epanechnikov kernel**, while the best Random Forest model consisted of **500 trees** with a **max depth of 2**. To obtain the crossvalidated mean squared error, I again used a 5-fold crossvalidation. Random states were used to ensure the results are reproducible. 

### Locally Weighted Regression
When calculating the crossvalidated mse, the data was always normalized before being passed to the 'lowess_reg' function. The model with **tau = 0.3** and the **Epanechnikov kernel** identified above was used with every fold. The Locally Weighted Regression produced a **crossvalidated MSE =  36.932** on the Boston Housing data. The code used to obtain this is shown below. 

```python
kf = KFold(n_splits=5, random_state=13, shuffle=True)

scores = []

for train_idx, test_idx in kf.split(x):
    Xtrain, Xtest = x[train_idx], x[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    scale = StandardScaler()
    Xtrain_ss = scale.fit_transform(Xtrain.reshape(-1, 1))
    Xtest_ss = scale.transform(Xtest.reshape(-1, 1))
    
    y_pred = lowess_reg(Xtrain_ss.reshape(Xtrain_ss.shape[0]), ytrain, Xtest_ss.reshape(Xtest_ss.shape[0]), 
                                            best_params_lr[1], best_params_lr[0])
    mse = MSE(ytest, y_pred)
    
    scores.append(mse)
    
print("Locally Weighted Regression Crossvalidated MSE: ", np.average(scores))
```

### Random Forest
Unnormalized data was again used for Random Forest calculations. The model with **n_estimators = 500** and  **max_depth = 2** identified earlier was used with every fold. The Random Forest Regression produced a **crossvalidated MSE =  36.328** on the Boston Housing data. The code used to obtain this is shown below.

```python
kf = KFold(n_splits=5, random_state=13, shuffle=True)

scores = []

for train_idx, test_idx in kf.split(x):
    Xtrain, Xtest = x[train_idx], x[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    model = RandomForestRegressor(n_estimators=best_params_rf[0], max_depth=best_params_rf[1], random_state=13)
    model.fit(Xtrain.reshape(-1, 1), ytrain)
    mse = MSE(ytest, model.predict(Xtest.reshape(-1, 1)))
    
    scores.append(mse)
    
print("Random Forest Crossvalidated MSE: ", np.average(scores))
```
### Conclusion 
As a recap, the Locally Weighted Regression has a crossvalidated MSE = **36.932** while the Random Forest had a crossvalidated MSE = **36.328**. This indicates the **Random Forest may be slightly more reliable when using the Boston Housing dataset**. Unlike the cars dataset, the Locally Wieghted Regression appears to be more sporadic, particularly towards the extremes of the data, when compared to the Random Forest. HOwever, I don't think I would be willin to claim that the graph below shows Random Forest is clearly better than Locally Weighted Regression, because it may only appear that way since we know the crossvalidated mean squared error values for each model. 

![](HousingComparison.png)

# Final Discussion
At this point we have seen an example where Locally Weighted Regression performed slightly better than Random Forest and an example where Random Forest showed slightly better predictive power. This suggests that the properties of the data being used will impact which model is the best to use. Like most cases in Data Science, I don't believe there is one model that will always achieve better results over the other. *I believe Locally Weighted Regression will be better for less noisy datasets, while Random Forest will perform betetr on datasets with a lot of noise*. This was highlighted by the performance on the Cars and Boston Housing datasets, but I don't think one example is good enough to support the claim. One of the major differences between Locally Weighted Regression and Random Forest is that Locally Weighted Regression uses a distance metric when determining the weights while Random Forest doesn't. In noisy datasets, the noisy points will have a large impact on the weights because their distance from other points is so large. On the other hand, Random Forest is able to counteract that because instead of distance, decision trees use cutoof points to create boxes that data either fit into or don't. *For this reasons I believe Random Forest will tend to perform better on noisy datasets, but can't compete with the precision of Locaaly Weighted Regression on data sets with minial noise*.
