# Project 2

This page explores the similarities and differences of Locally Weighted Linear Regression and Random Forests for Project 2 in Data 410: Advanced Applied Machine Learning. All analysis was performed by Bryce Whitney. 

## Theoretical  Descriptions

### Locally Weighted Regression

### Random Forest

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

This process of splitting the data into trainging and testing sets and scanning for optimal hyperparameters was exectued for both the cars dataset and the Boston housing dataset. The optimal parameters were then used when calculating the crossvalidated mean squared error for each model. 

## Cars Dataset

For both models I performed a 5-fold crossvalidation to obtain the crossvalidated mean squared error. I chose to use 5 folds because there were 392 observations in the car data, leaving roughly 78 observations for each fold. Anything less than this would have been too small a validation sample in my opinion. The code for obtaining each crossvalidated mean squared error are shown in their respective sections below. 

### Locally Weighted Regression
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

## Boston Housing Dataset

### Locally Weighted Regression

### Random Forest

### Conclusion 

## Discussion
- Maybe Locally weighted regression is more sensitive to noise
