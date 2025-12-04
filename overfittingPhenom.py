import pandas as pd
import numpy as np
from PolynomialRegression import regress
from Grapher import plotPolynomialRegression as plot1
from Grapher import plotGoodVsOverFit as plot2
from Grapher import polynomToString

# Reading the training data
training_df = pd.read_csv("icecream_train.csv")

# Extracting vectors from dataset to input to regression
X_train = training_df["Temperature (C)"].to_numpy()
Y_train = training_df["Ice Cream Sales (units)"].to_numpy()

# Degree 2 regression (is not overfitting)
R_fit = regress(X_train, Y_train, 2)
print(polynomToString(R_fit))

# Degree 6 regression (is overfitting)
R_overfit = regress(X_train, Y_train, 6)
print(polynomToString(R_overfit))

# Plotting regressions side-by-side with training data
plot2(X_train, Y_train, R_fit, R_overfit)

# Reading the test data
test_df = pd.read_csv("icecream_test.csv")

# Extracting independent variable vector from dataset
X_test = test_df["Temperature (C)"].to_numpy()

# Calculating 2 sets of predicted dependent variable values for the 
# input vector X_test using the good fit and overfit polynomial
Y_test_goodpred = np.polyval(np.flip(R_fit), X_test)
Y_test_badpred = np.polyval(np.flip(R_overfit), X_test)

# Extracting the actual dependent variable vector from the dataset
Y_test_actual = test_df["Ice Cream Sales (units)"].to_numpy()

# Plotting regressions side-by-side with test data
plot2(X_test, Y_test_actual, R_fit, R_overfit)

# Calculating mean squared error for both the good and bad prediction
mse_fit = np.mean((Y_test_goodpred - Y_test_actual)**2) # value turns out to be 12.44
print("error for good fit: ", mse_fit)
mse_overfit = np.mean((Y_test_badpred - Y_test_actual)**2) # value turns out to be 22.50
print("error for overfit: ", mse_overfit)