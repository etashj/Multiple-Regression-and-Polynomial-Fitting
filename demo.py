import numpy as np
import Grapher

### Polynomial Regression ###
import PolynomialRegression as PR

# Create a dataset represeting approximately degree 4 polynomial
x = np.array( [-3, -2, -1, 0, 1, 2, 3, ] )
y = np.array( [-14, -1, 4, 5, 2, 1, -2 ] )

# Try a degree 1 polynomial and graph (underfitting)
ans = PR.regress_qr(x, y, 1)
print("1st degree polynomial: ", ans)
Grapher.plotPolynomialRegression(x,y,ans)

# Try a degree 15 polynomial and graph (gross overfitting)
ans = PR.regress_qr(x, y, 15)
print("15th degree polynomial: ", ans)
Grapher.plotPolynomialRegression(x,y,ans)

# Try a degree 4 polynomial and graph (reasonable fit)
ans = PR.regress_qr(x, y, 4)
print("4th degree polynomial: ", ans)
Grapher.plotPolynomialRegression(x,y,ans)


### Multiple Regression ###
import MultipleRegression as MR

# Example 1 (3 dependent variables)
# Create a dataset with two observations (3 features each)
n = np.array([
    [1, 2, 3], 
    [4, 5, 6],
])

# Create outputs (2 outputs)
m = np.array( [1,2] )

# Regress
R = MR.regress(n, m)
print(R)
# Note, you cannot graph higher than 3 dimensions, 
# to plot this data we would require a 4 dimensional plot
# You may choose to plot this by plotting each variable independently

# Example 2 (2 dependent variables)
# Create a dataset with 4 observations (2 features each)
n = np.array([
    [1, 0, ], 
    [2, 1, ], 
    [3, 1, ], 
    [4, 2, ], 
])

# Create outputs (4 outputs)
m = np.array( [4, 6, 9, 12, ] )

# Regress and graph
R = MR.regress(n, m)
print(R)
Grapher.plotMultipleRegression(n, m, R)
