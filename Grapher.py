import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray

# Plotter for Multiple Regression (3-dim only)
'''
Parameters
----------
X: np.ndarray, shape(n_samples, 2_features)
    Dependent variable array. 
    Eg. [ 
          [x_11, x_12], 
          [x_21, x_22],   
                  ...            ,
          [x_n1, x_n2],   
        ]

Y: np.ndarray, shape(n_samples)
    Independent variable array. 
    Eg. [ Y_1, Y_2, ..., Y_n ]
    
B: np.ndarray, shape(3_features)
    Array of regression coefficient outputs (outputs of MultipleRegression.regress(X,Y))
    Eg. [ β_0, β_1, β_2, ]
    ==> β_0 + β_1*x_1 + β_2*x_2
    ==> [β]X.T (in matrix form)

Result
------
Window displayed with data and regression
'''
def plotMultipleRegression(X: NDArray[np.float64], 
                           Y: NDArray[np.float64], 
                           B: NDArray[np.float64]) -> None: 
    # Creating the plotting environment
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    
    # Checking the dimensions
    assert(X.shape[0] == Y.shape[0])
    assert(B.shape[0] == 3)
    
    # Plotting the points
    for i in range (X.shape[0]): 
        ax.scatter(X[i][0], X[i][1], Y[i])


    # Define the range for x and y
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x1_range = np.linspace(xmin, xmax, 50)
    x2_range = np.linspace(ymin, ymax, 50)

    # Create the meshgrid
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Calculate values
    Z = B[0] + B[1]*X1 + B[2]*X2

    # Plot the surface
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    
    # Show the plot
    plt.show()


# Plotter for Polynomial Regression (2-dim only)
'''
Parameters
----------
X: np.ndarray, shape(n_samples)
    Dependent variable array. 
    Eg. [ X_1, X_2, ..., X_n ]

Y: np.ndarray, shape(n_samples)
    Independent variable array. 
    Eg. [ Y_1, Y_2, ..., Y_n ]
    
B: np.ndarray, shape(m_features)
    Array of regression coefficient outputs
    Degree will be inferred from this length
    Eg. [ β_0, β_1, ..., β_deg ]
    ==> β_0 + β_1*x + β_2*x^2 + ... + β_m*x^deg

Result
------
Window displayed with data and regression
'''
def plotPolynomialRegression(X: NDArray[np.float64], 
                             Y: NDArray[np.float64], 
                             B: NDArray[np.float64]) -> None: 
    # Creating the plotting environment
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X, Y)
    
    # Define the range for x
    xmin, xmax = ax.get_xlim()
    Xp = np.linspace(xmin, xmax, 100)

    # Calculate values
    P = np.poly1d(B[::-1])
    Yp = P(Xp)

    ax.plot(Xp, Yp)

    plt.show()

'''
Parameters
----------
B: np.ndarray, shape(m_features)
    Array of regression coefficient outputs
    Degree will be inferred from this length
    Eg. [ β_0, β_1, ..., β_deg ]
    ==> β_0 + β_1*x + β_2*x^2 + ... + β_m*x^deg

Result
------
A string representation of the polynomial
"y = β_0 + β_1*x + β_2*x^2 + ... + β_m*x^deg"
'''
def polynomToString(B: NDArray[np.float64]) -> str: 
    result = "y = " + str(B[0])
    for i in range(1, B.shape[0]): 
        result += f" + {B[i]}(x^{i})"
    return result

'''
Parameters
----------
B: np.ndarray, shape(m_features)
    Array of regression coefficient outputs
    Variables will be inferred from this length
    Eg. [ β_0, β_1, ..., β_n ]
    ==> β_0 + β_1*x_1 + β_2*x_2 + ... + β_m*x_n

Result
------
A string representation of the polynomial
"y = β_0 + β_1*x_1 + β_2*x_2 + ... + β_m*x_n"
'''
def linearToString(B: NDArray[np.float64]) -> str: 
    result = "y = " + str(B[0])
    for i in range(1, B.shape[0]): 
        result += f" + {B[i]}(x_{i})"
    return result

# Plotter for 2 polynomial regressions on the same plot - 
# specifically for showing the difference between a good fit 
# regression and an overfit regression
'''
Parameters
----------
X: np.ndarray, shape(n_samples)
    Dependent variable array. 
    Eg. [ X_1, X_2, ..., X_n ]

Y: np.ndarray, shape(n_samples)
    Independent variable array. 
    Eg. [ Y_1, Y_2, ..., Y_n ]
    
B_good: np.ndarray, shape(m_features)
    Array of degree 2 regression coefficient outputs
    Degree will be inferred from this length
    Eg. [ β_0, β_1, β_2 ]
    ==> β_0 + β_1*x + β_2*x^2

B_over: np.ndarray, shape(m_features)
    Array of overfit regression coefficient outputs
    Degree will be inferred from this length
    Eg. [ β_0, β_1, ..., β_6 ]
    ==> β_0 + β_1*x + β_2*x^2 + ... + β_6*x^6

Result
------
Window displayed with data and side-by-side regressions
'''
def plotGoodVsOverFit(X: NDArray[np.float64], 
                      Y: NDArray[np.float64], 
                      B_good: NDArray[np.float64],
                      B_over: NDArray[np.float64]) -> None: 
    # Creating the plotting environment
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(X, Y)
    ax2.scatter(X, Y)
    
    # Define the range for x
    xmin, xmax = ax1.get_xlim()
    Xp = np.linspace(xmin, xmax, 100)

    # Calculate y-values of degree 2 polynomial
    P_good = np.poly1d(B_good[::-1])
    Yp_good = P_good(Xp)

    # Calculate y-values of degree 6 polynomial
    P_over = np.poly1d(B_over[::-1])
    Yp_over = P_over(Xp)

    # Adding a regression to each subplot with labels
    ax1.plot(Xp, Yp_good)
    ax1.set_title("Degree 2 Regression")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.plot(Xp, Yp_over)
    ax2.set_title("Degree 6 Regression")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    # Showing the graphs
    plt.tight_layout()
    plt.show()