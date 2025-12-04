import numpy as np
from numpy.typing import NDArray

# Regressor Function
'''
Parameters
----------
X: np.ndarray, shape(n_samples, m_features)
    Dependent variable array. 
    Eg. [ 
          [x_11, x_12, ..., x_1m], 
          [x_21, x_22, ..., x_2m],   
                  ...            ,
          [x_n1, x_n2, ..., x_nm],   
        ]

Y: np.ndarray, shape(n_samples)
    Independent variable array. 
    Eg. [ Y_1, Y_2, ..., Y_n ]
    
Result
------
np.ndarray, shape(m_features)
    Array of regression coefficient outputs
    Eg. [ β_0, β_1, ..., β_m ]
    ==> β_0 + β_1*x_1 + β_2*x_2 + ... + β_m*x_m
    ==> [β]X.T (in matrix form)
'''

def regress(X: NDArray[np.float64], 
            Y: NDArray[np.float64]) -> NDArray[np.float64]: 
    
    # Constructing the design matrix by appending a column of 1s
    xmat = np.vstack( (np.ones((1, X.shape[0])), np.transpose(X)) )
    xmat = np.asmatrix(np.transpose(xmat))
    
    # Constructing the output matrix
    ymat = np.transpose(np.asmatrix(Y))

    # Computing two sides of the normal equations
    ata = np.transpose(xmat) * xmat
    atb = np.transpose(xmat) * ymat

    # Attempts to solve the system, if there is a singular matrix
    # then we pick a arbitrary solution from the solution space 
    # using the lstsq function. We note that this function may 
    # have been used from the start
    try:
        B = np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError: 
        B, _, _, _ = np.linalg.lstsq(ata, atb)

    # Return out equation coefficients
    B = [ b[0] for b in np.asarray(B) ]
    return np.asarray(B)

