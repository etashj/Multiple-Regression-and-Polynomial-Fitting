import numpy as np
from numpy.typing import NDArray

# Regressor Function
'''
Parameters
----------
X: np.ndarray, shape(n_samples)
    Dependent variable array. 
    Eg. [ X_1, X_2, ..., X_n ]

Y: np.ndarray, shape(n_samples)
    Independent variable array. 
    Eg. [ Y_1, Y_2, ..., Y_n ]

deg: int
    Degree of the polynomial we wish to construct
    Eg. 3 (for a cubic)
    
Result
------
np.ndarray, shape(m_features)
    Array of regression coefficient outputs
    Eg. [ β_0, β_1, ..., β_deg ]
    ==> β_0 + β_1*x + β_2*x^2 + ... + β_m*x^deg
'''

def regress(X: NDArray[np.float64], Y: NDArray[np.float64], degree: int) -> NDArray[np.float64]: 
    xmat = np.ones((1, X.shape[0]))
    for i in range(1, degree+1): 
        xmat = np.vstack((xmat, X**i))
    xmat = np.asmatrix(np.transpose(xmat))
    
    ymat = np.transpose(np.asmatrix(Y))

    # Computing two sides of the normal equations
    ata = np.transpose(xmat) * xmat
    atb = np.transpose(xmat) * ymat

    # Attempts to solve the system, if there is a singular matrix
    # then we pick a arbitrary solution from the solution space using 
    # the lstsq function. We note that this function may have been used from the start
    try:
        B = np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError: 
        B, _, _, _ = np.linalg.lstsq(ata, atb)

    # Return out equation coefficients
    return B
    
def regress_qr(X: NDArray[np.float64], Y: NDArray[np.float64], degree: int) -> NDArray[np.float64]: 
    xmat = np.ones((1, X.shape[0]))
    for i in range(1, degree+1): 
        xmat = np.vstack((xmat, X**i))
    xmat = np.asmatrix(np.transpose(xmat))

    Q, R = np.linalg.qr(xmat, mode="reduced")
    
    ymat = np.transpose(np.asmatrix(Y))

    try:
        B = np.linalg.solve(R, np.transpose(Q) * ymat)
    except np.linalg.LinAlgError: 
        B, _, _, _ = np.linalg.lstsq(R, np.transpose(Q) * ymat)

    # Return out equation coefficients
    return B
