import MultipleRegression as MR
# import PolynomialRegression as PR


import numpy as np

n = np.array([
    [1, 2, 3], 
    [4, 5, 6],
])

m = np.array( [1,2] )


MR.regress(n, m)


n = np.array([
    [1, 0, ], 
    [2, 1, ], 
    [3, 1, ], 
    [4, 2, ], 
])

m = np.array( [4, 6, 9, 12, ] )

MR.regress(n, m)
