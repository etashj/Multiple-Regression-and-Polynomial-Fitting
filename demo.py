import MultipleRegression as MR
import PolynomialRegression as PR
import numpy as np

x = np.array( [-3, -2, -1, 0, 1, 2, 3, ] )
y = np.array( [-14, -1, 4, 5, 2, 1, -2 ] )

ans = PR.regress_qr(x, y, 4)
print(ans)

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
