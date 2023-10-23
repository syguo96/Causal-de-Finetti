import numpy as np
from scipy.stats import chi2
from itertools import product

def gtest(X, Y, Z):
    """
    The G test can test for goodness of fit to a distribution.
    We are here testing the null hypothesis X \perp Y |Z
    Note X, Y, Z are all binary variables
    Parameters
    ----------
    X : array
    Y : array
    Z : n*d matrix with d number of conditioned variables
    ddof : int, optional
        adjustment to the degrees of freedom for the p-value
    Returns
    -------
    chisquare statistic : float
        The chisquare test statistic
    p : float
        The p-value of the test.
    Notes
    -----
    """
    assert len(X) == len(Y)
    n = Z.shape[1]
    df = 2 ** n
    EmpTable = np.zeros((2, 2, df))
    conditionedset = list(product([0, 1], repeat=n))
    for x in [0, 1]:
        for y in [0, 1]:
            for z in conditionedset:
                xindex = set(np.where(X == x)[0])
                yindex = set(np.where(Y == y)[0])
                valid_index = xindex.intersection(yindex)
                for i in range(len(z)):
                    valid_index = valid_index.intersection(set(np.where(Z[:, i] == z[i])[0]))
                ind = conditionedset.index(z)
                EmpTable[x, y, ind] = len(valid_index)

    ExpTable = np.zeros((2, 2, df))
    for x in [0, 1]:
        for y in [0, 1]:
            for z in range(len(conditionedset)):
                ExpTable[x, y, z] = np.sum(EmpTable[x, :, z]) * np.sum(EmpTable[:, y, z])
                ExpTable[x, y, z] /= np.sum(EmpTable[:, :, z])
    prop = EmpTable.reshape(-1) * np.log((EmpTable / ExpTable).reshape(-1))
    prop = np.nan_to_num(prop)
    g = 2 * np.sum(prop)
    sig = chi2.sf(g, df)
    return g, sig