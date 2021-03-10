import numpy as np
from SparseMatrix import SparseMatrix

def fromDense(M):
    assert M.shape[0] == M.shape[1]
    sp = SparseMatrix(M.shape[0])
    for col in range(M.shape[0]):
        for row in range(M.shape[0]):
            if M[row,col] != 0:
                sp[row,col] = M[row,col]
    return sp

ZERO = np.array([1,0])
ONE = np.array([0,1])

H = fromDense(np.array([
    [1,1],
    [1,-1]
]) / np.sqrt(2))

X = fromDense(np.array([
    [0,1],
    [1,0]
]))

Z = fromDense(np.array([
    [1,0],
    [0,-1]
]))

I = fromDense(np.eye(2))

cX = fromDense(np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
]))

cZ = fromDense(np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,-1]
]))

def cnZ(n):
    # Matrix for controlled ^ n Z gate
    # There are n-1 control bits
    cnZ = np.eye(pow(2,n))
    cnZ[-1,-1] = -1
    return fromDense(cnZ)

def constructOracle(n,ws):
    # Create n qubit dummy oracle with ws as solution states
    oracle = np.eye(pow(2,n))
    for w in ws:
        oracle[w,w] = -1
    return fromDense(oracle)

def MeasureAll(register):
    cum_prob = np.cumsum(register**2)
    r = np.random.rand()
    measurement = np.searchsorted(cum_prob,r)
    print(f"Collapsed into basis state |{measurement}> (|{bin(measurement)[2:]}>)")
    return measurement