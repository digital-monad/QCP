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

def CROT(n,control, target, k):
    control = n - control - 1
    target = n - target - 1
    cROT = SparseMatrix(2**n)
    period = 2**control
    periodT = 2**target
    for col in range(2**n):
        if (col//period)%2 == 1 and (col//periodT)%2 == 1:
            cROT.cols[col][col] = np.exp(np.pi * 1j / 2**(k-1) )
        else:
            cROT.cols[col][col] = 1
    return cROT

def swap(n,bit1,bit2):
    return cXs(n,bit1,bit2) @ cXs(n,bit2,bit1) @ cXs(n,bit1,bit2)

def mod_exp(a,j,N):
    for i in range(j):
        a = np.mod(a**2,N)
    return a

def cU(t,n,control,a,j,N):
    control = t + n - control - 1
    period = 2**control
    cU = SparseMatrix(2**(t+n))
    for col in range(2**(t+n)):
        # print(f"Considering basis state {col}")
        if (col//period)%2 == 1:
            L_reg = col & (2**(n-1) - 1)
            if L_reg > N:
                applyUx = L_reg
            else:
                applyUx = np.mod(L_reg * a**(2**j),N)
            basisTransform = int(format(col,'#010b')[:-n] + format(applyUx,'#010b')[-n:],2)
            if basisTransform > 2**(t+n) - 1:
                print(f"control {control} j {j} col {col} L_reg {L_reg}")
            cU.cols[col][basisTransform] = 1.0
        else:
            cU.cols[col][col] = 1.0
    return cU
    

def U(n,a,j,N):
    U = SparseMatrix(2**n)
    for y in range(2**n):
        if y < N:
            U.cols[y][np.mod(a*y,N)] = 1.0
        else:
            U.cols[y][y] = 1.0
    return U

def Ux(n,a,j,N):
    U = SparseMatrix(2**n)
    for y in range(2**n):
        if y < N:
            U.cols[y][mod_exp(a,j,N) * np.mod(y,N)] = 1.0
        else:
            U.cols[y][y] = 1.0
    return U



def cnZ(n):
    # Matrix for controlled ^ n Z gate
    # There are n-1 control bits
    cnZ = np.eye(pow(2,n))
    cnZ[-1,-1] = -1
    return fromDense(cnZ)

def cXs(n,control,target):
    control = n - control - 1
    target = n - target - 1
    cX = SparseMatrix(2**n)
    period = 2**control
    for col in range(2**n):
        if (col//period)%2 == 1:
            idx = col ^ (1 << target)
        else:
            idx = col
        cX.cols[col][idx] = 1
    return cX

def cZs(n,control,target):
    control = n - control - 1
    target = n - target - 1
    cX = SparseMatrix(2**n)
    periodC = 2**control
    periodT = 2**target
    for col in range(2**n):
        if (col//periodC)%2 == 1 and (col//periodT)%2 == 1:
            cX.cols[col][col] = -1
        else:
            cX.cols[col][col] = 1
    return cX

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