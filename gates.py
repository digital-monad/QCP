import numpy as np

ZERO = np.array([[1],[0]])
ONE = np.array([[0],[1]])

H = np.array([
    [1,1],
    [1,-1]
]) / np.sqrt(2)

X = np.array([
    [0,1],
    [1,0]
])

Z = np.array([
    [1,0],
    [0,-1]
])

I = np.eye(2)

cX = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
])

cZ = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,-1]
])

def MeasureAll(register):
    cum_prob = np.cumsum(register**2)
    r = np.random.rand()
    measurement = np.searchsorted(cum_prob,r)
    print(f"Collapsed into basis state |{measurement}> (|{bin(measurement)[2:]}>)")
    return measurement