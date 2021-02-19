# QCP
import numpy as np
import cmath

# shorthand for i
i = complex(0,1)
_i = complex(0,-1)

# here I just coded a bunch of quantum gates of arbritrary size
# They need to be pieced together into a circuit
def oracle():
    # randomly has one -1 along the diagonal
    n = 3
    r = np.random.randint(0,n)
    O = np.zeros([n,n], dtype = complex)

    for i in range(0,n):
        if i != r:
            O[i,i] = 1
        else:
            O[i,i] = -1
    return O

def hadamard():
    n = 2
    init = np.ones([n,n], dtype = complex)
    init[1][1] = -1
    H = init*(1/np.sqrt(2))
    return H

def paulix():
    n = 2
    X = np.ones([n,n], dtype = complex)
    X[0][0] = 0
    X[1][1] = 0
    return X

def pauliy():
    n = 2
    Y = np.zeros([n,n], dtype = complex)
    Y[0][1] = _i
    Y[1][0] = i
    return Y

def pauliz():
    n = 2
    Z = np.zeros([n,n], dtype = complex)
    Z[0][0] = 1
    Z[1][1] = -1
    return Z

def cnot():
    n = 4
    C = np.zeros([n,n])
    C[0][0] = 1
    C[1][1] = 1
    C[2][3] = 1
    C[3][2] = 1
    return C

def phase(phi = 0.25):
    # angle can be changed easily, set in pi-radians
    n = 2
    P = np.zeros([n,n], dtype = complex)
    P[0][0] = 1
    P[1][1] = np.exp(i*phi*np.pi)
    return P

# initialised matricies to then be used
O = oracle()
H = hadamard()
X = paulix()
Y = pauliy()
Z = pauliz()
C = cnot()
P = phase()

# initialises |0> and |1> as matricies, I havent doubled checked these are correct yet
x = (1/np.sqrt(2)) * np.array([0,1])
y = (1/np.sqrt(2)) * np.array([1,0])

# some tests to see the Hadamard gate and phase gates acting on |0> and |1>
f1 = x * H
f2 = y * H
f3 = x * P
f4 = y * P

# print functions are sums of matricies, but also need to calc probability from complex numbers
print("Hadamard gate acting on |0> and |1>")
print(np.sum(f1))
print(np.sum(f2))
print()
print("Phase gate acting on |0> and |1>")
print(np.sum(f3))
print(np.sum(f4))
