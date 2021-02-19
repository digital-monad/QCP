import numpy as np

def tensorProduct(U,V):
    # Naïve tensor product implementation
    (rU,cU), (rV,cV) = U.shape, V.shape
    result = np.zeros((rU*rV,cU*cV))
    for r in range(rU):
        for c in range(cU):
            result[r*rV:(r+1)*rV,c*cV:(c+1)*cV] = U[r,c] * V
    return result

def tensorPower(U,n):
    result = U
    for i in range(n-1):
        result = tensorProduct(result,U)
    return result

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


def entangle():
    # Extremely simple circuit to entangle 2 qubits
    qRegister = tensorProduct(ZERO,ZERO) # Start with a register of 2 qubits
    operator1 = tensorProduct(H,H) # The first operator is an H gate on qubit 0
    qRegister = np.dot(operator1,qRegister) # Apply it to the register
    # print(qRegister)
    qRegister = np.dot(cNOT,qRegister) # Apply a CNOT gate to the register
    # print(qRegister)
    MeasureAll(qRegister) # Measure the state of the register

def grover2bit(q1,q2):
    # 2 qubit Grover
    qRegister = tensorProduct(q1,q2)
    HH = tensorProduct(H,H)
    qRegister = np.dot(HH,qRegister)
    qRegister = np.dot(cZ,qRegister)
    qRegister = np.dot(HH,qRegister)
    XX = tensorProduct(X,X)
    qRegister = np.dot(XX,qRegister)
    qRegister = np.dot(cZ,qRegister)
    qRegister = np.dot(XX,qRegister)
    qRegister = np.dot(HH,qRegister)
    print(f"State of quantum register : \n {qRegister}\n")
    print("Measuring...\n")
    MeasureAll(qRegister)

# grover2bit(ZERO,ZERO)

print(tensorProduct(X,I))