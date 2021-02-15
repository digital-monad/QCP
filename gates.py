import numpy as np

def tensorProduct(U,V):
    # Naïve tensor product implementation
    (rU,cU), (rV,cV) = U.shape, V.shape
    result = np.zeros((rU*rV,cU*cV))
    for r in range(rU):
        for c in range(cU):
            result[r*rV:(r+1)*rV,c*cV:(c+1)*cV] = U[r,c] * V
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

cNOT = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
])

def MeasureAll(register):
    cum_prob = np.cumsum(register**2)
    r = np.random.rand()
    measurement = np.searchsorted(cum_prob,r)
    print(f"Collapsed into basis state : |{measurement}>")
    return measurement



# Extremely simple circuit to entangle 2 qubits

qRegister = tensorProduct(ZERO,ZERO) # Start with a register of 2 qubits
operator1 = tensorProduct(H,I) # The first operator is an H gate on qubit 0
qRegister = np.dot(operator1,qRegister) # Apply it to the register
# print(qRegister)
qRegister = np.dot(cNOT,qRegister) # Apply a CNOT gate to the register
# print(qRegister)
MeasureAll(qRegister) # Measure the state of the register
