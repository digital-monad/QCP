import numpy as np
import gates

def tensorProduct(U,V):
    # Tensor product of 2 matrices
    (rU,cU), (rV,cV) = U.shape, V.shape
    result = np.zeros((rU*rV,cU*cV))
    for r in range(rU):
        for c in range(cU):
            result[r*rV:(r+1)*rV,c*cV:(c+1)*cV] = U[r,c] * V
    return result

def tensorPower(U,n):
    # Matrix U tensored with itself n times
    result = U
    for i in range(n-1):
        result = tensorProduct(result,U)
    return result

def constructOracle(n,ws):
    # Create n qubit dummy oracle with ws as solution states
    oracle = np.eye(pow(2,n))
    for w in ws:
        oracle[w,w] = -1
    return oracle

def cnZ(n):
    # Matrix for controlled ^ n Z gate
    # There are n-1 control bits
    cnZ = np.eye(pow(2,n))
    cnZ[-1,-1] = -1
    return cnZ

def grover(n,ws):
    # Initialise the register
    qState = tensorPower(gates.ZERO,n)
    hadamardAll = tensorPower(gates.H,n)
    qState = np.dot(hadamardAll,qState)
    # Perform rotation
    # Change number in for loop - getting weird results
    for iteration in range(20):
        print(f"Iteration {iteration}")
        # Apply the phase oracle
        xAll = tensorPower(gates.X,n)
        oracle = constructOracle(n,ws)

        qState = np.dot(oracle,qState)
        # Probability amplification (diffuser)
        qState = np.dot(hadamardAll,qState)
        qState = np.dot(xAll,qState)
        qState = np.dot(cnZ(n),qState)
        qState = np.dot(xAll,qState)
        qState = np.dot(hadamardAll,qState)
    # Operations introduce a global phase, hence the -
    # This does not matter when it is measured
    #print("Final register state:")
    #print(qState)
    print("Measurement after 4 iterations of grover: ")
    measurement = gates.MeasureAll(qState)
    print(f"P(|{measurement}>) = {qState[measurement]**2}")

n = 11
w = np.random.randint(0,pow(2,n))
print(f"Random marked state is |{w}>")
grover(n,[w])