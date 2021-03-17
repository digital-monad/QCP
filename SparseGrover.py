import numpy as np
from SparseMatrix import SparseMatrix
from QuantumRegister import QuantumRegister
import sparseGates as gates

def grover(n,ws):
    # Initialise the quantum register
    qState = QuantumRegister(n)
    # Pre-prepare some useful gates
    hadamardAll = gates.H**n
    xAll = gates.X**n
    cnZ = gates.cnZ(n)
    # Prepare register in equal superposition
    qState = hadamardAll @ qState
    oracle = gates.constructOracle(n,ws)
    # Perform rotation
    for iteration in range(int(np.pi*2**(n/2)/4)):
        print(f"Iteration {iteration}")
        # Apply the phase oracle
        qState = oracle @ qState
        # Probability amplification (diffuser)
        qState = hadamardAll @ qState
        qState = xAll @ qState
        qState = cnZ @ qState
        qState = xAll @ qState
        qState = hadamardAll @ qState
    print(f"Measurement after {int(np.pi*2**(n/2)/4)} iterations of grover: ")
    measurement = qState.MeasureAll()
    print(f"P(|{measurement}>) = {qState[measurement]**2}")

n = 10
w = np.random.randint(0,pow(2,n))
print(f"Random marked state is |{w}>")
grover(n,[w])


