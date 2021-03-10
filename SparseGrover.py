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
    qState = qState @ hadamardAll
    oracle = gates.constructOracle(n,ws)
    # Perform rotation
    for iteration in range(4):
        # Apply the phase oracle
        qState = qState @ oracle
        # Probability amplification (diffuser)
        qState = qState @ hadamardAll
        qState = qState @ xAll
        qState = qState @ cnZ
        qState = qState @ xAll
        qState = qState @ hadamardAll
    print("Measurement after 4 iterations of grover: ")
    measurement = qState.MeasureAll()
    print(f"P(|{measurement}>) = {qState[measurement]**2}")

n = 9
w = np.random.randint(0,pow(2,9))
print(f"Random marked state is |{w}>")
grover(n,[w])


