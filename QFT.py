import numpy as np
from QuantumRegister import QuantumRegister
import sparseGates as gates

# Start with an n qubit register
def QFT(n,state):
    for rot in range(n-1):
        hn = gates.H
        hn = gates.I**rot * gates.H if rot > 0 else hn
        hn = hn * gates.I**(n- rot - 1)
        state = hn @ state
        for qubit in range(n-1):
            rotation = gates.CROT(n,qubit+1,rot,qubit+2)
            state = rotation @ state
    # hadamard the last qubit
    hn = gates.I**(n-1) * gates.H
    state = hn @ state
    # Swap pairs of qubits
    for qubit in range(n//2):
        state = gates.swap(n,qubit,n-qubit-1) @ state
    print(np.sum(state**2))
    return state

state = QuantumRegister(4)
state = QFT(4,state)
state.MeasureAll()
