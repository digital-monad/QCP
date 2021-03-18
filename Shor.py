from QFT import QFT
import numpy as np
from QuantumRegister import QuantumRegister
import sparseGates as gates

# Test N = 15
N = 15
# Take a as 7
a = 7
# Create a register with 12 qubits
state = QuantumRegister(12)
# Put register 1 in the 0 state and register 2 in the 1 state
print("Hadamarding All")
# state = (gates.H**8 * gates.I**4) @ state
print("Starting Us")

for i in range(8):
    U_n = gates.cU(8,4,8 - 1 - i,a,i,N)
    state = U_n @ state
state = (gates.I**8 * gates.H) @ state

state.MeasureAll()
# Apply inverse QFT
