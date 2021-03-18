import sparseGates as gates

# Start with an n qubit register
def QFT(n,state,end):
    '''Runs a quantum fourier transformation for an n qubit input'''
    for rot in range(end-1):
        hn = gates.H
        hn = gates.I**rot * gates.H if rot > 0 else hn
        hn = hn * gates.I**(n - 1 - rot)
        state = hn @ state
        print(f"Apply H to bit {rot}")
        k = 2
        print(f"Apply on bit {rot}")
        for cqubit in range(rot+1,end):
            print(f"Controlled Rotation control {cqubit} target {rot}, k {k}")
            rotation = gates.CROT(n,cqubit, rot, k)
            state = rotation @ state
            k += 1
    # hadamard the last qubit
    hn = gates.I**(end-1) * gates.H
    hn = hn * gates.I**(n - end) if n != end else hn
    state = hn @ state
    # Swap pairs of qubits
    for qubit in range(end//2):
        state = gates.swap(n,qubit,end-qubit-1) @ state
    print("Transformed Vector:")
    print(state)
    state.MeasureAll()
    return state

# Example individual usage:
# state = QuantumRegister(4,init=1)
# print(np.round(state,3))
# state = QFT(4,state,4)
# print(np.round(state,3))
# state.MeasureAll()