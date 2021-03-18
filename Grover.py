def Grover(state, ws):
        '''Runs grovers on a state object, with the given ws winner states for the oracle'''

        state.hadamard_all()
        # Perform rotation
        # Change number in for loop - getting weird results

        # Perform rotation
        # Change number in for loop - getting weird results
        i = 4* int(state.qubit_n ** (1/2))  # iteration num = sqrt N * some factor
        print(i)
        for iteration in range(40):
            # Apply the phase oracle
            state.oracle_general(ws)
            # Probability amplification (diffuser)
            state.hadamard_all()
            state.x_all_fast()
            state.c_z_last()
            state.x_all_fast()
            state.hadamard_all()
        
        return state