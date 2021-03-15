# attempt 1 at structuring the code better
import numpy as np


# plan: have multiple classes, each with a suite of all the operations we are going to want, then only inherit/import
# the one we actually want to use (which can be switched really easily) to a class which has many algorithms implemented

# gate naming has to be consistent for this to work


class explicit_matrices:
    # this is going to be one implementation of the gates. Many will follow.
    # there may be better ways of creating this from a parent class if needed, depends how much can be reused between
    # this and sparse/lazy matrix implementations

    # weaknesses of this particular implementation (not exhaustive):
    # ~2 ^2n operations per gate application - v slow
    # gates generated each time- slow if they are used repeatedly (having the option to store may reduce time if say
    # we use hadamard again and again in one circuit, though storing adds to memory cost)

    ################################################
    # THIS CLASS IS A HUGE MESS RIGHT NOW
    # WE NEED A CLEAR NAMING SCHEME FOR THE METHODS (Multi/single qubit gates etc, so that everything is consistent
    # and easy to understand and use)
    ################################################

    def tensor_product(self, U, V):
        ###### should this be in the general class? should each of these classes inherit from an even further up class?
        # Naïve tensor product implementation
        (rU, cU), (rV, cV) = U.shape, V.shape
        result = np.zeros((rU * rV, cU * cV))
        for r in range(rU):
            for c in range(cU):
                result[r * rV:(r + 1) * rV, c * cV:(c + 1) * cV] = U[r, c] * V
        return result

    # gates that act upon all qubits with the same separate operation
    # hadamard is heavily annotated as an example

    def hadamard_all(self):
        '''applies the hadamard gate to each qubit in the entangled state (by reference)'''
        n = 2  # reminder- needs generalising

        init = np.ones([2, 2], dtype=complex)  # set up array
        init[1][1] = -1  # correct the values to get a hadamard gate

        H = init * (1 / np.sqrt(2))  # normalisation
        H_single = H
        for val in range(self.qubit_n - 1):  # v slow, but works
            H = self.tensor_product(H, H_single)
        # H now is the full matrix for a hadamard gate that will be applied to every qubit in an entangled state of
        # self.n size

        # applying the matrix to the current state of the system to
        self.state = H.dot(self.state)

    def x_all(self):
        n = 2
        X = np.ones([2, 2], dtype=complex)
        X[0][0] = 0
        X[1][1] = 0
        X_single = X

        for val in range(self.qubit_n - 1):  # v slow, but works
            X = self.tensor_product(X, X_single)

        self.state = X.dot(self.state)

    def y_all(self):
        n = 2
        Y = np.zeros([2, 2], dtype=complex)
        Y[0][1] = _i
        Y[1][0] = i
        Y_single = Y
        for val in range(self.qubit_n - 1):  # v slow, but works
            Y = self.tensor_product(Y, Y_single)

        self.state = Y.dot(self.state)

    def z_all(self):
        Z = np.zeros([2, 2], dtype=complex)
        Z[0][0] = 1
        Z[1][1] = -1
        Z_single = Z
        for val in range(self.qubit_n - 1):  # v slow, but works
            Z = self.tensor_product(Z, Z_single)

        self.state = Z.dot(self.state)

    def phase_all(self, phi):
        # angle can be changed easily, set in pi-radians
        n = 2
        P = np.zeros([2, 2], dtype=complex)
        P[0][0] = 1
        P[1][1] = np.exp(i * phi * np.pi)

        for val in range(self.qubit_n - 1):  # v slow, but works
            P = self.tensor_product(P, P)

        self.state = P.dot(self.state)

    # gates that act upon a single qubit with an operation (reminder to send the qubit number)

    # empty (for now, we want every single state in this form: compose them from a 1 qubit gate + identity and
    # repeated tensor products to combine if necessary. setup depends on qubit number and which qubit is acted on)

    # gates that act on the list of qubits sent:

    def hadamard_list(self, list):
        '''applies the hadamard gate to each qubit in the list indices sent (zero based)'''
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list

        init = np.ones([2, 2], dtype=complex)  # set up array
        init[1][1] = -1  # correct the values to get a hadamard gate
        I = np.array([[1, 0], [0, 1]])
        if 0 in list:
            H = init * (1 / np.sqrt(2))  # normalisation
        else:
            H = I
        H_single = init * (1 / np.sqrt(2))

        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                H = self.tensor_product(H, H_single)
            else:
                H = self.tensor_product(H, I)
        # H now is the full matrix for a hadamard gate that will be applied to every qubit in an entangled state of
        # self.n size

        # applying the matrix to the current state of the system to
        self.state = H.dot(self.state)

    def x_list(self, list):
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list
        X = np.ones([2, 2], dtype=complex)
        X[0][0] = 0
        X[1][1] = 0
        X_single = X
        I = np.array([[1, 0], [0, 1]])
        if not 0 in list:
            X = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                X = self.tensor_product(X, X_single)
            else:
                X = self.tensor_product(X, I)

        self.state = X.dot(self.state)

    def y_list(self, list):
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list
        Y = np.zeros([2, 2], dtype=complex)
        Y[0][1] = _i
        Y[1][0] = i
        Y_single = Y
        I = np.array([[1, 0], [0, 1]])
        if not 0 in list:
            Y = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                Y = self.tensor_product(Y, Y_single)
            else:
                Y = self.tensor_product(Y, I)

        self.state = Y.dot(self.state)

    def z_list(self, list):
        # approach: repeatedly tensor product either an identity or gate depending on if the item is in the list
        Z = np.zeros([2, 2], dtype=complex)
        Z[0][0] = 1
        Z[1][1] = -1
        Z_single = Z
        I = np.array([[1, 0], [0, 1]])
        if not 0 in list:
            Z = I  # special case for if the first qubit does not have an x gate across it (zero isn't in list)
        for val in range(self.qubit_n - 1):  # v slow, but works
            if val + 1 in list:
                Z = self.tensor_product(Z, Z_single)
            else:
                Z = self.tensor_product(Z, I)

        self.state = Z.dot(self.state)

    # controlled gates (act on multiple qubits, in some more complicated way)

    def c_z_last(self):
        '''controlled z operating on the last qubit'''
        C = np.zeros([self.n, self.n], dtype=complex)
        for i in range(self.n):
            C[i, i] = 1

        C[self.n - 1, self.n - 1] = -1
        # print("aaa")
        # print(self.n)
        # print(C)
        self.state = C.dot(self.state)

    def cnot(self):
        n = 4
        C = np.zeros([self.n, self.n])
        C[0][0] = 1
        C[1][1] = 1
        C[2][3] = 1
        C[3][2] = 1

        self.state = C.dot(self.state)

    def oracle_general(self, ws):
        # Create n qubit dummy oracle with ws as solution states
        oracle = np.eye(pow(2, int(self.qubit_n)))
        for w in ws:
            oracle[w, w] = -1
        return np.dot(oracle, self.state)


class programs:

    def amplifier(self, state):
        a = 1

    def grovers(state, ws):
        '''Runs grovers on a state object, with the given ws for the oracle'''
        state.hadamard_all()
        # Perform rotation
        # Change number in for loop - getting weird results
        for iteration in range(4):
            # Apply the phase oracle

            state.oracle_general(ws)
            # Probability amplification (diffuser)
            state.hadamard_all()
            state.x_all()
            state.c_z_last()
            state.hadamard_all()
        return state

    def measure(state):
        '''returns a quantum measurement on the state object'''
        cum_prob = np.cumsum(state.get_state() ** 2)
        r = np.random.rand()
        measurement = np.searchsorted(cum_prob, r)
        print(f"Collapsed into basis state |{measurement}> (|{bin(measurement)[2:]}>)")
        return measurement

    def print_register(state):
        print(f"The quantum register is in the state {state.get_state()}")

# everything above here could be in a separate file if wanted, then imported
######################################################################################################################
# the class state has to be defined below, and the entirety of the code in main should make the general control loop
# for the program. This is to allow for the user to choose the calculation method used during runtime.


def main():
    # if the gates are named the same in alternative classes, it should just be a quick swap and any written algorithms
    # should work as before (but potentially more/less efficiently)

    # user input goes here, to decide program, calculation method, inputs.

    input = [[1, 0], [0, 1], [0, 1]]  ###### placeholder: will need to read in data and validate
    methods = explicit_matrices  # by changing this, you can change which class is used for the processing of gates.

    # can be user defined from a list using a case statement possibly?

    class state(methods):
        def __init__(self, bits):
            self.state = self.entangle(bits)
            self.n = len(self.state)  # = 2 ^ num of entangled qubits
            self.qubit_n = len(bits)
            assert self.n == 2 ** self.qubit_n
            print("Initial state:")
            print(bits)
            print("entangled: ")
            print(self.state)
            print("-----------------------")

        def entangle(self, bits):
            """returns the entangled state for n Qubits"""
            # this is just a special case of the tensor product repeatedly applied

            state = [1]  # set up the start of the multiplication
            bits.reverse()  # to get the bits in reverse order, since the loop operates backwards on them
            for bit in bits:  # adds each qubit 1 by 1 to an entangled state
                for element, comparison in zip(bit, self.normalise(bit)):
                    # check that "bit" is normalised, otherwise throw error
                    assert element == comparison, "State not normalised: output will be incorrect"

                # add the next bit to the entangled state, using a tensor product:
                state = list(np.array(state).dot(bit[0])) + list(np.array(state).dot(bit[1]))

            # print("--------------------")
            # print(state)
            # a = bits[0] # testing
            # b = bits[1] # testing
            # print(np.array([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]) == state) # testing, only for 2bit case
            return np.array(state)

        def normalise(self, vector):
            vector = np.array(vector)  # incase state is sent as a list
            total = 0
            for component in vector:
                total += component ** 2
            return vector / (total ** (1 / 2))

        def get_state(self):
            return self.state

    np.set_printoptions(formatter={'complex_kind': '{:.5f}'.format})  # formatting, hides floating point errors from
    # making messy looking outputs

    # every time I write 'run on the state', I mean send the state object (not just tbe vector) to the method.

    # all the outpus/ calculations should go here: create a state with the given inputs,
    # run the needed programs on that state
    # to output a measurement, simply run programs.measure on the state

    # example of how to use the class:
    print("starting")
    print("...")
    input = [[1, 0], [0, 1], [0, 1]]
    my_state = state(input)
    final_state = programs.grovers(my_state, [5, 6])
    programs.print_register(final_state)
    programs.measure(final_state)
    print("")
    print("done")


main()
