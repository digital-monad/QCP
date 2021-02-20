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

    # controlled gates (act on multiple qubits, in some more complicated way)

    def c_z_last(self):
        '''controlled z operating on the last qubit'''
        C = np.zeros([self.n, self.n], dtype=complex)
        for i in range(self.n):
            C[i, i] = 1

        C[self.n - 1, self.n - 1] = -1
        #print("aaa")
        #print(self.n)
        #print(C)
        self.state = C.dot(self.state)

    def cnot(self):
        n = 4
        C = np.zeros([self.n, self.n])
        C[0][0] = 1
        C[1][1] = 1
        C[2][3] = 1
        C[3][2] = 1

        self.state = C.dot(self.state)

    def oracle(self):
        # which oracle is this?
        # randomly has one -1 along the diagonal
        n = 3
        r = np.random.randint(0, self.n)
        O = np.zeros([self.n, self.n], dtype=complex)

        for i in range(0, n):
            if i != r:
                O[i, i] = 1
            else:
                O[i, i] = -1

        self.state = O.dot(self.state)  # perform operation on entangle


def main():
    methods = explicit_matrices  # by changing this, you can change which class is used for the processing of gates

    # if the gates are named the same in alternative classes, it should just be a quick swap and any written algorithms
    # should work as before (but potentially more/less efficiently)

    class state(methods):
        def __init__(self, bits):
            self.state = self.entangle(bits)
            self.n = len(self.state)  # = 2 ^ num of entangled qubits
            self.qubit_n = len(bits)
            assert self.n == 2 ** self.qubit_n
            print("Initial state:")
            print(self.state)
            print(self.n)
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

    # example of how to use the class:

    # N qubit grovers (sifting for 0):
    test = state([[1, 0], [1, 0], [1, 0]])
    test.hadamard_all()
    test.c_z_last()
    test.z_all()
    test.c_z_last()
    test.hadamard_all()
    print(test.get_state())

    # right now, most of the gates are a huge mess so it all needs to be more consistent.


main()
