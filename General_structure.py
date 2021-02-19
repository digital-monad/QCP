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
    def __init__(self):
        # not much needed here?
        print("aaa")

    #######################################
    # these gates all need generalising

    def tensorProduct(self, U, V):
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

    def hadamard(self):
        '''applies the hadamard gate to each qubit in the entangled state (by reference)'''
        n = 2  # reminder- needs generalising

        init = np.ones([self.n, self.n], dtype=complex)  # set up array
        init[1][1] = -1  # correct the values to get a hadamard gate

        H = init * (1 / np.sqrt(2)) # normalisation

        # for 2 bit system only:
        H = self.tensorProduct(H, H)
        # H now is the full matrix for a hadamard gate that will be applied to every qubit in an entangled state of
        # self.n size

        # applying the matrix to the current state of the system to
        self.state = H.dot(self.state)

    def paulix(self):
        n = 2
        X = np.ones([self.n, self.n], dtype=complex)
        X[0][0] = 0
        X[1][1] = 0

        self.state = X.dot(self.state)

    def pauliy(self):
        n = 2
        Y = np.zeros([n, n], dtype=complex)
        Y[0][1] = _i
        Y[1][0] = i

        self.state = Y.dot(self.state)

    def pauliz(self):
        self.n = 2
        Z = np.zeros([self.n, self.n], dtype=complex)
        Z[0][0] = 1
        Z[1][1] = -1

        self.state = Z.dot(self.state)

    def phase(self, phi):
        # angle can be changed easily, set in pi-radians
        n = 2
        P = np.zeros([self.n, self.n], dtype=complex)
        P[0][0] = 1
        P[1][1] = np.exp(i * phi * np.pi)

        self.state = P.dot(self.state)

    # gates that act upon a single qubit with an operation (reminder to send the qubit number)

    # empty (for now, we want every single state in this form: compose them from a 1 qubit gate + identity and
    # repeated tensor products to combine if necessary. setup depends on qubit number and which qubit is acted on)

    # controlled gates (act on multiple qubits, in some more complicated way)

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

    class qstate(methods):
        def __init__(self, bits):
            self.state = self.entangle(bits)
            self.n = len(bits)    # = 2 ^ num of entangled qubits

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

            #print("--------------------")
            # print(state)
            #a = bits[0] # testing
            #b = bits[1] # testing
            #print(np.array([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]) == state) # testing, only for 2bit case
            return np.array(state)

        def normalise(self, vector):
            vector = np.array(vector)  # incase state is sent as a list
            total = 0
            for component in vector:
                total += component ** 2
            return vector / (total ** (1 / 2))

    # example of how to use the class:

    test = qstate([[1, 0], [0,1]])
    test.hadamard()
    print(test.state)

    # right now, most of the gates are a huge mess so it all needs to be more consistent.


main()