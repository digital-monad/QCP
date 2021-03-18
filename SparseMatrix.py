import numpy as np

class SparseMatrix:

    def __init__(self,dim):
        '''Initialise the sparse matrix with the column array'''
        self.dim = dim
        self.cols = np.array([dict() for _ in range(dim)])
    
    def __getitem__(self,indices):
        '''Overload indexing'''
        row, col = indices
        if row in self.cols[col]:
            return self.cols[col][row]
        else:
            return 0.0

    def __setitem__(self,indices,value):
        '''Overload indexing (set)'''
        row, col = indices
        self.cols[col][row] = value

    def __matmul__(self,other):
        '''sparse matrix multiplication'''
        if isinstance(other, SparseMatrix):
            result = SparseMatrix(self.dim)
            for col in range(len(other.cols)):
                for row in other.cols[col]:
                    thisColumn = self.cols[row]
                    for thisRow in thisColumn:
                        result[thisRow,col] = result[thisRow,col] + other.cols[col][row] * thisColumn[thisRow]
            return result
        elif isinstance(other,np.ndarray):
            newRegister = np.zeros_like(other,dtype=complex)
            for col in range(self.dim):
                for row in self.cols[col]:
                    # print(f"Row = {row}")
                    # print(f"Col = {col}")
                    newRegister[row] += self.cols[col][row] * other[col]
            return newRegister
        elif isinstance(other, (int,float)):
            for col in range(len(self.cols)):
                self.cols[col] = dict(map(lambda x : (x[0],other*x[1]), self.cols[col].items()))
            return self

    def __rmatmul__(self,other):
        if isinstance(other, (int,float)):
            return self.__mul__(other)


    def __str__(self):
        '''Pretty printing'''
        output = ""
        for row in range(self.dim):
            for col in range(self.dim):
                output += str(round(self[row,col],3))
                output += " "
            output += "\n"
        return output

    def __mul__(self,other):
        '''Sparse tensor product'''
        dim = self.dim*other.dim
        result = SparseMatrix(dim)
        for row in range(dim):
            for col in range(dim):
                result[row,col] = self[row//other.dim,col//other.dim] * other[row%other.dim,col%other.dim]
        return result

    def __pow__(self,power):
        '''Sparse tensor power'''
        result = self
        for _ in range(power-1):
            result *= self
        return result