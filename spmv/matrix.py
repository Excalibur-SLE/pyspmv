"""
Sparse matrix structure.

The following sparse matrix implementation
has been completely written with Numba. It uses
a CSR representation.

"""

import numpy
import numba
from scipy.sparse.linalg import LinearOperator


class Matrix(LinearOperator):
    """Implementation of a CSR matrix."""

    def __init__(self, data, indices, indptr, shape):
        """
        Definition of a CSR matrix.

        The matrix is initialized from data, indices and
        indptr. The shape is a tuple with the matrix shape.

        The matrix implements the Scipy linear operator
        protocol.

        """

        self._data = data
        self._indices = indices
        self._indptr = indptr
        self.shape = shape
        self.dtype = data.dtype

    def _matvec(self, vec):
        """Multiplication with a vector."""

        if vec.shape[0] != self.shape[1]:
            raise ValueError(
                f"Vector has dimension {vec.shape[0]}, but require {self.shape[1]}."
            )

        if vec.dtype != self.dtype:
            raise ValueError(
                f"Matrix has type {self.dtype}, but vector has type {vec.dtype}."
            )

        return _numba_matvec(
            vec, self._data, self._indices, self._indptr, self.shape[0]
        )

    @classmethod
    def from_coo(cls, row_indices, col_indices, data, shape):
        """Create sparse matrix from AIJ representation."""

        csr_data, indices, indptr = _numba_csr_from_aij(
            row_indices, col_indices, data, shape[0]
        )

        return cls(csr_data, indices, indptr, shape)


@numba.njit(parallel=True)
def _numba_matvec(vec, data, indices, indptr, nrows):
    """Numba implementation of CSR Matvec."""

    res = numpy.zeros(nrows, dtype=vec.dtype)

    for row_index in numba.prange(shape[0]):
        for index in range(indptr[row_index], indptr[row_index + 1]):
            col_index = indices[index]
            res[row_index] += data[index] * vec[col_index]

    return res


@numba.njit
def _numba_csr_from_coo(row_indices, col_indices, data, nrows):
    """Numba implementation of AIJ to CSR conversion."""

    nnz = data.shape[0]
    elems_per_row = numpy.zeros(nrows, dtype=np.int64)
    csr_data = numpy.empty(nnz, dtype=data.dtype)
    indices = numpy.empty(nnz, dtype=np.int64)
    indptr = numpy.empty(1 + nrows, dtype=np.int64)

    for index in range(nnz):
        elems_per_row[row_indices[index]] += 1

    count = 0

    for index in range(nrows):
        indptr[index] = count
        count += elems_per_row[index]

    indptr[-1] = count

    for index in range(nnz):
        row = row_indices[index]
        col = col_indices[index]
        value = data[index]
        new_index = indptr[row]
        csr_data[new_index] = value
        indices[new_index] = col
        indptr[row] += 1

    last = 0
    for index in range(nrows):
        indptr[i], last = last, indptr[i + 1]

    return csr_data, indices, indptr
