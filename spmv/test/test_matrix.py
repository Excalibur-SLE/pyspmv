import spmv
import numpy


def test_csr_matvec():
    """Test CSR matrix generation."""

    from scipy.sparse import random

    m = 1000
    n = 500

    rng = numpy.random.default_rng(0)
    scipy_mat = random(m, n, format="csr", random_state=rng)
    csr_mat = spmv.CsrMatrix(
        scipy_mat.data, scipy_mat.indices, scipy_mat.indptr, (m, n)
    )

    vec = rng.normal(size=n)

    expected = scipy_mat @ vec
    actual = csr_mat @ vec

    numpy.testing.assert_allclose(actual, expected)


def test_csr_from_coo():
    """Test the generation of a CSR from a COO vectors."""

    from scipy.sparse import random

    m = 1000
    n = 500

    rng = numpy.random.default_rng(0)
    scipy_coo_mat = random(m, n, format="coo", random_state=rng)
    scipy_csr_mat = scipy_coo_mat.tocsr()
    csr_mat = spmv.CsrMatrix.from_coo(
        scipy_coo_mat.row, scipy_coo_mat.col, scipy_coo_mat.data, (m, n)
    )

    vec = rng.normal(size=n)

    expected = scipy_csr_mat @ vec
    actual = csr_mat @ vec

    numpy.testing.assert_allclose(actual, expected)
