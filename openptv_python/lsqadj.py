"""Least squares adjustment of the camera parameters."""
import numpy as np


def ata(a, n):
    """
    Multiply transpose of a matrix A by matrix A itself, creating symmetric matrix.

    Args:
    ----
    a - matrix of doubles of the size (m x n_large).
    n - number of rows and columns in the output ata - the size of the sub-matrix

    Returns:
    -------
    ata - matrix of the result multiply(a.T,a) of size (n x n)
    """
    # Transpose the input matrix a
    a_T = np.transpose(a)

    # Compute the product of a.T and a
    ata = np.dot(a_T, a)

    # Take only the upper-left square submatrix of size n x n
    ata = ata[:n, :n]

    return ata


def atl(a, ll, n):
    """Multiply transpose of a matrix A by vector l, creating vector u.

    with the option of working with the sub-vector only, when n < n_large.

    Args:
    ----
    a - matrix of doubles of the size (m x n_large).
    l - vector of doubles (m x 1)
    n - length of the output u - the size of the sub-matrix

    Returns:
    -------
    u - vector of doubles of the size (n x 1)
    """
    # Transpose the input matrix a
    a_T = np.transpose(a)

    # Compute the product of a.T and l
    u = np.dot(a_T, ll)

    # Take only the first n elements of the result vector
    u = u[:n]

    return u


# # Define a matrix
# A = np.array([[1, 2], [3, 4]])

# # Calculate the inverse using numpy.linalg.inv
# A_inv = np.linalg.inv(A)

# # Print the result
# print(A_inv)


def matinv(a: np.ndarray, n: int, n_large: int):
    """Invert a matrix using Gauss-Jordan elimination.

    This is a Python function that calculates the inverse of a
    square matrix a of size n_large x n_large. The function uses
    the Gauss-Jordan method to invert the matrix. If the size of
    the sub-matrix is less than n_large, the function only works
    with the sub-matrix of size n.

    The function takes three arguments:

        a: a NumPy array that represents the matrix to be inverted.
        n: an integer that represents the size of the sub-matrix.
        n_large: an integer that represents the number of rows and
        columns in the matrix a.

    The function starts by iterating over the rows of the sub-matrix
    using the ipiv variable. It calculates the pivot element as
    pivot = 1.0 / a[ipiv * n_large + ipiv] and its negative as
    npivot = - pivot. Then it updates the elements of the sub-matrix to
    get the identity matrix on the left side of the original matrix.
    This is done using the nested for loops that iterate over the rows
    and columns of the sub-matrix. The inner loops update the elements
    of the sub-matrix using the pivot element.

    The next loop updates the elements below the diagonal to make them
    zero, as well as updating the diagonal elements to make them equal to 1.
    Finally, the function sets the diagonal element to pivot as it was
    updated to 1 in the previous loop.

    At the end of the function, the inverse of the matrix is stored
    in the original a array.


    """
    for ipiv in range(n):
        pivot = 1.0 / a[ipiv * n_large + ipiv]
        npivot = -pivot
        for irow in range(n):
            for icol in range(n):
                if irow != ipiv and icol != ipiv:
                    a[irow * n_large + icol] -= (
                        a[ipiv * n_large + icol] * a[irow * n_large + ipiv] * pivot
                    )
        for icol in range(n):
            if ipiv != icol:
                a[ipiv * n_large + icol] *= npivot
        for irow in range(n):
            if ipiv != irow:
                a[irow * n_large + ipiv] *= pivot
        a[ipiv * n_large + ipiv] = pivot

    return a


def matmul(b: np.ndarray, c: np.ndarray, m: int, n: int, k: int):
    """Multiply two matrices and store the result in a third matrix.

        /* Calculate dot product of a matrix 'b' of the size (m_large x n_large) with
    *  a vector 'c' of the size (n_large x 1) to get vector 'a' of the size
    *  (m x 1), when m < m_large and n < n_large
    *  i.e. one can get dot product of the submatrix of b with sub-vector of c
    *   when n_large > n and m_large > m
    *   Arguments:
    *   a - output vector of doubles of the size (m x 1).
    *   b - matrix of doubles of the size   (m x n)
    *   c - vector of doubles of the size (n x 1)
    *   m - integer, number of rows of a
    *   n - integer, number of columns in a
    *   k - integer, size of the vector output 'a', typically k = 1
    */

    """
    a = np.zeros((m, 1), dtype=np.float64)
    b_sub = b[:, :n]
    c_sub = c[:n, :k]
    a_sub = np.dot(b_sub, c_sub)
    a[:m, :k] = a_sub

    return a


def matmul2(b, c, m, n, k, m_large, n_large):
    """Multiply two matrices and store the result in a third matrix."""
    a = np.zeros((m, k), dtype=np.float64)
    for i in range(k):
        pb = b
        pa = a[:, i]
        for j in range(m):
            pc = c
            x = np.dot(pb, pc[0:n])
            pa[j] = x
            pb += n_large
        c += 1
        a[:, i] = np.pad(pa, (0, m_large - m), mode="constant", constant_values=0)
    return a[:, 1]
