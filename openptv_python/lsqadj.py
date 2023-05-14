"""Least squares adjustment of the camera parameters."""
import numpy as np

# def ata(a, n):
#     """
#     Multiply transpose of a matrix A by matrix A itself, creating symmetric matrix.

#     Args:
#     ----
#     a - matrix of doubles of the size (m x n_large).
#     n - number of rows and columns in the output ata - the size of the sub-matrix

#     Returns:
#     -------
#     ata - matrix of the result multiply(a.T,a) of size (n x n)
#     """
#     # Transpose the input matrix a
#     a_T = np.transpose(a)

#     # Compute the product of a.T and a
#     ata = np.dot(a_T, a)

#     # Take only the upper-left square submatrix of size n x n
#     ata = ata[:n, :n]

#     return ata


def ata(a: np.ndarray, ata: np.ndarray, m: int, n: int, n_large: int) -> None:
    """Return the product of the transpose of a matrix.

    and the matrix itself, creating symmetric matrix.
    matrix a and result matrix ata = at a
    a is m * n_large, ata is an output n * n.
    """
    if a.shape != (m, n_large):
        raise ValueError("a has wrong shape")

    if ata.shape != (n, n):
        raise ValueError("ata has wrong shape")

    # a = a.flatten(order='C')
    # ata = ata.flatten(order='C')

    for i in range(n):
        for j in range(n):
            ata.flat[i * n_large + j] = 0.0
            for k in range(m):
                ata.flat[i * n_large + j] += (
                    a.flat[k * n_large + i] * a.flat[k * n_large + j]
                )


def atl(u: np.ndarray, a: np.ndarray, b: np.ndarray, m: int, n: int, n_large: int):
    """Multiply transpose of a matrix A by vector b, creating vector u.

    with the option of working with the sub-vector only, when n < n_large.

    Arguments:
    ---------
    u -- vector of doubles of the size (n x 1)
    a -- matrix of doubles of the size (m x n_large)
    l -- vector of doubles (m x 1)
    m -- number of rows in matrix a
    n -- length of the output u - the size of the sub-matrix
    n_large -- number of columns in matrix a
    """
    for i in range(n):
        u.flat[i] = 0.0
        for k in range(m):
            u.flat[i] += a.flat[k * n_large + i] * b.flat[k]


# # Define a matrix
# A = np.array([[1, 2], [3, 4]])

# # Calculate the inverse using numpy.linalg.inv
# A_inv = np.linalg.inv(A)

# # Print the result
# print(A_inv)


def matinv(a: np.ndarray, n: int, n_large: int) -> np.ndarray:
    """
    Calculate inverse of a matrix A, with the option of working with the sub-vector only, when n < n_large.

    Arguments:
    ---------
    a - matrix of doubles of the size (n_large x n_large).
    n - size of the output - size of the sub-matrix, number of observations
    n_large - number of rows and columns in matrix a
    """
    if a.shape != (n_large, n_large):
        raise ValueError("a has wrong shape")

    a = a.flatten(order="C")

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

    return a.reshape(n, n)


# def matmul(b: np.ndarray, c: np.ndarray, m: int, n: int, k: int):
#     """Multiply two matrices and store the result in a third matrix.

#         /* Calculate dot product of a matrix 'b' of the size (m_large x n_large) with
#     *  a vector 'c' of the size (n_large x 1) to get vector 'a' of the size
#     *  (m x 1), when m < m_large and n < n_large
#     *  i.e. one can get dot product of the submatrix of b with sub-vector of c
#     *   when n_large > n and m_large > m
#     *   Arguments:
#     *   a - output vector of doubles of the size (m x 1).
#     *   b - matrix of doubles of the size   (m x n)
#     *   c - vector of doubles of the size (n x 1)
#     *   m - integer, number of rows of a
#     *   n - integer, number of columns in a
#     *   k - integer, size of the vector output 'a', typically k = 1
#     */

#     """
#     a = np.zeros((m, 1), dtype=np.float64)
#     b_sub = b[:, :n]
#     c_sub = c[:n, :k]
#     a_sub = np.dot(b_sub, c_sub)
#     a[:m, :k] = a_sub

#     return a

# def matmul(a, b, c, m, n, k, m_large, n_large) -> None:
#     """ Multiply two matrices and store the result in a third matrix."""
#     for i in range(k):
#         pb = b
#         pa = a + i
#         for j in range(m):
#             pc = c
#             x = 0.0
#             for l in range(n):
#                 x += pb[l] * pc
#                 pc += k
#             for l in range(n_large-n):
#                 pb += 1
#                 pc += k
#             pa[0] = x
#             pa += k
#         for j in range(m_large-m):
#             pa += k
#         c += 1


# def matmul(
#     b: np.ndarray, c: np.ndarray, m: int, n: int, k: int, m_large: int, n_large: int
# ) -> np.ndarray:
#     """Multiply two matrices and store the result in a third matrix."""
#     c = np.atleast_2d(c).T
#     if m_large < m or n_large < n:
#         raise ValueError("m_large < m or n_large < n")

#     return (b[:m, :n] @ c[:n, :k]).transpose()


# def matmul(a, b, c, m, n, k, m_large, n_large):
#     """
#     Calculate dot product of a matrix 'b' of the size (m_large x n_large) with

#     a vector 'c' of the size (n_large x 1) to get vector 'a' of the size (m x 1),
#     when m < m_large and n < n_large.
#     i.e. one can get dot product of the submatrix of b with sub-vector of c when
#     n_large > n and m_large > m.

#     Arguments:
#     ---------
#     a - output vector of doubles of the size (m x 1).
#     b - matrix of doubles of the size (m_large x n_large)
#     c - vector of doubles of the size (n x 1)
#     m - integer, number of rows of a
#     n - integer, number of columns in a
#     k - integer, size of the vector output 'a', typically k = 1
#     m_large - number of rows in matrix b
#     n_large - number of columns in matrix b

#     """
#     if b.shape != (m_large, n_large):
#         raise ValueError("b has wrong shape")

#     c = np.atleast_2d(c).T
#     if c.shape != (n_large, k):
#         raise ValueError("c has wrong shape")

#     a = np.atleast_2d(a).T
#     if a.shape != (m, k):
#         raise ValueError("a has wrong shape")

#     b = b.flatten(order="C")
#     # a = a.flatten(order="C")

#     for i in range(k):
#         print(f"i = {i}")
#         pb[i] = b[i]
#         pa[i] = a[i]
#         print(f"pb = {pb}, pa = {pa}, a={a}")
#         for j in range(m):
#             pc = c
#             x = 0.0
#             for ll in range(n):
#                 x += pb[ll] * pc[ll * k]
#             for ll in range(n_large - n):
#                 pb += 1
#                 pc += k
#             pa[0] = x
#             pa += k
#             pb += n
#         for j in range(m_large - m):
#             pa += k
#         c += 1


def matmul(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    m: int,
    n: int,
    k: int,
    m_large: int,
    n_large: int,
) -> None:
    """
    Calculate dot product of a matrix 'b' of size (m_large x n_large).

    with a vector 'c' of size (n_large x 1) to get
    vector 'a' of size (m x 1), when m < m_large and n < n_large.

    Arguments:
    ---------
    a -- output vector of doubles of size (m x 1).
    b -- matrix of doubles of size (m x n)
    c -- vector of doubles of size (n x 1)
    m -- integer, number of rows of 'a'
    n -- integer, number of columns in 'a'
    k -- integer, size of the vector output 'a', typically k = 1
    m_large -- integer, number of rows in matrix 'b'
    n_large -- integer, number of columns in matrix 'b'
    """
    if m_large < m or n_large < n:
        raise ValueError("m_large < m or n_large < n")

    for i in range(k):
        for j in range(m):
            x = 0.0
            for ll in range(n):
                x += b.flat[j * n_large + ll] * c.flat[ll * k + i]
            a.flat[j * k + i] = x
