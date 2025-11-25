import sympy as sp
from functools import lru_cache

def sym_basis_k_symbolic(k, d=3):
    basis = []

    # Factorials and sqrt become symbolic
    factorial = sp.factorial
    sqrt = sp.sqrt

    def generate(remain, parts):
        if len(parts) == d - 1:
            parts = parts + [remain]
            a, b, c = parts

            norm = sqrt(factorial(a) * factorial(b) * factorial(c) / factorial(k))
            basis.append((tuple(parts), sp.simplify(norm)))
            return

        for i in range(remain, -1, -1):
            generate(remain - i, parts + [i])

    generate(k, [])
    return basis

# # Test
# for vec, norm in sym_basis_k_symbolic(4):
#     print(vec, norm)

def pi_symmetric_multinomial_opt(M, basis, numeric=False):
    """
    Compute the symmetric multinomial projection of M on the given basis.

    Args:
        M (sp.Matrix): d x d symbolic matrix.
        basis (list): List of (vector, norm) tuples from sym_basis_k_symbolic.
        numeric (bool): If True, evaluates numerically to speed up large k.

    Returns:
        sp.Matrix: Projected matrix.
    """
    d = M.rows
    n = len(basis)
    piM = sp.zeros(n, n)
    factorial = sp.factorial

    # Maximum degree k (all basis vectors have same degree)
    max_k = sum(basis[0][0])
    factorials = [factorial(i) for i in range(max_k + 1)]

    # Precompute powers of M
    powers = {}
    for u in range(d):
        for v in range(d):
            powers[(u, v)] = [1]  # M[u,v]^0
            val = 1
            for t in range(1, max_k + 1):
                val *= M[u, v]
                powers[(u, v)].append(val)

    # Function to enumerate contingency matrices
    def enum_contingency(rows, cols):
        @lru_cache(maxsize=None)
        def _recurse(row_idx, cols_remaining):
            cols_remaining = list(cols_remaining)
            if row_idx == len(rows):
                if all(c == 0 for c in cols_remaining):
                    return [()]
                return []
            results = []
            target = rows[row_idx]

            def compose(pos, left, current):
                if pos == len(cols_remaining) - 1:
                    val = left
                    if val <= cols_remaining[pos]:
                        yield tuple(current + [val])
                    return
                maxv = min(left, cols_remaining[pos])
                for v in range(maxv, -1, -1):
                    yield from compose(pos + 1, left - v, current + [v])

            for row_choice in compose(0, target, []):
                new_cols = tuple(cols_remaining[j] - row_choice[j] for j in range(len(cols_remaining)))
                for tail in _recurse(row_idx + 1, new_cols):
                    results.append(tuple(row_choice) + tail)
            return results

        return _recurse(0, tuple(cols))

    # Compute projection matrix
    for i_idx, (vec_i, norm_i) in enumerate(basis):
        for j_idx, (vec_j, norm_j) in enumerate(basis):
            val = 0
            for mat_flat in enum_contingency(tuple(vec_i), tuple(vec_j)):
                term = 1
                denom = 1
                for u in range(d):
                    for v in range(d):
                        t = mat_flat[u * d + v]
                        if t:
                            term *= powers[(u, v)][t]
                            denom *= factorials[t]
                val += factorials[max_k] / denom * term
            piM[i_idx, j_idx] = norm_i * norm_j * val
            if numeric:
                piM[i_idx, j_idx] = sp.N(piM[i_idx, j_idx])

    return piM