import sympy as sp
import csv


# save
def encode(expr):
    if expr.is_Symbol:
        return str(expr)

    if expr.is_Number:
        return str(expr)

    if expr.is_Add:
        return "(" + "+".join(encode(a) for a in expr.args) + ")"

    if expr.is_Mul:
        return "(" + "*".join(encode(a) for a in expr.args) + ")"

    if expr.is_Pow:
        base, exp = expr.args
        # sqrt
        if exp == sp.Rational(1, 2):
            return f"k({encode(base)})"
        # 1/x
        if exp == -1:
            return f"l({encode(base)})"
        return f"({encode(base)}^{encode(exp)})"

    return str(expr)  # fallback


def save_sympy_matrix(piM_sym, filename="matrix.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in piM_sym.tolist():
            writer.writerow([encode(e) for e in row])


###########################################################################################x


# read back
def k(x):
    return sp.sqrt(x)


def l(x):
    return 1 / x


locals_dict = {"k": k, "l": l}


def read_sympy_matrix_csv(filename):
    """ """
    rows = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([sp.sympify(cell, locals=locals_dict) for cell in row])
    return sp.Matrix(rows)



#########################################################################################################
# NEW CODE

import numpy as np
import sympy as sp
import pickle
from scipy import sparse

vars9 = sp.symbols("a b c d e f g h i")

# --- Exponent generator ---
def gen_exponents(k, n=9):
    """Generate all exponent tuples of length n summing to k."""
    def recurse(pos, rem, cur):
        if pos == n - 1:
            yield tuple(cur + [rem])
            return
        for e in range(rem, -1, -1):
            yield from recurse(pos+1, rem-e, cur+[e])
    return list(recurse(0, k, []))

# --- Poly → coeff vector ---
def poly_to_coeff_vector(expr, exps, vars=vars9):
    expr = sp.expand(expr)
    poly = sp.Poly(expr, vars)
    coeffs = np.zeros(len(exps), dtype=float)
    exp_to_index = {exp: i for i, exp in enumerate(exps)}
    for exp_tuple, coeff in poly.as_dict().items():
        idx = exp_to_index.get(exp_tuple)
        if idx is None:
            raise ValueError(f"Monomial {exp_tuple} not in chosen degree basis")
        coeffs[idx] = float(coeff)
    return coeffs

vars9 = sp.symbols("a b c d e f g h i")

def gen_exponents(k, n=9):
    def recurse(pos, rem, cur):
        if pos == n - 1:
            yield tuple(cur + [rem])
            return
        for e in range(rem, -1, -1):
            yield from recurse(pos+1, rem-e, cur+[e])
    return list(recurse(0, k, []))

def save_matrix_direct_sparse(mat_sym, k, filename):
    """
    Converts symbolic matrix directly to sparse storage format without 
    allocating dense intermediate arrays.
    """
    print("Generating exponent basis...")
    exps = gen_exponents(k, 9)
    # Create a hash map for O(1) index lookups
    exp_to_idx = {e: i for i, e in enumerate(exps)}
    
    n = mat_sym.rows
    M = len(exps) # Total size of the polynomial basis
    
    # We still use the object array structure you defined
    T_sparse = np.empty((n, n), dtype=object)
    
    print(f"Processing {n}x{n} matrix entries...")
    
    for i in range(n):
        for j in range(n):
            expr = sp.expand(mat_sym[i, j])
            poly = sp.Poly(expr, vars9)
            poly_dict = poly.as_dict()
            
            # Lists to build the COO matrix directly
            data = []
            cols = []
            
            for exp_tuple, coeff in poly_dict.items():
                if coeff == 0: continue
                
                # Find where this monomial belongs in the basis
                idx = exp_to_idx.get(exp_tuple)
                
                if idx is None:
                    raise ValueError(f"Monomial {exp_tuple} not in basis (degree {k})")
                
                data.append(float(coeff))
                cols.append(idx)
            
            # Since these are row vectors (1 x M), the row index is always 0
            rows = [0] * len(data)
            
            # Create the sparse matrix directly. 
            # If data is empty (polynomial was 0), this creates an empty sparse matrix efficiently.
            T_sparse[i, j] = sparse.coo_matrix(
                (data, (rows, cols)), 
                shape=(1, M)
            )

    print("Saving to disk...")
    # Save using the same architecture as before
    with open(filename + "_T_sparse.pkl", "wb") as f:
        pickle.dump(T_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(filename + "_exps.pkl", "wb") as f:
        pickle.dump(exps, f, protocol=pickle.HIGHEST_PROTOCOL)

# --- Load sparse tensor ---
def load_sparse_coeff_tensor(filename):
    """
    Load sparse coefficient tensor as object array (n x n) and exponent list.
    """
    with open(filename + "_T_sparse.pkl", "rb") as f:
        T_sparse = pickle.load(f)
    with open(filename + "_exps.pkl", "rb") as f:
        exps = pickle.load(f)

    # If loaded as flat list or 1D array, reshape to n x n
    if isinstance(T_sparse, list) or T_sparse.ndim == 1:
        n = int(np.sqrt(len(T_sparse)))
        T_sparse = np.array(T_sparse, dtype=object).reshape(n, n)
    return T_sparse, exps

# --- Evaluate sparse tensor ---
def evaluate_sparse_tensor_matrix_input(T_sparse, exps, A):
    """
    Evaluate sparse coefficient tensor T_sparse with 3x3 matrix A.
    Returns numeric n x n array.
    Works for arbitrary symmetric reps (not just 3x3).
    """
    vals = np.array(A).reshape(-1)   # length 9
    exps_arr = np.array(exps)        # shape (Nmon, 9)
    monoms = np.prod(np.power(vals, exps_arr), axis=1)  # shape (Nmon,)

    n = T_sparse.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Each entry is a 1xNmon sparse row
            res = T_sparse[i,j].dot(monoms)
            # dot() may return a numpy scalar, 1D array of length 1, or 1x1 matrix.
            # Handle all cases robustly and convert to float.
            try:
                # Try to index common shapes
                if hasattr(res, 'shape'):
                    if res.shape == ():  # scalar
                        result[i,j] = float(res)
                    elif res.shape == (1,):
                        result[i,j] = float(res[0])
                    elif res.shape == (1,1):
                        result[i,j] = float(res[0,0])
                    else:
                        # Fallback: attempt to flatten and take first element
                        result[i,j] = float(np.asarray(res).ravel()[0])
                else:
                    result[i,j] = float(res)
            except Exception:
                # As a last resort, coerce to array then take 0th element
                arr = np.asarray(res)
                if arr.size == 0:
                    result[i,j] = 0.0
                else:
                    result[i,j] = float(arr.ravel()[0])
    return result
