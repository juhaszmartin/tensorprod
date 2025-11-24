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
