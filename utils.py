import numpy as np

# ---------- KERNEL FUNCTION ---------- #
class KernelFunction:
  """
  K(x_l, x_k) = exp{ -1/2 * sum_j [ ((x_lj - x_kj)^2) / s_j^2 ] }
  """
  def __init__(self, s):
    """
    Params
    ----------
    s : array-like de shape (p,)
    """
    self.s = np.asarray(s, dtype=float)
  
  def __call__(self, x_l, x_k):
    """
    Kernel between two vectors.
    """

    x_l = np.asarray(x_l, dtype=float)
    x_k = np.asarray(x_k, dtype=float)
    
    diff = x_l - x_k
    
    weighted_sq = np.sum((diff ** 2) / (self.s ** 2))
    
    return np.exp(-0.5 * weighted_sq)
  
  def matrix(self, X):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
      for j in range(n_samples):
        K[i, j] = self(X[i], X[j])

    return K


# ---------- WIDTH PARAMS CLASS ---------- #

class KernelWidthParameters:
  def __init__(self, kernel, s_sums_history):
    """"
    Eq. (14)
    """
    self.kernel, self.s_sums_history = kernel, np.asarray(s_sums_history, dtype=float)

  def compute(self, X, G, U, m, threshold=1e-10):
    """
    s_j width

    Params:
    X : (n, p)
        input data
    G : (c, p)
        centroids
    U : (n, c)
        pertinence matrix u_ki
    m : float
        fuzzy exponent
    """
    n, p = X.shape
    c = G.shape[0]

    K_matrix = np.zeros((n, c))

    for k in range(n):
      for i in range(c):
        K_matrix[k, i] = self.kernel(X[k], G[i])

    denominator_terms = np.zeros(p)

    for j in range(p):
      denom = 0.0

      for i in range(c):
        for k in range(n):
          weight = (U[k, i] ** m) * K_matrix[k, i]
          denom += weight * (X[k, j] - G[i, j]) ** 2

      denominator_terms[j] = denom if denom > threshold else max(denom, self.s_sums_history[-1, j])

    # product of all terms
    product_term = 1.0

    for h in range(p):
      product_term *= denominator_terms[h]

    numerator_root = product_term ** (1.0 / p)

    # final formula: 1/s_j^2 = (numerador_root / denominador_j)
    inv_s_sq = numerator_root / denominator_terms
    s = 1.0 / np.sqrt(inv_s_sq)
    
    return (s, denominator_terms)


# ---------- compute fuzzy clusters prototypes ---------- #

class FuzzyClusterPrototypes:
  """
  Eq. (15a)
  """

  def __init__(self, kernel):
    self.kernel = kernel

  def compute(self, X, U, m, G_init):
    """
    update centroids

    params
    X : (n, p)
      input data
    U : shape (n, c)
      pertinence matrix u_ki
    m :
      fuzzy exponent
    G_init : (c, p)
      current prototypes

    Returns new prototypes
    """
    n, p = X.shape
    c = U.shape[1]
    G_new = np.zeros_like(G_init)

    for i in range(c):
      numer = np.zeros(p)
      denom = 0.0

      for k in range(n):
        w = (U[k, i] ** m) * self.kernel(X[k], G_init[i])
        numer += w * X[k]
        denom += w

      if denom > 1e-12: # small number to avoid 0 divisions
        G_new[i] = numer / denom
      else:
        G_new[i] = G_init[i]

    return G_new


# ---------- compute membership degrees ---------- #

class FuzzyMemberships:
  """
  Eq. (16a).
  """

  def __init__(self, kernel):
    self.kernel = kernel

  def compute(self, X, G, m):
    """
    updates membership degrees u_ki

    params
    ----------
    X : (n, p)
      input data
    G : (c, p)
      fuzzy prototypes
    m : 
      fuzzy exponent

    returns new matrix
    """
    n, _ = X.shape
    c = G.shape[0]
    U_new = np.zeros((n, c))

    Kxg = np.zeros((n, c))

    for k in range(n):
      for i in range(c):
        Kxg[k, i] = self.kernel(X[k], G[i])

    # each u_ki
    for k in range(n):
      for i in range(c):
        summation = 0.0

        if np.array_equal(G[i], X[k]):
          U_new[k] = np.eye(c, dtype=float)[i]
          break
        
        for h in range(c):
          if np.array_equal(G[h], X[k]): summation += 0
          else: summation += ((2 - 2 * Kxg[k, i])/(2 - 2 * Kxg[k, h])) ** (1/(m-1))

        U_new[k, i] = 1.0 / summation

    return U_new
    

# ---------- COST FUNCTION ---------- #

class CostFunction:
  """
  Eq. 11
  """

  def __init__(self, kernel):
    self.kernel = kernel

  def compute(self, X, U, G, m):
    n, c = U.shape
    J = 0.0

    for i in range(c):
      for k in range(n):
        J += (U[k, i] ** m) * (2 - 2 * self.kernel(X[k], G[i]))
    
    return J


# ---- this is just an example of how to use the functions and to know if the cost function j decreased ---- #
# ----- so you can just ignore this part ----- #

# usage example
# np.random.seed(0)

# # ex data
# X = np.random.rand(5, 3)
# U = np.random.rand(5, 2)
# U = U / U.sum(axis=1, keepdims=True)
# m = 2.0
# G_init = np.random.rand(2, 3)

# s = np.ones(X.shape[1])
# kernel = KernelFunction(s)

# width_calc = KernelWidthParameters(kernel)
# s_new = width_calc.compute(X, G_init, U, m)
# print("new s: ", s_new)

# new_kernel = KernelFunction(s_new)

# proto_calc = FuzzyClusterPrototypes(new_kernel)
# G_new = proto_calc.compute(X, U, m, G_init)

# print("\old prototyoes (G_init):\n", G_init)
# print("new prototypes (G_new):\n", G_new)

# J_old = CostFunction(kernel).compute(X, U, G_init, m)
# J_new = CostFunction(new_kernel).compute(X, U, G_new, m)

# print("J old: ", J_old)
# print("J new: ", J_new)
# print("did it get better? ", J_new < J_old)
