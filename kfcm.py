# // KFCM-K-W-1 algorithm // #
from utils import KernelFunction, KernelWidthParameters, FuzzyClusterPrototypes, FuzzyMemberships, CostFunction
import numpy as np

class KFCM_K_W_1:
  def __init__(self, n_clusters, m=2.0, max_iter=100, tol=1e-6, random_state=None):
    self.c = n_clusters
    self.m = m
    self.max_iter = max_iter
    self.tol = tol
    
    if random_state is not None:
      self.rng = np.random.default_rng(random_state)

    self.s = None
    self.kernel = None
    self.G = None
    self.U = None
    self.cost_history = []

  def _initialize(self, X):
    n, p = X.shape
    
    # kernel width parameters initiate in 1
    # line 7. set 1/(S_j^2) -> 1 (1 <= j <= p):
    self.s = np.ones(p)
    self.kernel = KernelFunction(self.s)

    # random prototypes selection c
    idx = self.rng.choice(n, self.c, replace=False)
    self.G = X[idx, :]

    # compute the membership degree u_ki using Equation (16a)
    membership_calc = FuzzyMemberships(self.kernel)
    self.U = membership_calc.compute(X, self.G, self.m)

  def fit(self, X):
    self._initialize(X)

    width_calc = KernelWidthParameters(self.kernel)
    proto_calc = FuzzyClusterPrototypes(self.kernel)
    membership_calc = FuzzyMemberships(self.kernel)
    cost_calc = CostFunction(self.kernel)

    J_old = cost_calc.compute(X, self.U, self.G, self.m)
    self.cost_history.append(J_old)

    print(f"[Init] J = {J_old:.6f}")

    for t in range(self.max_iter):
      # Step 1: computation of the width parameters - eq. 14a
      self.s = width_calc.compute(X, self.G, self.U, self.m)
      self.kernel = KernelFunction(self.s)

      # Step 2: Computation of the fuzzy cluster prototypes - eq. 15a
      proto_calc.kernel = self.kernel
      self.G = proto_calc.compute(X, self.U, self.m, self.G)

      # Step 3: Computation of the membership degrees - eq. 16a
      membership_calc.kernel = self.kernel
      self.U = membership_calc.compute(X, self.G, self.m)

      # Compute J_new from equation (11)
      cost_calc.kernel = self.kernel
      J_new = cost_calc.compute(X, self.U, self.G, self.m)
      self.cost_history.append(J_new)

      print(f"[Iter {t+1}] J_old={J_old:.6f}  J_new={J_new:.6f}  Δ={abs(J_new - J_old):.6e}")

      # stop criterion
      if abs(J_new - J_old) < self.tol:
        print(f"convergence in {t+1} iterations (ΔJ < {self.tol})")
        break

      J_old = J_new

    return self

  def predict(self, X):
    membership_calc = FuzzyMemberships(self.kernel)
    U_new = membership_calc.compute(X, self.G, self.m)
    
    # crisp partition
    return np.argmax(U_new, axis=1)


  def get_cost_history(self):
    return np.array(self.cost_history)

# execution example:
# np.random.seed(24)
# X = np.random.rand(8, 3)
# X_test = np.random.rand(4, 3)

# model = KFCM_K_W_1(n_clusters=3, m=2.0, max_iter=50, tol=1e-6, random_state=24)
# model.fit(X)

# print("pred", model.predict(X_test))

# print("\nFinal widths (s):", model.s)
# print("Final prototypes (G):\n", model.G)
# print("Final membership matrix (U):\n", model.U)
# print("Cost history:\n", model.get_cost_history())
