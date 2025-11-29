#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import functools

# -------------------- Logging Decorator --------------------
def log_execution(func):
    """
    Decorator that executes the function, then writes the
    input 'x' and output 'value' to 'dumpfile.dat'.
    Format: x1,x2,...,xn | value
    """
    @functools.wraps(func)
    def wrapper(x):
        # 1. Execute the actual mathematical function
        # CORRECTION: The defined functions return a scalar 'value', not a tuple.
        # We capture the value here and pair it with x manually.
        result_val = func(x)
        result_x = x

        # 2. Prepare data for logging
        # Ensure x is flat and converted to string
        if hasattr(result_x, "flatten"):
            x_flat = result_x.flatten()
        elif isinstance(result_x, list):
            x_flat = np.array(result_x).flatten()
        else:
            x_flat = np.array([result_x])
            
        x_str = ",".join(map(str, x_flat))

        # 3. Write to dumpfile.dat
        with open("dumpfile.dat", "a") as f:
            f.write(f"{x_str} | {result_val}\n")

        # 4. Return the result as a tuple so MCTS/Plotter keeps working
        return result_x, result_val
    return wrapper


# -------------------- Plotting helper (optional) --------------------
def plot_3d_function(func, lb, ub, resolution=100):
    """
    Plot a 2D function as a 3D surface.

    Assumes `func(x)` returns (x, f(x)).
    `lb`, `ub` are 1D arrays of lower/upper bounds.
    """
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            # The decorator ensures this returns a tuple (x, val)
            _, Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    plt.tight_layout()
    plt.show()


# ====================================================================
# F1 – F23 benchmark functions
# ====================================================================

@log_execution
def sphere(x):
    """
    F1 – Sphere Function
    f(x) = sum(x_i^2)
    - Bounds: [-100, 100]
    - Dimension: any (here: 2 for plotting)
    - Global Minimum: f(x) = 0 at x = 0
    """
    value = np.sum(x**2)
    return value


@log_execution
def schwefel_2_22(x):
    """
    F2 – Schwefel 2.22 Function
    f(x) = sum(|x_i|) + prod(|x_i|)
    - Bounds: [-10, 10]
    - Global Minimum: f(x) = 0 at x = 0
    """
    x = np.asarray(x)
    value = np.sum(np.abs(x)) + np.prod(np.abs(x))
    return value


@log_execution
def schwefel_1_2(x):
    """
    F3 – Schwefel 1.2 Function
    f(x) = sum_{i=1}^{n} (sum_{j=1}^{i} x_j)^2
    - Bounds: [-100, 100]
    - Global Minimum: f(x) = 0 at x = 0
    """
    value = np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])
    return value


@log_execution
def max_value(x):
    """
    F4 – Maximum Value Function
    f(x) = max(|x_i|)
    - Bounds: [-100, 100]
    - Global Minimum: f(x) = 0 at x = 0
    """
    value = np.max(np.abs(x))
    return value


@log_execution
def rosenbrock(x):
    """
    F5 – Rosenbrock Function
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    - Bounds: [-30, 30]
    - Global Minimum: f(x) = 0 at x = [1, 1, ..., 1]
    """
    value = np.sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1)])
    return value


@log_execution
def step_function(x):
    """
    F6 – Step Function
    f(x) = sum(floor(x_i + 0.5)^2)
    - Bounds: [-100, 100]
    - Global Minimum: f(x) = 0 at x = ( -0.5 ≤ x_i < 0.5 ) (or at x_i = 0 in the alternative form)
    """
    value = np.sum(np.abs(x + 0.5)**2)
    return value


@log_execution
def quartic_noise(x):
    """
    F7 – Quartic Function with Noise
    f(x) = sum(i * x_i^4) + random[0, 1)
    - Bounds: [-1.28, 1.28]
    - Global Minimum: f(x) ≈ 0 at x = 0
    """
    x = np.asarray(x)  # Handle lists/arrays
    indices = np.arange(1, len(x) + 1)  # 1-based indexing
    value = np.sum(indices * x**4) + np.random.uniform(0, 1)
    return value


@log_execution
def schwefel(x):
    """
    F8 – Schwefel Function
    f(x) = sum(-x_i * sin(sqrt(|x_i|)))
    - Bounds: [-500, 500]
    - Global Minimum: f(x) ≈ -418.9829 * n at x_i = 420.9687
    """
    value = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    return value


@log_execution
def rastrigin(x):
    """
    F9 – Rastrigin Function
    f(x) = sum(x_i^2 - 10 * cos(2πx_i) + 10)
    - Bounds: [-5.12, 5.12]
    - Global Minimum: f(x) = 0 at x = 0
    """
    value = np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
    return value


@log_execution
def ackley(x):
    """
    F10 – Ackley Function
    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(x_i^2))) 
           - exp(1/n * sum(cos(2πx_i))) + 20 + e
    - Bounds: [-32, 32]
    - Global Minimum: f(x) = 0 at x = 0
    """
    x = np.asarray(x)
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n) + np.e
    value = term1 + term2 + 20
    return value


@log_execution
def griewank(x):
    """
    F11 – Griewank Function
    f(x) = 1/4000 * sum(x_i^2) - prod(cos(x_i / sqrt(i))) + 1
    - Bounds: [-600, 600]
    - Global Minimum: f(x) = 0 at x = 0
    """
    sum_sq = np.sum(x**2)
    prod_cos = np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x))])
    
    value = sum_sq / 4000 - prod_cos + 1
    return value


@log_execution
def penalized1(x):
    """
    F12 – Penalized Function 1
    f(x) = π/n * {10*sin²(π*y1) + sum((y_i - 1)^2 * [1 + 10sin²(π*y_{i+1})]) + (y_n - 1)^2}
           + sum(u(x_i, 10, 100, 4))
    where y_i = 1 + (x_i + 1)/4
          u(x_i, a, k, m) is a penalty function
    - Bounds: [-50, 50]
    - Global Minimum: f(x) = 0
    """
    def u(xi, a=10, k=100, m=4):
        if xi > a:
            return k * (xi - a)**m
        elif xi < -a:
            return k * (-xi - a)**m
        else:
            return 0

    y = 1 + (x + 1) / 4
    term1 = 10 * np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2
    penalty = np.sum([u(xi) for xi in x])
    value = (np.pi / len(x)) * (term1 + term2 + term3) + penalty
    return value


@log_execution
def penalized2(x):
    """
    F13 – Penalized Function 
    f(x) = 0.1 * [sin²(3πx_1) + sum((x_i - 1)^2 * (1 + sin²(3πx_{i+1}))) 
           + (x_n - 1)^2 * (1 + sin²(2πx_n))] + sum(u(x_i, 5, 100, 4))
    - Bounds: [-50, 50]
    - Global Minimum: f(x) = 0
    """
    def u(xi, a=5, k=100, m=4):
        if xi > a:
            return k * (xi - a)**m
        elif xi < -a:
            return k * (-xi - a)**m
        else:
            return 0

    term1 = np.sin(3 * np.pi * x[0])**2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    penalty = np.sum([u(xi) for xi in x])

    value = 0.1 * (term1 + term2 + term3) + penalty
    return value


@log_execution
def shekel_foxholes(x):
    """
    F14 – Shekel’s Foxholes (De Jong’s Function No. 5)
    f(x) = [0.002 + sum_{i=1}^{25} 1 / (i + (x1 - a1_i)^6 + (x2 - a2_i)^6)]^(-1)
    - Bounds: [-65, 65]
    - Dimension: 2
    - Global Minimum: f(x) ≈ 1 at x = [-32, -32]
    """
    A = [-32., -16., 0., 16., 32.]
    a1 = np.array(A * 5)
    a2 = np.repeat(A, 5)
    val = 0
    for i in range(25):
        val += 1 / (i + 1 + (x[0] - a1[i])**6 + (x[1] - a2[i])**6)
    value = (0.002 + val)**(-1)
    return value


@log_execution
def kowalik(x):
    """
    F15 – Kowalik Function
    f(x) = sum_{i=1}^{11} [a_i - (x1(bi^2 + bi*x2)) / (bi^2 + bi*x3 + x4)]^2
    - Bounds: [-5, 5]
    - Dimension: 4
    - Global Minimum: ≈ 0.000307 at x ≈ [0.1928, 0.1908, 0.1231, 0.1358]
    """
    b = np.array([4.0, 2.0, 1.0, 1/2.0, 1/4.0,
                  1/6.0, 1/8.0, 1/10.0, 1/12.0, 1/14.0, 1/16.0])
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844,
                  0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])

    val = 0.0
    for i in range(11):
        bi = b[i]
        num = x[0] * (bi**2 + bi * x[1])
        denom = bi**2 + bi * x[2] + x[3]
        val += (a[i] - num / denom) ** 2

    value = val
    return value


@log_execution
def six_hump_camel(x):
    """
    F16 – Six-Hump Camel Function
    f(x) = (4 - 2.1*x_1^2 + x_1^4/3)*x_1^2 + x_1*x_2 + (4*x_2^2 - 4)*x_2^2
    - Bounds: [-5, 5]
    - Global Minimum: ≈ -1.0316 at [±0.0898, ∓0.7126]
    """
    value = (4 - 2.1 * x[0]**2 + (x[0]**4) / 3) * x[0]**2 + x[0]*x[1] + (4 * x[1]**2 - 4) * x[1]**2
    return value


@log_execution
def branin(x):
    """
    F17 – Branin Function
    f(x) = (x2 - (5.1 / (4π²)) x1² + (5 / π)x1 - 6)² + 10(1 - 1 / (8π))cos(x1) + 10
    - Bounds: [-5, 5]
    - Dimension: 2
    - Global Minimum: ≈ 0.397887 at x = [-π, 12.275], [π, 2.275], or [9.42478, 2.475]
    """
    value = (x[1] - (5.1 / (4 * np.pi**2)) * x[0]**2 + 5 * x[0] / np.pi - 6)**2 + \
            10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

    return value


@log_execution
def goldstein_price(x):
    """
    F18 – Goldstein–Price Function
    f(x) = [1 + (x1 + x2 + 1)^2 * (19 - 14x1 + 3x1² - 14x2 + 6x1x2 + 3x2²)]
           × [30 + (2x1 - 3x2)^2 * (18 - 32x1 + 12x1² + 48x2 - 36x1x2 + 27x2²)]
    - Bounds: [-2, 2]
    - Dimension: 2
    - Global Minimum: f(x) = 3 at x = [0, -1]
    """
    a = 1 + (x[0] + x[1] + 1)**2 * (
        19 - 14 * x[0] + 3 * x[0]**2 -
        14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2
    )
    b = 30 + (2 * x[0] - 3 * x[1])**2 * (
        18 - 32 * x[0] + 12 * x[0]**2 +
        48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2
    )

    value = a * b
    return value


@log_execution
def hartmann_3d(x):
    """
    F19 – Hartmann 3D Function
    f(x) = -sum_{i=1}^4 α_i * exp(-sum_{j=1}^3 A_ij * (x_j - P_ij)^2)
    - Bounds: [0, 1]
    - Dimension: 3
    - Global Minimum: ≈ -3.86278 at specific x
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10, 30],
        [0.1, 10, 35],
        [3.0, 10, 30],
        [0.1, 10, 35]
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828]
    ])
    
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        outer += alpha[i] * np.exp(-inner)

    value = -outer
    return value


@log_execution
def hartmann_6d(x):
    """
    F20 – Hartmann 6D Function
    f(x) = -sum_{i=1}^4 c_i * exp(-sum_{j=1}^6 a_ij * (x_j - p_ij)^2)
     - Bounds: [0, 1]
     - Dimension: 6
     - Global Minimum: ≈ -3.32
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        outer += alpha[i] * np.exp(-inner)

    value = -outer
    return value


@log_execution
def shekel_5(x):
    """
    F21 – Shekel Function (m = 5)
    f(x) = -sum_{i=1}^5 [1 / (||x - a_i||^2 + c_i)]
    - Bounds: [0, 10]
    - Dimension: 4
    - Global Minimum: ≈ -10.1532 at x_i = 4 for all i
    """
    A = np.array([
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0]
    ])
    C = np.array([0.1, 0.2, 0.2, 0.4, 0.4])

    value = -np.sum([1.0 / (np.sum((x - A[i])**2) + C[i]) for i in range(5)])
    
    return value


@log_execution
def shekel_7(x):
    """
    F22 – Shekel Function (m = 7)
    f(x) = -sum_{i=1}^7 [1 / (||x - a_i||^2 + c_i)]
    - Bounds: [0, 10]
    - Dimension: 4
    - Global Minimum: ≈ -10.4038 at x_i = 4 for all i
    """
    A = np.array([
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0],
        [2.0, 9.0, 2.0, 9.0],
        [5.0, 5.0, 3.0, 3.0]
    ])
    C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])

    value = -np.sum([1.0 / (np.sum((x - A[i])**2) + C[i]) for i in range(7)])
    
    return value


@log_execution
def shekel_10(x):
    """
    F23 – Shekel Function (m = 10)
    f(x) = -sum_{i=1}^{10} [1 / (||x - a_i||^2 + c_i)]
    - Bounds: [0, 10]
    - Dimension: 4
    - Global Minimum: ≈ -10.5363 at x_i = 4 for all i
    """
    A = np.array([
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0],
        [2.0, 9.0, 2.0, 9.0],
        [5.0, 5.0, 3.0, 3.0],
        [8.0, 1.0, 8.0, 1.0],
        [6.0, 2.0, 6.0, 2.0],
        [7.0, 3.6, 7.0, 3.6]
    ])
    C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    value = -np.sum([1.0 / (np.sum((x - A[i])**2) + C[i]) for i in range(10)])
    return value



# ====================================================================
# Aliases and lookup table: F1..F23
# ====================================================================

F1 = sphere
F2 = schwefel_2_22
F3 = schwefel_1_2
F4 = max_value
F5 = rosenbrock
F6 = step_function
F7 = quartic_noise
F8 = schwefel
F9 = rastrigin
F10 = ackley
F11 = griewank
F12 = penalized1
F13 = penalized2
F14 = shekel_foxholes
F15 = kowalik
F16 = six_hump_camel
F17 = branin
F18 = goldstein_price
F19 = hartmann_3d
F20 = hartmann_6d
F21 = shekel_5
F22 = shekel_7
F23 = shekel_10

FUNCTIONS = {
    "F1": F1,
    "F2": F2,
    "F3": F3,
    "F4": F4,
    "F5": F5,
    "F6": F6,
    "F7": F7,
    "F8": F8,
    "F9": F9,
    "F10": F10,
    "F11": F11,
    "F12": F12,
    "F13": F13,
    "F14": F14,
    "F15": F15,
    "F16": F16,
    "F17": F17,
    "F18": F18,
    "F19": F19,
    "F20": F20, 
    "F21": F21,
    "F22": F22,
    "F23": F23,
}


def get_benchmark(name: str):
    """
    Return benchmark function by ID, e.g. get_benchmark("F7").

    Each function has signature: f(x) -> (x, value).
    """
    return FUNCTIONS[name]