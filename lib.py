"""This small library contains Quantum Analytic Descent and variants thereof,
as well as helper functions for analysis of MaxCut-QAOA costs.

Functions in this file:
    build_extended_qad_model
    build_qad_model
    generalized_parshift
    trig_interpolation_qad
    parshift_analysis
    get_unique_differences
    graph_generator
    maxcut_hamiltonian
    goemans_williamson_upper_bound
    goemans_williamson_solution
    better_sdp_bound
    max_degree_pair
"""
import copy
from functools import reduce
from itertools import chain, product, combinations, combinations_with_replacement

import pennylane as qml
from pennylane import numpy as np
import cvxpy as cp
import cvxgraphalgs as cvxgr  # For Goemans-Williamson implementation
import networkx as nx

from sinc_autograd import sinc

# The following signs are used to compute the coefficients
# E_D, E_E, E_F, and E_G for the extended QAD model.
signs_coeffs_extended_QAD_model = np.array(
    [
        [1, -1, -1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, 1, 1, 1],
    ]
)

signs_shifts_extended_QAD_model = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])


def build_extended_qad_model(fun, params):
    r"""Computes the coefficients for the extended classical model landscape
    of extended Quantum Analytic Descent, E_A to E_G, and creates a function
    to compute the model that takes the same input parameters as the
    original function.

    Args:
        fun (callable): Function to be modeled.
        params (array[float]): Parameter position at which to build the model.

    Returns:
        callable: The model for the function fun.
    """
    num_params = len(params)
    # Initialize the coefficients of the extended QAD model
    E_A = fun(params)

    # The coefficients E_B will capture the cost function gradient
    E_B = np.zeros(num_params)
    # The coefficients E_C will capture the diagonal of the cost function Hessian
    # All entries contain the term -E_A, which we initialize here already.
    E_C = np.ones(num_params) * (-E_A)

    # The following tensor subsums all coefficients for the bivariate part of the
    # extended model, that is E_D, E_E, E_F and E_G.
    # The first axis determines which of the four coefficients is accessed.
    E_bivariate = np.zeros((4, num_params, num_params))

    # We use the fixed shift amplitude of \pi/2 for the QAD model,
    # as is used in the standard parameter-shift rule for gates
    # with a single frequency.
    shifts = np.eye(num_params) * 0.5 * np.pi

    # Determine the univariate coefficients E_B and E_C
    for k in range(num_params):
        eval_plus = fun(params + shifts[k])
        eval_minus = fun(params - shifts[k])
        E_B[k] = eval_plus - eval_minus
        # Note that the term -E_A already is contained in the initialization
        # of E_C above.
        E_C[k] += eval_plus + eval_minus

    # Determine the bivariate coefficients - we only require their upper right
    # triangular part because they are symmetric
    for k in range(num_params):
        for l in range(k + 1, num_params):
            # Compute the four combinations of shifting the k-th and l-th
            # parameter simultaneously in positive or negative direction.
            shifts_ = signs_shifts_extended_QAD_model @ np.array([shifts[k], shifts[l]])
            shifted_evals = np.array([fun(params + shift) for shift in shifts_])
            # Compute the bivariate coefficients from the shifted
            # evaluations, using hard-coded signs.
            E_bivariate[:, k, l] = signs_coeffs_extended_QAD_model @ shifted_evals

            # In addition to the doubly-shifted evaluations, the bivariate coefficients
            # contain additional contributions from the univariate coefficients and E_A.
            E_bivariate[1, k, l] -= E_B[k]
            E_bivariate[2, k, l] -= E_B[l]
            E_bivariate[3, k, l] -= E_A + E_C[k] + E_C[l]

    def model(par):
        r"""Extended Quantum Analytic Descent energy model.
        Args:
            par (array[float]): Input parameters
        Returns:
            float: Model energy at par
        Comments:
            This model includes all terms that can be
            obtained from the four evaluations used
            for the Hessian.
        """
        A = np.prod(0.5 * (1 + np.cos(par)))
        B_over_A = np.tan(0.5 * par)
        C_over_A = B_over_A ** 2
        terms_without_A = [
            # Constant part
            E_A,
            # Univariate part
            np.dot(E_B, B_over_A),
            np.dot(E_C, C_over_A),
            # Bivariate part
            B_over_A @ E_bivariate[0] @ B_over_A,
            B_over_A @ E_bivariate[1] @ C_over_A,
            C_over_A @ E_bivariate[2] @ B_over_A,
            C_over_A @ E_bivariate[3] @ C_over_A,
        ]
        cost = A * np.sum(terms_without_A)
        return cost

    return model


def generalized_parshift(Omegas, shifts, y_plus, y_minus, y_0=None, order=1):
    r"""Compute any univariate derivative of a trigonometric function using
    shift rules based on a full univariate reconstruction.

    Args:
        Omegas (array[float]): Frequencies present in the Fourier series.
        shifts (array[float]): Parameter shifts that were used for evaluations.
        y_plus (array[float]): Function values at positively shifted parameters.
        y_minus (array[float]): Function values at negatively shifted parameters.
        y_0 (float): Function value at ``center``. Only required if ``order`` is even.
        order (int): Order of the derivative.
    Returns
        float: The derivative of the function that was queried for ``y_plus``, ``y_minus``
        and ``y_0`` of order ``order`` at the center of the queries.
    """
    R = len(Omegas)
    if order % 2:
        # Compute the odd contributions by subtracting the shifted evaluations
        y = (y_plus - y_minus) / 2
        # Construct the coefficient matrix N_{kl} = -\sin(\Omega_l (x_k-center)) for the odd terms
        mat = -np.sin(np.array([[O * s for O in Omegas] for s in shifts]))
        sign = (-1) ** ((order + 1) // 2)
    else:
        if y_0 is None:
            raise ValueError(f"y_0 is required for derivatives of even order.")
        if order == 0:
            return y_0

        y = np.array([y_0, *((y_plus + y_minus) / 2)])
        # Construct the coefficient matrix M_{kl} = -\cos(\Omega_l (x_k-center)) for the even terms
        mat = np.ones((R + 1, R + 1))
        #  All terms except for the first row (k=0, x_k=center) and first column (l=0, \Omega=0)
        mat[1:, 1:] = np.cos(np.array([[O * s for O in Omegas] for s in shifts]))

        sign = (-1) ** (order // 2)

    # Solve the linear equations mat @ coeffs = y
    coeffs = np.linalg.inv(mat) @ y
    derivative = sign * np.dot(coeffs[-R:], np.array(Omegas) ** order)

    return derivative


def parshift_analysis(fun, params, Omegas):
    r"""Compute the gradient and the Hessian using parameter shifts
    for known equidistant frequencies in the original function.

    Args:
        fun (callable): Function to compute the Hessian of.
        params (array[float]): Parameters with respect to which to compute the Hessian.
        Omegas (array[array[float]]): Frequencies in function (per parameter).
            Each entry is expected to take the form
            ``Omegas[i] = [j*Omega_0 for j in range(1, R+1)]``
    Returns:
        float: funtion fun at params
        array[float]: gradient of fun
        array[float]: Hessian of fun
    """
    num_params = len(params)
    Rs = [len(Omega) for Omega in Omegas]
    # Remap the function such that all frequencies are integers, assuming
    # the frequencies to be equidistant already.
    scales = np.array([Omega[0] for Omega in Omegas])
    _fun = lambda par: fun(params + (par - params) / scales)

    # Reference cost function value
    y_0 = _fun(params)

    # Shift vectors
    vecs = np.eye(num_params)
    grad = np.zeros(num_params)
    hessian = np.zeros((num_params, num_params))

    # # # # # # # # Univariate section # # # # # # # #
    # Shift angles for parameter shift gradient
    first_order_shifts = [
        np.array([(2 * i - 1) * np.pi / (2 * R) for i in range(1, R + 1)]) for R in Rs
    ]

    # Compute gradient and Hessian diagonal
    for k, fo_shifts in enumerate(first_order_shifts):
        y_plus = []
        y_minus = []
        for i, shift in enumerate(fo_shifts, start=1):
            y_plus.append(_fun(params + vecs[k] * shift))
            y_minus.append(_fun(params - vecs[k] * shift))
        # First order derivative
        grad[k] = generalized_parshift(
            list(range(1, Rs[k] + 1)),
            fo_shifts,
            np.array(y_plus),
            np.array(y_minus),
            y_0=None,
            order=1,
        )
        # Second order (univariate) derivative
        hessian[k, k] = generalized_parshift(
            list(range(1, Rs[k] + 1)),
            fo_shifts,
            np.array(y_plus),
            np.array(y_minus),
            y_0=y_0,
            order=2,
        )

    # # # # # # # # Bivariate section # # # # # # # #
    idx_pairs_hessian = list(combinations(list(range(num_params)), 2))

    # Use equidistant shifts for bivariate second order derivatives if none are given
    second_order_shifts = {
        (k, m): [i / (Rs[k] + Rs[m]) * np.pi for i in range(1, Rs[k] + Rs[m] + 1)]
        for k, m in idx_pairs_hessian
    }
    for k, m in idx_pairs_hessian:
        y_plus = []
        y_minus = []
        for shift in second_order_shifts[(k, m)]:
            y_plus.append(_fun(params + (vecs[k] + vecs[m]) * shift))
            # Using symmetry of rescaled function: f(x+pi*v_k)=f(x-pi*v_k)
            if np.isclose(shift, np.pi):
                y_minus.append(y_plus[-1])
            else:
                y_minus.append(_fun(params - (vecs[k] + vecs[m]) * shift))
        # Directional derivative across diagonal in (x_k,x_m) plane
        dir_der = generalized_parshift(
            list(range(1, Rs[k] + Rs[m] + 1)),
            second_order_shifts[(k, m)],
            np.array(y_plus),
            np.array(y_minus),
            y_0=y_0,
            order=2,
        )
        # Extract second order (bivariate) derivative from directional derivative
        hessian[k, m] = hessian[m, k] = (dir_der - hessian[k, k] - hessian[m, m]) / 2

    # Rescale gradient and Hessian back to original frequencies
    grad *= scales
    hessian *= np.outer(scales, scales)

    return y_0, grad, hessian


def build_qad_model(fun, params):
    r"""Computes the coefficients for the classical model landscape
    of Quantum Analytic Descent, E_A, E_B, E_C, and E_D, and creates
    a function to compute the model that takes the same input parameters
    as the original function.

    Args:
        fun (callable): Function to be modeled.
        params (array[float]): Parameter position at which to build the model.
    Returns:
        callable: The model for the function fun.

    *Warning:*
        This QAD model follows a different prefactor convention than the QAD paper.
    """
    num_params = len(params)

    # The simple QAD model only works for one frequency per gate, which
    # here is assumed to be 1 (this can always be achieved by rescaling)
    Omegas = np.ones((num_params, 1))

    # Compute the function, its gradient and its Hessian at the reconstruction point.
    # The former two are stored in the model coefficients E_A and E_B.
    E_A, E_B, hessian = parshift_analysis(fun, params, Omegas=Omegas)

    # Extract the upper right triangle and the diagonal of the Hessian as E_C and E_D
    # E_C additionally is shifted by E_A / 2
    E_C = np.diag(hessian) + E_A / 2
    E_D = np.triu(hessian, 1)

    def model(par):
        r"""Quantum Analytic Descent cost function model.
        Args:
            par (array[float]): Input parameters
        Returns:
            float: Model energy at par
        """
        A = np.prod(0.5 * (1 + np.cos(par)))
        B_over_A = 2 * np.tan(0.5 * par)
        C_over_A = B_over_A ** 2 / 2
        D_over_A = np.outer(B_over_A, B_over_A)
        terms_without_A = [
            E_A,
            np.dot(E_B, B_over_A),
            np.dot(E_C, C_over_A),
            np.trace(E_D @ D_over_A),
        ]
        cost = A * np.sum(terms_without_A)
        return cost

    return model


def trig_interpolation_qad(fun, params, R, extended=False):
    r"""Query a function and construct a model function based on trigonometric interpolation.
    Args:
        fun (callable): Original cost function to be modelled.
        params (array[float]): Parameter position at which to model ``fun``.
        R (array[int]): Number of frequencies to assume (per parameter).
        extended (bool): Whether to add the third- and fourth-order
            terms that are available through the (conventional) evaluations for the Hessian.
            Originally, this had no additional cost but due to the cheaper Hessian, it
            now does.
    Returns:
        callable: Model cost function for ``fun``.
    Comments:
        The model function will coincide with ``fun`` on all queried points and match
            all derivatives of ``fun`` (at least) of order ``order`` at ``params``.
        The number of quantum evaluations to construct the model with order :math:`p`
            and :math:`n` parameters scale as :math:`\mathcal{O}(n^p)`.
    """
    num_params = len(params)
    fun_at_params = fun(params)

    N = [(2 * _R + 1) for _R in R]
    vecs = np.eye(num_params)

    evaluations = {tuple(): fun_at_params}
    for k in range(num_params):
        for i in range(1, N[k]):
            evaluations[(k, i)] = fun(params + vecs[k] * i * 2 * np.pi / N[k])

    idx_pairs_hessian = list(combinations(list(range(num_params)), 2))

    f = lambda x, N, l: sinc(N / 2 * x - l * np.pi) / sinc(0.5 * x - l / N * np.pi)
    g = lambda x, N, l: f(x, N, l) * np.cos(0.5 * x - l / N * np.pi)

    for k, m in idx_pairs_hessian:
        for i in range(1, N[k]):
            for j in range(1, N[m]):
                evaluations[(k, m, i, j)] = fun(
                    params + vecs[k] * i * 2 * np.pi / N[k] + vecs[m] * j * 2 * np.pi / N[m],
                )

    def reconstructed_fun(param):
        """Model cost function based on trigonometric interpolation.
        Args:
            param (array[float]): Input parameters for original cost function.
        Returns:
            float: Model cost.
        Comments:
            This function uses 4R[k]R[m] evaluations per Hessian entry and recycles
                these evaluations.
        """
        F = [f(param[k], N[k], np.array(list(range(1, N[k])))) for k in range(num_params)]
        prod0 = np.prod([f(_x, _N, 0) for _x, _N in zip(param, N)])
        out = evaluations[()] * prod0
        for k in range(num_params):
            prod1 = np.prod([f(_x, _N, 0) for _k, (_x, _N) in enumerate(zip(param, N)) if _k != k])
            out = (
                out + np.dot(F[k], np.array([evaluations[(k, i)] for i in range(1, N[k])])) * prod1
            )
        for k, m in idx_pairs_hessian:
            prod2 = np.prod(
                [f(_x, _N, 0) for _k, (_x, _N) in enumerate(zip(param, N)) if _k not in {k, m}]
            )
            evals = np.array(
                [[evaluations[(k, m, i, j)] for j in range(1, N[m])] for i in range(1, N[k])]
            )
            out = out + np.dot(F[k], evals @ F[m]) * prod2

        return out

    return reconstructed_fun


def get_unique_differences(omegas, prec=1e-10):
    r"""Find unique pairwise differences of the input iterable.

    Args:
        omegas (array[float]): Values to take differences between.
        prec (float): Precision with which to distinguish float values.

    Returns:
        array[float]: (Sorted) unique pairwise differences.

    Naturally, for any difference d, -d would be in the set of differences
    as well. Those values are *not* included in the output.
    Also, the value 0 would be in the output for any input, we skip this
    one as well. If `omegas` was sorted in ascending (descending) order,
    all values in the output will be positive (negative). Omegas will be sorted
    in ascending order. The choice of all the above matches the relevant
    transition from the gate generator eigenvalues to the frequencies
    in the cost function for our framework.
    """
    # Make omegas unique (rounding to precision prec)
    omegas = sorted(
        [O * prec for O in set([int(O / prec) for O in np.round(omegas, int(np.log10(1 / prec)))])]
    )
    # Compute pairwise differences
    Omegas = list(
        chain.from_iterable(
            [[o2 - o1 for o1 in omegas[: len(omegas) - i - 1]] for i, o2 in enumerate(omegas[::-1])]
        )
    )
    # Make Omegas unique (rounding to precision prec)
    Omegas = sorted(
        [O * prec for O in set([int(O / prec) for O in np.round(Omegas, int(np.log10(1 / prec)))])]
    )

    return Omegas


def graph_generator(graph_type, max_num_vertices):
    """Create a graph-generating function as well as the
    applicable set of vertex counts for a given graph type."""
    if graph_type.endswith("regular"):
        l = int(graph_type.split("-")[0])
        _nums_wires = list(range(l + 1, max_num_vertices + 1))

        def gen_graph(num_wires):
            try:
                g = nx.random_regular_graph(l, num_wires)
            except nx.NetworkXError:
                print(f"Skipping graph of type {graph_type} for {num_wires} nodes.")
                return None
            return g

    elif graph_type.startswith("complete"):
        _nums_wires = list(range(3, max_num_vertices + 1))

        def gen_graph(num_wires):
            g = nx.complete_graph(num_wires)
            return g

    elif "edges per vertex" == " ".join(graph_type.split()[1:4]):
        edge_per_vertex = int(graph_type.split()[0])
        # The following excludes complete graphs
        _nums_wires = list(range(3, max_num_vertices + 1))

        def gen_graph(num_wires):
            num_edges = edge_per_vertex * num_wires
            g = nx.dense_gnm_random_graph(num_wires, num_edges)
            return g

    return gen_graph, _nums_wires


def maxcut_hamiltonian(graph):
    """Generate the Hamiltonian diagonal for a given unweighted graph."""
    I = [1, 1]
    Z = [1, -1]
    diag = 0.0
    num_wires = len(graph.nodes)
    for (a, b) in graph.edges():
        a, b = min((a, b)), max((a, b))
        local_diags = [I] * a + [Z] + [I] * (b - a - 1) + [Z] + [I] * (num_wires - b - 1)
        diag += 0.5 * (1 - reduce(np.kron, local_diags))

    return diag


def goemans_williamson_upper_bound(graph):
    """Upper bound for MaxCut(G) of a graph G via the (unrounded) Goemans-Williamson algorithm.

    Args:
        graph (nx.Graph): Graph to bound.
    Returns:
        float: The unrounded result of the GW algorithm.
    """
    adjacency = nx.linalg.adjacency_matrix(graph).toarray()
    size = len(adjacency)
    ones_matrix = np.ones((size, size))
    products = cp.Variable((size, size), PSD=True)
    cut_size = 0.25 * cp.sum(cp.multiply(adjacency, ones_matrix - products))

    objective = cp.Maximize(cut_size)
    constraints = [cp.diag(products) == 1]
    problem = cp.Problem(objective, constraints)
    sol = problem.solve()

    return problem.solution.opt_val


def goemans_williamson_solution(graph, number=1):
    """Approximately solve MaxCut using the (rounded) Goemans-Williamson algorithm.

    Args:
        graph (nx.Graph): Graph to solve.
        number (int): Number of randomized roundings over which to maximize.
    Returns:
        int: The best of the obtained values via rounding.
    Comments:
        The semi-definite program also is solved ``number`` times.
    """
    values = []
    for _ in range(number):
        sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(graph)
        values.append(sdp_cut.evaluate_cut_size(graph))
    return max(values)


def better_sdp_bound(graph):
    """An improved upper bound to MaxCut(G) based on the Goemans-Williamson algorithm.

    Based on https://link.springer.com/content/pdf/10.1023/A:1014895808844.pdf
    """
    if len(graph.nodes) > 20:
        raise ValueError()
    adjacency = nx.linalg.adjacency_matrix(graph).toarray()
    size1 = len(adjacency)
    size2 = int(size1 * (size1 + 1) / 2 + 1)
    # The matrix to multiply and trace with Q
    sub_Y = cp.Variable((size1, size1), PSD=True)
    Y = cp.Variable((size2, size2), PSD=True)
    Q = 1 / 4 * (np.diag(adjacency @ np.ones(size1)) - adjacency)

    # Objective function
    cut_size = cp.trace(Q @ sub_Y)
    objective = cp.Maximize(cut_size)

    # Indexing
    T_fn = lambda i, j: j * (j - 1) / 2 + i if i <= j else i * (i - 1) / 2 + j
    S_fn = lambda i, j: j * (j + 1) / 2 + i if i <= j else i * (i + 1) / 2 + j
    quartic_indices = []
    for i, j in combinations(list(range(1, size1)), r=2):
        quartic_indices += [(T_fn(i, j), T_fn(i, k), T_fn(k, j)) for k in range(1, size1)]
    sub_Y_indices = []
    Y_indices = []
    for i, j in combinations_with_replacement(list(range(size1)), r=2):
        sub_Y_indices.append((i, j))
        Y_indices.append((0, 1 + S_fn(i, j)))

    # Constraints
    constraints = [
        # Diagonal of Y is 1.
        cp.diag(Y) == 1,
        # Diagonal of sub_Y is 1.
        cp.diag(sub_Y) == 1,
    ]
    # Constraints based on quartic equalities
    constraints += [Y[0, tup[0]] == Y[tup[1], tup[2]] for tup in quartic_indices]
    # Constraints that sub_Y is a part of Y
    constraints += [sub_Y[ind1] == Y[ind2] for ind1, ind2 in zip(sub_Y_indices, Y_indices)]

    # Solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return objective.value


def max_degree_pair(graph):
    """Find the largest sum of degrees of two vertices that belong to an
    edge in a given graph.

    Args:
        graph (nx.Graph): Graph to analyze.

    Returns:
        int: Largest sum of degrees of any two vertices that are joined by
            an edge in the given graph
    """
    max_max = sum([deg for _, deg in sorted(graph.degree, key=lambda x: x[1], reverse=True)[:2]])
    max_degree_pair = 0
    for e in graph.edges:
        degree_pair = graph.degree[e[0]] + graph.degree[e[1]]
        if degree_pair > max_degree_pair:
            max_degree_pair = degree_pair
            if max_degree_pair == max_max:
                break
    return max_degree_pair
