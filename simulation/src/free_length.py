import cvxpy as cp
import numpy as np


def _require_params(params: dict, keys):
    missing = [key for key in keys if key not in params]
    if missing:
        raise KeyError(f"Missing required parameters: {missing}")


def solve_free_length_primal_fixed_N(
    N: int,
    params: dict,
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Solve the unrestricted primal GP where both W_i and l_i are optimized.
    """
    _require_params(
        params,
        [
            "R0",
            "r",
            "c",
            "Cg0",
            "Cp0",
            "CL",
            "Tclk",
            "Wmin",
            "Wmax",
            "eps",
            "lmax",
            "A",
        ],
    )

    R0 = float(params["R0"])
    r = float(params["r"])
    c = float(params["c"])
    Cg0 = float(params["Cg0"])
    Cp0 = float(params["Cp0"])
    CL = float(params["CL"])
    Tclk = float(params["Tclk"])
    Wmin = float(params["Wmin"])
    Wmax = float(params["Wmax"])
    eps = float(params["eps"])
    lmax = float(params["lmax"])
    A = float(params["A"])

    W = cp.Variable(N, pos=True)
    L = cp.Variable(N, pos=True)

    obj_terms = []
    for i in range(N - 1):
        obj_terms.append(Cg0 * W[i + 1] + Cp0 * W[i] + c * L[i])
    obj_terms.append(CL + Cp0 * W[N - 1] + c * L[N - 1])
    unscaled_objective = cp.sum(obj_terms)
    objective = cp.Minimize(A * unscaled_objective)

    timing_terms = []
    for i in range(N - 1):
        Wi = W[i]
        Wip1 = W[i + 1]
        Li = L[i]
        timing_terms += [
            R0 * Cg0 * (Wip1 / Wi),
            R0 * Cp0,
            R0 * c * (Li / Wi),
            r * Cg0 * (Li * Wip1),
            r * Cp0 * (Li * Wi),
            r * c * (Li**2),
        ]

    Wi = W[N - 1]
    Li = L[N - 1]
    timing_terms += [
        R0 * CL * (1.0 / Wi),
        R0 * Cp0,
        R0 * c * (Li / Wi),
        r * CL * Li,
        r * Cp0 * (Li * Wi),
        r * c * (Li**2),
    ]
    timing = cp.sum(timing_terms)

    constraints = [timing / Tclk <= 1]
    for i in range(N):
        constraints += [
            Wmin / W[i] <= 1,
            W[i] / Wmax <= 1,
            eps / L[i] <= 1,
            L[i] / lmax <= 1,
        ]

    problem = cp.Problem(objective, constraints)
    problem.solve(gp=True, solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            "Free-length primal solve failed. "
            f"status={problem.status}, N={N}, Tclk={Tclk}"
        )

    unscaled_value = float(unscaled_objective.value)
    return {
        "status": problem.status,
        "obj": float(problem.value),
        "obj_unscaled": unscaled_value,
        "log_obj_unscaled": float(np.log(unscaled_value)),
        "W": np.asarray(W.value, dtype=float),
        "L": np.asarray(L.value, dtype=float),
        "timing": float(timing.value),
        "timing_ratio": float(timing.value) / Tclk,
        "N": int(N),
        "model": "free_length_primal",
    }


def solve_free_length_dual_fixed_N(
    N: int,
    params: dict,
    solver=cp.CLARABEL,
    verbose: bool = False,
):
    """
    Solve the dual associated with the unrestricted free-length formulation.

    The objective is implemented in the log-domain, matching the derivation in
    dual.pdf. To compare with the primal GP objective, use log_obj_unscaled.
    """
    _require_params(
        params,
        [
            "R0",
            "r",
            "c",
            "Cg0",
            "Cp0",
            "CL",
            "Tclk",
            "Wmin",
            "Wmax",
            "eps",
            "lmax",
            "A",
        ],
    )

    R0 = float(params["R0"])
    r = float(params["r"])
    c = float(params["c"])
    Cg0 = float(params["Cg0"])
    Cp0 = float(params["Cp0"])
    CL = float(params["CL"])
    Tclk = float(params["Tclk"])
    Wmin = float(params["Wmin"])
    Wmax = float(params["Wmax"])
    eps = float(params["eps"])
    lmax = float(params["lmax"])
    A = float(params["A"])

    if N < 2:
        raise ValueError("The free-length dual is implemented for N >= 2")

    rho0 = cp.Variable(nonneg=True)
    theta0 = cp.Variable(nonneg=True)
    theta_tail = cp.Variable(N - 1, nonneg=True)
    phi = cp.Variable(N, nonneg=True)
    eta = cp.Variable((N, 6), nonneg=True)
    lam = cp.Variable(nonneg=True)
    alpha = cp.Variable(N, nonneg=True)
    beta = cp.Variable(N, nonneg=True)
    gamma = cp.Variable(N, nonneg=True)
    delta = cp.Variable(N, nonneg=True)

    constraints = [
        rho0 + theta0 + cp.sum(theta_tail) + cp.sum(phi) == 1,
        cp.sum(eta) == lam,
    ]

    constraints.append(
        theta0 - eta[0, 0] - eta[0, 2] + eta[0, 4] - alpha[0] + beta[0] == 0
    )
    for i in range(1, N - 1):
        constraints.append(
            theta_tail[i - 1]
            + eta[i - 1, 0]
            + eta[i - 1, 3]
            - eta[i, 0]
            - eta[i, 2]
            + eta[i, 4]
            - alpha[i]
            + beta[i]
            == 0
        )
    constraints.append(
        theta_tail[N - 2]
        + eta[N - 2, 0]
        + eta[N - 2, 3]
        - eta[N - 1, 0]
        - eta[N - 1, 2]
        + eta[N - 1, 4]
        - alpha[N - 1]
        + beta[N - 1]
        == 0
    )
    for i in range(N):
        constraints.append(
            phi[i] + eta[i, 2] + eta[i, 3] + eta[i, 4] + 2 * eta[i, 5] - gamma[i] + delta[i] == 0
        )

    def _x_log_c_over_x(var, const):
        return cp.entr(var) + var * np.log(const)

    obj_expr = _x_log_c_over_x(rho0, CL)
    obj_expr += _x_log_c_over_x(theta0, Cp0)
    obj_expr += cp.sum(cp.entr(theta_tail) + cp.multiply(theta_tail, np.log(Cg0 + Cp0)))
    obj_expr += cp.sum(cp.entr(phi) + cp.multiply(phi, np.log(c)))

    eta_constants = np.zeros((N, 6), dtype=float)
    if N > 1:
        eta_constants[:-1, :] = np.array(
            [R0 * Cg0, R0 * Cp0, R0 * c, r * Cg0, r * Cp0, r * c],
            dtype=float,
        )
    eta_constants[-1, :] = np.array(
        [R0 * CL, R0 * Cp0, R0 * c, r * CL, r * Cp0, r * c],
        dtype=float,
    )
    obj_expr += cp.sum(cp.multiply(eta, np.log(eta_constants)))

    eta_flat = cp.reshape(eta, (N * 6,), order="C")
    lam_stack = cp.hstack([lam] * (N * 6))
    obj_expr += -cp.sum(cp.rel_entr(eta_flat, lam_stack))

    obj_expr += -lam * np.log(Tclk)
    obj_expr += cp.sum(alpha) * np.log(Wmin)
    obj_expr += -cp.sum(beta) * np.log(Wmax)
    obj_expr += cp.sum(gamma) * np.log(eps)
    obj_expr += -cp.sum(delta) * np.log(lmax)

    problem = cp.Problem(cp.Maximize(obj_expr), constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            "Free-length dual solve failed. "
            f"status={problem.status}, N={N}, Tclk={Tclk}"
        )

    log_obj_unscaled = float(problem.value)
    obj_unscaled = float(np.exp(log_obj_unscaled))
    return {
        "status": problem.status,
        "dual_log_obj_unscaled": log_obj_unscaled,
        "dual_obj_unscaled": obj_unscaled,
        "dual_obj": float(A * obj_unscaled),
        "rho0": float(rho0.value),
        "theta0": float(theta0.value),
        "theta_tail": np.asarray(theta_tail.value, dtype=float),
        "phi": np.asarray(phi.value, dtype=float),
        "eta": np.asarray(eta.value, dtype=float),
        "lambda": float(lam.value),
        "alpha": np.asarray(alpha.value, dtype=float),
        "beta": np.asarray(beta.value, dtype=float),
        "gamma": np.asarray(gamma.value, dtype=float),
        "delta": np.asarray(delta.value, dtype=float),
        "N": int(N),
        "model": "free_length_dual",
    }


def solve_free_length_primal_and_dual_fixed_N(
    N: int,
    params: dict,
    primal_solver=cp.SCS,
    dual_solver=cp.CLARABEL,
    verbose: bool = False,
):
    """
    Solve the unrestricted primal and dual, then compute the duality gap.
    """
    primal = solve_free_length_primal_fixed_N(
        N,
        params,
        solver=primal_solver,
        verbose=verbose,
    )
    dual = solve_free_length_dual_fixed_N(
        N,
        params,
        solver=dual_solver,
        verbose=verbose,
    )

    log_gap = primal["log_obj_unscaled"] - dual["dual_log_obj_unscaled"]
    return {
        "primal": primal,
        "dual": dual,
        "log_gap": float(log_gap),
        "unscaled_gap": float(primal["obj_unscaled"] - dual["dual_obj_unscaled"]),
        "scaled_gap": float(primal["obj"] - dual["dual_obj"]),
    }
