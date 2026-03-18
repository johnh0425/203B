import cvxpy as cp
import numpy as np


def _require_params(params: dict, keys):
    missing = [key for key in keys if key not in params]
    if missing:
        raise KeyError(f"Missing required parameters: {missing}")


def solve_boyd_style_fixed_N(
    N: int,
    params: dict,
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Solve a simplified GP baseline without wire RC effects.

    This baseline is inspired by classical gate-sizing GP formulations and is
    used as a controlled ablation against the wire-aware model.
    """
    _require_params(
        params,
        ["R0", "Cg0", "Cp0", "CL", "Tclk", "Wmin", "Wmax", "A"],
    )

    R0 = float(params["R0"])
    Cg0 = float(params["Cg0"])
    Cp0 = float(params["Cp0"])
    CL = float(params["CL"])
    Tclk = float(params["Tclk"])
    Wmin = float(params["Wmin"])
    Wmax = float(params["Wmax"])
    A = float(params["A"])

    W = cp.Variable(N, pos=True)

    obj_terms = []
    for i in range(N - 1):
        obj_terms.append(Cg0 * W[i + 1] + Cp0 * W[i])
    obj_terms.append(CL + Cp0 * W[N - 1])
    objective = cp.Minimize(A * cp.sum(obj_terms))

    timing_terms = []
    for i in range(N - 1):
        timing_terms += [R0 * Cg0 * (W[i + 1] / W[i]), R0 * Cp0]

    timing_terms += [R0 * CL * (1.0 / W[N - 1]), R0 * Cp0]
    timing = cp.sum(timing_terms)

    constraints = [timing / Tclk <= 1]
    for i in range(N):
        constraints += [Wmin / W[i] <= 1, W[i] / Wmax <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(gp=True, solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solve failed. status={problem.status}")

    return {
        "status": problem.status,
        "obj": float(problem.value),
        "W": np.asarray(W.value, dtype=float),
        "timing": float(timing.value),
        "timing_ratio": float(timing.value) / Tclk,
        "N": int(N),
        "l_profile": None,
        "model": "boyd_style_baseline",
    }


def solve_boyd_style_over_N(
    N_list,
    params: dict,
    solver=cp.SCS,
    verbose: bool = False,
):
    """Solve the simplified baseline across multiple stage counts."""
    results = []
    for N in N_list:
        result = solve_boyd_style_fixed_N(N, params, solver=solver, verbose=verbose)
        results.append(result)

    best = min(results, key=lambda item: item["obj"])
    return best, results

