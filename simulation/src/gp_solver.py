import cvxpy as cp
import numpy as np


def _require_params(params: dict, keys):
    missing = [key for key in keys if key not in params]
    if missing:
        raise KeyError(f"Missing required parameters: {missing}")


def _resolve_profile_spec(N: int, l_profile):
    if callable(l_profile):
        profile = np.asarray(l_profile(N), dtype=float)
    elif isinstance(l_profile, dict):
        if N not in l_profile:
            raise KeyError(f"No wire profile provided for N={N}")
        profile = np.asarray(l_profile[N], dtype=float)
    else:
        profile = np.asarray(l_profile, dtype=float)

    if profile.shape != (N,):
        raise ValueError(f"Expected wire profile of shape ({N},), got {profile.shape}")
    if np.any(profile <= 0):
        raise ValueError("Wire profile entries must be strictly positive")
    return profile


def solve_full_gp_fixed_N(
    N: int,
    params: dict,
    l_profile,
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Solve the wire-aware geometric program for a fixed number of stages.

    The only decision variables are the gate widths W_1, ..., W_N.
    The wire lengths l_i are treated as fixed experiment inputs.
    """
    _require_params(
        params,
        ["R0", "r", "c", "Cg0", "Cp0", "CL", "Tclk", "Wmin", "Wmax", "A"],
    )

    profile = _resolve_profile_spec(N, l_profile)

    R0 = float(params["R0"])
    r = float(params["r"])
    c = float(params["c"])
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
        obj_terms.append(Cg0 * W[i + 1] + Cp0 * W[i] + c * profile[i])
    obj_terms.append(CL + Cp0 * W[N - 1] + c * profile[N - 1])
    objective = cp.Minimize(A * cp.sum(obj_terms))

    timing_terms = []
    for i in range(N - 1):
        Wi = W[i]
        Wip1 = W[i + 1]
        li = profile[i]
        timing_terms += [
            R0 * Cg0 * (Wip1 / Wi),
            R0 * Cp0,
            R0 * c * (li / Wi),
            r * Cg0 * (li * Wip1),
            r * Cp0 * (li * Wi),
            r * c * (li**2),
        ]

    Wi = W[N - 1]
    li = profile[N - 1]
    timing_terms += [
        R0 * CL * (1.0 / Wi),
        R0 * Cp0,
        R0 * c * (li / Wi),
        r * CL * li,
        r * Cp0 * (li * Wi),
        r * c * (li**2),
    ]
    timing = cp.sum(timing_terms)

    constraints = [timing / Tclk <= 1]
    for i in range(N):
        constraints += [Wmin / W[i] <= 1, W[i] / Wmax <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(gp=True, solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            "Solve failed. "
            f"status={problem.status}. "
            f"Try a looser Tclk or a smaller total wire length. "
            f"N={N}, Tclk={Tclk}, "
            f"min_l={profile.min():.4f}, max_l={profile.max():.4f}"
        )

    return {
        "status": problem.status,
        "obj": float(problem.value),
        "W": np.asarray(W.value, dtype=float),
        "timing": float(timing.value),
        "timing_ratio": float(timing.value) / Tclk,
        "N": int(N),
        "l_profile": profile.copy(),
        "model": "full_wire_aware_gp",
    }


def solve_full_gp_over_N(
    N_list,
    params: dict,
    l_profile,
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Solve the full GP across multiple stage counts.

    The l_profile argument may be:
    - a callable that accepts N and returns an array
    - a dict keyed by N
    - a fixed array for a single-N call
    """
    results = []
    for N in N_list:
        result = solve_full_gp_fixed_N(
            N,
            params,
            l_profile,
            solver=solver,
            verbose=verbose,
        )
        results.append(result)

    best = min(results, key=lambda item: item["obj"])
    return best, results
