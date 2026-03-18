import copy
from typing import Optional

import cvxpy as cp
import numpy as np

from src.baselines import solve_boyd_style_fixed_N
from src.free_length import solve_free_length_primal_and_dual_fixed_N
from src.gp_solver import solve_full_gp_fixed_N


def default_params():
    """Return the default technology and experiment parameters."""
    return {
        "R0": 1.085,
        "r": 0.35,
        "c": 0.228,
        "Cg0": 1.34,
        "Cp0": 0.85,
        "CL": 10.0,
        "Tclk": 100.0,
        "Wmin": 0.5,
        "Wmax": 50.0,
        "A": 0.1 * 1.0 * (0.7**2),
    }


def default_n_list():
    return list(range(2, 11))


def default_tclk_list():
    return [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]


def _normalize_weights(weights, total_length: float):
    if total_length <= 0:
        raise ValueError("total_length must be positive")

    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0):
        raise ValueError("All profile weights must be positive")
    return total_length * weights / weights.sum()


def make_uniform_profile(N: int, total_length: float):
    return _normalize_weights(np.ones(N), total_length)


def make_back_loaded_profile(N: int, total_length: float, severity: float = 3.0):
    if severity <= 0:
        raise ValueError("severity must be positive")
    weights = np.linspace(1.0, float(severity), N)
    return _normalize_weights(weights, total_length)


def make_single_hotspot_profile(
    N: int,
    total_length: float,
    hotspot_idx: Optional[int] = None,
    hotspot_scale: float = 4.0,
):
    if hotspot_scale <= 0:
        raise ValueError("hotspot_scale must be positive")
    if hotspot_idx is None:
        hotspot_idx = N // 2
    if hotspot_idx < 0 or hotspot_idx >= N:
        raise ValueError("hotspot_idx is out of range")

    weights = np.ones(N)
    weights[hotspot_idx] = float(hotspot_scale)
    return _normalize_weights(weights, total_length)


def build_profile_suite(
    N: int,
    total_length: float,
    back_loaded_severity: float = 3.0,
    hotspot_idx: Optional[int] = None,
    hotspot_scale: float = 4.0,
):
    return {
        "Uniform": make_uniform_profile(N, total_length),
        "Back-loaded": make_back_loaded_profile(N, total_length, severity=back_loaded_severity),
        "Single-hotspot": make_single_hotspot_profile(
            N,
            total_length,
            hotspot_idx=hotspot_idx,
            hotspot_scale=hotspot_scale,
        ),
    }


def _copy_params(params: Optional[dict]):
    return copy.deepcopy(default_params() if params is None else params)


def _infeasible_result(
    N: int,
    profile_name: Optional[str],
    params: dict,
    error: Exception,
    Tclk: Optional[float] = None,
    l_profile=None,
    model: str = "full_wire_aware_gp",
):
    result = {
        "status": "infeasible",
        "obj": np.nan,
        "W": None,
        "timing": np.nan,
        "timing_ratio": np.nan,
        "N": int(N),
        "l_profile": None if l_profile is None else np.asarray(l_profile, dtype=float).copy(),
        "model": model,
        "error": str(error),
        "params": copy.deepcopy(params),
    }
    if profile_name is not None:
        result["profile_name"] = profile_name
    if Tclk is not None:
        result["Tclk"] = float(Tclk)
    return result


def _solve_full_or_mark_infeasible(
    N: int,
    current_params: dict,
    profile_name: str,
    profile,
    solver,
    verbose: bool,
    Tclk: Optional[float] = None,
):
    try:
        result = solve_full_gp_fixed_N(
            N,
            current_params,
            profile,
            solver=solver,
            verbose=verbose,
        )
        result["profile_name"] = profile_name
        result["params"] = copy.deepcopy(current_params)
        if Tclk is not None:
            result["Tclk"] = float(Tclk)
        return result
    except RuntimeError as error:
        return _infeasible_result(
            N,
            profile_name,
            current_params,
            error,
            Tclk=Tclk,
            l_profile=profile,
            model="full_wire_aware_gp",
        )


def _solve_baseline_or_mark_infeasible(
    N: int,
    current_params: dict,
    solver,
    verbose: bool,
    Tclk: Optional[float] = None,
):
    try:
        result = solve_boyd_style_fixed_N(
            N,
            current_params,
            solver=solver,
            verbose=verbose,
        )
        result["params"] = copy.deepcopy(current_params)
        if Tclk is not None:
            result["Tclk"] = float(Tclk)
        return result
    except RuntimeError as error:
        return _infeasible_result(
            N,
            None,
            current_params,
            error,
            Tclk=Tclk,
            l_profile=None,
            model="boyd_style_baseline",
        )


def _width_activity_metrics(result: dict, wmin: float):
    if result["W"] is None:
        return {
            "max_width_delta": np.nan,
            "mean_width_delta": np.nan,
            "width_span": np.nan,
            "active_width_count": 0,
        }

    deltas = np.asarray(result["W"], dtype=float) - float(wmin)
    return {
        "max_width_delta": float(np.max(deltas)),
        "mean_width_delta": float(np.mean(deltas)),
        "width_span": float(np.max(result["W"]) - np.min(result["W"])),
        "active_width_count": int(np.sum(deltas > 1e-3)),
    }


def run_single_case(
    N: int = 6,
    params: Optional[dict] = None,
    total_length: float = 30.0,
    profile_name: str = "Uniform",
    solver=cp.SCS,
    verbose: bool = False,
):
    current_params = _copy_params(params)
    profiles = build_profile_suite(N, total_length)
    if profile_name not in profiles:
        raise KeyError(f"Unknown profile name: {profile_name}")

    result = solve_full_gp_fixed_N(
        N,
        current_params,
        profiles[profile_name],
        solver=solver,
        verbose=verbose,
    )
    result["profile_name"] = profile_name
    result["params"] = current_params
    return result


def scan_parameter_grid(
    N: int = 6,
    params: Optional[dict] = None,
    total_length_values=None,
    Tclk_values=None,
    CL_values=None,
    profile_name: str = "Single-hotspot",
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Scan a small parameter grid to find a more informative simulation regime.

    The main goal is to identify feasible settings where the optimized widths
    are not all pinned at Wmin.
    """
    base_params = _copy_params(params)
    total_length_values = (
        [20.0, 25.0, 30.0, 35.0, 40.0]
        if total_length_values is None
        else [float(x) for x in total_length_values]
    )
    Tclk_values = (
        [70.0, 80.0, 90.0, 100.0, 110.0]
        if Tclk_values is None
        else [float(x) for x in Tclk_values]
    )
    CL_values = (
        [10.0, 15.0, 20.0, 30.0]
        if CL_values is None
        else [float(x) for x in CL_values]
    )

    records = []
    for total_length in total_length_values:
        profiles = build_profile_suite(N, total_length)
        if profile_name not in profiles:
            raise KeyError(f"Unknown profile name: {profile_name}")

        profile = profiles[profile_name]
        for Tclk in Tclk_values:
            for CL in CL_values:
                current_params = _copy_params(base_params)
                current_params["Tclk"] = float(Tclk)
                current_params["CL"] = float(CL)
                result = _solve_full_or_mark_infeasible(
                    N,
                    current_params,
                    profile_name,
                    profile,
                    solver,
                    verbose,
                    Tclk=float(Tclk),
                )
                record = {
                    "N": int(N),
                    "profile_name": profile_name,
                    "total_length": float(total_length),
                    "Tclk": float(Tclk),
                    "CL": float(CL),
                    "status": result["status"],
                    "obj": result["obj"],
                    "result": result,
                }
                record.update(_width_activity_metrics(result, current_params["Wmin"]))
                records.append(record)
    return records


def rank_informative_cases(
    scan_records,
    min_active_width_count: int = 1,
):
    """
    Rank feasible parameter settings by how strongly they activate width sizing.
    """
    feasible = [
        record
        for record in scan_records
        if record["status"] in ("optimal", "optimal_inaccurate")
    ]
    filtered = [
        record
        for record in feasible
        if record["active_width_count"] >= int(min_active_width_count)
    ]
    ranked = sorted(
        filtered if filtered else feasible,
        key=lambda item: (
            item["active_width_count"],
            item["max_width_delta"],
            item["width_span"],
            -item["Tclk"],
            item["CL"],
        ),
        reverse=True,
    )
    return ranked


def choose_informative_defaults(
    N: int = 6,
    params: Optional[dict] = None,
    total_length_values=None,
    Tclk_values=None,
    CL_values=None,
    profile_name: str = "Single-hotspot",
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Return the highest-ranked candidate from a small grid search.
    """
    records = scan_parameter_grid(
        N=N,
        params=params,
        total_length_values=total_length_values,
        Tclk_values=Tclk_values,
        CL_values=CL_values,
        profile_name=profile_name,
        solver=solver,
        verbose=verbose,
    )
    ranked = rank_informative_cases(records)
    if not ranked:
        raise RuntimeError("No feasible parameter setting was found in the scan grid.")
    return ranked[0], ranked


def scan_common_parameter_grid(
    N: int = 6,
    params: Optional[dict] = None,
    total_length_values=None,
    Tclk_values=None,
    CL_values=None,
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Scan for parameter settings that are feasible across all predefined profiles.
    """
    base_params = _copy_params(params)
    total_length_values = (
        [20.0, 25.0, 30.0, 35.0, 40.0]
        if total_length_values is None
        else [float(x) for x in total_length_values]
    )
    Tclk_values = (
        [70.0, 80.0, 90.0, 100.0, 110.0]
        if Tclk_values is None
        else [float(x) for x in Tclk_values]
    )
    CL_values = (
        [10.0, 15.0, 20.0, 30.0]
        if CL_values is None
        else [float(x) for x in CL_values]
    )

    records = []
    for total_length in total_length_values:
        profiles = build_profile_suite(N, total_length)
        for Tclk in Tclk_values:
            for CL in CL_values:
                current_params = _copy_params(base_params)
                current_params["Tclk"] = float(Tclk)
                current_params["CL"] = float(CL)

                profile_results = {}
                all_feasible = True
                total_active_width_count = 0
                total_max_width_delta = 0.0
                total_width_span = 0.0

                for profile_name, profile in profiles.items():
                    result = _solve_full_or_mark_infeasible(
                        N,
                        current_params,
                        profile_name,
                        profile,
                        solver,
                        verbose,
                        Tclk=float(Tclk),
                    )
                    metrics = _width_activity_metrics(result, current_params["Wmin"])
                    profile_results[profile_name] = {
                        "result": result,
                        **metrics,
                    }
                    if result["status"] not in ("optimal", "optimal_inaccurate"):
                        all_feasible = False
                    total_active_width_count += metrics["active_width_count"]
                    if not np.isnan(metrics["max_width_delta"]):
                        total_max_width_delta += metrics["max_width_delta"]
                    if not np.isnan(metrics["width_span"]):
                        total_width_span += metrics["width_span"]

                records.append(
                    {
                        "N": int(N),
                        "total_length": float(total_length),
                        "Tclk": float(Tclk),
                        "CL": float(CL),
                        "all_feasible": all_feasible,
                        "profile_results": profile_results,
                        "total_active_width_count": int(total_active_width_count),
                        "total_max_width_delta": float(total_max_width_delta),
                        "total_width_span": float(total_width_span),
                    }
                )
    return records


def rank_common_informative_cases(
    scan_records,
    require_all_feasible: bool = True,
):
    """
    Rank settings that work across all profiles and activate nontrivial sizing.
    """
    candidates = scan_records
    if require_all_feasible:
        candidates = [record for record in candidates if record["all_feasible"]]

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["total_active_width_count"],
            item["total_max_width_delta"],
            item["total_width_span"],
            -item["Tclk"],
            item["CL"],
        ),
        reverse=True,
    )
    return ranked


def choose_common_informative_defaults(
    N: int = 6,
    params: Optional[dict] = None,
    total_length_values=None,
    Tclk_values=None,
    CL_values=None,
    solver=cp.SCS,
    verbose: bool = False,
):
    """
    Choose a single parameter setting that is feasible across all three profiles.
    """
    records = scan_common_parameter_grid(
        N=N,
        params=params,
        total_length_values=total_length_values,
        Tclk_values=Tclk_values,
        CL_values=CL_values,
        solver=solver,
        verbose=verbose,
    )
    ranked = rank_common_informative_cases(records, require_all_feasible=True)
    if not ranked:
        raise RuntimeError(
            "No common feasible parameter setting was found across all profiles."
        )
    best = ranked[0]
    representative_result = best["profile_results"]["Single-hotspot"]["result"]
    return {
        **best,
        "result": representative_result,
    }, ranked


def default_free_length_params(base_params: Optional[dict] = None):
    """
    Add the bounds needed by the unrestricted free-length formulation.
    """
    params = _copy_params(base_params)
    params.setdefault("eps", 0.5)
    params.setdefault("lmax", 15.0)
    return params


def sweep_free_length_primal_dual_over_Tclk(
    N: int = 6,
    Tclk_list=None,
    params: Optional[dict] = None,
    primal_solver=cp.SCS,
    dual_solver=cp.CLARABEL,
    verbose: bool = False,
):
    """
    Compare unrestricted primal and dual values across timing budgets.
    """
    current_params = default_free_length_params(params)
    timing_values = default_tclk_list() if Tclk_list is None else [float(x) for x in Tclk_list]

    results = []
    for Tclk in timing_values:
        current_params["Tclk"] = float(Tclk)
        comparison = solve_free_length_primal_and_dual_fixed_N(
            N,
            current_params,
            primal_solver=primal_solver,
            dual_solver=dual_solver,
            verbose=verbose,
        )
        results.append(
            {
                "Tclk": float(Tclk),
                "params": copy.deepcopy(current_params),
                **comparison,
            }
        )
    return results


def sweep_over_N(
    N_list=None,
    params: Optional[dict] = None,
    total_length: float = 30.0,
    solver=cp.SCS,
    verbose: bool = False,
):
    current_params = _copy_params(params)
    stage_counts = default_n_list() if N_list is None else list(N_list)

    results_by_profile = {"Uniform": [], "Back-loaded": [], "Single-hotspot": []}
    for N in stage_counts:
        profiles = build_profile_suite(N, total_length)
        for profile_name, profile in profiles.items():
            result = _solve_full_or_mark_infeasible(
                N,
                current_params,
                profile_name,
                profile,
                solver,
                verbose,
            )
            results_by_profile[profile_name].append(result)
    return results_by_profile


def sweep_over_Tclk(
    N: int = 6,
    Tclk_list=None,
    params: Optional[dict] = None,
    total_length: float = 30.0,
    solver=cp.SCS,
    verbose: bool = False,
):
    current_params = _copy_params(params)
    timing_values = default_tclk_list() if Tclk_list is None else [float(x) for x in Tclk_list]
    profiles = build_profile_suite(N, total_length)

    results_by_profile = {"Uniform": [], "Back-loaded": [], "Single-hotspot": []}
    for Tclk in timing_values:
        current_params["Tclk"] = float(Tclk)
        for profile_name, profile in profiles.items():
            result = _solve_full_or_mark_infeasible(
                N,
                current_params,
                profile_name,
                profile,
                solver,
                verbose,
                Tclk=float(Tclk),
            )
            results_by_profile[profile_name].append(result)
    return results_by_profile


def sweep_full_vs_baseline(
    N: int = 6,
    Tclk_list=None,
    params: Optional[dict] = None,
    total_length: float = 30.0,
    profile_name: str = "Single-hotspot",
    solver=cp.SCS,
    verbose: bool = False,
):
    current_params = _copy_params(params)
    timing_values = default_tclk_list() if Tclk_list is None else [float(x) for x in Tclk_list]
    profiles = build_profile_suite(N, total_length)
    if profile_name not in profiles:
        raise KeyError(f"Unknown profile name: {profile_name}")

    full_profile = profiles[profile_name]
    comparison_results = []

    for Tclk in timing_values:
        current_params["Tclk"] = float(Tclk)
        full_result = _solve_full_or_mark_infeasible(
            N,
            current_params,
            profile_name,
            full_profile,
            solver,
            verbose,
            Tclk=float(Tclk),
        )
        baseline_result = _solve_baseline_or_mark_infeasible(
            N,
            current_params,
            solver,
            verbose,
            Tclk=float(Tclk),
        )
        comparison_results.append(
            {
                "Tclk": float(Tclk),
                "profile_name": profile_name,
                "full": full_result,
                "baseline": baseline_result,
            }
        )
    return comparison_results
