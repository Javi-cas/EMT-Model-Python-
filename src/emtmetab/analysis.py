"""Analyses built on the model: nullclines, steady states, ATP accounting, parameter sweeps,
and solver diagnostics. These functions only call the model; they do not change it."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root

from .model import (
    atp_from_fluxes,
    default_params,
    eqs_at,
    flux_root_system,
    simulate,
    solve_fluxes,
)

FLUX_NAMES = ["G1", "G2", "F", "Q1", "Q3", "G3", "F2", "Q2"]
FLUX_LABELS = {
    "G1": "glucose oxidation",
    "G2": "glycolysis",
    "F": "fatty-acid oxidation",
    "Q1": "glutamine oxidation",
    "Q3": "reductive glutamine",
    "G3": "glucose anabolism",
    "F2": "fatty-acid synthesis",
    "Q2": "glutamine anabolism",
}
RESULT_KEYS = {"Q2": "Q2f"}  # simulate() stores the Q2 flux under "Q2f"


def nullcline_grid(p=None, a_max: float = 750.0, h_max: float = 750.0, n: int = 60):
    """dA/dt and dH/dt on an (A, H) grid, as in the notebook's nullcline section.

    Returns (A_vec, H_vec, dA, dH) with dA and dH indexed [H, A].
    """
    p = default_params() if p is None else p
    a_vec = np.linspace(0, a_max, n)
    h_vec = np.linspace(0, h_max, n)
    dA = np.zeros((n, n))
    dH = np.zeros((n, n))
    for i, a in enumerate(a_vec):
        for j, h in enumerate(h_vec):
            dA[j, i], dH[j, i] = eqs_at(a, h, p)
    return a_vec, h_vec, dA, dH


def steady_state(p=None, y0=None, t_end: float = 400.0) -> dict:
    """Integrate to t_end and report the final state with its residual derivatives."""
    p = default_params() if p is None else p
    res = simulate(p, t_span=(0, t_end), y0=y0)
    a, h = float(res["A"][-1]), float(res["H"][-1])
    da, dh = eqs_at(a, h, p)
    return {"A": a, "H": h, "ATP": float(res["ATP"][-1]), "dA_dt": float(da),
            "dH_dt": float(dh)}


def atp_contributions(res: dict, p=None) -> dict[str, np.ndarray]:
    """ATP produced (+) or consumed (-) by each flux over time (eqs. 23 to 31)."""
    p = default_params() if p is None else p
    return {k: p[f"ATP_{k}"] * res[RESULT_KEYS.get(k, k)] for k in FLUX_NAMES}


def flux_solver_diagnostics(A: float, H: float, p=None) -> dict:
    """Did the algebraic flux system converge at (A, H)? The model itself uses the solver
    output without checking; this helper makes convergence visible."""
    p = default_params() if p is None else p
    x0 = np.ones(8)
    sol = root(lambda v: flux_root_system(v, A, H, p), x0, method="hybr")
    resid = flux_root_system(sol.x, A, H, p)
    return {"success": bool(sol.success), "max_abs_residual": float(np.max(np.abs(resid))),
            "min_flux": float(sol.x.min()), "message": sol.message}


def sweep(param: str, values, p=None, t_end: float = 400.0, y0=None) -> list[dict]:
    """Steady state as one parameter varies (one-at-a-time sensitivity)."""
    base = default_params() if p is None else p
    if param not in base:
        raise KeyError(f"Unknown parameter '{param}'")
    out = []
    for v in values:
        q = dict(base)
        q[param] = float(v)
        ss = steady_state(q, y0=y0, t_end=t_end)
        fx, _ = solve_fluxes(ss["A"], ss["H"], q)
        ss.update({param: float(v)}, **dict(zip(FLUX_NAMES, map(float, fx), strict=True)))
        ss["ATP_check"] = float(atp_from_fluxes(fx, q))
        out.append(ss)
    return out


def _reduced_rhs(z, p):
    return np.array(eqs_at(z[0], z[1], p))


def jacobian(A: float, H: float, p=None, eps: float = 1e-4) -> np.ndarray:
    """Numerical Jacobian of the reduced (A, H) system, in which ROS and fluxes are at
    their algebraic quasi-steady state (as in the notebook's nullcline analysis)."""
    p = default_params() if p is None else p
    z = np.array([A, H], dtype=float)
    J = np.zeros((2, 2))
    for k in range(2):
        dz = np.zeros(2)
        dz[k] = eps * max(1.0, abs(z[k]))
        J[:, k] = (_reduced_rhs(z + dz, p) - _reduced_rhs(z - dz, p)) / (2 * dz[k])
    return J


def fixed_points(p=None, a_max: float = 750.0, h_max: float = 750.0, n: int = 25,
                 tol: float = 1e-6) -> list[dict]:
    """Fixed points of the reduced (A, H) system in [0, a_max] x [0, h_max].

    Newton-type root finding from an n x n grid of starting points; duplicates merged.
    Stability from the Jacobian eigenvalues (stable if all real parts are negative).
    """
    p = default_params() if p is None else p
    found: list[np.ndarray] = []
    for a0 in np.linspace(1, a_max, n):
        for h0 in np.linspace(1, h_max, n):
            sol = root(lambda z: _reduced_rhs(z, p), [a0, h0], method="hybr")
            z = sol.x
            if not sol.success or np.max(np.abs(_reduced_rhs(z, p))) > tol:
                continue
            if not (0 <= z[0] <= a_max and 0 <= z[1] <= h_max):
                continue
            if all(np.linalg.norm(z - f) > 1e-3 * max(1.0, np.linalg.norm(f)) for f in found):
                found.append(z)
    out = []
    for a, h in sorted(found, key=lambda z: z[0]):
        ev = np.linalg.eigvals(jacobian(a, h, p))
        stable = bool(np.all(ev.real < 0))
        kind = "stable" if stable else ("saddle" if np.prod(np.sign(ev.real)) < 0 else "unstable")
        out.append({"A": float(a), "H": float(h), "stability": kind,
                    "eigenvalues": [complex(e) for e in ev]})
    return out
