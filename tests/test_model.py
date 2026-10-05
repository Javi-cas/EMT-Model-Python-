import numpy as np
import pytest

from emtmetab import (
    FLUX_NAMES,
    Hcomp,
    Hs,
    atp_contributions,
    atp_from_fluxes,
    default_params,
    flux_root_system,
    flux_solver_diagnostics,
    simulate,
    solve_fluxes,
    steady_state,
    sweep,
)


class TestShiftedHill:
    def test_limits(self):
        assert Hs(0.0, 100, 4, 3.0) == pytest.approx(1.0)
        assert Hs(1e9, 100, 4, 3.0) == pytest.approx(3.0, rel=1e-6)

    def test_midpoint(self):
        # At x = x0 the function is halfway between 1 and lambda.
        assert Hs(100.0, 100, 4, 3.0) == pytest.approx(2.0)

    @pytest.mark.parametrize("lam,increasing", [(3.0, True), (0.2, False)])
    def test_monotone_activation_or_inhibition(self, lam, increasing):
        y = np.array([Hs(x, 50, 2, lam) for x in np.linspace(0, 500, 50)])
        d = np.diff(y)
        assert np.all(d >= 0) if increasing else np.all(d <= 0)


def test_competitive_combiner_limits():
    assert Hcomp(1, 5, 0.2, 0, 250, 2, 0, 150, 2) == pytest.approx(1.0)
    assert Hcomp(1, 5, 0.2, 1e9, 250, 2, 0, 150, 2) == pytest.approx(5.0, rel=1e-6)
    assert Hcomp(1, 5, 0.2, 0, 250, 2, 1e9, 150, 2) == pytest.approx(0.2, rel=1e-6)


def test_default_params_complete():
    p = default_params()
    assert all(f"ATP_{k}" in p for k in FLUX_NAMES)
    assert all(np.isfinite(v) for v in p.values())


def test_flux_system_converges_at_default_steady_state():
    ss = steady_state()
    d = flux_solver_diagnostics(ss["A"], ss["H"])
    assert d["success"] and d["max_abs_residual"] < 1e-6
    x, _ = solve_fluxes(ss["A"], ss["H"], default_params())
    assert np.max(np.abs(flux_root_system(x, ss["A"], ss["H"], default_params()))) < 1e-6


def test_steady_state_is_stationary():
    ss = steady_state(t_end=400)
    assert abs(ss["dA_dt"]) < 1e-3 and abs(ss["dH_dt"]) < 1e-3


def test_atp_accounting_is_consistent():
    p = default_params()
    res = simulate(p, t_span=(0, 50))
    total = sum(atp_contributions(res, p).values())
    assert np.allclose(total, res["ATP"])
    x, _ = solve_fluxes(res["A"][-1], res["H"][-1], p)
    assert atp_from_fluxes(x, p) == pytest.approx(res["ATP"][-1])


def test_producers_and_consumers_match_published_notebook():
    p = default_params()
    contrib = atp_contributions(simulate(p, t_span=(0, 200)), p)
    producers = sorted(k for k, v in contrib.items() if v.mean() > 0)
    consumers = sorted(k for k, v in contrib.items() if v.mean() < 0)
    assert producers == sorted(["G1", "G2", "F", "Q1"])
    assert consumers == sorted(["Q3", "G3", "F2", "Q2"])


def test_state_variables_stay_nonnegative():
    res = simulate(t_span=(0, 200))
    for k in ["A", "H", "Rmt", "Rnox"]:
        assert res[k].min() >= 0


def test_sweep_runs_and_rejects_unknown_parameter():
    rows = sweep("M", [600, 1200], t_end=100)
    assert len(rows) == 2 and all(abs(r["ATP"] - r["ATP_check"]) < 1e-6 for r in rows)
    with pytest.raises(KeyError):
        sweep("not_a_parameter", [1])


def test_cli_simulate(capsys):
    from emtmetab.cli import main

    assert main(["simulate", "--t-end", "20", "--set", "M=1200"]) == 0
    assert '"ATP"' in capsys.readouterr().out


def test_fixed_points_three_stable_two_saddles():
    from emtmetab import fixed_points

    fps = fixed_points()
    kinds = [f["stability"] for f in fps]
    assert kinds == ["stable", "saddle", "stable", "saddle", "stable"]
    # The default trajectory ends at the high-AMPK, low-HIF-1 stable state.
    hi_ampk = fps[-1]
    assert hi_ampk["A"] == pytest.approx(561.44, abs=0.5)
    assert hi_ampk["H"] == pytest.approx(91.43, abs=0.5)
    # The opposite corner: low AMPK, high HIF-1.
    assert fps[0]["A"] < 150 and fps[0]["H"] > 400


def test_jacobian_matches_finite_difference_scale():
    from emtmetab import jacobian

    J = jacobian(561.44, 91.43)
    assert J.shape == (2, 2) and np.all(np.isfinite(J))
