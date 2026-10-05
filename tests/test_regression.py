"""Regression tests: the package must reproduce the published notebook exactly."""

import json
from pathlib import Path

import numpy as np
import pytest

from emtmetab import default_params, simulate

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "1_emt_metabolism_eq1_31_notebook.ipynb"

# Printed by cell 14 of the published notebook (Python 3.10.19, numpy 2.2.6, scipy 1.15.2).
NOTEBOOK_FINAL = {"A": 561.4449775216146, "H": 91.42650009077921, "ATP": 1675.0035738037714}


@pytest.fixture(scope="module")
def default_run():
    return simulate(t_span=(0, 200))


def test_matches_published_notebook_output(default_run):
    for k, v in NOTEBOOK_FINAL.items():
        assert float(default_run[k][-1]) == pytest.approx(v, rel=1e-8), k


def test_notebook_still_prints_the_pinned_values():
    """Guards the pinned values themselves against silent drift in the notebook."""
    nb = json.loads(NOTEBOOK.read_text())
    text = "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    assert "Final A,H,ATP: 561.4449775216146 91.42650009077921 1675.0035738037714" in text


def _notebook_namespace():
    nb = json.loads(NOTEBOOK.read_text())
    code = [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]
    # Model-definition cells only (skip the pip cell and the plotting cells).
    defs = [c for c in code if c.lstrip().startswith(("import", "def"))]
    ns: dict = {}
    exec("\n\n".join(defs), ns)  # noqa: S102 - trusted repository notebook
    return ns


def test_package_functions_identical_to_notebook_functions():
    """Same inputs give bit-identical outputs from the notebook code and the package."""
    import emtmetab.model as pkg

    nb = _notebook_namespace()
    p = default_params()
    rng = np.random.default_rng(0)
    for a, h in rng.uniform(1, 700, size=(25, 2)):
        x_nb, _ = nb["solve_fluxes"](a, h, p)
        x_pk, _ = pkg.solve_fluxes(a, h, p)
        assert np.array_equal(x_nb, x_pk)
        assert nb["eqs_at"](a, h, p) == pkg.eqs_at(a, h, p)
        y = [a, h, 50.0, 50.0]
        assert nb["odes_1_to_31"](0.0, y, p) == pkg.odes_1_to_31(0.0, y, p)


def test_package_parameters_identical_to_notebook_parameters():
    nb = _notebook_namespace()
    assert nb["default_params"]() == default_params()
