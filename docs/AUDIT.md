# Implementation audit

Notes from reviewing the published notebook while packaging it. **Nothing listed here was changed**: the model code in `src/emtmetab/model.py` is a verbatim copy of the notebook, and the regression test pins its output. These are items for the author to decide on.

| # | Location | Observation | Effect on published results | Suggested action |
|:--|:--|:--|:--|:--|
| 1 | `solve_fluxes` | The root solver's convergence flag (`sol.success`) is not checked; a failed solve would pass silently into the ODEs. | None found: at the default steady state the solver converges with max residual below 1e-6. | Use `emtmetab.flux_solver_diagnostics(A, H)` when exploring new parameter regions; consider warning on non-convergence. |
| 2 | `flux_root_system`, Q2 equation | The Q2 capacity is the literal `30.0` rather than a named parameter, so it cannot be changed through `default_params()`. | None at default parameters. | Promote it to a parameter (e.g. `q2`) in a future version, with the default value 30.0. |
| 3 | `flux_root_system`, glucose uptake G0 | Contains a disabled term `0.0 * p['g00'] * Hs(M, ...)`. | None (multiplied by zero). | Document whether the metabolite (M) dependence of glucose uptake was intentionally switched off. |
| 4 | `solve_fluxes` | Each call starts the root solver from a vector of ones rather than the previous solution. | Correct, but slower; in multistable regions the starting point can in principle select a different root of the flux equations. | Leave as is for reproducibility; a warm start could be an option. |
| 5 | Repository | No license file. | Others cannot legally reuse the code. | Choose a license (for example MIT or BSD-3-Clause) with your co-authors. |
| 6 | Repository name | `EMT-Model-Python-` while the content is the metabolism model. | Discoverability only. | Consider renaming (GitHub keeps a redirect). |
