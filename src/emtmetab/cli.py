"""Command line: ``emtmetab simulate``, ``emtmetab sweep``, ``emtmetab figures``."""

from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="emtmetab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="Run the default model and print the final state")
    s.add_argument("--t-end", type=float, default=200.0)
    s.add_argument("--set", nargs="*", default=[], metavar="PARAM=VALUE",
                   help="Override parameters, e.g. --set M=600 kh=0.5")
    w = sub.add_parser("sweep", help="Steady state across values of one parameter")
    w.add_argument("param")
    w.add_argument("--start", type=float, required=True)
    w.add_argument("--stop", type=float, required=True)
    w.add_argument("--n", type=int, default=11)
    w.add_argument("--out", default=None, help="CSV path (default: print)")
    f = sub.add_parser("figures", help="Write the README figures")
    f.add_argument("--out", default="docs/img")
    a = ap.parse_args(argv)

    from .model import default_params, simulate

    if a.cmd == "simulate":
        p = default_params()
        for kv in a.set:
            k, v = kv.split("=", 1)
            if k not in p:
                ap.error(f"unknown parameter {k}")
            p[k] = float(v)
        r = simulate(p, t_span=(0, a.t_end))
        print(json.dumps({k: float(r[k][-1]) for k in ["A", "H", "Rmt", "Rnox", "ATP"]},
                         indent=1))
    elif a.cmd == "sweep":
        import pandas as pd

        from .analysis import sweep

        df = pd.DataFrame(sweep(a.param, np.linspace(a.start, a.stop, a.n)))
        if a.out:
            df.to_csv(a.out, index=False)
            print(a.out)
        else:
            print(df.round(3).to_string(index=False))
    elif a.cmd == "figures":
        from .plots import make_all

        print(make_all(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
