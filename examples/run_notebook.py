"""Execute the published notebook headlessly (used in CI).

The notebook's first cell runs `pip install -r requirements.txt`; it is skipped here because
dependencies are already installed. The notebook file itself is not modified.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.read(ROOT / "1_emt_metabolism_eq1_31_notebook.ipynb", as_version=4)
nb.cells = [c for c in nb.cells if "pip install" not in c.source]
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()
out = "".join(o.get("text", "") for c in nb.cells for o in c.get("outputs", []))
print(next(line for line in out.splitlines() if line.startswith("Final A,H,ATP")))
