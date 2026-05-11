# Project instructions for Codex

This is a Python quantitative finance project using Conda/Miniconda.

Always use the conda environment named `carma` for every Python command, script execution, package check, test run, and notebook execution. Do not run project code with plain `python`, `python3`, or the Microsoft Store Python launcher, because those may point to the wrong interpreter.

Use this command format by default in PowerShell:

```powershell
conda run -n carma python ...
```

If `conda` is not recognized in the shell `PATH`, use the full Conda executable path instead:

```powershell
& "C:\Users\gabri\miniforge3\Scripts\conda.exe" run -n carma python ...
```

The expected Python executable for the project environment is:

```text
C:\Users\gabri\miniforge3\envs\carma\python.exe
```

Before running substantial code, Codex may verify the environment with:

```powershell
& "C:\Users\gabri\miniforge3\Scripts\conda.exe" run -n carma python -c "import sys; print(sys.executable); print(sys.version)"
```

To run a Python script, use:

```powershell
& "C:\Users\gabri\miniforge3\Scripts\conda.exe" run -n carma python path\to\script.py
```

To run inline Python, use:

```powershell
& "C:\Users\gabri\miniforge3\Scripts\conda.exe" run -n carma python -c "print('hello')"
```

For notebooks, execute them using Python/Jupyter tooling from the `carma` environment. Prefer `nbclient` or Jupyter through `conda run`, and set the notebook working directory correctly so relative paths such as `../data/...` resolve as expected.

Example notebook execution pattern:

```powershell
& "C:\Users\gabri\miniforge3\Scripts\conda.exe" run -n carma python scratch\execute_notebook.py
```

Example `nbclient` script:

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path("notebooks/09_monthly_lambda.ipynb")
nb = nbformat.read(path, as_version=4)

client = NotebookClient(
    nb,
    timeout=1200,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
)

client.execute()
nbformat.write(nb, path)
```

Important notes for Codex: use PowerShell syntax; prefix full executable paths with `&`; do not rely on `conda activate` in non-interactive shells; keep all project Python, package checks, tests, and notebook execution inside the `carma` environment.
```



