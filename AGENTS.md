# Project instructions for Codex

This is a Python quantitative finance project using Conda/Miniconda.

Always use the conda environment named `carma` for every Python command, script execution, package check, test run, and notebook execution. Do not run project code with plain `python`, `python3`, or the Microsoft Store Python launcher, because those may point to the wrong interpreter.

Preferred fast PowerShell setup
===============================

Prefer running the `carma` environment Python executable directly after adding the environment DLL paths to `PATH`. This avoids the overhead and multiline quoting issues of `conda run` while still using the correct Conda environment.

```powershell
$env:CARMA_ENV = "C:\Users\gabri\miniforge3\envs\carma"
$env:CONDA_PREFIX = $env:CARMA_ENV
$env:PATH = "$env:CARMA_ENV;$env:CARMA_ENV\Library\mingw-w64\bin;$env:CARMA_ENV\Library\usr\bin;$env:CARMA_ENV\Library\bin;$env:CARMA_ENV\Scripts;$env:CARMA_ENV\bin;$env:PATH"

& "$env:CARMA_ENV\python.exe" ...
```

The expected Python executable for the project environment is:

```text
C:\Users\gabri\miniforge3\envs\carma\python.exe
```

Before running substantial code, Codex may verify the environment with:

```powershell
$env:CARMA_ENV = "C:\Users\gabri\miniforge3\envs\carma"
$env:CONDA_PREFIX = $env:CARMA_ENV
$env:PATH = "$env:CARMA_ENV;$env:CARMA_ENV\Library\mingw-w64\bin;$env:CARMA_ENV\Library\usr\bin;$env:CARMA_ENV\Library\bin;$env:CARMA_ENV\Scripts;$env:CARMA_ENV\bin;$env:PATH"
& "$env:CARMA_ENV\python.exe" -c "import sys; print(sys.executable); print(sys.version)"
```

To run a Python script, use:

```powershell
$env:CARMA_ENV = "C:\Users\gabri\miniforge3\envs\carma"
$env:CONDA_PREFIX = $env:CARMA_ENV
$env:PATH = "$env:CARMA_ENV;$env:CARMA_ENV\Library\mingw-w64\bin;$env:CARMA_ENV\Library\usr\bin;$env:CARMA_ENV\Library\bin;$env:CARMA_ENV\Scripts;$env:CARMA_ENV\bin;$env:PATH"
& "$env:CARMA_ENV\python.exe" path\to\script.py
```

To run inline Python, use:

```powershell
$env:CARMA_ENV = "C:\Users\gabri\miniforge3\envs\carma"
$env:CONDA_PREFIX = $env:CARMA_ENV
$env:PATH = "$env:CARMA_ENV;$env:CARMA_ENV\Library\mingw-w64\bin;$env:CARMA_ENV\Library\usr\bin;$env:CARMA_ENV\Library\bin;$env:CARMA_ENV\Scripts;$env:CARMA_ENV\bin;$env:PATH"
& "$env:CARMA_ENV\python.exe" -c "print('hello')"
```

Fallback with `conda run`
=========================

If the direct executable approach fails, fall back to `conda run`. Prefer writing larger snippets to a temporary or scratch script before running them, because `conda run ... python -c` can fail on Windows when the command contains multiline code.

```powershell
& "C:\Users\gabri\miniforge3\Scripts\conda.exe" run -n carma python path\to\script.py
```

For short one-line checks, this fallback is acceptable:

```powershell
& "C:\Users\gabri\miniforge3\Scripts\conda.exe" run -n carma python -c "print('hello')"
```

Notebooks
=========

For notebooks, execute them using Python/Jupyter tooling from the `carma` environment. Prefer `nbclient` or Jupyter through the direct `carma` Python setup above, and set the notebook working directory correctly so relative paths such as `../data/...` resolve as expected.

Example notebook execution pattern:

```powershell
$env:CARMA_ENV = "C:\Users\gabri\miniforge3\envs\carma"
$env:CONDA_PREFIX = $env:CARMA_ENV
$env:PATH = "$env:CARMA_ENV;$env:CARMA_ENV\Library\mingw-w64\bin;$env:CARMA_ENV\Library\usr\bin;$env:CARMA_ENV\Library\bin;$env:CARMA_ENV\Scripts;$env:CARMA_ENV\bin;$env:PATH"
& "$env:CARMA_ENV\python.exe" scratch\execute_notebook.py
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

Important notes for Codex: use PowerShell syntax; prefix full executable paths with `&`; do not rely on `conda activate` in non-interactive shells; keep all project Python, package checks, tests, and notebook execution inside the `carma` environment. Do not call the environment `python.exe` without first adding the environment DLL paths above, because native packages such as NumPy/SciPy may crash silently on Windows.
