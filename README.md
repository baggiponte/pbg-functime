# Introduction to time series forecasting with [functime](https://functime.ai/)

1. Install dependencies with any PEP517/518 compliant package manager (e.g. not Poetry).

```bash
# pdm
pdm install -G=ide
```

```bash
# pip
python -m venv .venv
source .venv/bin/activate
python -m pip install ".[ide]"
```

**Note**. If `lightgbm` fails to install, check [here](https://github.com/microsoft/LightGBM/tree/master/python-package#install-from-pypi).

2. Run the notebooks:

```bash
# pdm
pdm run marimo edit notebooks/intro.py

# inside a venv
python -m marimo edit notebooks/intro.py
```
