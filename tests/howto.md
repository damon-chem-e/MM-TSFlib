# How to pass all tests

1. Install uv.
    - If on Engaging cluster, create a conda environment first, then run `pip install uv`
2. In the root directory of the repo, run `uv sync`
3. Activate the virtual environment.
    - If on windows, `.venv/Scripts/activate`
    - If on linux, `source .venv/bin/activate`
4. Run tests with `pytest -s`
    - `-s` flag to enable console prints, which is useful during training. If you do not want to train, comment out the relevant cases in `test_baseline.py` (line 33), or skip `test_baseline.py` all together. 
    - Run a specific test with `pytest tests/test_thisone.py`