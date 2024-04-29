# lc3-ensemble Python Backend

## Development Setup

1. Create a Python virtual environment with `python -m venv .env`
2. Activate the environment by running the activate script
    - Windows: `.env\Scripts\activate`
    - Other: `source .env/bin/activate`
3. Install maturin through pip
4. Run `maturin develop`
5. Import the `ensemble_test.core` or `ensemble_test.autograder` modules while inside the virtual environment
