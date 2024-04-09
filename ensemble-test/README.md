# lc3-ensemble Python Backend

## Setup

1. Create a Python virtual environment with `python -m venv .env`
2. Activate the environment by running the activate script
    - Windows: `.env\Scripts\activate`
    - Other: `source .env/bin/activate`
3. Install maturin through pip
4. Run `maturin develop`
5. Import the `ensemble_test.core` or `ensemble_test.autograder` modules while inside the virtual environment

## Differences from complx

This Python wrapper tries to maintain many of the same APIs as `complx` and `pylc3`.

Notably:

- `ensemble_test.core` inherits its API from `pylc3.core`,
- `ensemble_test.autograder` inherits its API from `pylc3.autograder`.

However, some breaking changes have been made to better fit the ensemble backend, which are noted below:

### core

- `pylc3.core.LC3State` -> `ensemble.core.Simulator`
- A various number of functions have been renamed or removed:
  - `LC3State::memory_read` -> `Simulator::read_mem`
  - `LC3State::memory_write` -> `Simulator::write_mem`
  - `LC3State::get_memory` -> `Simulator::get_mem`
  - `LC3State::set_memory` -> `Simulator::set_mem`
  - `LC3State::get_r*`, `LC3State::get_register` -> `Simulator::get_reg`
  - `LC3State::set_r*`, `LC3State::set_register` -> `Simulator::set_reg`
  - `LC3State::previous_line` -> `Simulator::prev_line`
  - `LC3State::add_symbol` -> `Simulator::add_label`
  - `LC3State::delete_label` -> `Simulator::delete_label`
  - `LC3State::add_breakpoint` -> `Simulator::add_breakpoint_by_addr`, `Simulator::add_breakpoint_by_label`
  - `LC3State::add_watchpoint` -> `Simulator::add_watchpoint_by_reg_or_addr`, `Simulator::add_watchpoint_by_label`
  - `LC3State::add_blackbox` -> `Simulator::add_blackbox_by_addr`, `Simulator::add_blackbox_by_label`
  - `LC3State::remove_breakpoint` -> `Simulator::remove_breakpoint_by_addr`, `Simulator::remove_breakpoint_by_label`
  - `LC3State::remove_watchpoint` -> `Simulator::remove_watchpoint_by_reg_or_addr`, `Simulator::remove_watchpoint_by_label`
  - `LC3State::remove_blackbox` -> `Simulator::remove_blackbox_by_addr`, `Simulator::remove_blackbox_by_label`

### autograder

- `pylc3.autograder.lc3_unit_test_case.*` -> `ensemble.autograder.*`
- The `six` compatibility layer has been removed. Users should use the ensemble backend on Python 3.
