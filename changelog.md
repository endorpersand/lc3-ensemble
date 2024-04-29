# 0.2.1 (April 29, 2024)

- Lower the Rust MSRV to 1.70 (from 1.77).

# 0.2.0 (April 24, 2024)

- Move Simulator configuration flags into `sim::SimFlags`
- Add a `use_real_halt` configuration flag
- Simplify `WordCreateStrategy` and tailor it to pylc3's default configurations
- `instructions_run` counter to allow for runtime limits
- Fix bug where `step_over` and `step_out` would run one command unconditionally without checking for breakpoints

# 0.1.0 (April 6, 2024)

- Initial release
- Implements the parser, assembler, simulator, etc.
- I dunno what else to say about this release
