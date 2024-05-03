# 0.3.0 (May 2, 2024)

- Various comment patches
- Rename `SymbolTable`'s methods to be clearer and more consistent
  - `SymbolTable::get_label` -> `SymbolTable::lookup_label`
  - `SymbolTable::get_line` -> `SymbolTable::lookup_line`
  - `SymbolTable::find_label_source` -> `SymbolTable::get_label_source`
  - `SymbolTable::find_line_source` -> `SymbolTable::rev_lookup_line`
  - Add `SymbolTable::rev_Lookup_label`
- IO revisions
  - Create `BufferedIO`, which is a buffered implementation of IO.
    - This was previously handled by `BiChannelIO` + `BlockingQueue`, but that implementation did not have easy access to the input. `BufferedIO` allows access to input (and hopefully is easier to use in general).
  - Remove `BlockingQueue`
    - No longer necessary due to introduction of `BufferedIO`.

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
