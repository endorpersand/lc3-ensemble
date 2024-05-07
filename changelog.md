# 0.4.0 (May 7, 2024)

- IO revisions
  - Revise `BiChannelIO` to use `Read` and `Write` traits
  - Change method definitions for `IODevice`
    - `IODevice::io_read` and `IODevice::io_write` take `&mut self` instead of `&self`
    - `IODevice::close` is removed (use `Drop` instead)
- `crate::sim::debug` revisions
  - Restrict `Breakpoint::PC` to only accept `u16` instead of `Comparator` (as comparing against addresses is unnecessary)
  - Revise `Comparator` to be an enum
  - Generalize `Breakpoint::And` and `Breakpoint::Or` to more than 2 breakpoints
  - Create `BreakpointList` to replace the `Vec<Breakpoint>` field of the simulator. This generates an ID that can be used to keep track of a given breakpoint.
- `FrameStack` (and `crate::sim::frame`)
  - Intended to aid in keeping track of call frames (for both subroutines and traps)
- Miscellaneous API changes
  - Create `Word::get_if_init` and `Word::set_if_init`, which replace and expand upon `Word::copy_word`
  - Delete `WordCreateStrategy::SeededFill`
  - Add `Simulator::run_while`
  - Add `Reg::try_from` and `Reg::reg_no`
  - Move `WordCreateStrategy` back into `crate::sim::mem`

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
