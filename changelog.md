# 0.7.2 (October 2, 2024)

- Fixed bug where `.fill -32` (or any negative number) would error.

# 0.7.1 (September 20, 2024)

- Added new SourceInfo APIs that allow constructing SourceInfo structs from source strings 
  - `SourceInfo::new`, `SourceInfo::from(String)`

# 0.7.0 (September 19, 2024)

- Revised MMIO system (should be last time!)
  - Allows IO to be modularized into separate devices
  - Allows IO devices to also trigger interrupts
  - Introduces timer interrupt device! (`device::TimerDevice`)
- Expand `use_real_halt` to also handle exceptions (via the `x00`, `x01`, `x02` interrupt vectors)
- Added more helpers to observe object file/symbol table state
  - `ObjectFile::addr_iter`: Maps each address in the object file to its value (if not uninitialized)
  - `SymbolTable::line_iter`: Maps each line defined in source to its corresponding address
  - `ast::asm::disassemble_line` and `ast::asm::try_disassemble_line`: Disassembles a single instruction.
- Added a simple memory observer API that tracks what memory addresses are written to during an execution.
  - `observer::ChangeObserver::mem_changed`: Gets whether an address changed during an execution
  - `observer::ChangeObserver::take_mem_changes`: Creates an ordered iterator of all memory address changes (and clears the observer)
- **Breaking changes** (like, everything):
  - IO related breaking changes
    - `io` module -> `device` module
    - `Simulator::open_io` and `Simulator::close_io` have been moved into `sim.device_handler` field (`device::DeviceHandler::add_device` and `device::DeviceHandler::remove_device`)
    - `Simulator::add_external_interrupt` and `Simulator::clear_external_interrupt` are removed (use `device::InterruptFromFn` with `device::Interrupt::External`)
    - `io::IODevice` -> `device::ExternalDevice`
    - `io::BufferedIO` -> `device::BufferedKeyboard`, `device::BufferedDisplay`
    - `io::BiChannelIO` removed with no current alternative
  - `Simulator::use_real_halt` -> `Simulator::use_real_traps`
  - `mem::Mem` -> `mem::MemArray`, `Simulator::read_mem`, `Simulator::write_mem`
  - Removed `ensemble-cli`

# 0.6.0 (August 13, 2024)

- Converted `Reg` into enum to better support niche optimizations
- Deleted `crate::ast::reg_consts`
- Fixed bug where program consisting only of new lines wouldn't assemble
- Added `ignore_privilege` flag

# 0.5.0 (June 29, 2024)

- Added significantly more documentation and examples
- Removed unnecessary APIs
  - Majority of `ObjectFile`'s methods
  - `Simulator::set_pc`, `Simulator::offset_pc` (just set `sim.pc` directly)
  - `Simulator::load_os` (useless to expose as this is done always)
  - `InterruptErr::new` (adjusted `Simulator::add_external_interrupt` to no longer require)
  - `SimErr::ProgramHalted` (useless to expose as it is never returned as an error)
  - Debug API
    - Removed `Breakpoint::Generic`, `Breakpoint::And`, and `Breakpoint::Or` (too niche)
    - Removed `BreakpointList` (simplification of above allows for use of `HashSet<Breakpoint>` instead)
  - IO API
    - Removed `SimIO`, `CustomIO` (adjusted `Simulator::open_io` to no longer require)
- Renamed `WordCreateStrategy` to `MachineInitStrategy`
- `sim.breakpoints` is now a `HashSet` and can be appended to directly
- `Simulator::open_io` now accepts all `IODevice`s without having to box beforehand
- `Simulator::add_external_interrupt` now accepts all `Errors` without having to be wrapped in `InterruptErr` beforehand
- `PSR::is_*`, `PSR::set_cc_*` for n, z, p

# 0.4.2 (May 15, 2024)

- Replace `slotmap` implementation of BreakpointList with an incremental one
- Add `Simulator::hit_halt`
- Revise SymbolTable's Debug implementation (does not affect its Debug output)
- External interrupts
  - Useful for stopping execution of the program when needed by an external handle around the Simulator (e.g., by a Python binding)
  - Interrupts that are triggered are raised as `SimErr::Interrupt`, which holds an opaque Error type which can be downcast and handled properly.

# 0.4.1 (May 8, 2024)

- Fix bug from 0.4.0 where the frame pointer wasn't being computed correctly when tracking a subroutine under standard calling convention
- Fix [the JSRR R7 bug](https://github.com/gt-cs2110/lc3tools/commit/fa9a23f62106eeee9fef7d2a278ba989356c9ee2)
- Create `Simulator::call_subroutine`

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
