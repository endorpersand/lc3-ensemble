use std::collections::HashMap;

use lc3_ensemble::asm::assemble_debug;
use lc3_ensemble::parse::parse_ast;
use lc3_ensemble::sim::Simulator;
use lc3_ensemble::sim::mem::{MemAccessCtx, Word, WordCreateStrategy};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// LC-3 tester framework, built on [`lc3_ensemble`].
#[pymodule]
fn ensemble_test(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    m.add_class::<Reg>()?;

    Ok(())
}

/// A register number.
/// 
/// Used for [`Simulator::get_reg`] and [`Simulator::set_reg`].
/// 
/// [`Simulator::get_reg`]: [`PySimulator::get_reg`]
/// [`Simulator::set_reg`]: [`PySimulator::set_reg`]
#[derive(Clone, Copy)]
#[pyclass(module="ensemble_test")]
enum Reg {
    R0, R1, R2, R3, R4, R5, R6, R7
}
#[pymethods]
impl Reg {
    /// Creates a new register from an integer.
    #[new]
    fn new(n: u8) -> PyResult<Self> {
        match n {
            0 => Ok(Self::R0),
            1 => Ok(Self::R1),
            2 => Ok(Self::R2),
            3 => Ok(Self::R3),
            4 => Ok(Self::R4),
            5 => Ok(Self::R5),
            6 => Ok(Self::R6),
            7 => Ok(Self::R7),
            _ => Err(PyValueError::new_err(format!("register {n} out of bounds")))
        }
    }
}
impl From<Reg> for lc3_ensemble::ast::Reg {
    fn from(value: Reg) -> Self {
        use lc3_ensemble::ast::reg_consts::{R0, R1, R2, R3, R4, R5, R6, R7};

        match value {
            Reg::R0 => R0,
            Reg::R1 => R1,
            Reg::R2 => R2,
            Reg::R3 => R3,
            Reg::R4 => R4,
            Reg::R5 => R5,
            Reg::R6 => R6,
            Reg::R7 => R7,
        }
    }
}

#[pyclass(name="Simulator", module="ensemble_test")]
struct PySimulator {
    sim: Simulator,
}

#[pymethods]
impl PySimulator {
    /// Constructs a new simulator in Python.
    #[new]
    fn constructor() -> Self {
        PySimulator { sim: Simulator::new(WordCreateStrategy::Unseeded) }
    }

    /// Initializes simulator's state.
    fn init(&mut self, src_fp: &str) -> PyResult<()> {
        self.sim = Simulator::new(WordCreateStrategy::Unseeded);
        
        let src = std::fs::read_to_string(src_fp)?;
        let ast = parse_ast(&src).unwrap();
        let obj = assemble_debug(ast, &src).unwrap();

        self.sim.load_obj_file(&obj);
        Ok(())
    }

    #[pyo3(signature=(
        src_fp,
        disable_plugins = false,
        process_debug_comments = true,
        multiple_errors = true,
        enable_warnings = false,
        warnings_as_errors = false,
    ))]
    /// Assembles (ported from pylc3, may need to revise)
    /// 
    /// `src_fp`: Full path of the file to load.
    /// `disable_plugins`: True to disable lc3 plugins.
    /// `process_debug_comments`: True to enable processing of @ statements in comments.
    /// `multiple_errors`: Assembling doesn't end with the first error message.
    /// `enable_warnings`: Enable assembler warnings.
    /// `warnings_as_errors`: Treat assembler warnings as errors.
    fn load(
        &mut self,
        src_fp: &str, 
        disable_plugins: bool, 
        process_debug_comments: bool, 
        multiple_errors: bool, 
        enable_warnings: bool, 
        warnings_as_errors: bool
    ) -> PyResult<String> {
        // most of these arguments don't actually apply, so we'd probably need to revise this
        todo!()
    }

    // Test only (ported from pylc3, may need to revise)
    fn load_code(&mut self, lc3_code: &str) -> PyResult<bool> {
        todo!()
    }

    /// Runs the simulator until the program halts or `n` steps have been executed.
    /// 
    /// This was originally `LC3State::run` in complx's pylc3.
    fn run(&mut self, n: Option<usize>) -> PyResult<()> {
        // TODO, use n to keep track of number of steps.
        // TODO: don't panic here
        self.sim.run().unwrap();
        Ok(())
    }

    /// Performs one step in.
    /// 
    /// This was originally `LC3State::step` in complx's pylc3.
    fn step(&mut self) -> PyResult<()> {
        // TODO: don't panic here
        self.sim.step_in().unwrap();
        Ok(())
    }

    /// @see lc3_back (ported from pylc3, may need to revise)
    fn back(&mut self) -> PyResult<()> {
        todo!()
    }

    /// @see lc3_rewind (ported from pylc3, may need to revise)
    fn rewind(&mut self, n: Option<usize>) -> PyResult<()> {
        todo!()
    }

    /// Performs one step out.
    /// 
    /// This was originally `LC3State::finish` in complx's pylc3.
    fn finish(&mut self) -> PyResult<()> {
        // TODO: don't panic here
        self.sim.step_out().unwrap();
        Ok(())
    }

    /// @see lc3_next_line (ported from pylc3, may need to revise)
    fn next_line(&mut self, n: Option<usize>) -> PyResult<()> {
        todo!()
    }
    
    /// @see lc3_prev_line (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::previous_line]
    fn prev_line(&mut self, n: Option<usize>) -> PyResult<()> {
        todo!()
    }
    
    /// Reads the value at the given address in memory with privileged access.
    /// 
    /// Note that this will trigger IO updates (unlike [`PySimulator::get_mem`]) and should **not** be used
    /// for querying the state of memory. If desired, use the other function instead.
    /// 
    /// This was originally `LC3State::memory_read` in complx's pylc3.
    fn read_mem(&mut self, addr: u16) -> PyResult<u16> {
        // TODO: strict?
        // TODO: don't panic here
        Ok(self.sim.mem.read(addr, MemAccessCtx { privileged: true, strict: false }).unwrap().get())
    }
    

    /// Writes the value at the given address in memory with privileged access.
    /// 
    /// Note that this will trigger IO updates (unlike [`PySimulator::set_mem`]) and should **not** be used
    /// for directly setting the state of memory. If desired, use the other function instead.
    /// 
    /// This was originally `LC3State::memory_write` in complx's pylc3.
    fn write_mem(&mut self, addr: u16, val: u16) -> PyResult<()> {
        // TODO: strict?
        // TODO: don't panic here
        self.sim.mem.write(addr, Word::new_init(val), MemAccessCtx { privileged: true, strict: false }).unwrap();
        Ok(())
    }

    /// @see lc3_sym_lookup (ported from pylc3, may need to revise)
    fn lookup(&mut self, label: &str) -> PyResult<u16> {
        todo!()
    }

    /// @see lc3_sym_rev_lookup (ported from pylc3, may need to revise)
    fn reverse_lookup(&mut self, addr: u16) -> PyResult<&str> {
        todo!()
    }

    /// @see lc3_sym_add (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::add_symbol]
    fn add_label(&mut self, label: &str, addr: u16) -> PyResult<bool> {
        todo!()
    }

    /// @see lc3_sym_delete (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::delete_label]
    fn delete_label(&mut self, label: &str) -> PyResult<()> {
        todo!()
    }

    /// Gets the value at the given address in memory.
    /// 
    /// Note that this reads the state of the memory and will not trigger IO updates (unlike [`PySimulator::read_mem`]).
    /// If these are desired, use the other function instead.
    /// 
    /// This was originally `LC3State::get_memory` in complx's pylc3.
    fn get_mem(&self, addr: u16) -> u16 {
        self.sim.mem.get_raw(addr).get()
    }

    /// Sets the value at the given address in memory.
    /// 
    /// Note that this reads the state of the memory and will not trigger IO updates (unlike [`PySimulator::write_mem`]).
    /// If these are desired, use the other function instead.
    /// 
    /// This was originally `LC3State::set_memory` in complx's pylc3.
    fn set_mem(&mut self, addr: u16, val: u16) {
        self.sim.mem.get_raw_mut(addr).set(val);
    }

    /// @see lc3_disassemble (ported from pylc3, may need to revise)
    fn disassemble(&mut self, addr: u16, level: i32) -> PyResult<&str> {
        todo!()
    }
    
    /// @see lc3_disassemble (ported from pylc3, may need to revise)
    fn disassemble_data(&mut self, addr: u16, level: i32) -> PyResult<&str> {
        todo!()
    }
    
    /// @see lc3_add_break (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::add_breakpoint]
    #[pyo3(signature=(
        addr,
        condition = "1",
        times = -1,
        label = ""
    ))]
    fn add_breakpoint_by_addr(&mut self, addr: u16, condition: &str, times: i32, label: &str) -> PyResult<bool> {
        todo!()
    }

    /// @see lc3_add_break (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::add_breakpoint]
    #[pyo3(signature=(
        label,
        condition = "1",
        times = -1,
        bp_label = ""
    ))]
    fn add_breakpoint_by_label(&mut self, label: u16, condition: &str, times: i32, bp_label: &str) -> PyResult<bool> {
        todo!()
    }

    /// @see lc3_add_watch (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::add_watchpoint]
    #[pyo3(signature=(
        is_reg,
        data,
        condition = "1",
        times = -1,
        label = ""
    ))]
    fn add_watchpoint_by_reg_or_addr(&mut self, is_reg: bool, data: u16, condition: &str, times: i32, label: &str) -> PyResult<bool> {
        todo!()
    }

    #[pyo3(signature=(
        label,
        condition = "1",
        times = -1,
        wp_label = ""
    ))]
    /// @see lc3_add_watch (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::add_watchpoint]
    fn add_watchpoint_by_label(&mut self, label: &str, condition: &str, times: i32, wp_label: &str) -> PyResult<bool> {
        todo!()
    }

    #[pyo3(signature=(
        label,
        condition = "1",
        bb_label = ""
    ))]
    /// @see lc3_add_blackbox (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::add_blackbox]
    fn add_blackbox_by_label(&mut self, label: &str, condition: &str, bb_label: &str) -> PyResult<bool> {
        todo!()
    }

    #[pyo3(signature=(
        addr,
        condition = "1",
        bb_label = ""
    ))]
    /// @see lc3_add_blackbox (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::add_blackbox]
    fn add_blackbox_by_addr(&mut self, addr: u16, condition: &str, bb_label: &str) -> PyResult<bool> {
        todo!()
    }


    /// @see lc3_remove_break (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::remove_breakpoint]
    fn remove_breakpoint_by_addr(&mut self, addr: u16) -> PyResult<bool> {
        todo!()
    }
    /// @see lc3_remove_break (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::remove_breakpoint]
    fn remove_breakpoint_by_label(&mut self, label: &str) -> PyResult<bool> {
        todo!()
    }
    /// @see lc3_remove_watch (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::remove_watchpoint]
    fn remove_watchpoint_by_reg_or_addr(&mut self, is_reg: bool, data: u16) -> PyResult<bool> {
        todo!()
    }
    /// @see lc3_remove_watch (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::remove_watchpoint]
    fn remove_watchpoint_by_label(&mut self, label: &str) -> PyResult<bool> {
        todo!()
    }
    /// @see lc3_remove_blackbox (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::remove_blackbox]
    fn remove_blackbox_by_addr(&mut self, addr: u16) -> PyResult<bool> {
        todo!()
    }
    /// @see lc3_remove_blackbox (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::remove_blackbox]
    fn remove_blackbox_by_label(&mut self, label: &str) -> PyResult<bool> {
        todo!()
    }

    /// Adds metadata for the subroutine specified (ported from pylc3, may need to revise)
    fn add_subroutine_info(&mut self, subroutine_label: &str, n_params: u32) -> PyResult<bool> {
        todo!()
    }

    /// @see srand (ported from pylc3, may need to revise)
    fn seed(&mut self, seed: u32) {
        todo!()
    }

    /// @see lc3_random (ported from pylc3, may need to revise)
    fn random(&mut self) -> u16 {
        todo!()
    }

    /// Gets the value stored at the provided register.
    /// 
    /// This was originally the following functions in complx's pylc3:
    /// - `LC3State::get_r0`
    /// - `LC3State::get_r1`
    /// - `LC3State::get_r2`
    /// - `LC3State::get_r3`
    /// - `LC3State::get_r4`
    /// - `LC3State::get_r5`
    /// - `LC3State::get_r6`
    /// - `LC3State::get_r7`
    /// - `LC3State::get_register`
    fn get_reg(&self, index: Reg) -> u16 {
        self.sim.reg_file[index.into()].get()
    }

    /// Sets the value stored at the provided register.
    /// 
    /// This was originally the following functions in complx's pylc3:
    /// - `LC3State::set_r0`
    /// - `LC3State::set_r1`
    /// - `LC3State::set_r2`
    /// - `LC3State::set_r3`
    /// - `LC3State::set_r4`
    /// - `LC3State::set_r5`
    /// - `LC3State::set_r6`
    /// - `LC3State::set_r7`
    /// - `LC3State::set_register`
    fn set_reg(&mut self, index: Reg, val: u16) {
        self.sim.reg_file[index.into()].set(val)
    }

    /// Gets the n condition code.
    /// 
    /// This was originally `LC3State::get_n` from complx's pylc3.
    fn get_n(&self) -> bool {
        let cc = self.sim.psr().cc();
        cc & 0b100 != 0
    }
    /// Gets the z condition code.
    /// 
    /// This was originally `LC3State::get_z` from complx's pylc3.
    fn get_z(&self) -> bool {
        let cc = self.sim.psr().cc();
        cc & 0b010 != 0
    }
    /// Gets the p condition code.
    /// 
    /// This was originally `LC3State::get_p` from complx's pylc3.
    fn get_p(&self) -> bool {
        let cc = self.sim.psr().cc();
        cc & 0b001 != 0
    }

    /// Gets the current value of the PC.
    /// 
    /// This was originally `LC3State::get_pc` from complx's pylc3.
    fn get_pc(&self) -> u16 {
        self.sim.pc
    }

    /// Sets the current value of the PC.
    /// 
    /// This was originally `LC3State::set_pc` from complx's pylc3.
    fn set_pc(&mut self, addr: u16) {
        self.sim.set_pc(Word::new_init(addr), false)
            .unwrap_or_else(|_| unreachable!("set_pc cannot error with initialized word"))
    }
    fn has_halted(&mut self) {
        todo!()
    }
    fn get_executions(&mut self) -> u32 {
        todo!()
    }
    
    /// @see lc3_state.memory_ops (ported from pylc3, may need to revise)
    fn get_memory_ops(&mut self) -> HashMap<u16, (/* lc3_memory_stats */)> {
        todo!()
    }

    /// @see lc3_state.comments (ported from pylc3, may need to revise)
    fn comment(&mut self) -> String {
        todo!()
    }

    fn get_breakpoints(&mut self) -> HashMap<u16, ( /* lc3_breakpoint_info& */ )> {
        todo!()
    }
    fn get_blackboxes(&mut self) -> HashMap<u16, ( /* lc3_blackbox_info& */ )> {
        todo!()
    }
    fn get_memory_watchpoints(&mut self) -> HashMap<u16, ( /* lc3_watchpoint_info& */ )> {
        todo!()
    }
    fn get_register_watchpoints(&mut self) -> HashMap<u16, ( /* lc3_watchpoint_info& */ )> {
        todo!()
    }

    fn get_max_undo_stack_size(&mut self) -> u32 {
        todo!()
    }
    fn set_max_undo_stack_size(&mut self, size: u32) {
        todo!()
    }

    fn get_max_call_stack_size(&mut self) -> u32 {
        todo!()
    }
    fn set_max_call_stack_size(&mut self, size: u32) {
        todo!()
    }

    fn get_true_traps(&mut self) -> bool {
        todo!()
    }
    fn set_true_traps(&mut self, status: bool) {
        todo!()
    }

    fn get_lc3_version(&mut self) -> i32 {
        todo!()
    }
    fn set_lc3_version(&mut self, version: i32) {
        todo!()
    }
    
    fn get_interrupts(&mut self) -> bool {
        todo!()
    }
    fn set_interrupts(&mut self, status: bool) {
        todo!()
    }
    
    fn enable_keyboard_interrupt(&mut self) {
        todo!()
    }
    
    fn get_keyboard_interrupt_delay(&mut self) -> u32 {
        todo!()
    }
    fn set_keyboard_intget_keyboard_interrupt_delay(&mut self, delay: u32) {
        todo!()
    }

    fn get_strict_execution(&mut self) -> bool {
        todo!()
    }
    fn set_strict_execution(&mut self, status: bool) {
        todo!()
    }
    
    fn setup_replay(&mut self, file: &str, replay_str: &str) {
        todo!()
    }
    fn describe_replay(&mut self, replay_str: &str) {
        todo!()
    }

    /// The following accessors are only meaningful if testing_mode was set (ported from pylc3, may need to revise)
    fn get_input(&mut self) -> &str {
        todo!()
    }
    fn set_input(&mut self, input: String) {
        todo!()
    }

    fn get_output(&mut self) -> &str {
        todo!()
    }
    fn set_output(&mut self, output: String) {
        todo!()
    }

    fn get_warnings(&mut self) -> &str {
        todo!()
    }
    fn set_warnings(&mut self, warnings: String) {
        todo!()
    }
    

    fn first_level_calls(&mut self) -> Vec<(/* lc3_subroutine_call_info */)> {
        todo!()
    }
    fn first_level_traps(&mut self) -> Vec<(/* lc3_trap_call_info */)> {
        todo!()
    }
}