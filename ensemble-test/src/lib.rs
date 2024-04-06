use std::collections::HashMap;

use lc3_ensemble::asm::assemble_debug;
use lc3_ensemble::ast::reg_consts::{R0, R1, R2, R3, R4, R5, R6, R7};
use lc3_ensemble::parse::parse_ast;
use lc3_ensemble::sim::Simulator;
use lc3_ensemble::sim::mem::WordCreateStrategy;
use pyo3::prelude::*;
use pyo3::exceptions::PyIndexError;

#[pyclass(name="Simulator")]
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

    /// @see lc3_run (ported from pylc3, may need to revise)
    fn run(&mut self, n: Option<usize>) -> PyResult<()> {
        todo!()
    }

    /// @see lc3_step (ported from pylc3, may need to revise)
    fn step(&mut self, ) -> PyResult<()> {
        todo!()
    }

    /// @see lc3_back (ported from pylc3, may need to revise)
    fn back(&mut self, ) -> PyResult<()> {
        todo!()
    }

    /// @see lc3_rewind (ported from pylc3, may need to revise)
    fn rewind(&mut self, n: Option<usize>) -> PyResult<()> {
        todo!()
    }

    /// @see lc3_finish (ported from pylc3, may need to revise)
    fn finish(&mut self, ) -> PyResult<()> {
        todo!()
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
    
    /// @see lc3_mem_read (ported from pylc3, may need to revise)
    fn read_mem(&mut self, addr: u16) -> PyResult<u16> {
        todo!()
    }
    
    /// @see lc3_mem_write (ported from pylc3, may need to revise)
    fn write_mem(&mut self, addr: u16, val: u16) -> PyResult<()> {
        todo!()
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

    /// Gets value at address, note that the difference between this and memory_read is that memory_read will trigger plugins and devices (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::get_memory]
    fn get_mem(&mut self, addr: u16) -> PyResult<u16> {
        todo!()
    }

    /// Sets value at address, note that the difference between this and memory_write is that memory_write will trigger plugins and devices (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::set_memory]
    fn set_mem(&mut self, addr: u16, val: u16) -> PyResult<()> {
        todo!()
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


    /// Gets value at address, note that the difference between this and memory_read is that memory_read will trigger plugins and devices (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::get_r0, ::get_r1, ::get_r2, ::get_r3, ::get_r4, ::get_r5, ::get_r6, ::get_r7, ::get_register]
    fn get_reg(&mut self, index: u16) -> PyResult<u16> {
        todo!()
    }

    /// Sets value at address, note that the difference between this and memory_write is that memory_write will trigger plugins and devices (ported from pylc3, may need to revise)
    /// [ORIGINALLY LC3State::set_r0, ::set_r1, ::set_r2, ::set_r3, ::set_r4, ::set_r5, ::set_r6, ::set_r7, ::set_register]
    fn set_reg(&mut self, index: u16, val: u16) -> PyResult<()> {
        todo!()
    }

    /// Gets the n condition code.
    fn get_n(&mut self) -> PyResult<bool> {
        todo!()
    }
    /// Gets the z condition code.
    fn get_z(&mut self) -> PyResult<bool> {
        todo!()
    }
    /// Gets the p condition code.
    fn get_p(&mut self) -> PyResult<bool> {
        todo!()
    }

    fn get_pc(&mut self, ) -> u16 {
        todo!()
    }
    fn set_pc(&mut self, addr: u16) {
        todo!()
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

    fn run_(&mut self) -> PyResult<()> {
        self.sim.run().unwrap();
        Ok(())
    }

    fn get_reg_(&self, reg: i32) -> PyResult<u16> {
        match reg {
            0 => Ok(self.sim.reg_file[R0].get()),
            1 => Ok(self.sim.reg_file[R1].get()),
            2 => Ok(self.sim.reg_file[R2].get()),
            3 => Ok(self.sim.reg_file[R3].get()),
            4 => Ok(self.sim.reg_file[R4].get()),
            5 => Ok(self.sim.reg_file[R5].get()),
            6 => Ok(self.sim.reg_file[R6].get()),
            7 => Ok(self.sim.reg_file[R7].get()),
            _ => Err(PyErr::new::<PyIndexError, _>("Invalid Register Specified"))
        }
    }

    fn get_memory(&mut self, addr: u16) -> PyResult<u16>{
        Ok(self.sim.mem.get_raw(addr).get())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ensemble_test(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    //m.add_function(wrap_pyfunction!(load_source, m)?)?;
    Ok(())
}
