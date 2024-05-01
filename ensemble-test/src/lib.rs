use lc3_ensemble::asm::{assemble_debug, ObjectFile};
use lc3_ensemble::ast::reg_consts::{R0, R1, R2, R3, R4, R5, R6, R7};
use lc3_ensemble::parse::parse_ast;
use lc3_ensemble::sim::debug::{Breakpoint, Comparator};
use lc3_ensemble::sim::mem::{MemAccessCtx, Word};
use lc3_ensemble::sim::{SimErr, Simulator, WordCreateStrategy};
use pyo3::{create_exception, prelude::*};
use pyo3::exceptions::{PyIndexError, PyValueError};

/// A LC-3 simulator and unit tester, backed by [`lc3-ensemble`].
#[pymodule]
fn ensemble_test(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    m.add("LoadError", py.get_type_bound::<LoadError>())?;
    m.add("SimError", py.get_type_bound::<SimError>())?;
    m.add_class::<MemoryFillType>()?;

    Ok(())
}

create_exception!(ensemble_test, LoadError, PyValueError);
create_exception!(ensemble_test, SimError, PyValueError);

impl LoadError {
    fn from_lc3_err(e: impl lc3_ensemble::err::Error) -> PyErr {
        struct ErrDisplay<'e, E>(&'e E);
        impl<E: lc3_ensemble::err::Error> std::fmt::Display for ErrDisplay<'_, E> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(self.0, f)?;
                if let Some(span) = self.0.span() {
                    write!(f, "({span:?})")?;
                }
                Ok(())
            }
        }
        
        LoadError::new_err(ErrDisplay(&e).to_string())
    }
}
impl SimError {
    fn from_lc3_err(e: SimErr, pc: u16) -> PyErr {
        SimError::new_err(format!("{e} (PC: x{:04X})", pc))
    }
}

#[derive(FromPyObject)]
enum BreakpointLocation {
    Address(u16),
    Label(String)
}

#[derive(Clone, Copy)]
#[pyclass(module="ensemble_test")]
enum MemoryFillType {
    Random,
    Single
}

#[pyclass(name="Simulator", module="ensemble_test")]
struct PySimulator {
    sim: Simulator,
    obj: Option<ObjectFile>
}

#[pymethods]
impl PySimulator {
    #[new]
    fn constructor() -> Self {
        Self {
            sim: Simulator::new(Default::default()),
            obj: None
        }
    }

    fn init(&mut self, fill: MemoryFillType, value: Option<u64>) -> u64 {
        let (strat, ret_value) = match fill {
            MemoryFillType::Random => {
                let seed = value.unwrap_or_else(rand::random);
                (WordCreateStrategy::Seeded { seed }, seed)
            },
            MemoryFillType::Single => {
                let value = value.unwrap_or(0);
                (WordCreateStrategy::Known { value: value as u16 }, value)
            },
        };
        
        self.sim.flags.word_create_strat = strat;
        self.sim.reset();
        ret_value
    }

    fn load_file(&mut self, src_fp: &str) -> PyResult<()> {
        self.sim.reset();
        self.obj.take();

        let src = std::fs::read_to_string(src_fp)?;
        let ast = parse_ast(&src)
            .map_err(LoadError::from_lc3_err)?;
        let obj = assemble_debug(ast, &src)
            .map_err(LoadError::from_lc3_err)?;
        
        self.sim.load_obj_file(&obj);
        self.obj.replace(obj);
        Ok(())
    }
    fn load_code(&mut self, src: &str) -> PyResult<()> {
        self.sim.reset();
        self.obj.take();

        let ast = parse_ast(src)
            .map_err(LoadError::from_lc3_err)?;
        let obj = assemble_debug(ast, src)
            .map_err(LoadError::from_lc3_err)?;
        
        self.sim.load_obj_file(&obj);
        Ok(())
    }

    fn run(&mut self, limit: Option<u64>) -> PyResult<()> {
        let result = if let Some(lim) = limit {
            self.sim.run_with_limit(lim)
        } else {
            self.sim.run()
        };

        result
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    fn step_in(&mut self) -> PyResult<()> {
        self.sim.step_in()
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    fn step_out(&mut self) -> PyResult<()> {
        self.sim.step_out()
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    fn step_over(&mut self) -> PyResult<()> {
        self.sim.step_over()
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
        }
    #[pyo3(signature=(
        addr,
        *,
        privileged = true,
        strict = false
    ))]
    fn read_mem(&mut self, addr: u16, privileged: bool, strict: bool) -> PyResult<u16> {
        let word = self.sim.mem.read(addr, MemAccessCtx { privileged, strict })
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))?;

        Ok(word.get())
    }
        #[pyo3(signature=(
        addr,
        val,
        *,
        privileged = true,
        strict = false
    ))]
    fn write_mem(&mut self, addr: u16, val: u16, privileged: bool, strict: bool) -> PyResult<()> {
        self.sim.mem.write(addr, Word::new_init(val), MemAccessCtx { privileged, strict })
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    fn get_mem(&self, addr: u16) -> u16 {
        self.sim.mem.get_raw(addr).get()
    }
    fn set_mem(&mut self, addr: u16, val: u16) {
        self.sim.mem.get_raw_mut(addr).set(val);
    }

    #[getter]
    fn get_r0(&self) -> u16 {
        self.sim.reg_file[R0].get()
    }
    #[setter]
    fn set_r0(&mut self, value: u16) {
        self.sim.reg_file[R0].set(value)
    }
    #[getter]
    fn get_r1(&self) -> u16 {
        self.sim.reg_file[R1].get()
    }
    #[setter]
    fn set_r1(&mut self, value: u16) {
        self.sim.reg_file[R1].set(value)
    }
    #[getter]
    fn get_r2(&self) -> u16 {
        self.sim.reg_file[R2].get()
    }
    #[setter]
    fn set_r2(&mut self, value: u16) {
        self.sim.reg_file[R2].set(value)
    }
    #[getter]
    fn get_r3(&self) -> u16 {
        self.sim.reg_file[R3].get()
    }
    #[setter]
    fn set_r3(&mut self, value: u16) {
        self.sim.reg_file[R3].set(value)
    }
    #[getter]
    fn get_r4(&self) -> u16 {
        self.sim.reg_file[R4].get()
    }
    #[setter]
    fn set_r4(&mut self, value: u16) {
        self.sim.reg_file[R4].set(value)
    }
    #[getter]
    fn get_r5(&self) -> u16 {
        self.sim.reg_file[R5].get()
    }
    #[setter]
    fn set_r5(&mut self, value: u16) {
        self.sim.reg_file[R5].set(value)
    }
    #[getter]
    fn get_r6(&self) -> u16 {
        self.sim.reg_file[R6].get()
    }
    #[setter]
    fn set_r6(&mut self, value: u16) {
        self.sim.reg_file[R6].set(value)
    }
    #[getter]
    fn get_r7(&self) -> u16 {
        self.sim.reg_file[R7].get()
    }
    #[setter]
    fn set_r7(&mut self, value: u16) {
        self.sim.reg_file[R7].set(value)
    }
    fn get_reg(&self, index: usize) -> PyResult<u16> {
        let reg = match index {
            0 => R0,
            1 => R1,
            2 => R2,
            3 => R3,
            4 => R4,
            5 => R5,
            6 => R6,
            7 => R7,
            _ => return Err(PyIndexError::new_err(format!("register {index} out of bounds")))
        };

        Ok(self.sim.reg_file[reg].get())
    }
    fn set_reg(&mut self, index: usize, val: u16) -> PyResult<()> {
        let reg = match index {
            0 => R0,
            1 => R1,
            2 => R2,
            3 => R3,
            4 => R4,
            5 => R5,
            6 => R6,
            7 => R7,
            _ => return Err(PyIndexError::new_err(format!("register {index} out of bounds")))
        };

        self.sim.reg_file[reg].set(val);
        Ok(())
    }

    fn lookup(&self, label: &str) -> Option<u16> {
        self.obj.as_ref()?.symbol_table()?.get_label(label)
    }
    fn reverse_lookup(&self, addr: u16) -> Option<&str> {
        // TODO: more efficient label access
        let (label, _) = self.obj.as_ref()?.symbol_table()?.label_iter()
            .find(|&(_, a)| a == addr)?;

        Some(label)
    }

    fn add_breakpoint(&mut self, break_loc: BreakpointLocation) {
        let m_addr = match break_loc {
            BreakpointLocation::Address(addr) => Some(addr),
            BreakpointLocation::Label(label)  => self.lookup(&label),
        };

        // TODO: error if label not found?
        if let Some(addr) = m_addr {
            self.sim.breakpoints.push(Breakpoint::PC(Comparator::eq(addr)));
        }
    }
    fn remove_breakpoint(&mut self, break_loc: BreakpointLocation) {
        let m_addr = match break_loc {
            BreakpointLocation::Address(addr) => Some(addr),
            BreakpointLocation::Label(label)  => self.lookup(&label),
        };
    
        // TODO: error if label not found?
        // TODO: error if breakpoint not found?
        if let Some(addr) = m_addr {
            self.sim.breakpoints.retain(|bp| {
                bp != &Breakpoint::PC(Comparator::eq(addr))
            })
        }
    }
    
    fn breakpoints(&self) -> Vec<u16> {
        todo!()
    }

    #[getter]
    fn get_n(&self) -> bool {
        self.sim.psr().cc() & 0b100 != 0
    }
    #[getter]
    fn get_z(&self) -> bool {
        self.sim.psr().cc() & 0b010 != 0
    }
    #[getter]
    fn get_p(&self) -> bool {
        self.sim.psr().cc() & 0b001 != 0
    }

    #[getter]
    fn get_pc(&self) -> u16 {
        self.sim.pc
    }
    #[setter]
    fn set_pc(&mut self, addr: u16) {
        self.sim.pc = addr;
    }

    #[getter]
    fn get_executions(&self) -> u64 {
        self.sim.instructions_run
    }

    #[getter]
    fn get_use_real_halt(&self) -> bool {
        self.sim.flags.use_real_halt
    }
    #[setter]
    fn set_use_real_halt(&mut self, status: bool) {
        self.sim.flags.use_real_halt = status;
    }
    
    #[getter]
    fn get_strict_mem_accesses(&self) -> bool {
        self.sim.flags.strict
    }
    #[setter]
    fn set_strict_mem_accesses(&mut self, status: bool) {
        self.sim.flags.strict = status;
    }
    
    #[getter]
    fn get_input(&self) -> &str {
        todo!()
    }
    #[setter]
    fn set_input(&mut self, input: &str) {
        todo!()
    }
    
    #[getter]
    fn get_output(&self) -> &str {
        todo!()
    }
    #[setter]
    fn set_output(&mut self, output: &str) {
        todo!()
    }
}