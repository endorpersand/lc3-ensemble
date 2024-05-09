use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use lc3_ensemble::asm::{assemble_debug, ObjectFile};
use lc3_ensemble::ast::reg_consts::{R0, R1, R2, R3, R4, R5, R6, R7};
use lc3_ensemble::ast::Reg;
use lc3_ensemble::parse::parse_ast;
use lc3_ensemble::sim::debug::Breakpoint;
use lc3_ensemble::sim::frame::{Frame, ParameterList};
use lc3_ensemble::sim::io::BufferedIO;
use lc3_ensemble::sim::mem::{MemAccessCtx, Word, WordCreateStrategy};
use lc3_ensemble::sim::{InterruptErr, SimErr, SimFlags, Simulator};
use pyo3::types::PyInt;
use pyo3::{create_exception, prelude::*};
use pyo3::exceptions::{PyIndexError, PyValueError};

/// Bindings for the LC3 simulator.
#[pymodule]
fn ensemble_test(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    m.add("LoadError", py.get_type_bound::<LoadError>())?;
    m.add("SimError", py.get_type_bound::<SimError>())?;
    m.add_class::<MemoryFillType>()?;
    m.add_class::<SubroutineType>()?;
    m.add_class::<PyFrame>()?;

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
    fn from_display(err: impl std::fmt::Display, pc: u16) -> PyErr {
        SimError::new_err(format!("{err} (PC: x{pc:04X})"))
    }

    fn from_lc3_err(err: SimErr, pc: u16) -> PyErr {
        match err {
            SimErr::Interrupt(e) => match e.into_inner().downcast() {
                Ok(py_err) => *py_err,
                Err(e) => SimError::from_display(e, pc),
            },
            e => SimError::from_display(e, pc)
        }
    }
}

#[derive(FromPyObject)]
enum MemLocation {
    Address(u16),
    Label(String)
}

#[derive(Clone, Copy)]
#[pyclass(module="ensemble_test")]
/// Strategies to fill the memory on initializing the simulator.
enum MemoryFillType {
    /// Fill the memory with random values.
    Random,
    /// Fill the memory with a single known value.
    Single
}

fn reg_from_py_int(reg_no: Bound<'_, PyInt>) -> PyResult<Reg> {
    let m_reg = reg_no.extract::<u8>().ok()
        .and_then(|i| Reg::try_from(i).ok());

    m_reg.ok_or_else(|| PyIndexError::new_err(format!("register {reg_no} out of bounds")))
}

#[derive(Clone, Copy)]
#[pyclass(module="ensemble_test")]
/// Strategies to fill the memory on initializing the simulator.
enum SubroutineType {
    /// Arguments are passed by standard LC-3 calling convention.
    CallingConvention,
    /// Arguments are passed by register.
    PassByRegister,
}

struct PyParamListWrapper(ParameterList);
impl IntoPy<PyObject> for PyParamListWrapper {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            ParameterList::CallingConvention { params } => (SubroutineType::CallingConvention, params).into_py(py),
            ParameterList::PassByRegister { params, ret } => {
                let params: Vec<_> = params.into_iter()
                    .map(|(s, r)| (s, r.reg_no()))
                    .collect();
                let ret = ret.map(Reg::reg_no);

                (SubroutineType::PassByRegister, params, ret).into_py(py)
            },
        }
    }
}
impl<'py> FromPyObject<'py> for PyParamListWrapper {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        match ob.get_item(0)?.extract::<SubroutineType>()? {
            SubroutineType::CallingConvention => {
                let (_, params) = ob.extract::<(SubroutineType, _)>()?;
                Ok(Self(ParameterList::CallingConvention { params }))
            },
            SubroutineType::PassByRegister => {
                let (_, params, ret) = ob.extract::<(SubroutineType, Vec<_>, Option<_>)>()?;
                let params = params.into_iter()
                    .map(|(s, r)| Ok((s, reg_from_py_int(r)?)))
                    .collect::<PyResult<_>>()?;

                let ret = match ret {
                    Some(r) => Some(reg_from_py_int(r)?),
                    None => None
                };
                
                Ok(Self(ParameterList::PassByRegister { params, ret }))
            }
        }
    }
}

#[repr(transparent)]
#[pyclass(name="Frame", module="ensemble_test")]
struct PyFrame(Frame);

#[pymethods]
impl PyFrame {
    #[getter]
    fn get_caller_addr(&self) -> u16 {
        self.0.caller_addr
    }
    #[getter]
    fn get_callee_addr(&self) -> u16 {
        self.0.callee_addr
    }
    #[getter]
    fn get_frame_type(&self) -> u16 {
        self.0.frame_type as u16
    }
    #[getter]
    fn get_frame_ptr(&self) -> Option<(u16, bool)> {
        let word = self.0.frame_ptr?;
        Some((word.get(), word.is_init()))
    }
    #[getter]
    fn get_arguments(&self) -> Vec<(u16, bool)> {
        self.0.arguments
            .iter()
            .map(|w| (w.get(), w.is_init()))
            .collect()
    }
}

/// The simulator!
#[pyclass(name="Simulator", module="ensemble_test")]
struct PySimulator {
    sim: Simulator,
    obj: Option<ObjectFile>,
    input: Arc<RwLock<VecDeque<u8>>>,
    output: Arc<RwLock<Vec<u8>>>
}

impl PySimulator {
    fn reset(&mut self) {
        self.sim.reset();

        let io = BufferedIO::with_bufs(Arc::clone(&self.input), Arc::clone(&self.output));
        self.sim.open_io(io);
        self.sim.add_external_interrupt(|_| {
            Python::with_gil(|py| py.check_signals())
                .map_err(InterruptErr::new)
        });
        self.obj.take();

        self.input.write().unwrap_or_else(|e| e.into_inner()).clear();
        self.output.write().unwrap_or_else(|e| e.into_inner()).clear();
    }

    fn resolve_location(&self, loc: MemLocation) -> Result<u16, String> {
        match loc {
            MemLocation::Address(addr) => Ok(addr),
            MemLocation::Label(label) => self.lookup(&label).ok_or(label),
        }
    }
}
#[pymethods]
impl PySimulator {
    #[new]
    fn constructor() -> Self {
        let flags = SimFlags {
            debug_frames: true,
            ..Default::default()
        };
        
        let mut this = Self {
            sim: Simulator::new(flags),
            obj: None,
            input: Default::default(),
            output: Default::default()
        };

        this.reset();
        this
    }

    /// Initialize the register files and memory of the simulator with the provided fill type and seed.
    /// 
    /// The following argument patterns are allowed:
    /// - `(MemoryFillType.Random, None)` -> randomly fill memory, with an arbitrary seed
    /// - `(MemoryFillType.Random, int)`  -> randomly fill memory, with the provided seed
    /// - `(MemoryFillType.Single, None)` -> fill memory with 0
    /// - `(MemoryFillType.Single, int)`  -> fill memory with the provided value
    /// 
    /// This method returns the seed/value that is used to initialize the simulator.
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
        self.reset();
        ret_value
    }

    /// Loads ASM code from a file, assembles it, 
    /// and loads the resulting object file into the simulator.
    /// 
    /// This can raise a [`LoadError`] if assembling fails.
    fn load_file(&mut self, src_fp: &str) -> PyResult<()> {
        let src = std::fs::read_to_string(src_fp)?;
        self.load_code(&src)
    }

    /// Assembles ASM code from a provided string, 
    /// and loads the resulting object file into the simulator.
    /// 
    /// This can raise a [`LoadError`] if assembling fails.
    fn load_code(&mut self, src: &str) -> PyResult<()> {
        self.reset();

        let ast = parse_ast(src)
            .map_err(LoadError::from_lc3_err)?;
        let obj = assemble_debug(ast, src)
            .map_err(LoadError::from_lc3_err)?;
        
        self.sim.load_obj_file(&obj);
        self.obj.replace(obj);
        Ok(())
    }

    /// Runs the simulator.
    /// 
    /// A `limit` parameter can be specified to limit the number of executions ran.
    /// 
    /// This can raise a [`SimError`] if an error occurs while simulating.
    fn run(&mut self, limit: Option<u64>) -> PyResult<()> {
        let result = if let Some(lim) = limit {
            self.sim.run_with_limit(lim)
        } else {
            self.sim.run()
        };

        result
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    /// Perform a step in.
    /// 
    /// This can raise a [`SimError`] if an error occurs while simulating.
    fn step_in(&mut self) -> PyResult<()> {
        self.sim.step_in()
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    /// Perform a step out.
    /// 
    /// This can raise a [`SimError`] if an error occurs while simulating.
    fn step_out(&mut self) -> PyResult<()> {
        self.sim.step_out()
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    /// Perform a step over.
    /// 
    /// This can raise a [`SimError`] if an error occurs while simulating.
    fn step_over(&mut self) -> PyResult<()> {
        self.sim.step_over()
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    /// Runs until a frame changes.
    /// 
    /// This is not meant for general use.
    fn _run_until_frame_change(&mut self) -> PyResult<()> {
        let frame = self.sim.frame_stack.len();
        self.sim.run_while(|s| s.frame_stack.len() == frame)
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }
    
    #[pyo3(signature=(
        addr,
        *,
        privileged = true,
        strict = false
    ))]
    /// Reads a value from memory, triggering any I/O devices if applicable.
    /// 
    /// See `get_mem` if you wish to get the memory directly without triggering I/O devices.
    /// 
    /// This function also accepts optional `privileged` and `strict` parameters.
    /// These designate whether to read memory in privileged mode and with strict memory access.
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

    /// Writes a value to memory, triggering any I/O devices if applicable.
    /// 
    /// See `set_mem` if you wish to set the memory directly without triggering I/O devices.
    /// 
    /// This function also accepts optional `privileged` and `strict` parameters.
    /// These designate whether to write memory in privileged mode and with strict memory access.
    fn write_mem(&mut self, addr: u16, val: u16, privileged: bool, strict: bool) -> PyResult<()> {
        self.sim.mem.write(addr, Word::new_init(val), MemAccessCtx { privileged, strict })
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }

    /// Gets a given value from memory without triggering I/O devices.
    /// 
    /// This function does not activate any I/O devices (and therefore can result in incorrect I/O values).
    /// If you wish to trigger I/O devices, use `read_mem`.
    fn get_mem(&self, addr: u16) -> u16 {
        self.sim.mem.get_raw(addr).get()
    }
    /// Sets a given value from memory without triggering I/O devices.
    /// 
    /// This function does not activate any I/O devices (and therefore can result in incorrect I/O values).
    /// If you wish to trigger I/O devices, use `write_mem`.
    fn set_mem(&mut self, addr: u16, val: u16) {
        self.sim.mem.get_raw_mut(addr).set(val);
    }

    /// The value of register 0.
    #[getter]
    fn get_r0(&self) -> u16 {
        self.sim.reg_file[R0].get()
    }
    #[setter]
    fn set_r0(&mut self, value: u16) {
        self.sim.reg_file[R0].set(value)
    }
    /// The value of register 1.
    #[getter]
    fn get_r1(&self) -> u16 {
        self.sim.reg_file[R1].get()
    }
    #[setter]
    fn set_r1(&mut self, value: u16) {
        self.sim.reg_file[R1].set(value)
    }
    /// The value of register 2.
    #[getter]
    fn get_r2(&self) -> u16 {
        self.sim.reg_file[R2].get()
    }
    #[setter]
    fn set_r2(&mut self, value: u16) {
        self.sim.reg_file[R2].set(value)
    }
    /// The value of register 3.
    #[getter]
    fn get_r3(&self) -> u16 {
        self.sim.reg_file[R3].get()
    }
    #[setter]
    fn set_r3(&mut self, value: u16) {
        self.sim.reg_file[R3].set(value)
    }
    /// The value of register 4.
    #[getter]
    fn get_r4(&self) -> u16 {
        self.sim.reg_file[R4].get()
    }
    #[setter]
    fn set_r4(&mut self, value: u16) {
        self.sim.reg_file[R4].set(value)
    }
    #[getter]
    /// The value of register 5.
    fn get_r5(&self) -> u16 {
        self.sim.reg_file[R5].get()
    }
    #[setter]
    fn set_r5(&mut self, value: u16) {
        self.sim.reg_file[R5].set(value)
    }
    /// The value of register 6.
    #[getter]
    fn get_r6(&self) -> u16 {
        self.sim.reg_file[R6].get()
    }
    #[setter]
    fn set_r6(&mut self, value: u16) {
        self.sim.reg_file[R6].set(value)
    }
    /// The value of register 7.
    #[getter]
    fn get_r7(&self) -> u16 {
        self.sim.reg_file[R7].get()
    }
    #[setter]
    fn set_r7(&mut self, value: u16) {
        self.sim.reg_file[R7].set(value)
    }

    /// Gets a value from a register.
    /// 
    /// This raises an error if the index is not between 0 and 7, inclusive.
    fn get_reg(&self, index: Bound<'_, PyInt>) -> PyResult<u16> {
        let reg = reg_from_py_int(index)?;
        Ok(self.sim.reg_file[reg].get())
    }

    /// Sets a value to a register.
    /// 
    /// This raises an error if the index is not between 0 and 7, inclusive.
    fn set_reg(&mut self, index: Bound<'_, PyInt>, val: u16) -> PyResult<()> {
        let reg = reg_from_py_int(index)?;
        self.sim.reg_file[reg].set(val);
        Ok(())
    }

    /// Looks up the address of a given label, returning None if the label is not defined.
    fn lookup(&self, label: &str) -> Option<u16> {
        self.obj.as_ref()?.symbol_table()?.lookup_label(label)
    }
    /// Looks up the label at a given address, returning None if no label is at the given address.
    fn reverse_lookup(&self, addr: u16) -> Option<&str> {
        self.obj.as_ref()?.symbol_table()?.rev_lookup_label(addr)
    }

    /// Adds a breakpoint to the given location.
    fn add_breakpoint(&mut self, break_loc: MemLocation) -> PyResult<u32> {
        let addr = self.resolve_location(break_loc)
            .map_err(|label| PyValueError::new_err(format!("cannot add a breakpoint at non-existent label {label:?}")))?;
            

        Ok(self.sim.breakpoints.insert(Breakpoint::PC(addr)))
    }
    /// Removes a breakpoint with the given ID.
    /// 
    /// This returns whether the removal was successful (i.e., whether there is a breakpoint at the given ID).
    fn remove_breakpoint(&mut self, break_id: u32) -> bool {
        self.sim.breakpoints.remove(break_id).is_some()
    }
    
    /// Gets a list of currently defined breakpoints.
    fn breakpoints(&self) -> std::collections::HashMap<u16, u32> {
        self.sim.breakpoints.iter()
            .filter_map(|(k, bp)| match *bp {
                Breakpoint::PC(addr) => Some((addr, k)),
                _ => None
            })
            .collect()
    }

    /// The n condition code.
    #[getter]
    fn get_n(&self) -> bool {
        self.sim.psr().cc() & 0b100 != 0
    }
    /// The z condition code.
    #[getter]
    fn get_z(&self) -> bool {
        self.sim.psr().cc() & 0b010 != 0
    }
    /// The p condition code.
    #[getter]
    fn get_p(&self) -> bool {
        self.sim.psr().cc() & 0b001 != 0
    }

    /// The program counter.
    #[getter]
    fn get_pc(&self) -> u16 {
        self.sim.pc
    }
    #[setter]
    fn set_pc(&mut self, addr: u16) {
        self.sim.pc = addr;
    }

    /// The number of instructions run since the simulator started running.
    #[getter]
    fn get_instructions_run(&self) -> u64 {
        self.sim.instructions_run
    }

    /// Configuration setting to designate whether to use real HALT or virtual HALT.
    #[getter]
    fn get_use_real_halt(&self) -> bool {
        self.sim.flags.use_real_halt
    }
    #[setter]
    fn set_use_real_halt(&mut self, status: bool) {
        self.sim.flags.use_real_halt = status;
    }
    
    /// Configuration setting to designate whether to use strict memory accesses during execution.
    #[getter]
    fn get_strict_mem_accesses(&self) -> bool {
        self.sim.flags.strict
    }
    #[setter]
    fn set_strict_mem_accesses(&mut self, status: bool) {
        self.sim.flags.strict = status;
    }
    
    /// The I/O input.
    #[getter]
    fn get_input(&self) -> String {
        let data: Vec<_> = self.input.read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .collect();

        String::from_utf8_lossy(&data).into_owned()
    }
    #[setter]
    fn set_input(&mut self, input: &str) {
        let mut inp = self.input.write()
            .unwrap_or_else(|e| e.into_inner());

        inp.clear();
        inp.extend(input.as_bytes());
    }
    fn append_to_input(&mut self, input: &str) {
        self.input.write()
            .unwrap_or_else(|e| e.into_inner())
            .extend(input.as_bytes())
    }

    /// The I/O output.
    #[getter]
    fn get_output(&self) -> String {
        String::from_utf8_lossy({
            &self.output.read()
                .unwrap_or_else(|e| e.into_inner())
        }).into_owned()
    }
    #[setter]
    fn set_output(&mut self, output: &str) {
        let mut out = self.output.write()
        .unwrap_or_else(|e| e.into_inner());

        out.clear();
        out.extend(output.as_bytes());
    }

    // Subroutine definitions and frames!

    /// Gets the definition of the subroutine located at the provided location, or None if no definition has been made.
    /// 
    /// A definition needs to be made with the `set_subroutine_def` method in order for this method to return a
    /// non-None value.
    fn get_subroutine_def(&self, loc: MemLocation) -> PyResult<Option<PyParamListWrapper>> {
        let addr = self.resolve_location(loc)
            .map_err(|label| PyValueError::new_err(format!("cannot get subroutine at non-existent label {label:?}")))?;

        Ok({
            self.sim.frame_stack.get_subroutine_def(addr)
                .map(|pl| PyParamListWrapper(pl.clone()))
        })
    }

    /// Sets the definiton of the subroutine located at the provided location.
    fn set_subroutine_def(&mut self, loc: MemLocation, pl: PyParamListWrapper) -> PyResult<()> {
        let addr = self.resolve_location(loc)
            .map_err(|label| PyValueError::new_err(format!("cannot define subroutine at non-existent label {label:?}")))?;

        self.sim.frame_stack.set_subroutine_def(addr, pl.0);
        Ok(())
    }

    /// Calls the subroutine located at the provided location, updating the PC and return address.
    fn call_subroutine(&mut self, loc: MemLocation) -> PyResult<()> {
        let addr = self.resolve_location(loc)
            .map_err(|label| PyValueError::new_err(format!("cannot call subroutine at non-existent label {label:?}")))?;

        self.sim.call_subroutine(addr)
            .map_err(|e| SimError::from_lc3_err(e, self.sim.prefetch_pc()))
    }

    /// Gets the total number of frames entered (the number of subroutine/trap calls we're currently deep in)
    #[getter]
    fn get_frame_number(&self) -> u64 {
        self.sim.frame_stack.len()
    }

    /// Gets a list of the current frames in the frame stack.
    #[getter]
    fn get_frames(&self) -> Option<Vec<PyFrame>> {
        let frames = self.sim.frame_stack.frames()?
            .iter()
            .cloned()
            .map(PyFrame)
            .collect();

        Some(frames)
    }

    /// Returns true if last execution hit HALT.
    fn hit_halt(&self) -> bool {
        self.sim.hit_halt()
    }
    /// Returns true if last execution hit a breakpoint.
    fn hit_breakpoint(&self) -> bool {
        self.sim.hit_breakpoint()
    }
}