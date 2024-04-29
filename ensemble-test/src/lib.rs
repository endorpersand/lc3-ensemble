use lc3_ensemble::asm::assemble_debug;
use lc3_ensemble::ast::reg_consts::{R0, R1, R2, R3, R4, R5, R6, R7};
use lc3_ensemble::parse::parse_ast;
use lc3_ensemble::sim::Simulator;
use pyo3::{create_exception, prelude::*};
use pyo3::exceptions::{PyIndexError, PyValueError};

/// A LC-3 simulator and unit tester, backed by [`lc3-ensemble`].
#[pymodule]
fn ensemble_test(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    m.add("LoadError", py.get_type_bound::<LoadError>())?;
    m.add("SimError", py.get_type_bound::<SimError>())?;
    
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

#[pyclass(name="Simulator")]
struct PySimulator {
    sim: Simulator,
}

#[pymethods]
impl PySimulator {
    #[new]
    fn constructor() -> Self {
        PySimulator { sim: Simulator::new(Default::default()) }
    }

    fn init(&mut self, src_fp: &str) -> PyResult<()> {
        self.sim = Simulator::new(Default::default());
        
        let src = std::fs::read_to_string(src_fp)?;
        let ast = parse_ast(&src)
            .map_err(LoadError::from_lc3_err)?;
        let obj = assemble_debug(ast, &src)
            .map_err(LoadError::from_lc3_err)?;

        self.sim.load_obj_file(&obj);
        Ok(())
    }

    fn run(&mut self) -> PyResult<()> {
        self.sim.run()
            .map_err(|e| SimError::new_err(format!("{e} (PC: 0x{})", self.sim.prefetch_pc())))
    }

    fn get_reg(&self, reg: i32) -> PyResult<u16> {
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

    fn get_memory(&mut self, address: u16) -> PyResult<u16>{
        Ok(self.sim.mem.get_raw(address).get())
    }
}