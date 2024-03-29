use lc3_ensemble::{sim::mem::MemAccessCtx, asm::assemble_debug, ast::reg_consts::{R0, R1, R2, R3, R4, R5, R6, R7}, parse::parse_ast};
use pyo3::{exceptions::PyIndexError, prelude::*};
use std::path::Path;

#[pyclass]
struct Simulator {
    sim: lc3_ensemble::sim::Simulator,
}

#[pymethods]
impl Simulator {
    #[new]
    fn constructor() -> Self {
        Simulator { sim: lc3_ensemble::sim::Simulator::new() }
    }

    fn init(&mut self, file: String) -> PyResult<()> {
        self.sim = lc3_ensemble::sim::Simulator::new();
        
        let source_path = Path::new(&file);
        let src = std::fs::read_to_string(source_path)?;
        let ast = parse_ast(&src).unwrap();
        let obj = assemble_debug(ast, &src).unwrap();

        self.sim.load_obj_file(&obj);
        Ok(())
    }

    fn run(&mut self) -> PyResult<()> {
        self.sim.run().unwrap();
        Ok(())
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
        Ok(self.sim.mem.get(address, MemAccessCtx { privileged: true, strict: false }).unwrap().get())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ensemble_test(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Simulator>()?;
    //m.add_function(wrap_pyfunction!(load_source, m)?)?;
    Ok(())
}
