//! A LC-3 parser, assembler, and simulator.
//! 
//! This is meant to be a general suite to use LC-3 assembly 
//! (meant as a backend for Georgia Tech's CS 2110 LC3Tools).
//!
//! # Usage
//! 
//! To convert LC-3 source code to an object file, it must be parsed and assembled:
//! ```
//! use lc3_ensemble::parse::parse_ast;
//! use lc3_ensemble::asm::{assemble, assemble_debug, ObjectFile};
//! 
//! let code = "
//!     .orig x3000
//!     AND R0, R0, #0
//!     AND R0, R0, #7
//!     HALT
//!     .end
//! ";
//! let ast = parse_ast(code).unwrap();
//! 
//! // Assemble AST into object file:
//! # {
//! # let ast = ast.clone();
//! let obj_file: ObjectFile = assemble(ast).unwrap();
//! # }
//! // OR:
//! let obj_file: ObjectFile = assemble_debug(ast, code).unwrap();
//! ```
//! 
//! Once an object file has been created, it can be executed with the simulator:
//! ```
//! # // Parsing and assembling was shown in the previous example, so this doesn't need to be shown again.
//! # use lc3_ensemble::parse::parse_ast;
//! # use lc3_ensemble::asm::{assemble_debug};
//! # 
//! # let code = ".orig x3000\nHALT\n.end";
//! # let ast = parse_ast(code).unwrap();
//! # let obj_file = assemble_debug(ast, code).unwrap();
//! # 
//! use lc3_ensemble::sim::Simulator;
//! 
//! let mut simulator = Simulator::new(Default::default());
//! simulator.load_obj_file(&obj_file);
//! simulator.run().unwrap(); // <-- Result can be handled accordingly
//! ```
//! 
//! If more granularity is needed for simulation, there are also step-in and step-out functions. 
//! See the [`sim`] module for more details.
#![warn(missing_docs)]

pub mod parse;
pub mod ast;
pub mod asm;
pub mod sim;
pub mod err;
