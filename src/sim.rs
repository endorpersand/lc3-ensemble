//! Simulating and execution for LC-3 assembly.
//! 
//! This module is focused on executing fully assembled code (i.e., [`ObjectFile`]).
//! 
//! This module consists of:
//! - [`Simulator`]: The struct that simulates assembled code.
//! - [`mem`]: The module handling memory relating to the registers.
//! - [`io`]: The module handling simulator IO.
//! - [`debug`]: The module handling types of breakpoints for the simulator.
//! - [`frame`]: The module handling the frame stack and call frame management.
//! 
//! # Usage
//! 
//! To simulate some code, you need to instantiate a Simulator and load an object file to it:
//! 
//! ```no_run
//! use lc3_ensemble::sim::Simulator;
//! 
//! # let obj_file = panic!("don't actually make an object file");
//! let mut simulator = Simulator::new(Default::default());
//! simulator.load_obj_file(&obj_file);
//! simulator.run().unwrap();
//! ```
//! 
//! ## Flags
//! 
//! Here, we define `simulator` to have the default flags. 
//! We could also configure the simulator by editing the flags. For example,
//! if we wish to enable real HALT, we can edit the flags like so:
//! 
//! ```no_run
//! # use lc3_ensemble::sim::{Simulator, SimFlags};
//! let mut simulator = Simulator::new(SimFlags { use_real_halt: true, ..Default::default() });
//! ```
//! 
//! All of the available flags can be found in [`SimFlags`].
//! 
//! ## Execution
//! 
//! Beyond the basic [`Simulator::run`] (which runs until halting),
//! there are also: 
//! - [`Simulator::step_in`], [`Simulator::step_out`], [`Simulator::step_over`]: manual step-by-step simulation
//! - [`Simulator::run_while`], [`Simulator::run_with_limit`]: more advanced programmatic execution
//! 
//! ```
//! use lc3_ensemble::parse::parse_ast;
//! use lc3_ensemble::asm::assemble;
//! use lc3_ensemble::sim::Simulator;
//! use lc3_ensemble::ast::reg_consts::R0;
//! 
//! let src = "
//!     .orig x3000
//!     AND R0, R0, #0
//!     ADD R0, R0, #1
//!     ADD R0, R0, #1
//!     ADD R0, R0, #1
//!     HALT
//!     .end
//! ";
//! let ast = parse_ast(src).unwrap();
//! let obj_file = assemble(ast).unwrap();
//! 
//! let mut sim = Simulator::new(Default::default());
//! sim.load_obj_file(&obj_file);
//! 
//! // Running step by step:
//! sim.step_in().unwrap();
//! assert_eq!(sim.reg_file[R0].get(), 0);
//! sim.step_in().unwrap();
//! assert_eq!(sim.reg_file[R0].get(), 1);
//! sim.step_in().unwrap();
//! assert_eq!(sim.reg_file[R0].get(), 2);
//! sim.step_in().unwrap();
//! assert_eq!(sim.reg_file[R0].get(), 3);
//! ```
//! 
//! ## Querying State
//! 
//! You can query (or set) a variety of different state values from the simulator.
//! 
//! - If you wish to access the PC, it can simply be done through the `sim.pc` field.
//! - If you wish to access the PSR or MCR, the [`Simulator::psr`] and [`Simulator::mcr`] methods are present to query those values.
//! - If you wish to access the register file, you can access it through the `sim.reg_file` field.
//! 
//! [`RegFile`] holds its values in a [`Word`], which is a memory cell which keeps track of initialization state.
//! Accessing can simply be done with [`Word::get`] and [`Word::set`]:
//! ```
//! use lc3_ensemble::sim::Simulator;
//! use lc3_ensemble::ast::reg_consts::R0;
//! 
//! let mut sim = Simulator::new(Default::default());
//! 
//! sim.reg_file[R0].set(0x1234);
//! assert_eq!(sim.reg_file[R0].get(), 0x1234);
//! ```
//! 
//! - If you wish to access the memory, [`Mem`] provides two pairs of memory access:
//!     - [`Mem::read`] and [`Mem::write`] are used for accesses which should trigger access violations and IO.
//!     - [`Mem::get_raw`] and [`Mem::get_raw_mut`] are used for accesses which directly access the memory.
//! ```
//! use lc3_ensemble::sim::Simulator;
//! 
//! let mut sim = Simulator::new(Default::default());
//! 
//! // Raw memory access:
//! sim.mem.get_raw_mut(0x3000).set(0x5678);
//! assert_eq!(sim.mem.get_raw(0x3000).get(), 0x5678);
//! 
//! // Through read/write:
//! use lc3_ensemble::sim::mem::Word;
//! 
//! assert!(sim.mem.write(0x0000, Word::new_init(0x9ABC), sim.default_mem_ctx()).is_err());
//! assert!(sim.mem.write(0x3000, Word::new_init(0x9ABC), sim.default_mem_ctx()).is_ok());
//! assert!(sim.mem.read(0x0000, sim.default_mem_ctx()).is_err());
//! assert!(sim.mem.read(0x3000, sim.default_mem_ctx()).is_ok());
//! ```
//! 
//! See more details in [`Mem`].
//! 
//! - Other state can be accessed. Consult the [`Simulator`] docs for more information.
//! 
//! ### Frames
//! 
//! The simulator also keeps track of subroutine frame information, accessible on the `frame_stack` field of [`Simulator`].
//! 
//! **If `debug_frames` is not enabled in [`SimFlags`]**, the only information the [`Simulator`] keeps track of
//! is the number of subroutine frames deep the simulator is (via [`FrameStack::len`]):
//! - During a JSR instruction, the frame count increases by 1.
//! - During a RET instruction, the frame count decreases by 1.
//! 
//! **If `debug_frames` is enabled in [`SimFlags`]**, the frame information is significantly extended.
//! The simulator then keeps track of several frame values (such as caller and callee address).
//! These are accessible via the [`FrameStack::frames`] method.
//! 
//! Debug frame information by default includes caller and callee addresses, but can be
//! configured to also include frame pointer and argument information. See the [`frame`]
//! module for details.
//! 
//! ## Debugging with breakpoints
//! 
//! Breakpoints are accessible through the `breakpoints` field on [`Simulator`].
//! 
//! To add a `breakpoint`, simply insert a [`Breakpoint`] and 
//! it will break if its condition is met during all execution functions (except [`Simulator::step_in`]).
//! 
//! ```
//! use lc3_ensemble::parse::parse_ast;
//! use lc3_ensemble::asm::assemble;
//! use lc3_ensemble::sim::Simulator;
//! use lc3_ensemble::sim::debug::Breakpoint;
//! 
//! let src = "
//!     .orig x3000
//!     ADD R0, R0, #0
//!     ADD R0, R0, #1
//!     ADD R0, R0, #2
//!     ADD R0, R0, #3
//!     HALT
//!     .end
//! ";
//! let ast = parse_ast(src).unwrap();
//! let obj_file = assemble(ast).unwrap();
//! 
//! let mut sim = Simulator::new(Default::default());
//! sim.load_obj_file(&obj_file);
//! 
//! // Without breakpoint
//! sim.run().unwrap();
//! assert_eq!(sim.pc, 0x3004);
//! 
//! // With breakpoint
//! sim.reset();
//! sim.load_obj_file(&obj_file);
//! sim.breakpoints.insert(Breakpoint::PC(0x3002));
//! sim.run().unwrap();
//! assert_eq!(sim.pc, 0x3002);
//! ```
//! 
//! ## IO
//! 
//! IO is handled by an [`IODevice`]. When a load or store to a memory-mapped address (0xFE00-0xFFFF) occurs,
//! the IODevice is queried for that data.
//! 
//! The best IO for programmatic uses is [`BufferedIO`], which exposes the IO to [`Vec`] and [`VecDeque`] buffers
//! that can be modified.
//! 
//! If you wish to use the standard I/O (i.e., `stdin`/`stdout`), [`BiChannelIO`] can be used.
//! 
//! ```
//! use lc3_ensemble::parse::parse_ast;
//! use lc3_ensemble::asm::assemble;
//! use lc3_ensemble::sim::Simulator;
//! use lc3_ensemble::sim::io::BufferedIO;
//! use std::sync::Arc;
//! 
//! let src = "
//!     .orig x3000
//!     LOOP:
//!     GETC
//!     PUTC
//!     ADD R0, R0, #0
//!     BRnp LOOP
//!     HALT
//!     .end
//! ";
//! let ast = parse_ast(src).unwrap();
//! let obj_file = assemble(ast).unwrap();
//! 
//! let mut sim = Simulator::new(Default::default());
//! sim.load_obj_file(&obj_file);
//! 
//! let io = BufferedIO::new();
//! let input = Arc::clone(io.get_input());
//! let output = Arc::clone(io.get_output());
//! sim.open_io(io);
//! 
//! input.write().unwrap().extend(b"Hello, World!\0");
//! sim.run().unwrap();
//! 
//! assert_eq!(&*input.read().unwrap(), b"");
//! assert_eq!(&**output.read().unwrap(), b"Hello, World!\0");
//! ```
//! 
//! ## Strictness (experimental)
//! 
//! This simulator also features uninitialized memory access checks (via the `strict` flag).
//! 
//! These strict memory checks verify that unintialized data is not written to the register files, memory, 
//! or other areas that do not expect uninitialized data. Uninitialized data here is defined as
//! data that is unknown as it was never fully set and is dependent on the values the machine was initialized with.
//! 
//! The `strict` flag can currently detect:
//! - Loads of uninitialized data into a register (excluding uninitialized reads from `mem[R6 + offset]`).
//! - Stores of uninitialized data into memory (excluding uninitialized stores to `mem[R6 + offset]` and `.blkw`'d memory).
//! - Stores of uninitialized data into memory-mapped IO
//! - Loads and stores through an uninitialized memory address
//! - Jumping to an uninitialized address (e.g., via `JSRR` or `JMP`)
//! - Jumping to a memory location that is uninitialized
//! - Decoding an instruction from uninitialized data
//! - Setting the PSR to an uninitialized value
//! 
//! Note that this is considered *experimental* as false positives can still occur.
//! Also note that the exceptions for loads and stores of uninitialized data
//! are present to prevent typical value manipulation on the stack or in stored memory
//! from triggering a strictness error.
//! 
//! ```
//! use lc3_ensemble::parse::parse_ast;
//! use lc3_ensemble::asm::assemble;
//! use lc3_ensemble::sim::{Simulator, SimErr};
//! 
//! let src = "
//!     .orig x3000
//!     ADD R0, R0, #0
//!     ADD R0, R0, #15 ;; R0 = 15
//!     HALT
//!     .end
//! ";
//! let ast = parse_ast(src).unwrap();
//! let obj_file = assemble(ast).unwrap();
//! 
//! let mut sim = Simulator::new(Default::default());
//! sim.load_obj_file(&obj_file);
//! sim.flags.strict = true;
//! 
//! // Strictness check detects `R0` was set without first being cleared.
//! assert!(matches!(sim.run(), Err(SimErr::StrictRegSetUninit)));
//! assert_eq!(sim.prefetch_pc(), 0x3000);
//! ```
//! 
//! [`VecDeque`]: std::collections::VecDeque
//! [`Breakpoint`]: self::debug::Breakpoint
pub mod mem;
pub mod io;
pub mod debug;
pub mod frame;

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::asm::ObjectFile;
use crate::ast::reg_consts::{R6, R7};
use crate::ast::sim::SimInstr;
use crate::ast::ImmOrReg;
use debug::Breakpoint;
use io::*;

use self::frame::{FrameStack, FrameType};
use self::mem::{Mem, MemAccessCtx, RegFile, Word, MachineInitStrategy};

/// Errors that can occur during simulation.
#[derive(Debug)]
pub enum SimErr {
    /// Word was decoded, but the opcode was invalid.
    IllegalOpcode,
    /// Word was decoded, and the opcode is recognized,
    /// but the instruction's format is invalid.
    InvalidInstrFormat,
    /// A privileged instruction was called in user mode.
    PrivilegeViolation,
    /// A supervisor region was accessed in user mode.
    AccessViolation,
    /// Interrupt raised.
    Interrupt(InterruptErr),
    /// A register was loaded with a partially uninitialized value.
    /// 
    /// This will ignore loads from the stack (R6), because it is convention to push registers 
    /// (including uninitialized registers).
    /// This also ignores loads from allocated (`.blkw`) memory in case the program writer
    /// uses those as register stores.

    // IDEA: So currently, the way this is implemented is that LDR Rx, R6, OFF is accepted regardless of initialization.
    // We could make this stricter by keeping track of how much is allocated on the stack.
    StrictRegSetUninit,
    /// Memory was loaded with a partially uninitialized value.
    /// 
    /// This will ignore loads from the stack (R6), because it is convention to push registers 
    /// (including uninitialized registers).
    /// This also ignores loads from allocated (`.blkw`) memory in case the program writer
    /// uses those as register stores.

    // IDEA: See StrictRegSetUninit.
    StrictMemSetUninit,
    /// Data was stored into MMIO with a partially uninitialized value.
    StrictIOSetUninit,
    /// Address to jump to is coming from an uninitialized value.
    StrictJmpAddrUninit,
    /// Address to jump to (which is a subroutine or trap call) is coming from an uninitialized value.
    StrictSRAddrUninit,
    /// Address to read from memory is coming from an uninitialized value.
    StrictMemAddrUninit,
    /// PC is pointing to an uninitialized value.
    StrictPCCurrUninit,
    /// PC was set to an address that has an uninitialized value and will read from it next cycle.
    StrictPCNextUninit,
    /// The PSR was loaded with a partially uninitialized value (by RTI).
    StrictPSRSetUninit,
}
impl std::fmt::Display for SimErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimErr::IllegalOpcode       => f.write_str("simulator executed illegal opcode"),
            SimErr::InvalidInstrFormat  => f.write_str("simulator executed invalid instruction"),
            SimErr::PrivilegeViolation  => f.write_str("privilege violation"),
            SimErr::AccessViolation     => f.write_str("access violation"),
            SimErr::Interrupt(e)        => write!(f, "unhandled interrupt: {e}"),
            SimErr::StrictRegSetUninit  => f.write_str("register was set to uninitialized value (strict mode)"),
            SimErr::StrictMemSetUninit  => f.write_str("tried to write an uninitialized value to memory (strict mode)"),
            SimErr::StrictIOSetUninit   => f.write_str("tried to write an uninitialized value to memory-mapped IO (strict mode)"),
            SimErr::StrictJmpAddrUninit => f.write_str("PC address was set to uninitialized address (strict mode)"),
            SimErr::StrictSRAddrUninit  => f.write_str("Subroutine starts at uninitialized address (strict mode)"),
            SimErr::StrictMemAddrUninit => f.write_str("tried to access memory with an uninitialized address (strict mode)"),
            SimErr::StrictPCCurrUninit  => f.write_str("PC is pointing to uninitialized value (strict mode)"),
            SimErr::StrictPCNextUninit  => f.write_str("PC will point to uninitialized value when this instruction executes (strict mode)"),
            SimErr::StrictPSRSetUninit  => f.write_str("tried to set the PSR to an uninitialized value (strict mode)"),
        }
    }
}
impl std::error::Error for SimErr {}

/// Anything that can cause a step to abruptly fail to finish.
enum StepBreak {
    /// A virtual halt was executed.
    Halt,
    /// A simulation error occurred.
    Err(SimErr),
}
impl From<SimErr> for StepBreak {
    fn from(value: SimErr) -> Self {
        Self::Err(value)
    }
}

/// An interrupt occurred.
/// 
/// See [`Simulator::add_external_interrupt`].
#[derive(Debug)]
pub struct InterruptErr(Box<dyn std::error::Error + Send + Sync + 'static>);
impl InterruptErr {
    /// Creates a new [`InterruptErr`].
    fn new(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        InterruptErr(Box::new(e))
    }

    /// Get the internal error from this interrupt.
    /// 
    /// This can be downcast by the typical methods on `dyn Error`.
    pub fn into_inner(self) -> Box<dyn std::error::Error + Send + Sync + 'static> {
        self.0
    }
}
impl std::fmt::Display for InterruptErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for InterruptErr {}
impl From<InterruptErr> for SimErr {
    fn from(value: InterruptErr) -> Self {
        SimErr::Interrupt(value)
    }
}
impl From<InterruptErr> for StepBreak {
    fn from(value: InterruptErr) -> Self {
        StepBreak::Err(value.into())
    }
}

#[allow(clippy::type_complexity)]
struct SimInterrupt(Box<dyn Fn(&Simulator) -> Result<(), InterruptErr> + Send + Sync + 'static>);
impl std::fmt::Debug for SimInterrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimInterrupt").finish_non_exhaustive()
    }
}

/// Reason for why execution paused if it wasn't due to an error.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
enum PauseCondition {
    /// Program reached a halt.
    Halt,
    /// Program set MCR to off.
    MCROff,
    /// Program hit a breakpoint.
    Breakpoint,
    /// Program hit a tripwire condition.
    Tripwire,
    /// Program hit an error and did not pause successfully.
    #[default]
    Unsuccessful
}

/// Configuration flags for [`Simulator`].
/// 
/// These can be modified after the `Simulator` is created with [`Simulator::new`]
/// and their effects should still apply.
/// 
/// Read the field descriptions for more details.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct SimFlags {
    /// Whether strict mode is enabled.
    /// 
    /// Strict mode adds additional integrity checks to the simulator,
    /// such as verifying initialization state is normal for provided data.
    pub strict: bool,

    /// Whether to use the real HALT trap.
    /// 
    /// There are two implementations of HALT within `Simulator`:
    /// - **virtual HALT**: On execution of `HALT` or `TRAP x25`, the simulator is automatically
    /// halted before executing any true TRAP routine.
    /// - **real HALT**: On execution of `HALT` or `TRAP x25`, the TRAP routine for HALT
    /// implemented in the OS is run and executed as usual.
    /// 
    /// Real HALT is useful for maintaining integrity to the LC-3 ISA, whereas
    /// virtual HALT preserves the state of the machine prior to calling the OS's HALT routine.
    pub use_real_halt: bool,
    
    /// The creation strategy for uninitialized Words.
    /// 
    /// This is used to initialize the `mem` and `reg_file` fields.
    pub machine_init: MachineInitStrategy,

    /// Whether to store debugging information about call frames.
    /// 
    /// This flag only goes into effect after a `Simulator::new` or `Simulator::reset` call.
    pub debug_frames: bool
}
impl Default for SimFlags {
    /// The default flags.
    /// 
    /// They are defined as follows:
    /// - `strict`: false
    /// - `use_real_halt`: false
    /// - `machine_init`: default [`MachineInitStrategy`]
    /// - `debug_frames`: false
    fn default() -> Self {
        Self {
            strict: false,
            use_real_halt: false,
            machine_init: Default::default(),
            debug_frames: false
        }
    }
}

/// Executes assembled code.
#[derive(Debug)]
pub struct Simulator {
    // ------------------ SIMULATION STATE ------------------
    // Calling [`Simulator::reset`] resets these values.

    /// The simulator's memory.
    /// 
    /// Note that this is held in the heap, as it is too large for the stack.
    pub mem: Mem,

    /// The simulator's register file.
    pub reg_file: RegFile,

    /// The program counter.
    pub pc: u16,

    /// The processor status register. See [`PSR`] for more details.
    psr: PSR,

    /// Saved stack pointer (the one currently not in use)
    saved_sp: Word,

    /// The frame stack.
    pub frame_stack: FrameStack,

    /// Allocated blocks in object file.
    /// 
    /// This field keeps track of "allocated" blocks 
    /// (memory written to by instructions or directives like .blkw)
    /// in the current object file.
    /// 
    /// Loading and storing uninitialized data in an allocated block
    /// does not cause strictness errors because we're assuming
    /// the programmer is using those as data stores.
    /// 
    /// This is technically a bit lax, because it lets them write
    /// into instructions but oops.
    alloca: Box<[(u16, u16)]>,

    /// The number of instructions successfully run since this `Simulator` was initialized.
    /// 
    /// This can be set to 0 to reset the counter.
    pub instructions_run: u64,

    /// Indicates whether the PC has been incremented in the fetch stage yet.
    /// 
    /// This is just for error handling purposes. It's used to compute
    /// the PC of the instruction that caused an error. See [`Simulator::prefetch_pc`].
    prefetch: bool,

    /// Indicates the reason why the last execution (via [`Simulator::run_while`] and adjacent)
    /// had paused.
    pause_condition: PauseCondition,

    /// Indicates whether the OS has been loaded.
    os_loaded: bool,

    // ------------------ CONFIG/DEBUG STATE ------------------
    // Calling [`Simulator::reset`] does not reset these values.

    /// Machine control.
    /// If unset, the program stops.
    /// 
    /// This is publicly accessible via a reference through [`Simulator::mcr`].
    mcr: Arc<AtomicBool>,

    /// Configuration settings for the simulator.
    /// 
    /// These are preserved between resets.
    /// 
    /// See [`SimFlags`] for more details on what configuration
    /// settings are available.
    pub flags: SimFlags,

    /// Breakpoints for the simulator.
    pub breakpoints: HashSet<Breakpoint>,

    /// Functions that are run every step that can pause execution of the `Simulator`.
    /// 
    /// When an [`InterruptErr`] is raised, the simulation will raise [`SimErr::Interrupt`],
    /// which can be used to handle the resulting `InterruptErr`.
    external_interrupts: Vec<SimInterrupt>,

}
impl Simulator where Simulator: Send + Sync {}

impl Simulator {
    /// Creates a new simulator with the provided initializers
    /// and with the OS loaded, but without a loaded object file.
    /// 
    /// This also allows providing an MCR atomic which is used by the Simulator.
    fn new_with_mcr(flags: SimFlags, mcr: Arc<AtomicBool>) -> Self {
        let mut filler = flags.machine_init.generator();

        let mut sim = Self {
            mem: Mem::new(&mut filler),
            reg_file: RegFile::new(&mut filler),
            pc: 0x3000,
            psr: PSR::new(),
            saved_sp: Word::new_init(0x3000),
            frame_stack: FrameStack::new(flags.debug_frames),
            alloca: Box::new([]),
            instructions_run: 0,
            prefetch: false,
            pause_condition: Default::default(),
            os_loaded: false,
            mcr,
            flags,
            breakpoints: Default::default(),
            external_interrupts: vec![],
        };

        sim.load_os();
        sim
    }

    /// Creates a new simulator with the provided initializers
    /// and with the OS loaded, but without a loaded object file.
    pub fn new(flags: SimFlags) -> Self {
        Self::new_with_mcr(flags, Arc::default())
    }

    /// Loads and initializes the operating system.
    /// 
    /// Note that this is done automatically with [`Simulator::new`].
    /// 
    /// This will initialize kernel space and create trap handlers,
    /// however it will not load working IO. This can cause IO
    /// traps such as `GETC` and `PUTC` to hang. The only trap that 
    /// is assured to function without IO is `HALT`.
    /// 
    /// To initialize the IO, use [`Simulator::open_io`].
    fn load_os(&mut self) {
        use crate::parse::parse_ast;
        use crate::asm::assemble;
        use std::sync::OnceLock;

        static OS_OBJ_FILE: OnceLock<ObjectFile> = OnceLock::new();
        
        if !self.os_loaded {
            self.mem.io.mcr = Arc::clone(&self.mcr); // load to allow HALT to function

            let obj = OS_OBJ_FILE.get_or_init(|| {
                let os_file = include_str!("os.asm");
                let ast = parse_ast(os_file).unwrap();
                assemble(ast).unwrap()
            });
    
            self.load_obj_file(obj);
            self.os_loaded = true;
        }
    }
    
    /// Resets the simulator.
    /// 
    /// This resets the state of the `Simulator` back to before any execution calls,
    /// while preserving configuration and debug state.
    /// 
    /// Note that this function preserves:
    /// - Flags
    /// - Breakpoints
    /// - External interrupts
    /// - MCR reference (i.e., anything with access to the Simulator's MCR can still control it)
    /// - IO (however, note that it does not reset IO state, which must be manually reset)
    /// 
    /// This also does not reload object files. Any object file data has to be reloaded into the Simulator.
    pub fn reset(&mut self) {
        let mcr = Arc::clone(&self.mcr);
        let flags = self.flags;
        let breakpoints = std::mem::take(&mut self.breakpoints);
        let external_ints = std::mem::take(&mut self.external_interrupts);
        let io = std::mem::take(&mut self.mem.io.inner);

        *self = Simulator::new_with_mcr(flags, mcr);
        self.breakpoints = breakpoints;
        self.external_interrupts = external_ints;
        self.mem.io.inner = io;
    }
    
    /// Sets and initializes the IO handler.
    pub fn open_io<IO: IODevice>(&mut self, io: IO) {
        self.mem.io.inner = io._to_sim_io(io::internals::ToSimIOToken(()));
    }

    /// Closes the IO handler, waiting for it to close.
    pub fn close_io(&mut self) {
        self.open_io(EmptyIO) // the illusion of choice
    }

    /// Loads an object file into this simulator.
    pub fn load_obj_file(&mut self, obj: &ObjectFile) {
        use std::cmp::Ordering;

        let mut alloca = vec![];

        for (start, words) in obj.iter() {
            self.mem.copy_obj_block(start, words);

            // add this block to alloca
            let len = words.len() as u16;
            let end = start.wrapping_add(len);

            match start.cmp(&end) {
                Ordering::Less    => alloca.push((start, len)),
                Ordering::Equal   => {},
                Ordering::Greater => {
                    // push (start..) and (0..end) as blocks
                    alloca.push((start, start.wrapping_neg()));
                    if end != 0 { alloca.push((0, end)) };
                },
            }
        }

        alloca.sort_by_key(|&(start, _)| start);
        self.alloca = alloca.into_boxed_slice();
    }

    /// Sets the condition codes using the provided result.
    fn set_cc(&mut self, result: u16) {
        match (result as i16).cmp(&0) {
            std::cmp::Ordering::Less    => self.psr.set_cc(0b100),
            std::cmp::Ordering::Equal   => self.psr.set_cc(0b010),
            std::cmp::Ordering::Greater => self.psr.set_cc(0b001),
        }
    }

    /// Gets a reference to the PSR.
    pub fn psr(&self) -> &PSR {
        // This is not mutable because editing the PSR can cause crashes to occur if
        // privilege is tampered with during an interrupt.
        &self.psr
    }

    /// Gets a reference to the MCR.
    pub fn mcr(&self) -> &Arc<AtomicBool> {
        // The mcr field is not exposed because that allows someone to swap the MCR
        // with another AtomicBool, which would cause the simulator's MCR
        // to be inconsistent with any other component's 
        &self.mcr
    }

    /// Sets the PC to the given address, raising any errors that occur.
    /// 
    /// The `st_check_mem` parameter indicates whether the data at the PC should be verified in strict mode.
    /// This should be enabled when it is absolutely known that the PC will read from the provided address
    /// on the next cycle.
    /// 
    /// This should be true when this function is used for instructions like `BR` and `JSR` 
    /// and should be false when this function is used to increment PC during fetch.
    fn set_pc(&mut self, addr_word: Word, st_check_mem: bool) -> Result<(), SimErr> {
        let addr = addr_word.get_if_init(self.flags.strict, SimErr::StrictJmpAddrUninit)?;
        if self.flags.strict && st_check_mem {
            // Check next memory value is initialized:
            if !self.mem.read(addr, self.default_mem_ctx())?.is_init() {
                return Err(SimErr::StrictPCNextUninit);
            }
        }
        self.pc = addr;
        Ok(())
    }
    /// Adds an offset to the PC.
    /// 
    /// See [`Simulator::set_pc`] for details about `st_check_mem`.
    fn offset_pc(&mut self, offset: i16, st_check_mem: bool) -> Result<(), SimErr> {
        self.set_pc(Word::from(self.pc.wrapping_add_signed(offset)), st_check_mem)
    }
    /// Gets the value of the prefetch PC.
    /// 
    /// This function is useful as it returns the location of the currently
    /// executing instruction in memory.
    pub fn prefetch_pc(&self) -> u16 {
        self.pc - (!self.prefetch) as u16
    }

    /// Checks whether the address points to a memory location that was allocated
    /// in the currently loaded object file.
    fn in_alloca(&self, addr: u16) -> bool {
        let first_post = self.alloca.partition_point(|&(start, _)| start <= addr);
        if first_post == 0 { return false };
        
        // This is the last block where start <= addr.
        let (start, len) = self.alloca[first_post - 1];

        // We must also check that addr < end.
        // If start + len is None, that means end is greater than all possible lengths.
        match start.checked_add(len) {
            Some(e) => addr < e,
            None    => true
        }
    }

    /// Indicates whether the last execution of the simulator hit a breakpoint.
    pub fn hit_breakpoint(&self) -> bool {
        matches!(self.pause_condition, PauseCondition::Breakpoint)
    }

    /// Indicates whether the last execution of the simulator resulted in a HALT successfully occurring.
    /// 
    /// This is defined as:
    /// - `HALT` being executed while virtual HALTs are enabled
    /// - `MCR` being set to `x0000` during the execution of the program.
    pub fn hit_halt(&self) -> bool {
        matches!(self.pause_condition, PauseCondition::Halt | PauseCondition::MCROff)
    }

    /// Computes the default memory access context, 
    /// which are the default flags to use (see [`Mem::read`] and [`Mem::write`]).
    pub fn default_mem_ctx(&self) -> MemAccessCtx {
        MemAccessCtx { privileged: self.psr.privileged(), strict: self.flags.strict }
    }

    /// Calls a subroutine.
    /// 
    /// This does all the steps for calling a subroutine, namely:
    /// - Setting the PC to the subroutine's start address
    /// - Setting R7 to the original PC (return address)
    /// - Adding information to the frame stack
    pub fn call_subroutine(&mut self, addr: u16) -> Result<(), SimErr> {
        self.reg_file[R7].set(self.pc);
        self.frame_stack.push_frame(self.prefetch_pc(), addr, FrameType::Subroutine, &self.reg_file, &self.mem);
        self.set_pc(Word::new_init(addr), true)
    }

    /// Calls a trap or interrupt, adding information to the frame stack
    /// and setting the PC to the start of the trap/interrupt handler.
    /// 
    /// `0x00-0xFF` represents a trap,
    /// `0x100-0x1FF` represents an interrupt.
    fn call_interrupt(&mut self, vect: u16, ft: FrameType) -> Result<(), SimErr> {
        let addr = self.mem.read(vect, self.default_mem_ctx())?
            .get_if_init(self.flags.strict, SimErr::StrictSRAddrUninit)?;

        self.frame_stack.push_frame(self.prefetch_pc(), vect, ft, &self.reg_file, &self.mem);
        self.set_pc(Word::new_init(addr), true)
    }
    /// Interrupt, trap, and exception handler.
    /// 
    /// If priority is none, this will unconditionally initialize the trap or exception handler.
    /// If priority is not none, this will run the interrupt handler only if the interrupt's priority
    /// is greater than the PSR's priority.
    /// 
    /// The address provided is the address into the jump table (either the trap or interrupt vector ones).
    /// This function will always jump to `mem[vect]` at the end of this function.
    fn handle_interrupt(&mut self, vect: u16, priority: Option<u8>) -> Result<(), StepBreak> {
        if priority.is_some_and(|prio| prio <= self.psr.priority()) { return Ok(()) };
        
        // Virtual HALT
        if !self.flags.use_real_halt && vect == 0x25 {
            self.offset_pc(-1, false)?; // decrement PC so that execution goes back here
            return Err(StepBreak::Halt)
        };
        
        if !self.psr.privileged() {
            std::mem::swap(&mut self.saved_sp, &mut self.reg_file[R6]);
        }

        // Push PSR, PC to supervisor stack
        let old_psr = self.psr.0;
        let old_pc = self.pc;
        
        self.psr.set_privileged(true);
        let mctx = self.default_mem_ctx();

        // push PSR and PC to stack
        let sp = self.reg_file[R6]
            .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;

        self.reg_file[R6] -= 2u16;
        self.mem.write(sp.wrapping_sub(1), Word::new_init(old_psr), mctx)?;
        self.mem.write(sp.wrapping_sub(2), Word::new_init(old_pc), mctx)?;
        
        // set interrupt priority
        if let Some(prio) = priority {
            self.psr.set_priority(prio);
        }

        let ft = match priority.is_some() {
            true => FrameType::Interrupt,
            false => FrameType::Trap,
        };
        
        self.call_interrupt(vect, ft)
            .map_err(Into::into)
    }

    /// Registers an "external interrupt" to the simulator which is run every step.
    /// 
    /// An "external interrupt" is a function that can pause execution of the `Simulator`,
    /// which is not necessarily handled by the Simulator's OS.
    /// 
    /// When an error is raised by an external interrupt, the simulation will raise [`SimErr::Interrupt`],
    /// which can be used to handle the resulting `InterruptErr`.
    /// 
    /// One example where this is used is in Python bindings. 
    /// In that case, we want to be able to halt the Simulator on a `KeyboardInterrupt`.
    /// However, by default, Python cannot signal to the Rust library that a `KeyboardInterrupt`
    /// has occurred. Thus, we can add a signal handler as an external interrupt to allow the `KeyboardInterrupt`
    /// to be handled properly.
    pub fn add_external_interrupt<E, F>(&mut self, interrupt: F) 
        where E: std::error::Error + Send + Sync + 'static,
              F: Fn(&Self) -> Result<(), E> + Send + Sync + 'static
    {
        self.external_interrupts.push(SimInterrupt(Box::new({
            move |this| interrupt(this).map_err(InterruptErr::new)
        })))
    }

    /// Clears any external interrupts.
    pub fn clear_external_interrupts(&mut self) {
        self.external_interrupts.clear()
    }

    /// Runs until the tripwire condition returns false (or any of the typical breaks occur).
    /// 
    /// The typical break conditions are:
    /// - `HALT` is executed
    /// - the MCR is set to false
    /// - A breakpoint matches
    pub fn run_while(&mut self, mut tripwire: impl FnMut(&mut Simulator) -> bool) -> Result<(), SimErr> {
        use std::sync::atomic::Ordering;

        std::mem::take(&mut self.pause_condition);
        self.mcr.store(true, Ordering::Relaxed);

        // event loop
        // run until:
        // 1. the MCR is set to false
        // 2. the tripwire condition returns false
        // 3. any of the breakpoints are hit
        let result = loop {
            // MCR turned off:
            if !self.mcr.load(Ordering::Relaxed) {
                break Ok(PauseCondition::MCROff);
            }
            // Tripwire turned off:
            if !tripwire(self) {
                break Ok(PauseCondition::Tripwire);
            }
            
            // Run a step:
            match self.step() {
                Ok(_) => {},
                Err(StepBreak::Halt) => break Ok(PauseCondition::Halt),
                Err(StepBreak::Err(e)) => break Err(e)
            }

            // After executing, check that any breakpoints were hit.
            if self.breakpoints.iter().any(|bp| bp.check(self)) {
                break Ok(PauseCondition::Breakpoint);
            }
        };
    
        self.mcr.store(false, Ordering::Release);
        self.pause_condition = result?;
        Ok(())
    }

    /// Execute the program.
    /// 
    /// This blocks until the program ends. 
    /// If you would like to limit the maximum number of steps to execute, consider [`Simulator::run_with_limit`].
    pub fn run(&mut self) -> Result<(), SimErr> {
        self.run_while(|_| true)
    }

    /// Execute the program with a limit on how many steps to execute.
    /// 
    /// This blocks until the program ends or until the number of steps to execute has been hit.
    pub fn run_with_limit(&mut self, max_steps: u64) -> Result<(), SimErr> {
        let i = self.instructions_run;
        self.run_while(|sim| sim.instructions_run.wrapping_sub(i) < max_steps)
    }
    
    /// Simulate one step, executing one instruction.
    /// 
    /// This function is a library function and should be used when one step is needed.
    /// The difference between this function and [`Simulator::step_in`] is that this
    /// function can return [`SimErr::ProgramHalted`] as an error,
    /// whereas `step_in` will ignore `ProgramHalted` errors.
    fn step(&mut self) -> Result<(), StepBreak> {
        self.prefetch = true;

        // Call all external interrupts:
        for SimInterrupt(cb) in &self.external_interrupts {
            cb(self)?;
        }

        let word = self.mem.read(self.pc, self.default_mem_ctx())?
            .get_if_init(self.flags.strict, SimErr::StrictPCCurrUninit)?;
        let instr = SimInstr::decode(word)?;

        self.offset_pc(1, false)?;
        self.prefetch = false;

        match instr {
            SimInstr::BR(cc, off)  => {
                if cc & self.psr.cc() != 0 {
                    self.offset_pc(off.get(), true)?;
                }
            },
            SimInstr::ADD(dr, sr1, sr2) => {
                let val1 = self.reg_file[sr1];
                let val2 = match sr2 {
                    ImmOrReg::Imm(i2) => Word::from(i2.get()),
                    ImmOrReg::Reg(r2) => self.reg_file[r2],
                };

                let result = val1 + val2;
                self.reg_file[dr].set_if_init(result, self.flags.strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(result.get());
            },
            SimInstr::LD(dr, off) => {
                let ea = self.pc.wrapping_add_signed(off.get());
                let write_strict = self.flags.strict && !self.in_alloca(ea);

                let val = self.mem.read(ea, self.default_mem_ctx())?;
                self.reg_file[dr].set_if_init(val, write_strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(val.get());
            },
            SimInstr::ST(sr, off) => {
                let ea = self.pc.wrapping_add_signed(off.get());
                let write_ctx = MemAccessCtx {
                    strict: self.flags.strict && !self.in_alloca(ea),
                    ..self.default_mem_ctx()
                };

                let val = self.reg_file[sr];
                self.mem.write(ea, val, write_ctx)?;
            },
            SimInstr::JSR(op) => {
                // Note: JSRR R7 jumps to address at R7, then sets PC to R7.
                // Refer to: https://github.com/gt-cs2110/lc3tools/commit/fa9a23f62106eeee9fef7d2a278ba989356c9ee2

                let addr = match op {
                    ImmOrReg::Imm(off) => Word::from(self.pc.wrapping_add_signed(off.get())),
                    ImmOrReg::Reg(br)  => self.reg_file[br],
                }.get_if_init(self.flags.strict, SimErr::StrictSRAddrUninit)?;

                self.call_subroutine(addr)?;
            },
            SimInstr::AND(dr, sr1, sr2) => {
                let val1 = self.reg_file[sr1];
                let val2 = match sr2 {
                    ImmOrReg::Imm(i2) => Word::from(i2.get()),
                    ImmOrReg::Reg(r2) => self.reg_file[r2],
                };

                let result = val1 & val2;
                self.reg_file[dr].set_if_init(result, self.flags.strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(result.get());
            },
            SimInstr::LDR(dr, br, off) => {
                let ea = self.reg_file[br]
                    .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?
                    .wrapping_add_signed(off.get());
                let write_strict = self.flags.strict && br != R6 && !self.in_alloca(ea);
                
                let val = self.mem.read(ea, self.default_mem_ctx())?;
                self.reg_file[dr].set_if_init(val, write_strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(val.get());
            },
            SimInstr::STR(sr, br, off) => {
                let ea = self.reg_file[br]
                    .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?
                    .wrapping_add_signed(off.get());
                let write_ctx = MemAccessCtx {
                    strict: self.flags.strict && br != R6 && !self.in_alloca(ea),
                    ..self.default_mem_ctx()
                };
                
                let val = self.reg_file[sr];
                self.mem.write(ea, val, write_ctx)?;
            },
            SimInstr::RTI => {
                if self.psr.privileged() {
                    let mctx = self.default_mem_ctx();

                    // Pop PC and PSR from the stack
                    let sp = self.reg_file[R6]
                        .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;

                    let pc = self.mem.read(sp, mctx)?
                        .get_if_init(self.flags.strict, SimErr::StrictJmpAddrUninit)?;
                    let psr = self.mem.read(sp.wrapping_add(1), mctx)?
                        .get_if_init(self.flags.strict, SimErr::StrictPSRSetUninit)?;
                    self.reg_file[R6] += 2u16;

                    self.set_pc(Word::new_init(pc), true)?;
                    self.psr = PSR(psr);

                    if !self.psr.privileged() {
                        std::mem::swap(&mut self.saved_sp, &mut self.reg_file[R6]);
                    }

                    self.frame_stack.pop_frame();
                } else {
                    return Err(SimErr::PrivilegeViolation.into());
                }
            },
            SimInstr::NOT(dr, sr) => {
                let val = self.reg_file[sr];
                
                let result = !val;
                self.reg_file[dr].set_if_init(result, self.flags.strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(result.get());
            },
            SimInstr::LDI(dr, off) => {
                let shifted_pc = self.pc.wrapping_add_signed(off.get());
                let ea = self.mem.read(shifted_pc, self.default_mem_ctx())?
                    .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;
                let write_strict = self.flags.strict && !self.in_alloca(ea);

                let val = self.mem.read(ea, self.default_mem_ctx())?;
                self.reg_file[dr].set_if_init(val, write_strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(val.get());
            },
            SimInstr::STI(sr, off) => {
                let shifted_pc = self.pc.wrapping_add_signed(off.get());
                let ea = self.mem.read(shifted_pc, self.default_mem_ctx())?
                    .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;
                let write_ctx = MemAccessCtx {
                    strict: self.flags.strict && !self.in_alloca(ea),
                    ..self.default_mem_ctx()
                };

                let val = self.reg_file[sr];
                self.mem.write(ea, val, write_ctx)?;
            },
            SimInstr::JMP(br) => {
                let addr = self.reg_file[br];
                self.set_pc(addr, true)?;
                
                // if this is RET,
                // we must also handle frame information:
                if br.reg_no() == 7 {
                    self.frame_stack.pop_frame();
                }
            },
            SimInstr::LEA(dr, off) => {
                let ea = self.pc.wrapping_add_signed(off.get());
                self.reg_file[dr].set(ea);
            },
            SimInstr::TRAP(vect) => {
                self.handle_interrupt(vect.get(), None)?;
            },
        }

        self.instructions_run = self.instructions_run.wrapping_add(1);
        Ok(())
    }

    /// Simulate one step, executing one instruction.
    pub fn step_in(&mut self) -> Result<(), SimErr> {
        match self.step() {
            Ok(()) => Ok(()),
            Err(StepBreak::Halt) => Ok(()),
            Err(StepBreak::Err(e)) => Err(e)
        }
    }

    /// Simulate one step, executing one instruction and running through entire subroutines as a single step.
    pub fn step_over(&mut self) -> Result<(), SimErr> {
        let curr_frame = self.frame_stack.len();
        let mut first = Some(()); // is Some if this is the first instruction executed in this call

        // this function should do at least one step before checking its condition
        // condition: run until we have landed back in the same frame
        self.run_while(|sim| first.take().is_some() || curr_frame < sim.frame_stack.len())
    }

    /// Run through the simulator's execution until the subroutine is exited.
    pub fn step_out(&mut self) -> Result<(), SimErr> {
        let curr_frame = self.frame_stack.len();
        let mut first = Some(()); // is Some if this is the first instruction executed in this call
        
        // this function should do at least one step before checking its condition
        // condition: run until we've landed in a smaller frame
        if curr_frame != 0 {
            self.run_while(|sim| first.take().is_some() || curr_frame <= sim.frame_stack.len())?;
        }

        Ok(())
    }
}
impl Default for Simulator {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

/// A wrapper over `u16` in order to faciliate the PSR.
/// 
/// The word is encoded as the following:
/// - `PSR[15..16]`: Privilege mode (0 = supervisor, 1 = user)
/// - `PSR[8..11]`:  Interrupt priority
/// - `PSR[0..3]`:   Condition codes
/// 
/// ```text
///         privilege
///         |     interrupt priority
///         |     |         condition codes
///         |     |         |
///         V     V         V
/// 0x8002: 1000 0000 0000 0010
///         ~     ~~~       ~~~
/// ```
/// 
/// Each of these are exposed as the [`PSR::privileged`], [`PSR::priority`], and [`PSR::cc`] values.
#[allow(clippy::upper_case_acronyms)]
#[repr(transparent)]
pub struct PSR(pub u16);

impl PSR {
    /// Creates a PSR with a default value (user mode, `z` condition code).
    pub fn new() -> Self {
        PSR(0x8002)
    }

    /// Checks whether the simulator is in privileged mode.
    /// - `true` = supervisor mode
    /// - `false` = user mode
    pub fn privileged(&self) -> bool {
        (self.0 >> 15) == 0
    }
    /// Checks the current interrupt priority of the simulator.
    pub fn priority(&self) -> u8 {
        ((self.0 >> 8) & 0b111) as u8
    }
    /// Checks the condition code of the simulator.
    pub fn cc(&self) -> u8 {
        (self.0 & 0b111) as u8
    }
    /// Checks the condition code of the simulator is `n`.
    pub fn is_n(&self) -> bool {
        self.cc() & 0b100 != 0
    }
    /// Checks the condition code of the simulator is `z`.
    pub fn is_z(&self) -> bool {
        self.cc() & 0b010 != 0
    }
    /// Checks the condition code of the simulator is `p`.
    pub fn is_p(&self) -> bool {
        self.cc() & 0b001 != 0
    }
    /// Sets whether the simulator is in privileged mode.
    pub fn set_privileged(&mut self, privl: bool) {
        self.0 &= 0x7FFF;
        self.0 |= (!privl as u16) << 15;
    }
    /// Sets the current interrupt priority of the simulator.
    pub fn set_priority(&mut self, prio: u8) {
        self.0 &= 0xF8FF;
        self.0 |= ((prio & 0b111) as u16) << 8;
    }
    /// Sets the condition code of the simulator.
    pub fn set_cc(&mut self, cc: u8) {
        self.0 &= 0xFFF8;
        self.0 |= (cc & 0b111) as u16;
    }
    /// Sets the condition code of the simulator to `n`.
    pub fn set_cc_n(&mut self) {
        self.set_cc(0b100)
    }
    /// Sets the condition code of the simulator to `z`.
    pub fn set_cc_z(&mut self) {
        self.set_cc(0b010)
    }
    /// Sets the condition code of the simulator to `p`.
    pub fn set_cc_p(&mut self) {
        self.set_cc(0b001)
    }
}
impl Default for PSR {
    fn default() -> Self {
        Self::new()
    }
}
impl std::fmt::Debug for PSR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        struct CC(u8);

        impl std::fmt::Debug for CC {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if self.0 & 0b100 != 0 { f.write_char('N')?; };
                if self.0 & 0b010 != 0 { f.write_char('Z')?; };
                if self.0 & 0b001 != 0 { f.write_char('P')?; };
                Ok(())
            }
        }

        f.debug_struct("PSR")
            .field("privileged", &self.privileged())
            .field("priority", &self.priority())
            .field("cc", &CC(self.cc()))
            .finish()
    }
}