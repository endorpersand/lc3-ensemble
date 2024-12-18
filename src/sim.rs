//! Simulating and execution for LC-3 assembly.
//! 
//! This module is focused on executing fully assembled code (i.e., [`ObjectFile`]).
//! 
//! This module consists of:
//! - [`Simulator`]: The struct that simulates assembled code.
//! - [`mem`]: The module handling memory relating to the registers.
//! - [`device`]: The module handling simulator IO, interrupts, and general handling of external devices.
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
//! if we wish to enable real traps, we can edit the flags like so:
//! 
//! ```no_run
//! # use lc3_ensemble::sim::{Simulator, SimFlags};
//! let mut simulator = Simulator::new(SimFlags { use_real_traps: true, ..Default::default() });
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
//! use lc3_ensemble::ast::Reg::R0;
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
//! use lc3_ensemble::ast::Reg::R0;
//! 
//! let mut sim = Simulator::new(Default::default());
//! 
//! sim.reg_file[R0].set(0x1234);
//! assert_eq!(sim.reg_file[R0].get(), 0x1234);
//! ```
//! 
//! - If you wish to access the memory, the simulator provides two pairs of memory access:
//!     - Direct access to the memory array (via the `mem`) field, which does not trigger access violations or IO.
//!     - [`Simulator::read_mem`] and [`Simulator::write_mem`], which are used for accesses which do trigger access violations and IO.
//! ```
//! use lc3_ensemble::sim::Simulator;
//! 
//! let mut sim = Simulator::new(Default::default());
//! 
//! // Raw memory access:
//! sim.mem[0x3000].set(0x5678);
//! assert_eq!(sim.mem[0x3000].get(), 0x5678);
//! 
//! // Through read/write:
//! use lc3_ensemble::sim::mem::Word;
//! 
//! assert!(sim.write_mem(0x0000, Word::new_init(0x9ABC), sim.default_mem_ctx()).is_err());
//! assert!(sim.write_mem(0x3000, Word::new_init(0x9ABC), sim.default_mem_ctx()).is_ok());
//! assert!(sim.read_mem(0x0000, sim.default_mem_ctx()).is_err());
//! assert!(sim.read_mem(0x3000, sim.default_mem_ctx()).is_ok());
//! ```
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
//! ## IO, interrupts, and external devices
//! 
//! IO and interrupts are handled by "external devices" (the trait [`ExternalDevice`]).
//! 
//! These can be added by registering the device in the Simulator's device handler 
//! ([`DeviceHandler::add_device`] of the `device_handler` field).
//! 
//! When a load or store to a memory-mapped address (0xFE00-0xFFFF) occurs,
//! the device handler sends the corresponding load/store to the device for it to handle.
//! 
//! The best IO for programmatic uses is [`device::BufferedKeyboard`] and [`device::BufferedDisplay`],
//! which exposes the IO to memory buffers that can be modified.
//! 
//! ```
//! use lc3_ensemble::parse::parse_ast;
//! use lc3_ensemble::asm::assemble;
//! use lc3_ensemble::sim::Simulator;
//! use lc3_ensemble::sim::device::{BufferedKeyboard, BufferedDisplay};
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
//! let input = BufferedKeyboard::default();
//! let output = BufferedDisplay::default();
//! sim.device_handler.set_keyboard(input.clone());
//! sim.device_handler.set_display(output.clone());
//! 
//! input.get_buffer().write().unwrap().extend(b"Hello, World!\0");
//! sim.run().unwrap();
//! 
//! assert_eq!(&*input.get_buffer().read().unwrap(), b"");
//! assert_eq!(&**output.get_buffer().read().unwrap(), b"Hello, World!\0");
//! ```
//! 
//! These external devices also support interrupt-based IO.
//! 
//! If the [`device::BufferedKeyboard`] device is enabled, interrupts can be enabled by setting `KBSR[14]`.
//! Once enabled, the keyboard can interrupt the simulator and run its interrupt service routine.
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
pub mod debug;
pub mod frame;
pub mod device;
mod observer;

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::asm::ObjectFile;
use crate::ast::Reg::{R6, R7};
use crate::ast::sim::SimInstr;
use crate::ast::ImmOrReg;
use debug::Breakpoint;
use device::{DeviceHandler, ExternalDevice, ExternalInterrupt};

use self::frame::{FrameStack, FrameType};
use self::mem::{MemArray, RegFile, Word, MachineInitStrategy};

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
    /// Object file contained unresolved external symbols.
    UnresolvedExternal(String),
    /// Interrupt raised.
    Interrupt(ExternalInterrupt),
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
            SimErr::IllegalOpcode         => f.write_str("simulator executed illegal opcode"),
            SimErr::InvalidInstrFormat    => f.write_str("simulator executed invalid instruction"),
            SimErr::PrivilegeViolation    => f.write_str("privilege violation"),
            SimErr::AccessViolation       => f.write_str("access violation"),
            SimErr::UnresolvedExternal(s) => write!(f, "unresolved external label {s} in object file"),
            SimErr::Interrupt(e)          => write!(f, "unhandled interrupt: {e}"),
            SimErr::StrictRegSetUninit    => f.write_str("register was set to uninitialized value (strict mode)"),
            SimErr::StrictMemSetUninit    => f.write_str("tried to write an uninitialized value to memory (strict mode)"),
            SimErr::StrictIOSetUninit     => f.write_str("tried to write an uninitialized value to memory-mapped IO (strict mode)"),
            SimErr::StrictJmpAddrUninit   => f.write_str("PC address was set to uninitialized address (strict mode)"),
            SimErr::StrictSRAddrUninit    => f.write_str("Subroutine starts at uninitialized address (strict mode)"),
            SimErr::StrictMemAddrUninit   => f.write_str("tried to access memory with an uninitialized address (strict mode)"),
            SimErr::StrictPCCurrUninit    => f.write_str("PC is pointing to uninitialized value (strict mode)"),
            SimErr::StrictPCNextUninit    => f.write_str("PC will point to uninitialized value when this instruction executes (strict mode)"),
            SimErr::StrictPSRSetUninit    => f.write_str("tried to set the PSR to an uninitialized value (strict mode)"),
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

macro_rules! int_vect {
    ($Type:ident, {$($name:ident = $value:literal), +}) => {
        enum $Type {
            $($name = $value),+
        }
        impl TryFrom<u16> for $Type {
            type Error = ();

            fn try_from(value: u16) -> Result<Self, Self::Error> {
                match value {
                    $($value => Ok(Self::$name)),+,
                    _ => Err(())
                }
            }
        }
    }
}
int_vect!(RealIntVect, {
    Halt = 0x25,
    PrivilegeViolation = 0x100,
    IllegalOpcode = 0x101,
    AccessViolation = 0x102
});


/// The OS object file with symbols.
/// 
/// Do not rely on this existing.
#[doc(hidden)]
#[allow(non_snake_case)]
pub fn _os_obj_file() -> &'static ObjectFile {
    // This is public because LC3Tools UI needs it;
    // however, I don't think there's any particular other reason
    // that a developer would need this, so it's #[doc(hidden)].
    use crate::parse::parse_ast;
    use crate::asm::assemble_debug;
    use std::sync::OnceLock;

    static OS_OBJ_FILE: OnceLock<ObjectFile> = OnceLock::new();
    
    OS_OBJ_FILE.get_or_init(|| {
        let os_file = include_str!("os.asm");
        let ast = parse_ast(os_file).unwrap();
        assemble_debug(ast, os_file).unwrap()
    })
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
    /// 
    /// By default, this flag is `false`.
    pub strict: bool,

    /// Whether to use emulated version of certain traps.
    /// 
    /// Certain traps and exceptions have two separate implementations within `Simulator`, namely:
    /// - `HALT` or `TRAP x25`
    /// - Privilege mode exception
    /// - Illegal opcode exception
    /// - Access violation exception
    /// 
    /// This flag allows us to configure between the two implementations:
    /// - **virtual** (`false`): On execution of one of these interrupts, the simulator breaks
    ///     and prints its own error.
    /// - **real** (`true`): On execution of one of these interrupts, the simulator delegates
    ///     the error to the machine's OS and continues through the OS.
    /// 
    /// Activating real traps is useful for maintaining integrity to the LC-3 ISA, whereas
    /// virtual HALT preserves the state of the machine prior to calling the interrupt routines
    /// and can provide slightly more helpful error messages.
    /// 
    /// By default, this flag is `false`.
    pub use_real_traps: bool,
    
    /// The creation strategy for uninitialized Words.
    /// 
    /// This is used to initialize the `mem` and `reg_file` fields.
    /// 
    /// By default, this flag is [`MachineInitStrategy::default`].
    pub machine_init: MachineInitStrategy,

    /// Whether to store debugging information about call frames.
    /// 
    /// This flag only goes into effect after a `Simulator::new` or `Simulator::reset` call.
    /// 
    /// By default, this flag is `false`.
    pub debug_frames: bool,

    /// If true, privilege checks are ignored and the simulator runs as though
    /// the executor has supervisor level privilege.
    /// 
    /// By default, this flag is `false`.
    pub ignore_privilege: bool
}

#[allow(clippy::derivable_impls)]
impl Default for SimFlags {
    fn default() -> Self {
        Self {
            strict: false,
            use_real_traps: false,
            machine_init: Default::default(),
            debug_frames: false,
            ignore_privilege: false
        }
    }
}

const USER_START: u16 = 0x3000;
const IO_START: u16 = 0xFE00;
const PSR_ADDR: u16 = 0xFFFC;
const MCR_ADDR: u16 = 0xFFFE;

/// Context behind a memory access.
/// 
/// This struct is used by [`Simulator::read_mem`] and [`Simulator::write_mem`] to perform checks against memory accesses.
/// A default memory access context for the given simulator can be constructed with [`Simulator::default_mem_ctx`].
#[derive(Clone, Copy, Debug)]
pub struct MemAccessCtx {
    /// Whether this access is privileged (false = user, true = supervisor).
    pub privileged: bool,

    /// Whether writes to memory should follow strict rules 
    /// (no writing partially or fully uninitialized data).
    /// 
    /// This does not affect [`Simulator::read_mem`].
    pub strict: bool,

    /// Whether a read to memory-mapped IO should cause side effects.
    /// 
    /// This can be set to false to observe the value of IO.
    /// 
    /// This does not affect [`Simulator::write_mem`].
    pub io_effects: bool
}
impl MemAccessCtx {
    /// Allows any access and allows access to (effectless) IO.
    /// 
    /// Useful for reading state.
    pub fn omnipotent() -> Self {
        MemAccessCtx { privileged: true, strict: false, io_effects: false }
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
    pub mem: MemArray,

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

    /// Tracks changes in simulator state.
    pub observer: observer::ChangeObserver,

    /// Indicates whether the OS has been loaded.
    os_loaded: bool,

    // ------------------ CONFIG/DEBUG STATE ------------------
    // Calling [`Simulator::reset`] does not reset these values.

    /// Machine control.
    /// If unset, the program stops.
    /// 
    /// This is publicly accessible via a reference through [`Simulator::mcr`].
    mcr: MCR,

    /// Configuration settings for the simulator.
    /// 
    /// These are preserved between resets.
    /// 
    /// See [`SimFlags`] for more details on what configuration
    /// settings are available.
    pub flags: SimFlags,

    /// Breakpoints for the simulator.
    pub breakpoints: HashSet<Breakpoint>,

    /// All external devices connected to the system (IO and interrupting devices).
    pub device_handler: DeviceHandler

}
impl Simulator where Simulator: Send + Sync {}

impl Simulator {
    /// Creates a new simulator with the provided initializers
    /// and with the OS loaded, but without a loaded object file.
    /// 
    /// This also allows providing an MCR atomic which is used by the Simulator.
    fn new_with_mcr(flags: SimFlags, mcr: MCR) -> Self {
        let mut filler = flags.machine_init.generator();

        let mut sim = Self {
            mem: MemArray::new(&mut filler),
            reg_file: RegFile::new(&mut filler),
            pc: 0x3000,
            psr: PSR::new(),
            saved_sp: Word::new_init(0x3000),
            frame_stack: FrameStack::new(flags.debug_frames),
            alloca: Box::new([]),
            instructions_run: 0,
            prefetch: false,
            pause_condition: Default::default(),
            observer: Default::default(),
            os_loaded: false,

            mcr,
            flags,
            breakpoints: Default::default(),
            device_handler: Default::default()
        };

        sim.mem.as_slice_mut()[IO_START as usize..].fill(Word::new_init(0)); // clear IO section
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
        if !self.os_loaded {
            self.load_obj_file(_os_obj_file())
                .unwrap_or_else(|_| unreachable!("OS object file should not have externals"));
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
        let dev_handler = std::mem::take(&mut self.device_handler);

        *self = Simulator::new_with_mcr(flags, mcr);
        self.breakpoints = breakpoints;
        self.device_handler = dev_handler;
        self.device_handler.io_reset();
    }
    
    /// Fallibly reads the word at the provided index, erroring if not possible.
    /// 
    /// This accepts a [`MemAccessCtx`], that describes the parameters of the memory access.
    /// The simulator provides a default [`MemAccessCtx`] under [`Simulator::default_mem_ctx`].
    /// 
    /// The flags are used as follows:
    /// - `privileged`: if false, this access errors if the address is a memory location outside of the user range.
    /// - `strict`: not used for `read`
    /// 
    /// Note that this method is used for simulating a read to memory-mapped IO. 
    /// If you would like to query the memory's state, consider using `index` on [`MemArray`].
    pub fn read_mem(&mut self, addr: u16, ctx: MemAccessCtx) -> Result<Word, SimErr> {
        use std::sync::atomic::Ordering;

        if !ctx.privileged && !(USER_START..IO_START).contains(&addr) { return Err(SimErr::AccessViolation) };

        // Apply read to IO and write to mem array:
        match addr {
            // Supervisor range
            0..USER_START => { /* Non-IO read */ },
            // User range
            USER_START..IO_START => { /* Non-IO read */ },
            // IO range
            PSR_ADDR => self.mem[addr].set(self.psr.get()),
            MCR_ADDR => self.mem[addr].set(u16::from(self.mcr.load(Ordering::Relaxed)) << 15),
            IO_START.. => {
                if let Some(data) = self.device_handler.io_read(addr, ctx.io_effects) {
                    self.mem[addr].set(data);
                }
            }
        }

        // Load from mem array:
        Ok(self.mem[addr])
    }

    /// Fallibly writes the word at the provided index, erroring if not possible.
    /// 
    /// This accepts a [`MemAccessCtx`], that describes the parameters of the memory access.
    /// The simulator provides a default [`MemAccessCtx`] under [`Simulator::default_mem_ctx`].
    /// 
    /// The flags are used as follows:
    /// - `privileged`: if false, this access errors if the address is a memory location outside of the user range.
    /// - `strict`: If true, all accesses that would cause a memory location to be set with uninitialized data causes an error.
    /// 
    /// Note that this method is used for simulating a write to memory-mapped IO. 
    /// If you would like to edit the memory's state, consider using `index_mut` on [`MemArray`].
    pub fn write_mem(&mut self, addr: u16, data: Word, ctx: MemAccessCtx) -> Result<(), SimErr> {
        use std::sync::atomic::Ordering;

        if !ctx.privileged && !(USER_START..IO_START).contains(&addr) { return Err(SimErr::AccessViolation) };
        
        // Apply write to IO:
        let success = match addr {
            // Supervisor range (non-IO write)
            0..USER_START => true,
            // User range (non-IO write)
            USER_START..IO_START => true,
            // IO range
            PSR_ADDR => {
                let io_data = data.get_if_init(ctx.strict, SimErr::StrictIOSetUninit)?;
                self.psr.set(io_data);
                true
            },
            MCR_ADDR => {
                let io_data = data.get_if_init(ctx.strict, SimErr::StrictIOSetUninit)?;
                self.mcr.store((io_data as i16) < 0, Ordering::Relaxed);
                true
            },
            IO_START.. => {
                let io_data = data.get_if_init(ctx.strict, SimErr::StrictIOSetUninit)?;
                self.device_handler.io_write(addr, io_data)
            }
        };

        // Duplicate write in mem array:
        if success {
            if self.mem[addr] != data {
                self.observer.set_mem_changed(addr);
            }
            self.mem[addr]
                .set_if_init(data, ctx.strict, SimErr::StrictMemSetUninit)?;
        }

        Ok(())
    }

    /// Loads an object file into this simulator.
    pub fn load_obj_file(&mut self, obj: &ObjectFile) -> Result<(), SimErr> {
        use std::cmp::Ordering;

        // Reject any object files with external symbols.
        if let Some(ext) = obj.get_external_symbol() {
            return Err(SimErr::UnresolvedExternal(ext.to_string()));
        }

        let mut alloca = vec![];

        for (start, words) in obj.block_iter() {
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

        Ok(())
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
    pub fn mcr(&self) -> &MCR {
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
            if !self.read_mem(addr, self.default_mem_ctx())?.is_init() {
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
    /// which are the default flags to use (see [`Simulator::read_mem`] and [`Simulator::write_mem`]).
    pub fn default_mem_ctx(&self) -> MemAccessCtx {
        MemAccessCtx {
            privileged: self.psr.privileged() || self.flags.ignore_privilege,
            strict: self.flags.strict,
            io_effects: true
        }
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
        let addr = self.read_mem(vect, self.default_mem_ctx())?
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
        
        // Virtual traps.
        // See the flag for documentation.
        // Virtual HALT
        if !self.flags.use_real_traps {
            if let Ok(intv) = RealIntVect::try_from(vect) {
                if !self.prefetch {
                    // decrement PC so that if play is pressed again, it goes back here
                    self.offset_pc(-1, false)?;
                    self.prefetch = true;
                }
                let break_value = match intv {
                    RealIntVect::Halt => StepBreak::Halt,
                    RealIntVect::PrivilegeViolation => StepBreak::Err(SimErr::PrivilegeViolation),
                    RealIntVect::IllegalOpcode => StepBreak::Err(SimErr::IllegalOpcode),
                    RealIntVect::AccessViolation => StepBreak::Err(SimErr::AccessViolation),
                };
                return Err(break_value);
            }
        };
        
        if !self.psr.privileged() {
            std::mem::swap(&mut self.saved_sp, &mut self.reg_file[R6]);
        }

        // Push PSR, PC to supervisor stack
        let old_psr = self.psr.get();
        let old_pc = self.pc;
        
        self.psr.set_privileged(true);
        let mctx = self.default_mem_ctx();

        // push PSR and PC to stack
        let sp = self.reg_file[R6]
            .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;

        self.reg_file[R6] -= 2u16;
        self.write_mem(sp.wrapping_sub(1), Word::new_init(old_psr), mctx)?;
        self.write_mem(sp.wrapping_sub(2), Word::new_init(old_pc), mctx)?;
        
        // set PSR to z
        self.psr.set_cc_z();

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

    /// Runs until the tripwire condition returns false (or any of the typical breaks occur).
    /// 
    /// The typical break conditions are:
    /// - `HALT` is executed
    /// - the MCR is set to false
    /// - A breakpoint matches
    pub fn run_while(&mut self, mut tripwire: impl FnMut(&mut Simulator) -> bool) -> Result<(), SimErr> {
        use std::sync::atomic::Ordering;

        self.observer.clear();
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
    
        self.mcr.store(false, Ordering::Relaxed);
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
    /// Unlike [`Simulator::step`], this function does not handle the `use_real_traps` flag.
    /// Both of these functions are not meant for general stepping use. That should be done
    /// with [`Simulator::step_in`].
    fn _step_inner(&mut self) -> Result<(), StepBreak> {
        self.prefetch = true;

        if let Some(int) = self.device_handler.poll_interrupt() {
            match int.kind {
                // If priority passes, handle interrupt then skip FETCH:
                device::InterruptKind::Vectored { vect, priority } if priority > self.psr().priority() => {
                    return self.handle_interrupt(0x100 + u16::from(vect), Some(priority));
                },
                // If priority does not pass, move to FETCH:
                device::InterruptKind::Vectored { .. } => Ok(()),

                // External interrupt.
                device::InterruptKind::External(int) => Err(StepBreak::Err(SimErr::Interrupt(int))),
            }?;
        }

        let word = self.read_mem(self.pc, self.default_mem_ctx())?
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

                let val = self.read_mem(ea, self.default_mem_ctx())?;
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
                self.write_mem(ea, val, write_ctx)?;
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
                
                let val = self.read_mem(ea, self.default_mem_ctx())?;
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
                self.write_mem(ea, val, write_ctx)?;
            },
            SimInstr::RTI => {
                if self.psr.privileged() || self.flags.ignore_privilege {
                    let mctx = self.default_mem_ctx();

                    // Pop PC and PSR from the stack
                    let sp = self.reg_file[R6]
                        .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;

                    let pc = self.read_mem(sp, mctx)?
                        .get_if_init(self.flags.strict, SimErr::StrictJmpAddrUninit)?;
                    let psr = self.read_mem(sp.wrapping_add(1), mctx)?
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
                let ea = self.read_mem(shifted_pc, self.default_mem_ctx())?
                    .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;
                let write_strict = self.flags.strict && !self.in_alloca(ea);

                let val = self.read_mem(ea, self.default_mem_ctx())?;
                self.reg_file[dr].set_if_init(val, write_strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(val.get());
            },
            SimInstr::STI(sr, off) => {
                let shifted_pc = self.pc.wrapping_add_signed(off.get());
                let ea = self.read_mem(shifted_pc, self.default_mem_ctx())?
                    .get_if_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;
                let write_ctx = MemAccessCtx {
                    strict: self.flags.strict && !self.in_alloca(ea),
                    ..self.default_mem_ctx()
                };

                let val = self.reg_file[sr];
                self.write_mem(ea, val, write_ctx)?;
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
    ///
    /// This function properly handles the `use_real_traps` flag.
    /// 
    /// This function is a library function and should be used when one step is needed.
    /// The difference between this function and [`Simulator::step_in`] is that this
    /// function can return [`StepBreak::Halt`] as an error,
    /// whereas `step_in` will ignore that error.
    fn step(&mut self) -> Result<(), StepBreak> {
        match self._step_inner() {
            // Virtual traps don't need to go through handle_interrupt logic
            s if !self.flags.use_real_traps => s,
            // Real traps!
            Err(StepBreak::Halt) => self.handle_interrupt(RealIntVect::Halt as u16, None),
            Err(StepBreak::Err(SimErr::PrivilegeViolation)) => self.handle_interrupt(RealIntVect::PrivilegeViolation as u16, None),
            Err(StepBreak::Err(SimErr::IllegalOpcode)) => self.handle_interrupt(RealIntVect::IllegalOpcode as u16, None),
            Err(StepBreak::Err(SimErr::InvalidInstrFormat)) => self.handle_interrupt(RealIntVect::IllegalOpcode as u16, None),
            Err(StepBreak::Err(SimErr::AccessViolation)) => self.handle_interrupt(RealIntVect::AccessViolation as u16, None),
            s => s
        }
    }
    /// Simulate one step, executing one instruction.
    pub fn step_in(&mut self) -> Result<(), SimErr> {
        self.observer.clear();
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
pub struct PSR(u16);

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

    /// Gets the bit-representation of the PSR.
    pub fn get(&self) -> u16 {
        self.0
    }
    /// Sets the PSR to the provided data value.
    pub fn set(&mut self, data: u16) {
        const MASK: u16 = 0b1000_0111_0000_0111;
        
        self.0 = data & MASK;
        self.set_cc((data & 0b111) as u8);
    }
    /// Sets whether the simulator is in privileged mode.
    pub fn set_privileged(&mut self, privl: bool) {
        self.0 &= 0x7FFF;
        self.0 |= u16::from(!privl) << 15;
    }
    /// Sets the current interrupt priority of the simulator.
    pub fn set_priority(&mut self, prio: u8) {
        self.0 &= 0xF8FF;
        self.0 |= u16::from(prio & 0b111) << 8;
    }
    /// Sets the condition code of the simulator.
    pub fn set_cc(&mut self, mut cc: u8) {
        self.0 &= 0xFFF8;

        // Guard from invalid CC.
        cc &= 0b111;
        if cc.count_ones() != 1 { cc = 0b010 };
        self.0 |= u16::from(cc);
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

/// A type alias for MCR.
pub type MCR = Arc<AtomicBool>;
