//! Simulating and execution for LC-3 assembly.
//! 
//! This module is focused on executing fully assembled code (i.e., [`ObjectFile`]).
//! 
//! This module consists of:
//! - [`Simulator`]: The struct that simulates assembled code.
//! - [`mem`]: The module handling memory relating to the registers.
//! - [`io`]: The module handling simulator IO.
//! - [`debug`]: The module handling types of breakpoints for the simulator.

pub mod mem;
pub mod io;
pub mod debug;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::asm::ObjectFile;
use crate::ast::reg_consts::{R6, R7};
use crate::ast::sim::SimInstr;
use crate::ast::ImmOrReg;
use io::*;

use self::debug::Breakpoint;
use self::mem::{AssertInit as _, Mem, MemAccessCtx, RegFile, Word};

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
    /// Not an error, but HALT!
    ProgramHalted,
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
            SimErr::ProgramHalted       => f.write_str("program halted"),
            SimErr::StrictRegSetUninit  => f.write_str("register was set to uninitialized value (strict mode)"),
            SimErr::StrictMemSetUninit  => f.write_str("tried to write an uninitialized value to memory (strict mode)"),
            SimErr::StrictIOSetUninit   => f.write_str("tried to write an uninitialized value to memory-mapped IO (strict mode)"),
            SimErr::StrictJmpAddrUninit => f.write_str("PC address was set to uninitialized value (strict mode)"),
            SimErr::StrictMemAddrUninit => f.write_str("tried to access memory with an uninitialized address (strict mode)"),
            SimErr::StrictPCCurrUninit  => f.write_str("PC is pointing to uninitialized value (strict mode)"),
            SimErr::StrictPCNextUninit  => f.write_str("PC will point to uninitialized value when this instruction executes (strict mode)"),
            SimErr::StrictPSRSetUninit  => f.write_str("tried to set the PSR to an uninitialized value (strict mode)"),
        }
    }
}
impl std::error::Error for SimErr {}

/// Strategy used to initialize the `reg_file` and `mem` of the [`Simulator`].
/// 
/// These are used to set the initial state of the memory and registers,
/// which will be treated as uninitialized until they are properly initialized
/// by program code.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum WordCreateStrategy {
    /// Initializes each word randomly and non-deterministically.
    #[default]
    Unseeded,

    /// Initializes each word randomly and deterministically.
    Seeded {
        /// The seed the RNG was initialized with.
        seed: u64
    },

    /// Generates a random value with the given seed and sets each word to that generated value.
    SeededFill {
        /// The seed the RNG was initialized with.
        seed: u64
    },

    /// Initializes each word to a known value.
    Known {
        /// The value to initialize each value to.
        value: u16
    }
}

impl WordCreateStrategy {
    fn generator(&self) -> WCGenerator {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        match self {
            WordCreateStrategy::Unseeded => WCGenerator::Unseeded,
            WordCreateStrategy::Seeded { seed } => WCGenerator::Seeded(Box::new(StdRng::seed_from_u64(*seed))),
            WordCreateStrategy::SeededFill { seed } => WCGenerator::Known(StdRng::seed_from_u64(*seed).gen()),
            WordCreateStrategy::Known { value } => WCGenerator::Known(*value),
        }
    }
}

enum WCGenerator {
    Unseeded,
    Seeded(Box<rand::rngs::StdRng>),
    Known(u16)
}
impl mem::WordFiller for WCGenerator {
    fn generate(&mut self) -> u16 {
        match self {
            WCGenerator::Unseeded  => ().generate(),
            WCGenerator::Seeded(r) => r.generate(),
            WCGenerator::Known(k)  => k.generate(),
        }
    }
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
    pub word_create_strat: WordCreateStrategy
}
impl Default for SimFlags {
    /// The default flags.
    /// 
    /// They are defined as follows:
    /// - `strict`: false
    /// - `use_real_halt`: false
    fn default() -> Self {
        Self { strict: false, use_real_halt: false, word_create_strat: Default::default() }
    }
}

/// Executes assembled code.
#[derive(Debug)]
pub struct Simulator {
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

    /// The number of subroutines or trap calls we've entered.
    /// 
    /// This is incremented when JSR/JSRR/TRAP is called,
    /// and decremented when RET/JMP R7/RTI is called.
    /// 
    /// If this is 0, this is considered the global state,
    /// and as such, decrementing should have no effect.
    /// 
    /// I am hoping this is large enough that it doesn't overflow :)
    sr_entered: u64,

    /// Configuration settings for the simulator.
    /// 
    /// These are preserved between resets.
    /// 
    /// See [`SimFlags`] for more details on what configuration
    /// settings are available.
    pub flags: SimFlags,

    /// Machine control.
    /// If unset, the program stops.
    /// 
    /// This is publicly accessible via a reference through [`Simulator::mcr`].
    mcr: Arc<AtomicBool>,

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


    /// Breakpoints for the simulator.
    pub breakpoints: Vec<Breakpoint>,

    /// The number of instructions successfully run since this `Simulator` was initialized.
    /// 
    /// This can be set to 0 to reset the counter.
    pub instructions_run: u64,

    /// Indicates whether the PC has been incremented in the fetch stage yet.
    /// 
    /// This is just for error handling purposes. It's used to compute
    /// the PC of the instruction that caused an error. See [`Simulator::prefetch_pc`].
    prefetch: bool,

    /// Indicates whether the last execution hit a breakpoint.
    hit_breakpoint: bool,

    /// Indicates whether the OS has been loaded.
    os_loaded: bool
}
impl Simulator where Simulator: Send + Sync {}

impl Simulator {
    /// Creates a new simulator with the provided initializers
    /// and with the OS loaded, but without a loaded object file.
    pub fn new(flags: SimFlags) -> Self {
        let mut filler = flags.word_create_strat.generator();

        let mut sim = Self {
            mem: Mem::new(&mut filler),
            reg_file: RegFile::new(&mut filler),
            pc: 0x3000,
            psr: PSR::new(),
            saved_sp: Word::new_init(0x3000),
            sr_entered: 0,
            flags,
            alloca: Box::new([]),
            mcr: Arc::default(),
            breakpoints: vec![],
            instructions_run: 0,
            prefetch: false,
            hit_breakpoint: false,
            os_loaded: false
        };

        sim.load_os();
        sim
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
    pub fn load_os(&mut self) {
        use crate::parse::parse_ast;
        use crate::asm::assemble;
        use std::sync::OnceLock;

        static OS_OBJ_FILE: OnceLock<ObjectFile> = OnceLock::new();
        
        if !self.os_loaded {
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
    /// This sets the simulator back to the state it was in when [`Simulator::new`] was called.
    /// Essentially, this resets all state fields back to their defaults,
    /// and the memory and register file back to their original initialization values 
    /// (unless the creation strategy used to initialize this [`Simulator`] was [`WordCreateStrategy::Unseeded`]).
    pub fn reset(&mut self) {
        // FIXME: Do these need to be preserved:
        // Breakpoints
        // IO state

        let flags = self.flags;
        *self = Simulator::new(flags);
    }
    
    /// Sets and initializes the IO handler.
    pub fn open_io<IO: Into<SimIO>>(&mut self, io: IO) {
        let io = std::mem::replace(&mut self.mem.io, io.into());
        io.close()
    }

    /// Closes the IO handler, waiting for it to close.
    pub fn close_io(&mut self) {
        self.open_io(EmptyIO) // the illusion of choice
    }
    
    /// Loads an object file into this simulator.
    pub fn load_obj_file(&mut self, obj: &ObjectFile) {
        use std::cmp::Ordering;

        let mut alloca = Vec::with_capacity(obj.len());

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
    pub fn set_pc(&mut self, addr_word: Word, st_check_mem: bool) -> Result<(), SimErr> {
        let addr = addr_word.assert_init(self.flags.strict, SimErr::StrictJmpAddrUninit)?.get();
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
    pub fn offset_pc(&mut self, offset: i16, st_check_mem: bool) -> Result<(), SimErr> {
        self.set_pc(Word::from(self.pc.wrapping_add_signed(offset)), st_check_mem)
    }
    /// Gets the value of the prefetch PC.
    /// 
    /// This function returns the value of PC before it is incremented druing fetch,
    /// which is also the location of the currently executing instruction in memory.
    /// 
    /// This is useful for pointing to a given memory location in error handling,
    /// as this computation always points to the memory location of the instruction.
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
        self.hit_breakpoint
    }

    /// Computes the default memory access context, 
    /// which are the default flags to use (see [`Mem::read`] and [`Mem::write`]).
    pub fn default_mem_ctx(&self) -> MemAccessCtx {
        MemAccessCtx { privileged: self.psr.privileged(), strict: self.flags.strict }
    }

    /// Interrupt, trap, and exception handler.
    /// 
    /// If priority is none, this will unconditionally initialize the trap or exception handler.
    /// If priority is not none, this will run the interrupt handler only if the interrupt's priority
    /// is greater than the PSR's priority.
    /// 
    /// The address provided is the address into the jump table (either the trap or interrupt vector ones).
    /// This function will always jump to `mem[vect]` at the end of this function.
    fn handle_interrupt(&mut self, vect: u16, priority: Option<u8>) -> Result<(), SimErr> {
        if priority.is_some_and(|prio| prio <= self.psr.priority()) { return Ok(()) };
        
        // Virtual HALT
        if !self.flags.use_real_halt && vect == 0x25 {
            self.offset_pc(-1, false)?; // decrement PC so that execution goes back here
            return Err(SimErr::ProgramHalted)
        };
        
        if !self.psr.privileged() {
            std::mem::swap(&mut self.saved_sp, &mut self.reg_file[R6]);
        }

        // Push PSR, PC to supervisor stack
        let old_psr = self.psr.0;
        let old_pc = self.pc;
        
        self.psr.set_privileged(true);
        let mctx = self.default_mem_ctx();
        let sp = &mut self.reg_file[R6];

        *sp -= 1u16;
        self.mem.write(sp.get(), Word::new_init(old_psr), mctx)?;
        *sp -= 1u16;
        self.mem.write(sp.get(), Word::new_init(old_pc), mctx)?;
        
        // set interrupt priority
        if let Some(prio) = priority {
            self.psr.set_priority(prio);
        }

        self.sr_entered += 1;
        let addr = self.mem.read(vect, self.default_mem_ctx())?;
        self.set_pc(addr, true)
    }

    /// Runs until the tripwire condition returns false (or any of the typical breaks occur)
    fn run_while(&mut self, mut tripwire: impl FnMut(&mut Simulator) -> bool) -> Result<(), SimErr> {
        use std::sync::atomic::Ordering;

        self.hit_breakpoint = false;
        self.mcr.store(true, Ordering::Relaxed);
        // event loop
        // run until:
        // 1. the MCR is set to 0
        // 2. the tripwire condition returns false
        // 3. any of the breakpoints are hit
        while self.mcr.load(Ordering::Relaxed) && tripwire(self) {
            match self.step() {
                Ok(_) => {},
                Err(SimErr::ProgramHalted) => break,
                Err(e) => {
                    self.mcr.store(false, Ordering::Release);
                    return Err(e);
                }
            }

            // After executing, check that any breakpoints were hit.
            if self.breakpoints.iter().any(|bp| bp.check(self)) {
                self.hit_breakpoint = true;
                break;
            }
        }
    
        self.mcr.store(false, Ordering::Release);
        Ok(())
    }

    /// Execute the program.
    pub fn run(&mut self) -> Result<(), SimErr> {
        self.run_while(|_| true)
    }

    /// Execute the program with a limit on how many steps to execute.
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
    fn step(&mut self) -> Result<(), SimErr> {
        self.prefetch = true;
        let word = self.mem.read(self.pc, self.default_mem_ctx())?
            .assert_init(self.flags.strict, SimErr::StrictPCCurrUninit)?
            .get();
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
                self.reg_file[dr].copy_word(result, self.flags.strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(result.get());
            },
            SimInstr::LD(dr, off) => {
                let ea = self.pc.wrapping_add_signed(off.get());
                let write_strict = self.flags.strict && !self.in_alloca(ea);

                let val = self.mem.read(ea, self.default_mem_ctx())?;
                self.reg_file[dr].copy_word(val, write_strict, SimErr::StrictRegSetUninit)?;
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
                self.reg_file[R7].set(self.pc);

                let addr = match op {
                    ImmOrReg::Imm(off) => Word::from(self.pc.wrapping_add_signed(off.get())),
                    ImmOrReg::Reg(br)  => self.reg_file[br],
                };
                self.sr_entered += 1;
                self.set_pc(addr, true)?;
            },
            SimInstr::AND(dr, sr1, sr2) => {
                let val1 = self.reg_file[sr1];
                let val2 = match sr2 {
                    ImmOrReg::Imm(i2) => Word::from(i2.get()),
                    ImmOrReg::Reg(r2) => self.reg_file[r2],
                };

                let result = val1 & val2;
                self.reg_file[dr].copy_word(result, self.flags.strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(result.get());
            },
            SimInstr::LDR(dr, br, off) => {
                let ea = self.reg_file[br]
                    .assert_init(self.flags.strict, SimErr::StrictMemAddrUninit)?
                    .get()
                    .wrapping_add_signed(off.get());
                let write_strict = self.flags.strict && br != R6 && !self.in_alloca(ea);
                
                let val = self.mem.read(ea, self.default_mem_ctx())?;
                self.reg_file[dr].copy_word(val, write_strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(val.get());
            },
            SimInstr::STR(sr, br, off) => {
                let ea = self.reg_file[br]
                    .assert_init(self.flags.strict, SimErr::StrictMemAddrUninit)?
                    .get()
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
                    let sp = (&mut self.reg_file[R6])
                        .assert_init(self.flags.strict, SimErr::StrictMemAddrUninit)?;

                    let pc = self.mem.read(sp.get(), mctx)?
                        .assert_init(self.flags.strict, SimErr::StrictJmpAddrUninit)?
                        .get();
                    *sp += 1u16;
                    let psr = self.mem.read(sp.get(), mctx)?
                        .assert_init(self.flags.strict, SimErr::StrictPSRSetUninit)?
                        .get();
                    *sp += 1u16;

                    self.pc = pc;
                    self.psr = PSR(psr);

                    if !self.psr.privileged() {
                        std::mem::swap(&mut self.saved_sp, &mut self.reg_file[R6]);
                    }

                    self.sr_entered = self.sr_entered.saturating_sub(1);
                } else {
                    return Err(SimErr::PrivilegeViolation);
                }
            },
            SimInstr::NOT(dr, sr) => {
                let val = self.reg_file[sr];
                
                let result = !val;
                self.reg_file[dr].copy_word(result, self.flags.strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(result.get());
            },
            SimInstr::LDI(dr, off) => {
                let shifted_pc = self.pc.wrapping_add_signed(off.get());
                let ea = self.mem.read(shifted_pc, self.default_mem_ctx())?
                    .assert_init(self.flags.strict, SimErr::StrictMemAddrUninit)?
                    .get();
                let write_strict = self.flags.strict && !self.in_alloca(ea);

                let val = self.mem.read(ea, self.default_mem_ctx())?;
                self.reg_file[dr].copy_word(val, write_strict, SimErr::StrictRegSetUninit)?;
                self.set_cc(val.get());
            },
            SimInstr::STI(sr, off) => {
                let shifted_pc = self.pc.wrapping_add_signed(off.get());
                let ea = self.mem.read(shifted_pc, self.default_mem_ctx())?
                    .assert_init(self.flags.strict, SimErr::StrictMemAddrUninit)?
                    .get();
                let write_ctx = MemAccessCtx {
                    strict: self.flags.strict && !self.in_alloca(ea),
                    ..self.default_mem_ctx()
                };

                let val = self.reg_file[sr];
                self.mem.write(ea, val, write_ctx)?;
            },
            SimInstr::JMP(br) => {
                // check for RET
                if br.0 == 7 {
                    self.sr_entered = self.sr_entered.saturating_sub(1);
                }
                let addr = self.reg_file[br];
                self.set_pc(addr, true)?;
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
            Err(SimErr::ProgramHalted) => Ok(()),
            r => r
        }
    }

    /// Simulate one step, executing one instruction and running through entire subroutines as a single step.
    pub fn step_over(&mut self) -> Result<(), SimErr> {
        let curr_frame = self.sr_entered;
        let mut first = Some(()); // is Some if this is the first instruction executed in this call

        // this function should do at least one step before checking its condition
        // condition: run until we have landed back in the same frame
        self.run_while(|sim| first.take().is_some() || curr_frame < sim.sr_entered)
    }

    /// Run through the simulator's execution until the subroutine is exited.
    pub fn step_out(&mut self) -> Result<(), SimErr> {
        let curr_frame = self.sr_entered;
        let mut first = Some(()); // is Some if this is the first instruction executed in this call
        
        // this function should do at least one step before checking its condition
        // condition: run until we've landed in a smaller frame
        if curr_frame != 0 {
            self.run_while(|sim| first.take().is_some() || curr_frame <= sim.sr_entered)?;
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
/// Each of these are exposed as the [`PSR::privileged`], [`PSR::priority`], and [`PSR::cc`] values.
#[allow(clippy::upper_case_acronyms)]
#[repr(transparent)]
pub struct PSR(pub u16);

impl PSR {
    /// Creates a PSR with a default value.
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
    /// Sets whether the simulator is in privileged mode.
    pub fn set_privileged(&mut self, privl: bool) {
        self.0 &= 0x7FFF;
        self.0 |= (!privl as u16) << 15;
    }
    /// Sets the current interrupt priority of the simulator.
    pub fn set_priority(&mut self, prio: u8) {
        self.0 &= 0xF8FF;
        self.0 |= (prio as u16) << 8;
    }
    /// Sets the condition code of the simulator.
    pub fn set_cc(&mut self, cc: u8) {
        self.0 &= 0xFFF8;
        self.0 |= cc as u16;
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