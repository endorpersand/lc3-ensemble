//! Memory handling for the LC-3 simulator.
//! 
//! This module consists of:
//! - [`Word`]: A mutable memory location.
//! - [`Mem`]: The memory.
//! - [`RegFile`]: The register file.

use rand::rngs::StdRng;
use rand::Rng;

use crate::ast::Reg;

use super::{IODevice, SimErr, SimIOwMCR};

/// A memory location that can be read and written to.
/// 
/// # Reading
/// 
/// A word's value can be read with:
/// - [`Word::get`] to directly access the value, ignoring any initialization state
/// - [`Word::get_if_init`] to directly access the value after verifying initialization state
/// 
/// See the respective functions for more details.
/// 
/// Both functions return the unsigned representation of the word.
/// If needed, this can be converted to a signed integer with typical `as` casting (`data as i16`).
/// 
/// # Writing
/// 
/// A word can be written into with a value or with another word:
/// - [`Word::set`] to read a value into this word
/// - [`Word::set_if_init`] to read a word into this word
/// 
/// [`Word::set_if_init`] may be more useful in situations where initialization state needs to be preserved
/// or when it needs to be verified.
/// 
/// See the respective functions for more details.
/// 
/// Words can also be written to by applying assign operations (e.g., add, sub, and, etc.).
/// All arithmetic operations that can be applied to words are assumed to be wrapping.
/// See those implementations for more details.
/// 
/// # Initialization
/// 
/// Internally, each memory location keeps track of two fields:
/// 1. its data (i.e., the value stored at this location)
/// 2. which bits of its data are truly "initialized" (as in the program knows what values are present there)
/// 
/// This second field is not used except for when the simulator is set to strict mode.
/// Then, this second field is leveraged to detect if uninitialized memory is being
/// written to places it shouldn't be (e.g., PC, addresses, registers and memory).
/// 
/// When a `Word` is created for memory/register files (i.e., via [`Word::new_uninit`]), 
/// it is created with the initialization bits set to fully uninitialized.
/// The data associated with this `Word` is decided by the creation strategy 
/// (see [`super::WordCreateStrategy`] for details).
#[derive(Debug, Clone, Copy)]
pub struct Word {
    data: u16,
    init: u16
}

const NO_BITS:  u16 = 0;
const ALL_BITS: u16 = 1u16.wrapping_neg();

impl Word {
    /// Creates a new word that is considered uninitialized.
    pub fn new_uninit(fill: &mut impl WordFiller) -> Self {
        Self {
            data: fill.generate(),
            init: NO_BITS,
        }
    }
    /// Creates a new word that is initialized with a given data value.
    pub fn new_init(data: u16) -> Self {
        Self {
            data,
            init: ALL_BITS,
        }
    }

    /// Reads the word, returning its unsigned representation.
    /// 
    /// The data is returned without checking for initialization state.
    /// If the initialization state should be checked before trying to query the data,
    /// then [`Word::get_if_init`] should be used instead.
    pub fn get(&self) -> u16 {
        self.data
    }
    /// Reads the word if it is properly initialized under strictness requirements, returning its unsigned representation.
    /// 
    /// This function is more cognizant of word initialization than [`Word::get`].
    /// - In non-strict mode (`strict == false`), this function unconditionally allows access to the data regardless of initialization state.
    /// - In strict mode (`strict == true`), this function verifies `self` is fully initialized, raising the provided error if not.
    pub fn get_if_init(&self, strict: bool, err: SimErr) -> Result<u16, SimErr> {
        match !strict || self.is_init() {
            true  => Ok(self.data),
            false => Err(err)
        }
    }

    /// Writes to the word.
    /// 
    /// This sets the word to the `data` value assuming it is **fully** initialized
    /// and correspondingly sets the initialization state to be fully initialized.
    /// 
    /// If the initialization state of the `data` value should be checked before
    /// trying to write to the word, then [`Word::set_if_init`] should be used instead.
    pub fn set(&mut self, data: u16) {
        self.data = data;
        self.init = ALL_BITS;
    }
    /// Writes to the word while verifying the data stored is properly initialized under strictness requirements.
    /// 
    /// This function is more cognizant of word initialization than [`Word::set`].
    /// - In non-strict mode, this function preserves the initialization data of the `data` argument.
    /// - In strict mode, this function verifies `data` is fully initialized, raising the provided error if not.
    pub fn set_if_init(&mut self, data: Word, strict: bool, err: SimErr) -> Result<(), SimErr> {
        match !strict || data.is_init() {
            true => {
                *self = data;
                Ok(())
            },
            false => Err(err)
        }
    }

    /// Checks that a word is fully initialized
    pub fn is_init(&self) -> bool {
        self.init == ALL_BITS
    }
    /// Clears initialization of this word.
    pub fn clear_init(&mut self) {
        self.init = NO_BITS;
    }
}
impl From<u16> for Word {
    /// Creates a fully initialized word.
    fn from(value: u16) -> Self {
        Word::new_init(value)
    }
}
impl From<i16> for Word {
    /// Creates a fully initialized word.
    fn from(value: i16) -> Self {
        Word::new_init(value as u16)
    }
}

impl std::ops::Not for Word {
    type Output = Word;

    /// Inverts the data on this word, preserving any initialization state.
    fn not(self) -> Self::Output {
        // Initialization state should stay the same after this.
        let Self { data, init } = self;
        Self { data: !data, init }
    }
}


impl std::ops::Add for Word {
    type Output = Word;

    /// Adds two words together (wrapping if overflow occurs).
    /// 
    /// If the two words are fully initialized, 
    /// the resulting word will also be fully initialized.
    /// Otherwise, the resulting word is fully uninitialized.
    fn add(self, rhs: Self) -> Self::Output {
        let Self { data: ldata, init: linit } = self;
        let Self { data: rdata, init: rinit } = rhs;

        if rdata == 0 && rinit == ALL_BITS { return self; }
        if ldata == 0 && linit == ALL_BITS { return rhs; }

        let data = ldata.wrapping_add(rdata);

        // Close enough calculation:
        // If both are fully init, consider this word fully init.
        // Otherwise, consider it fully uninit.
        let init = match linit == ALL_BITS && rinit == ALL_BITS {
            true  => ALL_BITS,
            false => NO_BITS,
        };

        Self { data, init }
    }
}
impl std::ops::AddAssign for Word {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl std::ops::AddAssign<u16> for Word {
    /// Increments the word by the provided value.
    /// 
    /// If the word was fully initialized,
    /// its updated value is also fully initialized.
    /// Otherwise, the resulting word is fully uninitialized.
    fn add_assign(&mut self, rhs: u16) {
        *self = *self + Word::from(rhs);
    }
}
impl std::ops::AddAssign<i16> for Word {
    /// Increments the word by the provided value.
    /// 
    /// If the word was fully initialized,
    /// its updated value is also fully initialized.
    /// Otherwise, the resulting word is fully uninitialized.
    fn add_assign(&mut self, rhs: i16) {
        *self = *self + Word::from(rhs);
    }
}


impl std::ops::Sub for Word {
    type Output = Word;

    /// Subtracts two words together (wrapping if overflow occurs).
    /// 
    /// If the two words are fully initialized, 
    /// the resulting word will also be fully initialized.
    /// Otherwise, the resulting word is fully uninitialized.
    fn sub(self, rhs: Self) -> Self::Output {
        let Self { data: ldata, init: linit } = self;
        let Self { data: rdata, init: rinit } = rhs;

        // This is (self - 0) == self.
        if rdata == 0 && rinit == ALL_BITS { return self; }

        let data = ldata.wrapping_sub(rdata);
        // Very lazy initialization scheme.
        // If both are fully init, consider this word fully init.
        // Otherwise, consider it fully uninit.
        let init = match linit == ALL_BITS && rinit == ALL_BITS {
            true  => ALL_BITS,
            false => NO_BITS,
        };

        Self { data, init }
    }
}
impl std::ops::SubAssign for Word {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl std::ops::SubAssign<u16> for Word {
    /// Decrements the word by the provided value.
    /// 
    /// If the word was fully initialized,
    /// its updated value is also fully initialized.
    /// Otherwise, the resulting word is fully uninitialized.
    fn sub_assign(&mut self, rhs: u16) {
        *self = *self - Word::new_init(rhs);
    }
}
impl std::ops::SubAssign<i16> for Word {
    /// Decrements the word by the provided value.
    /// 
    /// If the word was fully initialized,
    /// its updated value is also fully initialized.
    /// Otherwise, the resulting word is fully uninitialized.
    fn sub_assign(&mut self, rhs: i16) {
        *self = *self - Word::new_init(rhs as _);
    }
}


impl std::ops::BitAnd for Word {
    type Output = Word;

    /// Applies a bitwise AND across two words.
    /// 
    /// This will also compute the correct initialization
    /// for the resulting word, taking into account bit clearing.
    fn bitand(self, rhs: Self) -> Self::Output {
        let Self { data: ldata, init: linit } = self;
        let Self { data: rdata, init: rinit } = rhs;

        let data = ldata & rdata;
        // A given bit of the result is init if:
        // - both the lhs and rhs bits are init
        // - either of the bits are data: 0, init: 1
        let init = (linit & rinit) | (!ldata & linit) | (!rdata & rinit);

        Self { data, init }
    }
}
impl std::ops::BitAndAssign for Word {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

/// Trait that describes types that can be used to create the data for an uninitialized [`Word`].
/// 
/// This is used with [`Word::new_uninit`] to create uninitialized Words.
pub trait WordFiller {
    /// Generate the data.
    fn generate(&mut self) -> u16;
}
impl WordFiller for () {
    /// This creates unseeded, non-deterministic values.
    fn generate(&mut self) -> u16 {
        rand::random()
    }
}
impl WordFiller for u16 {
    /// Sets each word to the given value.
    fn generate(&mut self) -> u16 {
        *self
    }
}
impl WordFiller for StdRng {
    /// This creates values from the standard random number generator.
    /// 
    /// This can be used to create deterministic, seeded values.
    fn generate(&mut self) -> u16 {
        self.gen()
    }
}
/// Strategy used to initialize the `reg_file` and `mem` of the [`Simulator`].
/// 
/// These are used to set the initial state of the memory and registers,
/// which will be treated as uninitialized until they are properly initialized
/// by program code.
/// 
/// [`Simulator`]: super::Simulator
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

    /// Initializes each word to a known value.
    Known {
        /// The value to initialize each value to.
        value: u16
    }
}

impl WordCreateStrategy {
    pub(super) fn generator(&self) -> impl WordFiller {
        use rand::SeedableRng;

        match self {
            WordCreateStrategy::Unseeded => WCGenerator::Unseeded,
            WordCreateStrategy::Seeded { seed } => WCGenerator::Seeded(Box::new(StdRng::seed_from_u64(*seed))),
            WordCreateStrategy::Known { value } => WCGenerator::Known(*value),
        }
    }
}

enum WCGenerator {
    Unseeded,
    Seeded(Box<rand::rngs::StdRng>),
    Known(u16)
}
impl WordFiller for WCGenerator {
    fn generate(&mut self) -> u16 {
        match self {
            WCGenerator::Unseeded  => ().generate(),
            WCGenerator::Seeded(r) => r.generate(),
            WCGenerator::Known(k)  => k.generate(),
        }
    }
}

/// Context behind a memory access.
/// 
/// This struct is used by [`Mem::read`] and [`Mem::write`] to perform checks against memory accesses.
/// A default memory access context for the given simulator can be constructed with [`Simulator::default_mem_ctx`].
/// 
/// [`Simulator::default_mem_ctx`]: super::Simulator::default_mem_ctx
#[derive(Clone, Copy)]
pub struct MemAccessCtx {
    /// Whether this access is privileged (false = user, true = supervisor).
    pub privileged: bool,
    /// Whether writes to memory should follow strict rules 
    /// (no writing partially or fully uninitialized data).
    /// 
    /// 
    /// [`Simulator::default_mem_ctx`]: super::Simulator::default_mem_ctx
    /// This does not affect [`Mem::read`].
    pub strict: bool
}

const N: usize = 1 << 16;
const IO_START: u16 = 0xFE00;
const USER_RANGE: std::ops::Range<u16> = 0x3000..0xFE00;

/// Memory. 
/// 
/// This can be addressed with any `u16` (16-bit address).
/// 
/// Note that this struct provides two methods of accessing memory:
/// - [`Mem::get_raw`] and [`Mem::get_raw_mut`]: direct access to memory values
/// - [`Mem::read`] and [`Mem::write`]: memory access with privilege checks, strictness checks, and IO updating
/// 
/// # `get_raw` and `get_raw_mut`
/// 
/// [`Mem::get_raw`] and [`Mem::get_raw_mut`]'s API is simple, it simply accesses the memory value at the address.
/// Note that this means:
/// - These functions do not trigger IO effects (and as a result, IO values will not be updated).
/// - These functions do not perform access violation checks.
/// 
/// ```
/// use lc3_ensemble::sim::mem::Mem;
/// 
/// let mut mem = Mem::new(&mut ()); // never should have to initialize mem
/// mem.get_raw_mut(0x3000).set(11);
/// assert_eq!(mem.get_raw(0x3000).get(), 11);
/// ```
/// 
/// # `read` and `write`
/// 
/// In contrast, [`Mem::read`] and [`Mem::write`] have to account for all of the possible conditions 
/// behind a memory access.
/// This means:
/// - These functions *do* trigger IO effects.
/// - These functions do perform access violation and strictness checks.
/// 
/// Additionally, these functions require a [`MemAccessCtx`], defining the configuration of the access, which consists of:
/// - `privileged`: if false, this access errors if the address is a memory location outside of the user range.
/// - `strict`: If true, all accesses that would cause a memory location to be set with uninitialized data causes an error (writes only).
/// 
/// The [`Simulator`] defines [`Simulator::default_mem_ctx`] to produce this value automatically based on the simulator's state.
/// ```
/// use lc3_ensemble::sim::Simulator;
/// use lc3_ensemble::sim::mem::Word;
/// 
/// let mut sim = Simulator::new(Default::default());
/// 
/// assert!(sim.mem.write(0x0000, Word::new_init(0x9ABC), sim.default_mem_ctx()).is_err());
/// assert!(sim.mem.write(0x3000, Word::new_init(0x9ABC), sim.default_mem_ctx()).is_ok());
/// assert!(sim.mem.read(0x0000, sim.default_mem_ctx()).is_err());
/// assert!(sim.mem.read(0x3000, sim.default_mem_ctx()).is_ok());
/// ```
/// 
/// [`Simulator`]: super::Simulator
/// [`Simulator::default_mem_ctx`]: super::Simulator::default_mem_ctx
#[derive(Debug)]
pub struct Mem {
    data: Box<[Word; N]>,
    pub(super) io: SimIOwMCR
}
impl Mem {
    /// Creates a new memory with a provided word creation strategy.
    pub fn new(filler: &mut impl WordFiller) -> Self {
        Self {
            data: std::iter::repeat_with(|| Word::new_uninit(filler))
                .take(N)
                .collect::<Box<_>>()
                .try_into()
                .unwrap_or_else(|_| unreachable!("iterator should have had {N} elements")),
            io: Default::default()
        }
    }

    /// Copies an object file block into this memory.
    pub fn copy_obj_block(&mut self, mut start: u16, data: &[Option<u16>]) {
        let mem = &mut self.data;

        // chunk_by was added in Rust 1.77
        struct ChunkBy<'s, T, F>(&'s [T], F);
        impl<'s, T, F: FnMut(&T, &T) -> bool> Iterator for ChunkBy<'s, T, F> {
            type Item = &'s [T];
        
            fn next(&mut self) -> Option<Self::Item> {
                let (first, rest) = self.0.split_first()?;

                // find the first element that doesn't match pred (+1 for the first el that was removed)
                let pos = match rest.iter().position(|n| !(self.1)(first, n)) {
                    Some(i) => i + 1,
                    None => self.0.len(),
                };

                let (chunk, rest) = self.0.split_at(pos);

                self.0 = rest;
                Some(chunk)
            }
        }

        // separate data into chunks of initialized/uninitialized
        for chunk in ChunkBy(data, |a: &Option<_>, b: &Option<_>| a.is_some() == b.is_some()) {
            let end = start.wrapping_add(chunk.len() as u16);

            let si = usize::from(start);
            let ei = usize::from(end);
            let block_is_contiguous = start <= end;

            if chunk[0].is_some() { // if chunk is init, copy the data over
                let ch: Vec<_> = chunk.iter()
                    .map(|&opt| opt.unwrap())
                    .map(Word::new_init)
                    .collect();

                if block_is_contiguous {
                    mem[si..ei].copy_from_slice(&ch);
                } else {
                    let (left, right) = ch.split_at(start.wrapping_neg() as usize);
                    mem[si..].copy_from_slice(left);
                    mem[..ei].copy_from_slice(right)
                }
            } else { // if chunk is uninit, clear the initialization state
                if block_is_contiguous {
                    for word in &mut mem[si..ei] {
                        word.clear_init();
                    }
                } else {
                    for word in &mut mem[si..] {
                        word.clear_init();
                    }
                    for word in &mut mem[..ei] {
                        word.clear_init();
                    }
                }
            }

            start = end;
        }
    }

    /// Gets a reference to a word from the memory's current state.
    /// 
    /// This is **only** meant to be used to query the state of the memory,
    /// not to simulate a read from memory.
    /// 
    /// Note the differences from [`Mem::read`]:
    /// - This function does not trigger IO effects (and as a result, IO values will not be updated).
    /// - This function does not require [`MemAccessCtx`].
    /// - This function does not perform access violation checks.
    /// 
    /// If any of these effects are necessary (e.g., when trying to execute instructions from the simulator),
    /// [`Mem::read`] should be used instead.
    pub fn get_raw(&self, addr: u16) -> &Word {
        // Mem could implement Index<u16>, but it doesn't as a lint against using this function incorrectly.
        &self.data[usize::from(addr)]
    }
    
    /// Gets a mutable reference to a word from the memory's current state.
    /// 
    /// This is **only** meant to be used to query/edit the state of the memory,
    /// not to simulate a write from memory.
    /// 
    /// Note the differences from [`Mem::write`]:
    /// - This function does not trigger IO effects (and as a result, IO values will not be updated).
    /// - This function does not require [`MemAccessCtx`].
    /// - This function does not perform access violation checks or strict uninitialized memory checking.
    /// 
    /// If any of these effects are necessary (e.g., when trying to execute instructions from the simulator),
    /// [`Mem::write`] should be used instead.
    pub fn get_raw_mut(&mut self, addr: u16) -> &mut Word {
        // Mem could implement IndexMut<u16>, but it doesn't as a lint against using this function incorrectly.
        &mut self.data[usize::from(addr)]
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
    /// Note that this method is used for simulating a read. If you would like to query the memory's state, 
    /// consider [`Mem::get_raw`].
    /// 
    /// [`Simulator::default_mem_ctx`]: super::Simulator::default_mem_ctx
    pub fn read(&mut self, addr: u16, ctx: MemAccessCtx) -> Result<Word, SimErr> {
        if !ctx.privileged && !USER_RANGE.contains(&addr) { return Err(SimErr::AccessViolation) };

        if addr >= IO_START {
            if let Some(new_data) = self.io.io_read(addr) {
                self.data[usize::from(addr)].set(new_data);
            }
        }
        Ok(self.data[usize::from(addr)])
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
    /// Note that this method is used for simulating a write. If you would like to edit the memory's state, 
    /// consider [`Mem::get_raw_mut`].
    /// 
    /// [`Simulator::default_mem_ctx`]: super::Simulator::default_mem_ctx
    pub fn write(&mut self, addr: u16, data: Word, ctx: MemAccessCtx) -> Result<(), SimErr> {
        if !ctx.privileged && !USER_RANGE.contains(&addr) { return Err(SimErr::AccessViolation) };
        
        let write_to_mem = if addr >= IO_START {
            let io_data = data.get_if_init(ctx.strict, SimErr::StrictIOSetUninit)?;
            self.io.io_write(addr, io_data)
        } else {
            true
        };
        if write_to_mem {
            self.data[usize::from(addr)]
                .set_if_init(data, ctx.strict, SimErr::StrictMemSetUninit)?;
        }
        Ok(())
    }
}

/// The register file. 
/// 
/// This struct can be indexed with a [`Reg`] 
/// (which can be constructed using the [`crate::ast::reg_consts`] module or via [`Reg::try_from`]).
/// 
/// # Example
/// 
/// ```
/// use lc3_ensemble::sim::mem::RegFile;
/// use lc3_ensemble::ast::reg_consts::R0;
/// 
/// let mut reg = RegFile::new(&mut ()); // never should have to initialize a reg file
/// reg[R0].set(11);
/// assert_eq!(reg[R0].get(), 11);
/// ```
#[derive(Debug, Clone)]
pub struct RegFile([Word; 8]);
impl RegFile {
    /// Creates a register file with uninitialized data.
    pub fn new(filler: &mut impl WordFiller) -> Self {
        Self(std::array::from_fn(|_| Word::new_uninit(filler)))
    }
}
impl std::ops::Index<Reg> for RegFile {
    type Output = Word;

    fn index(&self, index: Reg) -> &Self::Output {
        &self.0[usize::from(index)]
    }
}
impl std::ops::IndexMut<Reg> for RegFile {
    fn index_mut(&mut self, index: Reg) -> &mut Self::Output {
        &mut self.0[usize::from(index)]
    }
}