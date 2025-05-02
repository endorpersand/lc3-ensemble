//! Memory handling for the LC-3 simulator.
//! 
//! This module consists of:
//! - [`Word`]: A mutable memory location.
//! - [`MemArray`]: The memory array.
//! - [`RegFile`]: The register file.

use rand::rngs::StdRng;
use rand::Rng;

use crate::ast::Reg;

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
/// (see [`super::MachineInitStrategy`] for details).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Word {
    data: u16,
    init: u16
}

const NO_BITS:  u16 = 0;
const ALL_BITS: u16 = 1u16.wrapping_neg();

impl Word {
    /// Creates a new word that is considered uninitialized.
    pub fn new_uninit<F: WordFiller + ?Sized>(fill: &mut F) -> Self {
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
    pub fn get_if_init<E>(&self, strict: bool, err: E) -> Result<u16, E> {
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
    pub fn set_if_init<E>(&mut self, data: Word, strict: bool, err: E) -> Result<(), E> {
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
    /// Generate a word of data.
    fn generate(&mut self) -> u16;

    /// Generates an array of [`Word`]s.
    fn generate_array<const N: usize>(&mut self) -> [Word; N] {
        std::array::from_fn(|_| Word::new_uninit(self))
    }
    /// Generates a heap-allocated array of [`Word`]s.
    fn generate_boxed_array<const N: usize>(&mut self) -> Box<[Word; N]> {
        std::iter::repeat_with(|| Word::new_uninit(self))
            .take(N)
            .collect::<Box<_>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!("iterator should have had {N} elements"))
    }
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
        self.random()
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
pub enum MachineInitStrategy {
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

impl MachineInitStrategy {
    pub(super) fn generator(&self) -> impl WordFiller {
        use rand::SeedableRng;

        match self {
            MachineInitStrategy::Unseeded => WCGenerator::Unseeded,
            MachineInitStrategy::Seeded { seed } => WCGenerator::Seeded(Box::new(StdRng::seed_from_u64(*seed))),
            MachineInitStrategy::Known { value } => WCGenerator::Known(*value),
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

/// Memory array.
/// 
/// This can be addressed with any `u16` (16-bit address).
/// 
/// This memory array *does* expose memory locations 0xFE00-0xFFFF,
/// however they are not accessible through normal Simulator operation 
/// (i.e., via [`Simulator::read_mem`]) and [`Simulator::write_mem`].
/// 
/// They can be read and edited via the typical Index traits.
/// If you wish to see the handling of memory-mapped IO, see the above
/// [`Simulator`] methods.
/// 
/// [`Simulator`]: super::Simulator
/// [`Simulator::read_mem`]: super::Simulator::read_mem
/// [`Simulator::write_mem`]: super::Simulator::write_mem
/// [`Simulator::default_mem_ctx`]: super::Simulator::default_mem_ctx
#[derive(Debug)]
pub struct MemArray(Box<[Word; 1 << 16]>);
impl MemArray {
    /// Creates a new memory with a provided word creation strategy.
    pub fn new(filler: &mut impl WordFiller) -> Self {
        Self(filler.generate_boxed_array())
    }

    /// Copies an object file block into this memory.
    pub(super) fn copy_obj_block(&mut self, mut start: u16, data: &[Option<u16>]) {
        let mem = &mut self.0;

        // separate data into chunks of initialized/uninitialized
        for chunk in data.chunk_by(|a, b| a.is_some() == b.is_some()) {
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

    pub(super) fn as_slice_mut(&mut self) -> &mut [Word] {
        &mut *self.0
    }
}
impl std::ops::Index<u16> for MemArray {
    type Output = Word;

    fn index(&self, index: u16) -> &Self::Output {
        &self.0[index as usize]
    }
}
impl std::ops::IndexMut<u16> for MemArray {
    fn index_mut(&mut self, index: u16) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

/// The register file. 
/// 
/// This struct can be indexed with a [`Reg`].
/// 
/// # Example
/// 
/// ```
/// use lc3_ensemble::sim::mem::RegFile;
/// use lc3_ensemble::ast::Reg::R0;
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
        Self(filler.generate_array())
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