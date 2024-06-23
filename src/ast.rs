//! Components relating to the abstract syntax trees (ASTs) 
//! used in representing assembly instructions.
//! 
//! These components together are used to construct... 
//! - [`asm::AsmInstr`] (a data structure holding an assembly source code instruction),
//! - [`asm::Directive`] (a data structure holding an assembly source code directive),
//! - and [`sim::SimInstr`] (a data structure holding a bytecode instruction).

pub mod asm;
pub mod sim;

use std::fmt::Write as _;
use std::num::TryFromIntError;
use offset_base::OffsetBacking;

/// A register. Must be between 0 and 7.
/// 
/// This `Reg` struct can either be constructed by selecting a register from [`reg_consts`], 
/// or by using [`Reg::try_from`].
/// 
/// ## Examples
/// 
/// ```text
/// AND R0, R0, #0
///     ~~  ~~    
/// ADD R1, R1, R0
///     ~~  ~~  ~~
/// LD R2, VALUE
///    ~~         
/// NOT R1, R2
///     ~~  ~~    
/// ```
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Reg(pub(crate) u8);

/// Register constants!
pub mod reg_consts {
    use super::Reg;

    /// The 0th register in the register file.
    pub const R0: Reg = Reg(0);
    /// The 1st register in the register file.
    pub const R1: Reg = Reg(1);
    /// The 2nd register in the register file.
    pub const R2: Reg = Reg(2);
    /// The 3rd register in the register file.
    pub const R3: Reg = Reg(3);
    /// The 4th register in the register file.
    pub const R4: Reg = Reg(4);
    /// The 5th register in the register file.
    pub const R5: Reg = Reg(5);
    /// The 6th register in the register file.
    pub const R6: Reg = Reg(6);
    /// The 7th register in the register file.
    pub const R7: Reg = Reg(7);
}
impl Reg {
    /// Gets the register number of this [`Reg`]. This is always between 0 and 7.
    pub fn reg_no(self) -> u8 {
        self.0
    }
}
impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // padding should have no effect here
        write!(f, "R{}", self.0)
    }
}
impl From<Reg> for usize {
    // Used for indexing the reg file in [`ast::Sim`].
    fn from(value: Reg) -> Self {
        usize::from(value.0)
    }
}
impl TryFrom<u8> for Reg {
    type Error = TryFromIntError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=7 => Ok(Reg(value)),
            // HACKy, but there's no other way to create this error
            _     => u8::try_from(256).map(|_| unreachable!("should've been TryFromIntError")),
        }
    }
}

/// A condition code (used for `BR`), must be between 0 and 7.
/// 
/// The condition codes are listed below:
/// 
/// | instruction   | code (bin) |
/// |---------------|------------|
/// | `NOP`         | `000`      |
/// | `BRn`         | `100`      |
/// | `BRz`         | `010`      |
/// | `BRnz`        | `110`      |
/// | `BRp`         | `001`      |
/// | `BRnp`        | `101`      |
/// | `BRzp`        | `011`      |
/// | `BR`, `BRnzp` | `111`      |
/// 
pub type CondCode = u8;

/// A value representing a signed offset or a signed immediate value.
/// 
/// `N` indicates the maximum bit size of this offset/immediate value.
/// 
/// ## Examples
/// 
/// `IOffset<5>` is used to represent `ADD`/`AND`'s imm5 operand:
/// 
/// ```text
/// AND R0, R0, #0
///             ~~
/// ADD R1, R1, #1
///             ~~
/// ```
/// 
/// They are also used for numeric register or PC offset values:
/// ```text
/// BR x-F
///    ~~~
/// JSR #99
///     ~~~
/// LD R0, #10
///        ~~~
/// LDR R0, R0, #9
///             ~~
/// ```
pub type IOffset<const N: u32> = Offset<i16, N>;
/// An unsigned 8-bit trap vector (used for `TRAP`).
/// 
/// ## Examples
/// 
/// ```text
/// TRAP x25
///      ~~~
/// ```
pub type TrapVect8 = Offset<u16, 8>;

/// A value representing either an immediate value or a register.
/// 
/// This is used to handle cases where an operand can be either 
/// an immediate value or a register (e.g., in `AND` or `ADD`).
/// 
/// ## Examples
/// ```text
/// AND R0, R0, #0
/// AND R1, R1, R1
/// ADD R2, R2, #2
/// ADD R3, R3, R3
///             ^^
/// ```
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum ImmOrReg<const N: u32> {
    #[allow(missing_docs)]
    Imm(IOffset<N>),
    #[allow(missing_docs)]
    Reg(Reg)
}
impl<const N: u32> std::fmt::Display for ImmOrReg<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImmOrReg::Imm(imm) => imm.fmt(f),
            ImmOrReg::Reg(reg) => reg.fmt(f),
        }
    }
}

/// A value representing an offset or an immediate value.
/// 
/// The `OFF` type represents the backing type of this offset. 
/// The signedness of this offset type is dependent on the signedness of the `OFF` type:
/// - `Offset<i16, _>`: signed offset (also aliased as [`IOffset`])
/// - `Offset<u16, _>`: unsigned offset
/// 
/// `N` indicates the maximum bit size of this offset/immediate value.
/// 
/// ## Examples
/// 
/// - `Offset<i16, 5>`  is used to represent `ADD`/`AND`'s imm5 operand. 
///     See [`IOffset`] for more examples of its use.
/// - `Offset<u16, 8>` is used to represent the `trapvect8` operand of the `TRAP` instruction.
///     See [`TrapVect8`] for more examples of its use.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Offset<OFF, const N: u32>(OFF);

impl<OFF: std::fmt::Display, const N: u32> std::fmt::Display for Offset<OFF, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('#')?;
        self.0.fmt(f)
    }
}
impl<OFF: std::fmt::Binary, const N: u32> std::fmt::Binary for Offset<OFF, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('b')?;
        self.0.fmt(f)
    }
}
impl<OFF: std::fmt::LowerHex, const N: u32> std::fmt::LowerHex for Offset<OFF, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('x')?;
        self.0.fmt(f)
    }
}
impl<OFF: std::fmt::UpperHex, const N: u32> std::fmt::UpperHex for Offset<OFF, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('x')?;
        self.0.fmt(f)
    }
}

/// The errors that can result from calling [`Offset::new`].
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum OffsetNewErr {
    /// The provided offset cannot fit an unsigned integer of the given bitsize.
    CannotFitUnsigned(u32),
    /// The provided offset cannot fit a signed integer of the given bitsize.
    CannotFitSigned(u32)
}

impl std::fmt::Display for OffsetNewErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OffsetNewErr::CannotFitUnsigned(n) => write!(f, "value is too big for unsigned {n}-bit integer"),
            OffsetNewErr::CannotFitSigned(n) => write!(f, "value is too big for signed {n}-bit integer"),
        }
    }
}
impl std::error::Error for OffsetNewErr {}
impl crate::err::Error for OffsetNewErr {
    fn help(&self) -> Option<std::borrow::Cow<str>> {
        use std::borrow::Cow;

        let error = match self {
            OffsetNewErr::CannotFitUnsigned(n) => Cow::from(format!("the range for an unsigned {n}-bit integer is [0, {}]", (1 << n) - 1)),
            OffsetNewErr::CannotFitSigned(n) => Cow::from(format!("the range for a signed {n}-bit integer is [{}, {}]", (-1) << (n - 1), (1 << (n - 1)) - 1)),
        };

        Some(error)
    }
}

mod offset_base {
    use super::OffsetNewErr;

    /// Any type that could store a value for [`Offset`].
    /// 
    /// [`Offset`]: super::Offset
    pub trait OffsetBacking: Copy + Eq {
        /// How many bits are contained within this backing.
        /// 
        /// For example, `u16` has 16 bits and thus BITS == 16.
        const BITS: u32;

        /// Truncates the given value to the provided `bit_size`.
        /// 
        /// This bit size is always known to be less than BITS.
        fn truncate(self, bit_size: u32) -> Self;

        /// The error to raise if a given value doesn't match
        /// its provided value when truncated to a given `bit_size`.
        fn does_not_fit_error(bit_size: u32) -> OffsetNewErr;
    }
    
    macro_rules! impl_offset_backing_for_ints {
        ($($Int:ty: $Err:ident),*) => {
            $(
                impl OffsetBacking for $Int {
                    const BITS: u32 = Self::BITS;
                
                    fn truncate(self, bit_size: u32) -> Self {
                        (self << (Self::BITS - bit_size)) >> (Self::BITS - bit_size)
                    }

                    fn does_not_fit_error(bit_size: u32) -> OffsetNewErr {
                        OffsetNewErr::$Err(bit_size)
                    }
                }
            )*
        }
    }
    impl_offset_backing_for_ints! {
        u16: CannotFitUnsigned,
        i16: CannotFitSigned
    }
}

impl<OFF: OffsetBacking, const N: u32> Offset<OFF, N> {
    /// Creates a new offset value.
    /// This must fit within `N` bits of the representation, otherwise an error is raised.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use lc3_ensemble::ast::Offset;
    /// #
    /// // Signed:
    /// let neg5 = Offset::<i16, 5>::new(-5);
    /// let pos15 = Offset::<i16, 5>::new(15);
    /// let pos16 = Offset::<i16, 5>::new(16);
    /// assert!(neg5.is_ok());
    /// assert!(pos15.is_ok());
    /// assert!(pos16.is_err());
    /// 
    /// // Unsigned:
    /// let pos15 = Offset::<u16, 5>::new(15);
    /// let pos16 = Offset::<u16, 5>::new(16);
    /// let pos32 = Offset::<u16, 5>::new(32);
    /// assert!(pos15.is_ok());
    /// assert!(pos16.is_ok());
    /// assert!(pos32.is_err());
    /// ```
    /// 
    /// # Panics
    /// 
    /// This will panic if `N` is larger than the offset backing (e.g., for backing `u16`, larger than 16).
    /// 
    /// ```should_panic
    /// # use lc3_ensemble::ast::Offset;
    /// #
    /// let oh_no = Offset::<i16, 17>::new(18);
    /// ```
    pub fn new(n: OFF) -> Result<Self, OffsetNewErr> {
        assert!(N <= OFF::BITS, "bit size {N} exceeds size of backing ({})", OFF::BITS);
        match n == n.truncate(N) {
            true  => Ok(Offset(n)),
            false => Err(OFF::does_not_fit_error(N)),
        }
    }

    /// Creates a new offset by extending the first N bits of the integer,
    /// and discarding the rest.
    /// 
    /// The extension is considered sign-extended if the offset's backing is signed,
    /// and zero-extended if the offset's backing is unsigned.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use lc3_ensemble::ast::Offset;
    /// #
    /// // Signed:
    /// let neg5 = Offset::<i16, 5>::new_trunc(-5);  // 0b11111111111_11011
    /// let pos15 = Offset::<i16, 5>::new_trunc(15); // 0b00000000000_01111
    /// let pos16 = Offset::<i16, 5>::new_trunc(16); // 0b00000000000_10000
    /// assert_eq!(neg5.get(),   -5); // 0b11011
    /// assert_eq!(pos15.get(),  15); // 0b01111
    /// assert_eq!(pos16.get(), -16); // 0b10000
    /// 
    /// // Unsigned:
    /// let pos15 = Offset::<u16, 5>::new_trunc(15); // 0b00000000000_01111
    /// let pos16 = Offset::<u16, 5>::new_trunc(16); // 0b00000000000_10000
    /// let pos32 = Offset::<u16, 5>::new_trunc(32); // 0b00000000001_00000
    /// assert_eq!(pos15.get(), 15); // 01111
    /// assert_eq!(pos16.get(), 16); // 10000
    /// assert_eq!(pos32.get(),  0); // 00000
    /// ```
    /// 
    /// # Panics
    /// 
    /// This will panic if `N` is larger than the offset backing (e.g., for backing `u16`, larger than 16).
    /// 
    /// ```should_panic
    /// # use lc3_ensemble::ast::Offset;
    /// #
    /// let oh_no = Offset::<i16, 17>::new_trunc(18);
    /// ```
    pub fn new_trunc(n: OFF) -> Self {
        assert!(N <= OFF::BITS, "bit size {N} exceeds size of backing ({})", OFF::BITS);
        Self(n.truncate(N))
    }

    /// Gets the value of the offset.
    pub fn get(&self) -> OFF {
        self.0
    }
}

/// An offset or a label.
/// 
/// This is used to represent [`PCOffset`] operands 
/// (such as the `PCOffset9` operand in `LD` and `ST` 
/// and the `PCOffset11` operand in `JSR`).
/// 
/// During the first assembly pass, the label is resolved and
/// replaced with a regular [`Offset`] value.
/// 
/// ## Examples
/// ```text
/// LD R2, VALUE
///        ~~~~~
/// BRz END
///     ~~~
/// BR #-99
///    ~~~~
/// JSR SUBROUTINE
///     ~~~~~~~~~~
/// ```
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum PCOffset<OFF, const N: u32> {
    #[allow(missing_docs)]
    Offset(Offset<OFF, N>),
    #[allow(missing_docs)]
    Label(Label)
}
impl<OFF, const N: u32> std::fmt::Display for PCOffset<OFF, N> 
    where Offset<OFF, N>: std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PCOffset::Offset(off)  => off.fmt(f),
            PCOffset::Label(label) => label.fmt(f),
        }
    }
}

/// A label.
/// 
/// This struct stores the name of the label (accessible by the `name` field)
/// and the source code span indicating where the label is located in assembly source code.
/// 
/// # Examples
/// ```text
/// .orig x3000
/// AND R0, R0, #0
/// LD R2, VALUE
///        ~~~~~
/// LOOP:
/// ~~~~
///     NOT R1, R2
///     ADD R1, R1, #1
///     ADD R1, R1, R0
///     BRz END
///         ~~~
///     ADD R0, R0, #1
///     BR LOOP
///        ~~~~
/// END: HALT
/// ~~~
/// VALUE: .fill #8464
/// ~~~~~
/// .end
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Label {
    /// The label's identifier
    pub name: String,

    /// The start of the label in assembly source code.
    /// 
    /// Since name stores the length of the string,
    /// we don't need to store the whole span.
    /// 
    /// This saves like 8 bytes of space on a 64-bit machine, so ya know
    start: usize
}
impl Label {
    /// Creates a new label.
    pub fn new(name: String, span: std::ops::Range<usize>) -> Self {
        debug_assert_eq!(span.start + name.len(), span.end, "span should have the same length as name");
        Label { name, start: span.start }
    }
    /// Returns the span of the label in assembly source code.
    pub fn span(&self) -> std::ops::Range<usize> {
        self.start .. (self.start + self.name.len())
    }
}
impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name.fmt(f)
    }
}