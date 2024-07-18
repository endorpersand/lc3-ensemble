//! This module is used for holding simulation instructions ([`SimInstr`]),
//! which are instructions that directly map to bytecode.
//! 
//! For instructions that map to assembly source code 
//! (i.e., the text representation of assembly instructions), see [`asm::AsmInstr`].
//! 
//! [`asm::AsmInstr`]: crate::ast::asm::AsmInstr

use std::fmt::Write as _;
use std::ops::Range;

use crate::sim::SimErr;

use super::{CondCode, IOffset, ImmOrReg, Reg, TrapVect8};

const OP_BR: u16   = 0b0000; 
const OP_ADD: u16  = 0b0001; 
const OP_LD: u16   = 0b0010; 
const OP_ST: u16   = 0b0011; 
const OP_JSR: u16  = 0b0100; 
const OP_AND: u16  = 0b0101; 
const OP_LDR: u16  = 0b0110; 
const OP_STR: u16  = 0b0111; 
const OP_RTI: u16  = 0b1000; 
const OP_NOT: u16  = 0b1001; 
const OP_LDI: u16  = 0b1010; 
const OP_STI: u16  = 0b1011; 
const OP_JMP: u16  = 0b1100; 
const OP_LEA: u16  = 0b1110; 
const OP_TRAP: u16 = 0b1111; 

/// An enum representing all of the possible instructions in LC-3 bytecode.
/// 
/// The variants in this enum represent instructions after the both assembly passes.
/// There are no notions of aliases and labels in the variants in this enum.
/// 
/// For instructions that map to assembly source code 
/// (i.e., the text representation of assembly instructions), refer to [`asm::AsmInstr`].
/// 
/// [`asm::AsmInstr`]: crate::ast::asm::AsmInstr
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum SimInstr {
    #[allow(missing_docs)]
    BR(CondCode, IOffset<9>),
    #[allow(missing_docs)]
    ADD(Reg, Reg, ImmOrReg<5>),
    #[allow(missing_docs)]
    LD(Reg, IOffset<9>),
    #[allow(missing_docs)]
    ST(Reg, IOffset<9>),
    #[allow(missing_docs)]
    JSR(ImmOrReg<11>),
    #[allow(missing_docs)]
    AND(Reg, Reg, ImmOrReg<5>),
    #[allow(missing_docs)]
    LDR(Reg, Reg, IOffset<6>),
    #[allow(missing_docs)]
    STR(Reg, Reg, IOffset<6>),
    #[allow(missing_docs)]
    RTI,
    #[allow(missing_docs)]
    NOT(Reg, Reg),
    #[allow(missing_docs)]
    LDI(Reg, IOffset<9>),
    #[allow(missing_docs)]
    STI(Reg, IOffset<9>),
    #[allow(missing_docs)]
    JMP(Reg),
    // Reserved,
    #[allow(missing_docs)]
    LEA(Reg, IOffset<9>),
    #[allow(missing_docs)]
    TRAP(TrapVect8),
}
impl std::fmt::Display for SimInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BR(cc, off) => {
                if cc != &0 {
                    write!(f, "BR")?;
                    if cc & 0b100 != 0 { f.write_char('n')?; };
                    if cc & 0b010 != 0 { f.write_char('z')?; };
                    if cc & 0b001 != 0 { f.write_char('p')?; };
                } else {
                    write!(f, "NOP")?;
                }
                
                write!(f, " {off}")
            },
            Self::ADD(dr, sr1, sr2) => write!(f, "ADD {dr}, {sr1}, {sr2}"),
            Self::LD(dr, off) => write!(f, "LD {dr}, {off}"),
            Self::ST(sr, off) => write!(f, "ST {sr}, {off}"),
            Self::JSR(off) => match off {
                ImmOrReg::Imm(imm) => write!(f, "JSR {imm}"),
                ImmOrReg::Reg(reg) => write!(f, "JSRR {reg}"),
            },
            Self::AND(dr, sr1, sr2) => write!(f, "AND {dr}, {sr1}, {sr2}"),
            Self::LDR(dr, br, off) => write!(f, "LDR {dr}, {br}, {off}"),
            Self::STR(sr, br, off) => write!(f, "STR {sr}, {br}, {off}"),
            Self::RTI   => f.write_str("RTI"),
            Self::NOT(dr, sr) => write!(f, "NOT {dr}, {sr}"),
            Self::LDI(dr, off) => write!(f, "LDI {dr}, {off}"),
            Self::STI(sr, off) => write!(f, "STI {sr}, {off}"),
            Self::JMP(br) => write!(f, "JMP {br}"),
            Self::LEA(dr, off) => write!(f, "LEA {dr}, {off}"),
            Self::TRAP(vect) => write!(f, "TRAP {vect:02X}")
        }
    }
}

impl SimInstr {
    /// Gets the opcode for the given instruction. This is always 4 bits.
    pub fn opcode(&self) -> u16 {
        match self {
            SimInstr::BR(_, _) => OP_BR,
            SimInstr::ADD(_, _, _) => OP_ADD,
            SimInstr::LD(_, _) => OP_LD,
            SimInstr::ST(_, _) => OP_ST,
            SimInstr::JSR(_) => OP_JSR,
            SimInstr::AND(_, _, _) => OP_AND,
            SimInstr::LDR(_, _, _) => OP_LDR,
            SimInstr::STR(_, _, _) => OP_STR,
            SimInstr::RTI => OP_RTI,
            SimInstr::NOT(_, _) => OP_NOT,
            SimInstr::LDI(_, _) => OP_LDI,
            SimInstr::STI(_, _) => OP_STI,
            SimInstr::JMP(_) => OP_JMP,
            // reserved => 0b1101
            SimInstr::LEA(_, _) => OP_LEA,
            SimInstr::TRAP(_) => OP_TRAP,
        }
    }

    /// Encodes this instruction as 16-bit bytecode.
    pub fn encode(&self) -> u16 {
        match self {
            SimInstr::BR(cc, off) => join_bits([
                (self.opcode(),    12..16),
                (*cc as u16,       9..12),
                (off.get() as u16, 0..9)
            ]),
            SimInstr::ADD(dr, sr1, ImmOrReg::Imm(i2)) => join_bits([ // ADD DR, SR1, imm5
                (self.opcode(),       12..16),
                (dr.reg_no() as u16,  9..12),
                (sr1.reg_no() as u16, 6..9),
                (0b1,                 5..6),
                (i2.get() as u16,     0..5)
            ]),
            SimInstr::ADD(dr, sr1, ImmOrReg::Reg(r2)) => join_bits([ // ADD DR, SR1, SR2
                (self.opcode(),       12..16),
                (dr.reg_no() as u16,  9..12),
                (sr1.reg_no() as u16, 6..9),
                (0b000,               3..6),
                (r2.reg_no() as u16,  0..3)
            ]),
            SimInstr::LD(dr, off) => join_bits([
                (self.opcode(),      12..16),
                (dr.reg_no() as u16, 9..12),
                (off.get() as u16,   0..9)
            ]),
            SimInstr::ST(sr, off) => join_bits([
                (self.opcode(),      12..16),
                (sr.reg_no() as u16, 9..12),
                (off.get() as u16,   0..9)
            ]),
            SimInstr::JSR(ImmOrReg::Imm(off)) => join_bits([ // JSR
                (self.opcode(),    12..16),
                (0b1,              11..12),
                (off.get() as u16, 0..11)
            ]),
            SimInstr::JSR(ImmOrReg::Reg(br)) => join_bits([ // JSRR
                (self.opcode(),      12..16),
                (0b000,              9..12),
                (br.reg_no() as u16, 6..9),
                (0b000_000,          0..6)
            ]),
            SimInstr::AND(dr, sr1, ImmOrReg::Imm(i2)) => join_bits([ // AND DR, SR1, imm5
                (self.opcode(),       12..16),
                (dr.reg_no() as u16,  9..12),
                (sr1.reg_no() as u16, 6..9),
                (0b1,                 5..6),
                (i2.get() as u16,     0..5)
            ]),
            SimInstr::AND(dr, sr1, ImmOrReg::Reg(r2)) => join_bits([ // AND DR, SR1, SR2
                (self.opcode(),      12..16),
                (dr.reg_no() as u16,  9..12),
                (sr1.reg_no() as u16, 6..9),
                (0b000,               3..6),
                (r2.reg_no() as u16,  0..3)
            ]),
            SimInstr::LDR(dr, br, off) => join_bits([
                (self.opcode(),      12..16),
                (dr.reg_no() as u16, 9..12),
                (br.reg_no() as u16, 6..9),
                (off.get() as u16,   0..6)
            ]),
            SimInstr::STR(dr, br, off) => join_bits([
                (self.opcode(),      12..16),
                (dr.reg_no() as u16, 9..12),
                (br.reg_no() as u16, 6..9),
                (off.get() as u16,   0..6)
            ]),
            SimInstr::RTI => join_bits([
                (self.opcode(),    12..16),
                (0b0000_0000_0000, 0..12)
            ]),
            SimInstr::NOT(dr, sr) => join_bits([
                (self.opcode(),      12..16),
                (dr.reg_no() as u16, 9..12),
                (sr.reg_no() as u16, 6..9),
                (0b111_111,          0..6)
            ]),
            SimInstr::LDI(dr, off) => join_bits([
                (self.opcode(),      12..16),
                (dr.reg_no() as u16, 9..12),
                (off.get() as u16,   0..9)
            ]),
            SimInstr::STI(sr, off) => join_bits([
                (self.opcode(),      12..16),
                (sr.reg_no() as u16, 9..12),
                (off.get() as u16,   0..9)
            ]),
            SimInstr::JMP(br) => join_bits([
                (self.opcode(),      12..16),
                (0b000,              9..12),
                (br.reg_no() as u16, 6..9),
                (0b000_000,          0..6)
            ]),
            SimInstr::LEA(dr, off) => join_bits([
                (self.opcode(),      12..16),
                (dr.reg_no() as u16, 9..12),
                (off.get() as u16,   0..9)
            ]),
            SimInstr::TRAP(vect) => join_bits([
                (self.opcode(), 12..16),
                (0b0000,        8..12),
                (vect.get(),    0..8)
            ]),
        }
    }

    /// Converts a 16-bit bytecode representation into a `SimInstr`.
    /// This will error if the format is invalid.
    pub fn decode(word: u16) -> Result<Self, SimErr> {
        // Note, there's a lot of magic being used in this function.
        // Refer to `DecodeUtils` at the bottom of this file.
        let opcode = word.slice(12..16);

        match opcode {
            OP_BR => {
                let cc  = word.slice(9..12) as u8;
                let off = word.slice(0..9).interpret();
                Ok(Self::BR(cc, off))
            },
            OP_ADD => {
                let dr  = word.slice(9..12).interpret();
                let sr1 = word.slice(6..9).interpret();
                let sr2 = match word.slice(5..6) != 0 {
                    false => {
                        word.slice(3..5).assert_equals(0b00)?;
                        ImmOrReg::Reg(word.slice(0..3).interpret())
                    },
                    true  => ImmOrReg::Imm(word.slice(0..5).interpret()),
                };
                Ok(Self::ADD(dr, sr1, sr2))
            },
            OP_LD => {
                let dr  = word.slice(9..12).interpret();
                let off = word.slice(0..9).interpret();
                Ok(Self::LD(dr, off))
            }
            OP_ST => {
                let sr  = word.slice(9..12).interpret();
                let off = word.slice(0..9).interpret();
                Ok(Self::ST(sr, off))
            },
            OP_JSR => {
                let val = match word.slice(11..12) != 0 {
                    true  => ImmOrReg::Imm(word.slice(0..11).interpret()),
                    false => {
                        word.slice(9..11).assert_equals(0b00)?;
                        word.slice(0..6).assert_equals(0b000_000)?;
                        ImmOrReg::Reg(word.slice(6..9).interpret())
                    },
                };
                Ok(Self::JSR(val))
            },
            OP_AND => {
                let dr  = word.slice(9..12).interpret();
                let sr1 = word.slice(6..9).interpret();
                let sr2 = match word.slice(5..6) != 0 {
                    false => {
                        word.slice(3..5).assert_equals(0b00)?;
                        ImmOrReg::Reg(word.slice(0..3).interpret())
                    },
                    true  => ImmOrReg::Imm(word.slice(0..5).interpret()),
                };
                Ok(Self::AND(dr, sr1, sr2))
            },
            OP_LDR => {
                let dr  = word.slice(9..12).interpret();
                let br  = word.slice(6..9).interpret();
                let off = word.slice(0..6).interpret();
                Ok(Self::LDR(dr, br, off))
            },
            OP_STR => {
                let sr  = word.slice(9..12).interpret();
                let br  = word.slice(6..9).interpret();
                let off = word.slice(0..6).interpret();
                Ok(Self::STR(sr, br, off))
            },
            OP_RTI => {
                word.slice(0..12).assert_equals(0b0000_0000_0000)?;
                Ok(Self::RTI)
            },
            OP_NOT => {
                let dr = word.slice(9..12).interpret();
                let sr = word.slice(6..9).interpret();
                word.slice(0..6).assert_equals(0b111_111)?;
                Ok(Self::NOT(dr, sr))
            },
            OP_LDI => {
                let dr  = word.slice(9..12).interpret();
                let off = word.slice(0..9).interpret();
                Ok(Self::LDI(dr, off))
            },
            OP_STI => {
                let sr  = word.slice(9..12).interpret();
                let off = word.slice(0..9).interpret();
                Ok(Self::STI(sr, off))
            },
            OP_JMP => {
                word.slice(9..11).assert_equals(0b00)?;
                let reg = word.slice(6..9).interpret();
                word.slice(0..6).assert_equals(0b000_000)?;
                Ok(Self::JMP(reg))
            },
            OP_LEA => {
                let dr  = word.slice(9..12).interpret();
                let off = word.slice(0..9).interpret();
                Ok(Self::LEA(dr, off))
            },
            OP_TRAP => {
                word.slice(8..12).assert_equals(0b0000)?;
                let vect = word.slice(0..8).interpret();
                Ok(Self::TRAP(vect))
            },
            _ => Err(SimErr::IllegalOpcode)
        }
    }
}

/// Given a sequence of values and ranges, it writes each value into its corresponding range,
/// truncating the input value if it is too long.
fn join_bits<const N: usize>(bits: [(u16, Range<usize>); N]) -> u16 {
    bits.into_iter()
        .map(|(val, Range { start, end })| {
            let len = end - start;
            let mask = (1 << len) - 1;
            (val & mask) << start
        })
        .fold(0, std::ops::BitOr::bitor)
}

trait FromBits: Sized {
    /// Converts bits into self, assuming bits are the correct size.
    fn from_bits(bits: u16) -> Self;
}
trait DecodeUtils: Sized + Eq {
    /// Interprets the bits as another value.
    fn interpret<T: FromBits>(self) -> T;

    /// Asserts the two are equal or errors.
    fn assert_equals(self, bits: Self) -> Result<(), SimErr> {
        match self == bits {
            true  => Ok(()),
            false => Err(SimErr::InvalidInstrFormat),
        }
    }

    /// Gets the bits of `self` within the range provided,
    /// returning them as the least-significant bits
    fn slice(self, range: Range<usize>) -> Self;
}
impl DecodeUtils for u16 {
    fn interpret<T: FromBits>(self) -> T {
        T::from_bits(self)
    }

    fn slice(self, range: Range<usize>) -> Self {
        let len = range.end - range.start;
        (self >> range.start) & ((1 << len) - 1)
    }
}

impl FromBits for Reg {
    fn from_bits(bits: u16) -> Self {
        Reg::try_from(bits as u8).unwrap()
    }
}
impl<const N: u32> FromBits for IOffset<N> {
    fn from_bits(bits: u16) -> Self {
        Self::new_trunc(bits as i16)
    }
}
impl<const N: u32> FromBits for crate::ast::Offset<u16, N> {
    fn from_bits(bits: u16) -> Self {
        Self::new_trunc(bits)
    }
}