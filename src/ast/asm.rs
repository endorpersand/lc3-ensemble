//! This module holds the AST for statements from assembly source code.
//! 
//! For instructions that map to bytecode instructions, see [`sim::SimInstr`].
//! 
//! Useful structs in this module include:
//! - [`AsmInstr`]: An enum of all possible assembly source code instructions
//! - [`Directive`]: An enum of all possible assembly source code directives
//! - [`Stmt`]: The format for a single "statement" in assembly source code
//! 
//! [`sim::SimInstr`]: [`crate::ast::sim::SimInstr`]
use std::fmt::Write as _;

use super::{CondCode, IOffset, ImmOrReg, Label, Offset, PCOffset, Reg, TrapVect8};


type PCOffset9 = PCOffset<i16, 9>;
type PCOffset11 = PCOffset<i16, 11>;

/// An enum representing all of the possible instructions in LC-3 assembly code.
/// 
/// The variants in this enum represent instructions before assembly passes.
/// 
/// For instructions that map to typeable assembly code, refer to [`sim::SimInstr`].
/// 
/// [`sim::SimInstr`]: [`crate::ast::sim::SimInstr`]
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum AsmInstr {
    /// An ADD instruction.
    /// 
    /// # Operation
    /// 
    /// Evaluates the two operands, adds them, and stores the result to the destination register (`DR`).
    /// This also sets the condition code for the LC-3 machine.
    /// 
    /// # Syntax
    /// - `ADD DR, SR1, SR2`
    /// - `ADD DR, SR1, imm5`
    ADD(Reg, Reg, ImmOrReg<5>),

    /// An AND instruction.
    /// 
    /// # Operation
    /// 
    /// Evaluates the two operands, bitwise ANDs them, and stores the result to the destination register (`DR`).
    /// This also sets the condition code for the LC-3 machine.
    /// 
    /// # Syntax
    /// - `AND DR, SR1, SR2`
    /// - `AND DR, SR1, imm5`
    AND(Reg, Reg, ImmOrReg<5>),

    /// A BR instruction.
    /// 
    /// # Operation
    /// 
    /// Checks the current condition code and branches to the given `PCOffset9` 
    /// if the condition code matches one of the provided condition codes of the instruction.
    /// 
    /// # Syntax
    /// - `BR PCOffset9` (equivalent to `BRnzp`),
    /// - `BRn PCOffset9`
    /// - `BRz PCOffset9`
    /// - `BRnz PCOffset9`
    /// - `BRp PCOffset9`
    /// - `BRnp PCOffset9`
    /// - `BRzp PCOffset9`
    /// - `BRnzp PCOffset9`
    BR(CondCode, PCOffset9),
    
    /// A JMP instruction.
    /// 
    /// # Operation
    /// 
    /// Unconditionally jumps to the location stored in the given register (`BR`).
    /// 
    /// # Syntax
    /// - `JMP BR`
    JMP(Reg),
    
    /// A JSR instruction.
    /// 
    /// # Operation
    /// 
    /// Jumps to a given subroutine. This is done by storing the current PC into R7,
    /// and then unconditionally jumping to the location of the given `PCOffset11`.
    /// 
    /// # Syntax
    /// - `JSR PCOffset11`
    JSR(PCOffset11),
    
    /// A JSRR instruction.
    /// 
    /// # Operation
    /// 
    /// Jumps to a given subroutine. This is done by storing the current PC into R7,
    /// and then unconditionally jumping to the location stored in the given register (`BR`).
    /// 
    /// # Syntax
    /// - `JSRR BR`
    JSRR(Reg),
    
    /// A LD instruction.
    /// 
    /// # Operation
    /// 
    /// Computes an effective address (`PC + PCOffset9`), accesses the memory at that address,
    /// and stores it to the destination register (`DR`).
    /// This also sets the condition code for the LC-3 machine.
    /// 
    /// # Syntax
    /// - `LD DR, PCOffset9`
    LD(Reg, PCOffset9),
    
    /// A LDI instruction.
    /// 
    /// # Operation
    /// 
    /// Computes an effective address (`mem[PC + PCOffset9]`), accesses the memory at that address,
    /// and stores it to the destination register (`DR`).
    /// This also sets the condition code for the LC-3 machine.
    /// 
    /// # Syntax
    /// - `LDI DR, PCOffset9`
    LDI(Reg, PCOffset9),
    
    /// A LDR instruction.
    /// 
    /// # Operation
    /// 
    /// Computes an effective address (`mem[BR + offset6]`), accesses the memory at that address,
    /// and stores it to the destination register (`DR`).
    /// This also sets the condition code for the LC-3 machine.
    /// 
    /// # Syntax
    /// - `LDR DR, BR, offset6`
    LDR(Reg, Reg, IOffset<6>),
    
    /// A LEA instruction.
    /// 
    /// # Operation
    /// 
    /// Computes an effective address (`PC + PCOffset9`) and stores it to the destination register (`DR`).
    /// 
    /// # Syntax
    /// - `LEA DR, PCOffset9`
    LEA(Reg, PCOffset9),

    /// A NOT instruction.
    /// 
    /// # Operation
    /// 
    /// Evaluates the operand, bitwise NOTs them, and stores the result to the destination register (`DR`).
    /// This also sets the condition code for the LC-3 machine.
    /// 
    /// # Syntax
    /// - `NOT DR, SR`
    NOT(Reg, Reg),
    
    /// A RET instruction.
    /// 
    /// # Operation
    /// 
    /// Returns from a subroutine. This is an alias for `JMP R7`.
    /// 
    /// # Syntax
    /// - `RET`
    RET,
    
    /// A RTI instruction.
    /// 
    /// # Operation
    /// 
    /// Returns from a trap or interrupt.
    /// 
    /// # Syntax
    /// - `RTI`
    RTI,
    
    /// A ST instruction.
    /// 
    /// # Operation
    /// 
    /// Computes an effective address (`PC + PCOffset9`), and writes the value from the source register (`SR`)
    /// into the memory at that address,
    /// 
    /// # Syntax
    /// - `ST SR, PCOffset9`
    ST(Reg, PCOffset9),

    /// A STI instruction.
    /// 
    /// # Operation
    /// 
    /// Computes an effective address (`mem[PC + PCOffset9]`), and writes the value from the source register (`SR`)
    /// into the memory at that address,
    /// 
    /// # Syntax
    /// - `STI SR, PCOffset9`
    STI(Reg, PCOffset9),

    /// A STR instruction.
    /// 
    /// # Operation
    /// 
    /// Computes an effective address (`mem[BR + offset6]`), and writes the value from the source register (`SR`)
    /// into the memory at that address,
    /// 
    /// # Syntax
    /// - `STR SR, BR, offset6`
    STR(Reg, Reg, IOffset<6>),

    /// A TRAP instruction.
    /// 
    /// # Operation
    /// 
    /// Executes the trap with the given trap vector `TrapVect8`.
    /// 
    /// # Syntax
    /// - `TRAP TrapVect8`
    TRAP(TrapVect8),

    /* ALIASES AND TRAPS */

    /// A NOP instruction.
    /// 
    /// # Operation
    /// 
    /// Does nothing.
    /// 
    /// # Syntax
    /// - `NOP`
    /// - `NOP LABEL` (label is computed, but not used)
    /// - `NOP #99`
    NOP(PCOffset9),

    /// A GETC instruction.
    /// 
    /// # Operation
    /// 
    /// Gets a character from the keyboard, and store it into R0 (with the high 8 bits cleared). 
    /// This is an alias for `TRAP x20`.
    /// 
    /// # Syntax
    /// - `GETC`
    GETC,

    /// An OUT instruction.
    /// 
    /// # Operation
    /// 
    /// Writes a character from `R0[7:0]` to the display. This is an alias for `TRAP x21`.
    /// 
    /// # Syntax
    /// - `OUT`
    OUT,

    /// A PUTC instruction.
    /// 
    /// # Operation
    /// 
    /// Writes a character from `R0[7:0]` to the display. This is an alias for `TRAP x21`.
    /// 
    /// # Syntax
    /// - `PUTC`
    PUTC,

    /// A PUTS instruction.
    /// 
    /// # Operation
    /// 
    /// Prints characters in consecutive memory locations until a x00 character is read.
    /// This starts with the memory location pointed to by the address in `R0`.
    /// 
    /// This is an alias for `TRAP x22`.
    /// 
    /// # Syntax
    /// - `PUTS`
    PUTS,

    /// An IN instruction.
    /// 
    /// # Operation
    /// 
    /// Prompts the user for a character, stores the character into `R0` (with the high 8 bits cleared).
    /// Additionally, this prints the obtained character onto the display.
    /// 
    /// This is an alias for `TRAP x23`.
    /// 
    /// # Syntax
    /// - `IN`
    IN,

    /// A PUTSP instruction.
    /// 
    /// # Operation
    /// 
    /// Prints characters (two characters per memory location) until a x00 character is read.
    /// This starts with the memory location pointed to by the address in `R0`.
    /// This first prints the character in the low 8 bits, and then the character in the high 8 bits.
    /// 
    /// This is an alias for `TRAP x24`.
    /// 
    /// # Syntax
    /// - `PUTSP`
    PUTSP,

    /// A HALT instruction.
    /// 
    /// # Operation
    /// 
    /// Stops execution of the program. This is an alias for `TRAP x25`.
    /// 
    /// # Syntax
    /// - `HALT`
    HALT
}
impl std::fmt::Display for AsmInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ADD(dr, sr1, sr2) => write!(f, "ADD {dr}, {sr1}, {sr2}"),
            Self::AND(dr, sr1, sr2) => write!(f, "AND {dr}, {sr1}, {sr2}"),
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
            Self::JMP(br) => write!(f, "JMP {br}"),
            Self::JSR(off) => write!(f, "JSR {off}"),
            Self::JSRR(br) => write!(f, "JSRR {br}"),
            Self::LD(dr, off) => write!(f, "LD {dr}, {off}"),
            Self::LDI(dr, off) => write!(f, "LDI {dr}, {off}"),
            Self::LDR(dr, br, off) => write!(f, "LDR {dr}, {br}, {off}"),
            Self::LEA(dr, off) => write!(f, "LEA {dr}, {off}"),
            Self::NOT(dr, sr) => write!(f, "NOT {dr}, {sr}"),
            Self::RET   => f.write_str("RET"),
            Self::RTI   => f.write_str("RTI"),
            Self::ST(sr, off) => write!(f, "ST {sr}, {off}"),
            Self::STI(sr, off) => write!(f, "STI {sr}, {off}"),
            Self::STR(sr, br, off) => write!(f, "STR {sr}, {br}, {off}"),
            Self::TRAP(vect) => write!(f, "TRAP {vect:02X}"),
            Self::NOP(off) => write!(f, "NOP {off}"),
            Self::GETC  => f.write_str("GETC"),
            Self::OUT   => f.write_str("OUT"),
            Self::PUTC  => f.write_str("PUTC"),
            Self::PUTS  => f.write_str("PUTS"),
            Self::IN    => f.write_str("IN"),
            Self::PUTSP => f.write_str("PUTSP"),
            Self::HALT  => f.write_str("HALT"),
        }
    }
}

/// An enum representing all the possible directives in LC-3 assembly code.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Directive {
    /// An `.orig` directive.
    /// 
    /// # Operation
    /// 
    /// Starts a block of assembly.
    /// 
    /// # Syntax
    /// 
    /// `.orig ADDR`
    Orig(Offset<u16, 16>),

    /// A `.fill` directive.
    /// 
    /// # Operation
    /// 
    /// Writes some data into the given memory location.
    /// 
    /// # Syntax
    /// 
    /// `.fill DATA`
    /// `.fill LABEL`
    Fill(PCOffset<u16, 16>),
    
    
    /// A `.blkw` directive.
    /// 
    /// # Operation
    /// 
    /// Saves a provided number of memory locations for writing into.
    /// 
    /// # Syntax
    /// 
    /// `.blkw N`
    Blkw(Offset<u16, 16>),

    /// A `.stringz` directive.
    /// 
    /// # Operation
    /// 
    /// Writes a null-terminated string into the provided location.
    /// 
    /// # Syntax
    /// 
    /// `.stringz "A literal"`
    Stringz(String),

    /// A `.end` directive.
    /// 
    /// # Operation
    /// 
    /// Closes a block started by an `.orig`.
    /// 
    /// # Syntax
    /// 
    /// `.end`
    End,
    // External
}
impl std::fmt::Display for Directive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Orig(addr)   => write!(f, ".orig {addr:04X}"),
            Self::Fill(val)    => write!(f, ".fill {val}"),
            Self::Blkw(n)      => write!(f, ".blkw {n}"),
            Self::Stringz(val) => write!(f, ".stringz {val:?}"),
            Self::End          => write!(f, ".end"),
        }
    }
}

/// Either an instruction or a directive.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum StmtKind {
    #[allow(missing_docs)]
    Instr(AsmInstr),
    #[allow(missing_docs)]
    Directive(Directive)
}
impl std::fmt::Display for StmtKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StmtKind::Instr(i) => i.fmt(f),
            StmtKind::Directive(d) => d.fmt(f),
        }
    }
}

/// A "statement" in LC-3 assembly.
/// 
/// While not a defined term in LC-3 assembly, 
/// a statement here refers to either an instruction or a directive,
/// and the labels that are associated with it.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Stmt {
    /// The labels.
    pub labels: Vec<Label>,
    /// The instruction or directive.
    pub nucleus: StmtKind,
    /// The span of the nucleus.
    pub span: std::ops::Range<usize>
}
impl std::fmt::Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for label in &self.labels {
            label.fmt(f)?;
            f.write_char(' ')?;
        }
        self.nucleus.fmt(f)
    }
}
/**
 * Attempts to disassemble bytecode back into assembly instructions.
 */
pub fn disassemble(data: &[u16]) -> Vec<Stmt> {
    data.iter()
        .copied()
        .map(|word| {
            let msi = match word > 0x100 {
                true  => super::sim::SimInstr::decode(word).ok(),
                false => None,
            };

            let nucleus = match msi {
                Some(si) => {
                    let ai = match si {
                        super::sim::SimInstr::BR(cc, off) => AsmInstr::BR(cc, PCOffset::Offset(off)),
                        super::sim::SimInstr::ADD(dr, sr1, sr2) => AsmInstr::ADD(dr, sr1, sr2),
                        super::sim::SimInstr::LD(dr, off) => AsmInstr::LD(dr, PCOffset::Offset(off)),
                        super::sim::SimInstr::ST(sr, off) => AsmInstr::ST(sr, PCOffset::Offset(off)),
                        super::sim::SimInstr::JSR(off) => match off {
                            ImmOrReg::Imm(imm) => AsmInstr::JSR(PCOffset::Offset(imm)),
                            ImmOrReg::Reg(reg) => AsmInstr::JSRR(reg),
                        },
                        super::sim::SimInstr::AND(dr, sr1, sr2) => AsmInstr::AND(dr, sr1, sr2),
                        super::sim::SimInstr::LDR(dr, br, off) => AsmInstr::LDR(dr, br, off),
                        super::sim::SimInstr::STR(sr, br, off) => AsmInstr::STR(sr, br, off),
                        super::sim::SimInstr::RTI   => AsmInstr::RTI,
                        super::sim::SimInstr::NOT(dr, sr) => AsmInstr::NOT(dr, sr),
                        super::sim::SimInstr::LDI(dr, off) => AsmInstr::LDI(dr, PCOffset::Offset(off)),
                        super::sim::SimInstr::STI(sr, off) => AsmInstr::STI(sr, PCOffset::Offset(off)),
                        super::sim::SimInstr::JMP(super::reg_consts::R7) => AsmInstr::RET,
                        super::sim::SimInstr::JMP(br) => AsmInstr::JMP(br),
                        super::sim::SimInstr::LEA(dr, off) => AsmInstr::LEA(dr, PCOffset::Offset(off)),
                        super::sim::SimInstr::TRAP(vect) if vect.get() == 0x20 => AsmInstr::GETC,
                        super::sim::SimInstr::TRAP(vect) if vect.get() == 0x21 => AsmInstr::PUTC,
                        super::sim::SimInstr::TRAP(vect) if vect.get() == 0x22 => AsmInstr::PUTS,
                        super::sim::SimInstr::TRAP(vect) if vect.get() == 0x23 => AsmInstr::IN,
                        super::sim::SimInstr::TRAP(vect) if vect.get() == 0x24 => AsmInstr::PUTSP,
                        super::sim::SimInstr::TRAP(vect) if vect.get() == 0x25 => AsmInstr::HALT,
                        super::sim::SimInstr::TRAP(vect) => AsmInstr::TRAP(vect),
                    };

                    StmtKind::Instr(ai)
                },
                None => {
                    let fill = Directive::Fill(PCOffset::Offset(super::Offset::new_trunc(word)));
                    StmtKind::Directive(fill)
                },
            };

            Stmt { labels: vec![], nucleus, span: 0..0 }
        })
        .collect()
}