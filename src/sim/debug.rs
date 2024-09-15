//! Utilities to debug simulation.
//! 
//! The key type here is [`Breakpoint`], which can be appended to the [`Simulator`]'s
//! breakpoint field to cause the simulator to break.
use std::fmt::Write;

use crate::ast::Reg;

use super::Simulator;

/// Common breakpoints.
#[derive(PartialEq, Eq, Hash)]
pub enum Breakpoint {
    /// Break when the PC is equal to the given value.
    PC(u16),

    /// Break when the provided register is set to a given value.
    Reg {
        /// Register to check.
        reg: Reg,
        /// Predicate to break against.
        value: Comparator
    },
    /// Break when the provided memory address is written to with a given value.
    Mem {
        /// Address to check.
        addr: u16,
        /// Predicate to break against.
        value: Comparator
    },
}

impl Breakpoint where Breakpoint: Send + Sync { /* assert Breakpoint is send/sync */ }

impl Breakpoint {
    /// Checks if a break should occur.
    pub fn check(&self, sim: &Simulator) -> bool {
        match self {
            Breakpoint::PC(expected) => expected == &sim.pc,
            Breakpoint::Reg { reg, value: cmp } => cmp.check(sim.reg_file[*reg].get()),
            Breakpoint::Mem { addr, value: cmp } => cmp.check(sim.mem[*addr].get()), // do not trigger IO devices
        }
    }

    fn fmt_bp(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::PC(expected) => {
                write!(f, "PC == x{expected:04X}")?;
            },
            Self::Reg { reg, value } => {
                write!(f, "{reg} ")?;
                value.fmt_cmp(f)?;
            },
            Self::Mem { addr, value } => {
                write!(f, "mem[x{addr:04X}] ")?;
                value.fmt_cmp(f)?;
            },
        }
        Ok(())
    }
}
impl std::fmt::Debug for Breakpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Breakpoint(")?;
        self.fmt_bp(f)?;
        f.write_char(')')
    }
}
/// Predicate checking whether the current value is equal to the value.
#[derive(PartialEq, Eq, Hash, Debug)]
pub enum Comparator {
    /// Never breaks.
    Never,
    /// Break if the desired value is less than the provided value.
    Lt(u16),
    /// Break if the desired value is equal to the provided value.
    Eq(u16),
    /// Break if the desired value is less than or equal to the provided value.
    Le(u16),
    /// Break if the desired value is greater than the provided value.
    Gt(u16),
    /// Break if the desired value is not equal to the provided value.
    Ne(u16),
    /// Break if the desired value is greater than or equal to the provided value.
    Ge(u16),
    /// Always breaks.
    Always
}
impl Comparator {
    /// Checks if the operand passes the comparator.
    pub fn check(&self, operand: u16) -> bool {
        match *self {
            Comparator::Never  => false,
            Comparator::Lt(r)  => operand < r,
            Comparator::Eq(r)  => operand == r,
            Comparator::Le(r)  => operand <= r,
            Comparator::Gt(r)  => operand > r,
            Comparator::Ne(r)  => operand != r,
            Comparator::Ge(r)  => operand >= r,
            Comparator::Always => true,
        }
    }

    fn fmt_cmp(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Comparator::Never  => f.write_str("never"),
            Comparator::Lt(r)  => write!(f, "< {r}"),
            Comparator::Eq(r)  => write!(f, "== {r}"),
            Comparator::Le(r)  => write!(f, "<= {r}"),
            Comparator::Gt(r)  => write!(f, "> {r}"),
            Comparator::Ne(r)  => write!(f, "!= {r}"),
            Comparator::Ge(r)  => write!(f, ">= {r}"),
            Comparator::Always => f.write_str("always"),
        }
    }
}