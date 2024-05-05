//! Utilities to debug simulation.
//! 
//! The key type here is [`Breakpoint`], which can be appended to the [`Simulator`]'s
//! breakpoint field to cause the simulator to break.
use std::fmt::Write;

use slotmap::{new_key_type, Key, KeyData, SlotMap};

use crate::ast::Reg;

use super::Simulator;

/// Common breakpoints.
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

    /// Breaks based on an arbitrarily defined function.
    /// 
    /// This can be constructed with the [`Breakpoint::generic`] function.
    Generic(BreakpointFn),

    /// All conditions have to apply for the break to be applied.
    And(Box<[Breakpoint]>),
    /// One of these conditions have to apply for the break to be applied.
    Or(Box<[Breakpoint]>),
}

impl Breakpoint where Breakpoint: Send + Sync { /* assert Breakpoint is send/sync */ }
type BreakpointFn = Box<dyn Fn(&Simulator) -> bool + Send + Sync + 'static>;

impl Breakpoint {
    /// Creates a breakpoint out of a function.
    pub fn generic(f: impl Fn(&Simulator) -> bool + Send + Sync + 'static) -> Breakpoint {
        Breakpoint::Generic(Box::new(f))
    }

    /// Checks if a break should occur.
    pub fn check(&self, sim: &Simulator) -> bool {
        match self {
            Breakpoint::PC(expected) => expected == &sim.pc,
            Breakpoint::Reg { reg, value: cmp } => cmp.check(sim.reg_file[*reg].get()),
            Breakpoint::Mem { addr, value: cmp } => cmp.check(sim.mem.get_raw(*addr).get()), // do not trigger IO devices
            Breakpoint::Generic(pred) => (pred)(sim),
            Breakpoint::And(conds) => conds.iter().all(|b| b.check(sim)),
            Breakpoint::Or(conds) => conds.iter().any(|b| b.check(sim)),
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
            Self::Generic(_) => f.debug_struct("Generic").finish_non_exhaustive()?,
            Self::And(conds) => {
                let Some((last, rest)) = conds.split_last() else { return f.write_str("always") };

                for bp in rest {
                    f.write_char('(')?;
                    bp.fmt_bp(f)?;
                    f.write_str(") && ")?;
                }
                
                f.write_str("(")?;
                last.fmt_bp(f)?;
                f.write_char(')')?;
            },
            Self::Or(conds) => {
                let Some((last, rest)) = conds.split_last() else { return f.write_str("never") };

                for bp in rest {
                    f.write_char('(')?;
                    bp.fmt_bp(f)?;
                    f.write_str(") || ")?;
                }
                
                f.write_str("(")?;
                last.fmt_bp(f)?;
                f.write_char(')')?;
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
impl std::ops::BitAnd for Breakpoint {
    type Output = Breakpoint;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = vec![];
        match (self, rhs) {
            (Self::And(left), Self::And(right)) => {
                result.extend(Vec::from(left));
                result.extend(Vec::from(right));
            },
            (Self::And(left), right) => {
                result.extend(Vec::from(left));
                result.push(right);
            },
            (left, Self::And(right)) => {
                result.push(left);
                result.extend(Vec::from(right));
            },
            (left, right) => {
                result.push(left);
                result.push(right);
            }
        }
        Self::And(result.into_boxed_slice())
    }
}
impl std::ops::BitOr for Breakpoint {
    type Output = Breakpoint;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = vec![];
        match (self, rhs) {
            (Self::Or(left), Self::Or(right)) => {
                result.extend(Vec::from(left));
                result.extend(Vec::from(right));
            },
            (Self::Or(left), right) => {
                result.extend(Vec::from(left));
                result.push(right);
            },
            (left, Self::Or(right)) => {
                result.push(left);
                result.extend(Vec::from(right));
            },
            (left, right) => {
                result.push(left);
                result.push(right);
            }
        }
        Self::Or(result.into_boxed_slice())
    }
}
impl PartialEq for Breakpoint {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::PC(l0), Self::PC(r0)) => l0 == r0,
            (Self::Reg { reg: l_reg, value: l_value }, Self::Reg { reg: r_reg, value: r_value }) => l_reg == r_reg && l_value == r_value,
            (Self::Mem { addr: l_addr, value: l_value }, Self::Mem { addr: r_addr, value: r_value }) => l_addr == r_addr && l_value == r_value,
            (Self::Generic(_), Self::Generic(_)) => false, /* can't really figure this one out */
            (Self::And(l0), Self::And(r0)) => l0 == r0,
            (Self::Or(l0), Self::Or(r0)) => l0 == r0,
            _ => false,
        }
    }
}

/// Predicate checking whether the current value is equal to the value.
#[derive(PartialEq, Eq, Debug)]
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

#[cfg(test)]
mod test {
    use crate::ast::reg_consts::R7;
    use crate::sim::debug::{Breakpoint, Comparator};

    #[test]
    fn print() {
        println!("{:?}", Breakpoint::Reg { reg: R7, value: Comparator::Lt(14) } & Breakpoint::Reg { reg: R7, value: Comparator::Ge(10) });
        println!("{:?}", Breakpoint::Mem { addr: 0x4000, value: Comparator::Lt(14) } | Breakpoint::Mem { addr: 0x5000, value: Comparator::Ge(10) } & Breakpoint::PC(0x3000));
    }
}

new_key_type! {
    /// Key to index into a breakpoint list.
    pub struct BreakpointKey;
}
impl BreakpointKey {
    /// Converts key into a value that can be passed through FFI.
    pub fn as_ffi(self) -> u64 {
        self.data().as_ffi()
    }
    /// Converts FFI value to key.
    pub fn from_ffi(val: u64) -> Self {
        Self::from(KeyData::from_ffi(val))
    }
}
/// A list of breakpoints.
/// 
/// This works similarly to GDB breakpoints, in that creating a breakpoint
/// gives you an ID which you can use to query or remove the breakpoint later.
#[derive(Debug, Default)]
pub struct BreakpointList {
    inner: SlotMap<BreakpointKey, Breakpoint>
}

impl BreakpointList {
    /// Creates a new breakpoint list.
    pub fn new() -> Self {
        Default::default()
    }
    
    /// Gets an immutable reference to breakpoint with a given key, 
    /// returning None if it was already removed.
    pub fn get(&self, key: BreakpointKey) -> Option<&Breakpoint> {
        self.inner.get(key)
    }
    /// Gets a mutable reference to breakpoint with a given key, 
    /// returning None if it was already removed.
    pub fn get_mut(&mut self, key: BreakpointKey) -> Option<&mut Breakpoint> {
        self.inner.get_mut(key)
    }
    /// Checks if the key is currently associated with some breakpoint.
    pub fn contains_key(&self, key: BreakpointKey) -> bool {
        self.inner.contains_key(key)
    }

    /// Counts the number of defined breakpoints.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Checks if the breakpoint list is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Inserts a breakpoint into the list and returns its key.
    pub fn insert(&mut self, bpt: Breakpoint) -> BreakpointKey {
        self.inner.insert(bpt)
    }
    /// Remove breakpoint with given key.
    /// 
    /// If breakpoint was previously removed, then this returns None.
    pub fn remove(&mut self, key: BreakpointKey) -> Option<Breakpoint> {
        self.inner.remove(key)
    }

    /// Remove breakpoint that matches a given value.
    /// 
    /// This is a utility function to remove a breakpoint by value.
    /// If you'd like to remove by key, use [`BreakpointList::remove`].
    /// 
    /// Note that this can only reliably remove PC, Reg, and Mem breakpoints.
    /// The remaining 3 may fail to match even if an identical breakpoint
    /// appears in the list.
    pub fn remove_breakpoint(&mut self, breakpoint: &Breakpoint) -> Option<Breakpoint> {
        self.remove_breakpoint_by(|bpt| bpt == breakpoint)
    }
    /// Remove breakpoint that matches a given predicate.
    /// 
    /// This is a utility function to remove a breakpoint by value.
    /// If you'd like to remove by key, use [`BreakpointList::remove`].
    pub fn remove_breakpoint_by(&mut self, mut pred: impl FnMut(&mut Breakpoint) -> bool) -> Option<Breakpoint> {
        self.inner.iter_mut()
            .find_map(|(k, b)| pred(b).then_some(k))
            .and_then(|k| self.inner.remove(k))
    }
    /// Removes all breakpoints from the list.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

        /// An iterator visiting all breakpoints and their associated keys in arbitrary order.
    /// 
    /// This function must iterate over all slots, empty or not. In the face of many deleted elements it can be inefficient.
    pub fn iter(&self) -> slotmap::basic::Iter<BreakpointKey, Breakpoint> {
        self.inner.iter()
    }
    /// An iterator visiting all breakpoints and their associated keys in arbitrary order, with a mutable reference to each breakpoint.
    /// 
    /// This function must iterate over all slots, empty or not. In the face of many deleted elements it can be inefficient.
    pub fn iter_mut(&mut self) -> slotmap::basic::IterMut<BreakpointKey, Breakpoint> {
        self.inner.iter_mut()
    }
    /// An iterator visiting all keys in arbitrary order.
    /// 
    /// This function must iterate over all slots, empty or not. In the face of many deleted elements it can be inefficient.
    pub fn keys(&self) -> slotmap::basic::Keys<BreakpointKey, Breakpoint> {
        self.inner.keys()
    }
    /// An iterator visiting all breakpoints in arbitrary order.
    /// 
    /// This function must iterate over all slots, empty or not. In the face of many deleted elements it can be inefficient.
    pub fn values(&self) -> slotmap::basic::Values<BreakpointKey, Breakpoint> {
        self.inner.values()
    }
    /// An iterator visiting all breakpoints in arbitrary order, with a mutable reference to each breakpoint.
    /// 
    /// This function must iterate over all slots, empty or not. In the face of many deleted elements it can be inefficient.
    pub fn values_mut(&mut self) -> slotmap::basic::ValuesMut<BreakpointKey, Breakpoint> {
        self.inner.values_mut()
    }
}
impl std::ops::Index<BreakpointKey> for BreakpointList {
    type Output = Breakpoint;

    fn index(&self, index: BreakpointKey) -> &Self::Output {
        &self.inner[index]
    }
}
impl std::ops::IndexMut<BreakpointKey> for BreakpointList {
    fn index_mut(&mut self, index: BreakpointKey) -> &mut Self::Output {
        &mut self.inner[index]
    }
}