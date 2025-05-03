//! Module handles memory access observers,
//! which store which accesses occur at a given memory location.
//! 
//! You would typically access an observer via the [`Simulator::observer`] field.
//! This [`AccessObserver`] can be used to read or update accesses via its [`get_mem_accesses`]
//! and [`update_mem_accesses`] methods.
//! 
//! [`Simulator::observer`]: crate::sim::Simulator::observer
//! [`get_mem_accesses`]: AccessObserver::get_mem_accesses
//! [`update_mem_accesses`]: AccessObserver::update_mem_accesses

use std::collections::BTreeMap;

/// The set of accesses which have occurred at this location.
/// 
/// ## Example
/// 
/// ```
/// # use lc3_ensemble::sim::observer::AccessSet;
/// 
/// let accesses = AccessSet::READ;
/// assert!(accesses.accessed());
/// assert!(accesses.read());
/// assert!(!accesses.written());
/// assert!(!accesses.modified());
/// ```
#[derive(Default, Clone, Copy)]
pub struct AccessSet(u8);
impl AccessSet {
    /// Set with only the read flag enabled.
    pub const READ: Self = Self(1 << 0);
    /// Set with only the write flag enabled.
    pub const WRITTEN: Self = Self(1 << 1);
    /// Set with only the modify flag enabled.
    pub const MODIFIED: Self = Self(1 << 2);

    /// True if any access has occurred.
    pub fn accessed(&self) -> bool {
        self.0 != 0
    }
    
    /// True if a read has occurred.
    pub fn read(&self) -> bool {
        self.0 & Self::READ.0 != 0
    }
    /// True if a write has occurred (does not necessarily have to change data).
    pub fn written(&self) -> bool {
        self.0 & Self::WRITTEN.0 != 0
    }
    /// True if a write has occurred (data must change).
    pub fn modified(&self) -> bool {
        self.0 & Self::MODIFIED.0 != 0
    }
}
impl std::ops::BitOr for AccessSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for AccessSet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}
impl std::fmt::Debug for AccessSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccessFlags")
            .field("accessed", &self.accessed())
            .field("read", &self.read())
            .field("written", &self.written())
            .field("modified", &self.modified())
            .finish()
    }
}

/// A struct that tracks accesses in memory.
#[derive(Debug)]
pub struct AccessObserver {
    mem: BTreeMap<u16, AccessSet>
    // Maybe add reg, PC, PSR, MCR support here?
    //
    // Memory can be easily tracked through the write_mem method,
    // but reg/PC/PSR/MCR aren't exactly encapsulated in this way.
}
impl AccessObserver {
    /// Creates a new access observer.
    pub fn new() -> Self {
        Self {
            mem: Default::default()
        }
    }

    /// Clears all accesses.
    pub fn clear(&mut self) {
        std::mem::take(self);
    }

    /// Gets the access set for the given memory location.
    pub fn get_mem_accesses(&self, addr: u16) -> AccessSet {
        self.mem.get(&addr).copied().unwrap_or_default()
    }

    /// Adds new flags to the access set for the given memory location.
    pub fn update_mem_accesses(&mut self, addr: u16, set: AccessSet) {
        *self.mem.entry(addr).or_default() |= set;
    }
    
    /// Takes all memory accesses which have occurred since last clear,
    /// as well as clearing memory accesses.
    /// 
    /// This iterator is sorted in address order.
    pub fn take_mem_accesses(&mut self) -> impl Iterator<Item=(u16, AccessSet)> {
        std::mem::take(&mut self.mem).into_iter()
    }
}
impl Default for AccessObserver {
    fn default() -> Self {
        Self::new()
    }
}