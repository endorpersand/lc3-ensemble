//! A small module that tracks changes in memory,
//! which can be used for whatever purposes necessary.

use std::collections::BTreeSet;

/// A struct that tracks changes in memory.
#[derive(Debug)]
pub struct ChangeObserver {
    mem: BTreeSet<u16>
    // Maybe add reg, PC, PSR, MCR support here?
    //
    // Memory can be easily tracked through the write_mem method,
    // but reg/PC/PSR/MCR aren't exactly encapsulated in this way.
}
impl ChangeObserver {
    pub fn new() -> Self {
        Self {
            mem: BTreeSet::new()
        }
    }

    /// Clears all changes.
    pub fn clear(&mut self) {
        std::mem::take(self);
    }
    /// Sets the memory changed state at the given address to true.
    pub fn set_mem_changed(&mut self, addr: u16) {
        self.mem.insert(addr);
    }
    /// Gets the memory changed state at the given address.
    pub fn mem_changed(&mut self, addr: u16) -> bool {
        self.mem.contains(&addr)
    }
    /// Takes the memory changes that happened since last clear,
    ///  as well as clearing the memory.
    /// 
    /// This iterator is sorted in address order.
    pub fn take_mem_changes(&mut self) -> impl Iterator<Item=u16> {
        std::mem::take(&mut self.mem).into_iter()
    }
}
impl Default for ChangeObserver {
    fn default() -> Self {
        Self::new()
    }
}