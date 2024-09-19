use std::collections::VecDeque;
use std::sync::{Arc, RwLock, RwLockWriteGuard, TryLockError};

use super::{DevWrapper, ExternalDevice, Interrupt, KBDR, KBSR, KB_INTP, KB_INTV};

/// Scaffolding needed to implement [`ExternalDevice`] for keyboard devices.
trait KeyboardDevice: Send + Sync + 'static {
    /// State of interrupt enabled.
    fn interrupts_enabled(&self) -> bool;
    /// Sets interrupt enabled.
    fn set_interrupts_enabled(&mut self, value: bool);

    /// Whether the keyboard has input to take.
    fn ready(&self) -> bool;
    /// Reads a character from the input (but does not take it).
    fn get_input(&self) -> Option<u8>;
    /// Reads and removes a character from the input.
    fn pop_input(&mut self) -> Option<u8>;
    /// Clears the input completely.
    fn clear_input(&mut self);
}

impl<K: KeyboardDevice> ExternalDevice for DevWrapper<K, dyn KeyboardDevice> {
    fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16> {
        match addr {
            KBSR => {
                Some(u16::from(self.ready()) << 15 | u16::from(self.interrupts_enabled()) << 14)
            },
            KBDR if effectful => self.pop_input().map(u16::from),
            KBDR => self.get_input().map(u16::from),
            _ => None
        }
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        match addr {
            KBSR => {
                let ie = (data >> 14) & 1 != 0;
                self.set_interrupts_enabled(ie);
                true
            },
            _ => false
        }
    }
    
    fn io_reset(&mut self) {
        self.clear_input();
        self.set_interrupts_enabled(false);
    }

    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        match self.ready() && self.interrupts_enabled() {
            true  => Some(Interrupt::vectored(KB_INTV, KB_INTP)),
            false => None,
        }
    }
}

/// Keyboard that accesses input from a memory buffer.
#[derive(Default, Clone)]
pub struct BufferedKeyboard {
    buffer: Arc<RwLock<VecDeque<u8>>>,
    interrupts_enabled: bool
}
impl BufferedKeyboard {
    /// Creates a new keyboard, wrapping it around a given buffer.
    pub fn new(buffer: Arc<RwLock<VecDeque<u8>>>) -> Self {
        Self { buffer, interrupts_enabled: false }
    }

    /// Gets a reference to the internal buffer of this keyboard.
    pub fn get_buffer(&self) -> &Arc<RwLock<VecDeque<u8>>> {
        &self.buffer
    }
    
    fn try_input(&self) -> Option<RwLockWriteGuard<'_, VecDeque<u8>>> {
        match self.buffer.try_write() {
            Ok(g) => Some(g),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}
impl KeyboardDevice for BufferedKeyboard {
    fn interrupts_enabled(&self) -> bool {
        self.interrupts_enabled
    }

    fn set_interrupts_enabled(&mut self, value: bool) {
        self.interrupts_enabled = value;
    }

    fn ready(&self) -> bool {
        self.try_input().is_some_and(|buf| !buf.is_empty())
    }

    fn get_input(&self) -> Option<u8> {
        self.try_input()?.front().copied()
    }

    fn pop_input(&mut self) -> Option<u8> {
        self.try_input()?.pop_front()
    }
    
    fn clear_input(&mut self) {
        if let Some(mut inp) = self.try_input() {
            inp.clear();
        }
    }
}
impl ExternalDevice for BufferedKeyboard {
    fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16> {
        DevWrapper::wrap(self).io_read(addr, effectful)
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        DevWrapper::wrap(self).io_write(addr, data)
    }

    fn io_reset(&mut self) {
        DevWrapper::wrap(self).io_reset()
    }
    
    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        DevWrapper::wrap(self).poll_interrupt()
    }
    
    fn _to_sim_device(self, _: super::internals::ToSimDeviceToken) -> super::internals::SimDevice
        where Self: Sized
    {
        super::internals::SimDevice::Keyboard(self)
    }
}
