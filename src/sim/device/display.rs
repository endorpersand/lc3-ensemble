use std::sync::{Arc, RwLock, RwLockWriteGuard, TryLockError};

use super::{DevWrapper, ExternalDevice, Interrupt, DDR, DSR};

/// Scaffolding required to implement [`ExternalDevice`] for display devices.
trait DisplayDevice: Send + Sync + 'static {
    /// Whether the display is ready to take output.
    fn ready(&self) -> bool;
    /// Sends output, returns whether the output was successfully accepted.
    fn send_output(&mut self, byte: u8) -> bool;

    /// Clears all of the current output.
    fn clear_output(&mut self);
}
impl<D: DisplayDevice> ExternalDevice for DevWrapper<D, dyn DisplayDevice> {
    fn io_read(&mut self, addr: u16, _effectful: bool) -> Option<u16> {
        match addr {
            DSR => Some(u16::from(self.ready()) << 15),
            _   => None
        }
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        match addr {
            DDR => self.send_output(data as u8),
            _ => false
        }
    }

    fn io_reset(&mut self) {
        self.clear_output();
    }

    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        None
    }
}

/// A display that delegates its output to a buffer.
#[derive(Default, Clone)]
pub struct BufferedDisplay {
    buffer: Arc<RwLock<Vec<u8>>>
}
impl BufferedDisplay {
    /// Creates a new display, wrapping it around a given buffer.
    pub fn new(buffer: Arc<RwLock<Vec<u8>>>) -> Self {
        Self { buffer }
    }

    /// Gets a reference to the internal buffer of this display.
    pub fn get_buffer(&self) -> &Arc<RwLock<Vec<u8>>> {
        &self.buffer
    }

    fn try_output(&self) -> Option<RwLockWriteGuard<'_, Vec<u8>>> {
        match self.buffer.try_write() {
            Ok(g) => Some(g),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}
impl DisplayDevice for BufferedDisplay {
    fn ready(&self) -> bool {
        self.try_output().is_some()
    }

    fn send_output(&mut self, byte: u8) -> bool {
        match self.try_output() {
            Some(mut d) => {
                d.push(byte);
                true
            },
            None => false,
        }
    }
    
    fn clear_output(&mut self) {
        if let Some(mut out) = self.try_output() {
            out.clear();
        }
    }
}
impl ExternalDevice for BufferedDisplay {
    fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16> {
        DevWrapper::wrap(self).io_read(addr, effectful)
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        DevWrapper::wrap(self).io_write(addr, data)
    }

    fn io_reset(&mut self) {
        DevWrapper::wrap(self).io_reset();
    }

    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        DevWrapper::wrap(self).poll_interrupt()
    }
    
    fn _to_sim_device(self, _: super::internals::ToSimDeviceToken) -> super::internals::SimDevice
        where Self: Sized
    {
        super::internals::SimDevice::Display(self)
    }
}