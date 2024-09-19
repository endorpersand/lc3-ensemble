//! Handlers for external devices connected to the Simulator.
//! 
//! This handles IO devices (such as the keyboard and display)
//! and handles interrupts.
//! 
//! The core types here are:
//! - [`ExternalDevice`]: A device which can be connected to the Simulator.
//! - [`DeviceHandler`]: The handler for the Simulator's IO ports & interrupts.
//! 
//! This module also provides some IO devices:
//! - [`NullDevice`]: Does nothing.
//! - [`BufferedKeyboard`]: Keyboard device that reads off of an input buffer.
//! - [`BufferedDisplay`]: Display device that writes to an output buffer.

mod keyboard;
mod display;
mod timer;

use super::IO_START;
pub use keyboard::BufferedKeyboard;
pub use display::BufferedDisplay;

const KBSR: u16 = 0xFE00;
const KBDR: u16 = 0xFE02;
const DSR:  u16 = 0xFE04;
const DDR:  u16 = 0xFE06;
const KB_INTV: u8 = 0x80;
const KB_INTP: u8 = 0b100;
const DEVICE_SLOTS: usize = IO_START.wrapping_neg() as usize;

/// An external device, which can be accessed via memory-mapped IO or via interrupts.
pub trait ExternalDevice: Send + Sync + 'static {
    /// Reads the data at the given memory-mapped address.
    /// 
    /// If successful, this returns the value returned from that address.
    /// If unsuccessful, this returns `None`.
    fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16>;

    /// Writes the data to the given memory-mapped address.
    /// 
    /// This returns whether the write was successful or not.
    fn io_write(&mut self, addr: u16, data: u16) -> bool;

    /// Resets device.
    fn io_reset(&mut self);

    /// During each instruction cycle, this function is called once to see whether
    /// to trigger an interrupt.
    /// 
    /// Of course, in the real world, the devices would send an interrupt signal
    /// which would be detected, but that can't really be done here.
    fn poll_interrupt(&mut self) -> Option<Interrupt>;

    /// Hacky specialization.
    /// 
    /// This allows [`super::Simulator::open_io`]'s signature to just require an [`IODevice`]
    /// and does not require that we expose IO internals.
    #[doc(hidden)]
    fn _to_sim_device(self, _: internals::ToSimDeviceToken) -> internals::SimDevice
        where Self: Sized
    {
        internals::SimDevice::Custom(Box::new(self))
    }
}

fn _get_dev_id(ports: &[u16], port: u16) -> Option<u16> {
    ports.get(port.checked_sub(IO_START)? as usize).copied()
}

/// The central hub for all external devices for the Simulator.
#[derive(Debug)]
pub struct DeviceHandler {
    devices: Vec<internals::SimDevice>,
    io_ports: Box<[u16; DEVICE_SLOTS]>
}

impl DeviceHandler {
    const NULL_DEV: u16 = 0;
    const KB_DEV: u16 = 1;
    const DS_DEV: u16 = 2;
    const FIXED_DEVS: &'static [u16] = &[Self::NULL_DEV, Self::KB_DEV, Self::DS_DEV];

    /// Creates a new device handler.
    pub fn new() -> Self {
        use internals::SimDevice::Null;

        let mut handler = Self {
            devices: vec![Null, Null, Null],
            io_ports: vec![0; DEVICE_SLOTS]
                .try_into()
                .expect("array should have the correct number of elements")
        };

        handler.set_port(KBSR, Self::KB_DEV);
        handler.set_port(KBDR, Self::KB_DEV);
        handler.set_port(DSR,  Self::DS_DEV);
        handler.set_port(DDR,  Self::DS_DEV);

        handler
    }

    /// Gets the device ID bound to a given address.
    fn get_dev_id(&self, port: u16) -> Option<u16> {
        _get_dev_id(&*self.io_ports, port)
    }
    // Sets port, failing if not possible.
    fn set_port(&mut self, port: u16, dev_id: u16) -> bool {
        fn get_dev_id_mut(ports: &mut [u16], port: u16) -> Option<&mut u16> {
            ports.get_mut(port.checked_sub(IO_START)? as usize)
        }

        let dev_len = self.devices.len();

        if let Some(d @ 0) = get_dev_id_mut(&mut *self.io_ports, port) {
            if (dev_id as usize) < dev_len {
                *d = dev_id;
            }
        }

        false
    }

    /// Set the keyboard device.
    pub fn set_keyboard(&mut self, kb: impl ExternalDevice) {
        self.devices[Self::KB_DEV as usize] = kb._to_sim_device(internals::ToSimDeviceToken(()));
    }
    /// Set the display device.
    pub fn set_display(&mut self, ds: impl ExternalDevice) {
        self.devices[Self::DS_DEV as usize] = ds._to_sim_device(internals::ToSimDeviceToken(()));
    }
    /// Add a new device (which is not a keyboard or a display).
    /// 
    /// This accepts an external device and the addresses which the device should act on.
    /// If successful, the ID of the device is returned.
    /// 
    /// # Errors
    /// If the device cannot be added, it will be returned back to the user.
    /// 
    /// The cases where the device cannot be added include:
    /// - The number of devices ever added exceeds [`u16::MAX`].
    /// - At least one of the addresses provided are already occupied by another device or not an IO port.
    pub fn add_device<D: ExternalDevice>(&mut self, dev: D, addrs: &[u16]) -> Result<u16, D> {
        // Fail if too many devices
        let Ok(dev_id) = u16::try_from(self.devices.len()) else { return Err(dev) };
        // Fail if port is preoccupied or non-existence
        let all_valid_ports = addrs.iter()
            .copied()
            .map(|p| self.get_dev_id(p))
            .all(|m_dev_id| m_dev_id.is_some_and(|d| d == 0));
        if !all_valid_ports { return Err(dev) };

        // Success:
        self.devices.push(dev._to_sim_device(internals::ToSimDeviceToken(())));

        for &p in addrs {
            self.set_port(p, dev_id);
        }

        Ok(dev_id)
    }
    /// Removes the device at the given device ID.
    pub fn remove_device(&mut self, dev_id: u16) {
        if let Some(dev_ref) = self.devices.get_mut(dev_id as usize) {
            std::mem::take(dev_ref);

            // Only reset ports if they're not keyboard/display.
            if !Self::FIXED_DEVS.contains(&dev_id) {
                self.io_ports.iter_mut()
                    .filter(|p| **p == dev_id)
                    .for_each(|p| *p = Self::NULL_DEV);
            }
        }
    }
}
impl Default for DeviceHandler {
    fn default() -> Self {
        Self::new()
    }
}
impl ExternalDevice for DeviceHandler {
    /// Accesses the IO device mapped to the given address and tries [`ExternalDevice::io_read`] on it.
    fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16> {
        let dev_id = self.get_dev_id(addr)?;
        self.devices[dev_id as usize].io_read(addr, effectful)
    }
    
    /// Accesses the IO device mapped to the given address and tries [`ExternalDevice::io_write`] on it.
    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        let Some(dev_id) = self.get_dev_id(addr) else { return false };
        self.devices[dev_id as usize].io_write(addr, data)
    }
    
    /// Resets all the devices connected to this handler.
    fn io_reset(&mut self) {
        self.devices.iter_mut().for_each(ExternalDevice::io_reset)
    }

    /// Checks for interrupts on all devices.
    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        self.devices.iter_mut()
            .filter_map(|s| s.poll_interrupt())
            .max_by_key(|i| i.priority().unwrap_or(0b1000))
    }
}

/// An interrupt.
/// 
/// This is output by an implementation of [`ExternalDevice::poll_interrupt`] if an interrupt should occur.
#[derive(Debug)]
pub struct Interrupt {
    pub(super) kind: InterruptKind
}

#[derive(Debug)]
pub(super) enum InterruptKind {
    /// A vectored interrupt (one handled by delegating to a corresponding interrupt vector).
    Vectored {
        /// Interrupt vector
        vect: u8,
        /// Priority value from 0-7
        priority: u8
    },

    /// An external interrupt (one handled by some force outside of the Simulator).
    External(ExternalInterrupt)
}
impl Interrupt {
    /// Creates a new vectored interrupt.
    /// 
    /// Note that the priority is truncated to 3 bits.
    pub fn vectored(vect: u8, priority: u8) -> Self {
        Self { kind: InterruptKind::Vectored { vect, priority: priority & 0b111 } }
    }
    /// Creates a new external interrupt (one not handled by the OS).
    /// 
    /// When this type of interrupt is raised, the simulator raises [`SimErr::Interrupt`]
    /// which can be used to handle the resulting `InterruptErr`.
    /// 
    /// One example where this is used is in Python bindings. 
    /// In that case, we want to be able to halt the Simulator on a `KeyboardInterrupt`.
    /// However, by default, Python cannot signal to the Rust library that a `KeyboardInterrupt`
    /// has occurred. Thus, we can add a signal handler as an external interrupt to allow the `KeyboardInterrupt`
    /// to be handled properly.
    /// 
    /// [`SimErr::Interrupt`]: super::SimErr::Interrupt
    pub fn external(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self { kind: InterruptKind::External(ExternalInterrupt::new(e)) }
    }

    /// Calculates the priority for a given interrupt.
    /// 
    /// - For all vectored interrupts, this is the three least-significant bits of the priority value.
    /// - For external interrupts, this is None
    pub fn priority(&self) -> Option<u8> {
        match self.kind {
            InterruptKind::Vectored { vect: _, priority } => Some(priority & 0b111),
            InterruptKind::External(_) => None,
        }
    }
}

/// An interrupt not handled by the simulator occurred.
/// 
/// See [`Interrupt::external`].
#[derive(Debug)]
pub struct ExternalInterrupt(Box<dyn std::error::Error + Send + Sync + 'static>);
impl ExternalInterrupt {
    /// Creates a new [`ExternalInterrupt`].
    fn new(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        ExternalInterrupt(Box::new(e))
    }

    /// Get the internal error from this interrupt.
    /// 
    /// This can be downcast by the typical methods on `dyn Error`.
    pub fn into_inner(self) -> Box<dyn std::error::Error + Send + Sync + 'static> {
        self.0
    }
}
impl std::fmt::Display for ExternalInterrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for ExternalInterrupt {}


/// Small wrapper that handles the base impls for keyboard and display devices.
/// 
/// A little HACKy.
/// See `keyboard.rs` and `display.rs` for actual use.
#[repr(transparent)]
struct DevWrapper<D, T: ?Sized>(D, std::marker::PhantomData<T>);
impl<D, T: ?Sized> DevWrapper<D, T> {
    pub fn wrap(d: &mut D) -> &mut DevWrapper<D, T> {
        // SAFETY: DevWrapper<D, T> is transparent with D
        unsafe { &mut *(d as *mut D).cast::<DevWrapper<D, T>>() }
    }
}
impl<D, T: ?Sized> std::ops::Deref for DevWrapper<D, T> {
    type Target = D;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<D, T: ?Sized> std::ops::DerefMut for DevWrapper<D, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
//

mod internals {
    use super::{BufferedDisplay, BufferedKeyboard, ExternalDevice, NullDevice};

    #[derive(Default)]
    pub enum SimDevice {
        #[default]
        Null,
        Keyboard(BufferedKeyboard),
        Display(BufferedDisplay),
        Custom(Box<dyn ExternalDevice + Send + Sync + 'static>)
    }

    impl ExternalDevice for SimDevice {
        fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16> {
            match self {
                SimDevice::Null => NullDevice.io_read(addr, effectful),
                SimDevice::Keyboard(dev) => dev.io_read(addr, effectful),
                SimDevice::Display(dev) => dev.io_read(addr, effectful),
                SimDevice::Custom(dev) => dev.io_read(addr, effectful),
            }
        }
    
        fn io_write(&mut self, addr: u16, data: u16) -> bool {
            match self {
                SimDevice::Null => NullDevice.io_write(addr, data),
                SimDevice::Keyboard(dev) => dev.io_write(addr, data),
                SimDevice::Display(dev) => dev.io_write(addr, data),
                SimDevice::Custom(dev) => dev.io_write(addr, data),
            }
        }
    
        fn io_reset(&mut self) {
            match self {
                SimDevice::Null => NullDevice.io_reset(),
                SimDevice::Keyboard(dev) => dev.io_reset(),
                SimDevice::Display(dev) => dev.io_reset(),
                SimDevice::Custom(dev) => dev.io_reset(),
            }
        }

        fn poll_interrupt(&mut self) -> Option<super::Interrupt> {
            match self {
                SimDevice::Null => NullDevice.poll_interrupt(),
                SimDevice::Keyboard(dev) => dev.poll_interrupt(),
                SimDevice::Display(dev) => dev.poll_interrupt(),
                SimDevice::Custom(dev) => dev.poll_interrupt(),
            }
        }
    }
    impl std::fmt::Debug for SimDevice {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Null => write!(f, "Null"),
                Self::Keyboard(_) => f.debug_struct("Keyboard").finish_non_exhaustive(),
                Self::Display(_) => f.debug_struct("Display").finish_non_exhaustive(),
                Self::Custom(_) => f.debug_struct("Custom").finish_non_exhaustive(),
            }
        }
    }

    /// Allows `_to_sim_device` to be private to this file only.
    pub struct ToSimDeviceToken(pub ());
}

// Implementations of devices are found in the submodules. Here's one:

/// Does nothing.
/// 
/// Does not accept any reads nor writes and never interrupts.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct NullDevice;
impl ExternalDevice for NullDevice {
    fn io_read(&mut self, _addr: u16, _effectful: bool) -> Option<u16> {
        None
    }

    fn io_write(&mut self, _addr: u16, _data: u16) -> bool {
        false
    }

    fn io_reset(&mut self) {}

    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        None
    }
    
    fn _to_sim_device(self, _: internals::ToSimDeviceToken) -> internals::SimDevice
        where Self: Sized
    {
        internals::SimDevice::Null
    }
}

/// A device that handles interrupts with a function.
#[allow(clippy::type_complexity)]
pub struct InterruptFromFn(Box<dyn FnMut() -> Option<Interrupt> + Send + Sync + 'static>);
impl InterruptFromFn {
    /// Creates a new interrupt from a function.
    pub fn new(f: impl FnMut() -> Option<Interrupt> + Send + Sync + 'static) -> Self {
        Self(Box::new(f))
    }
}
impl std::fmt::Debug for InterruptFromFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterruptFromFn").finish_non_exhaustive()
    }
}
impl ExternalDevice for InterruptFromFn {
    fn io_read(&mut self, _addr: u16, _effectful: bool) -> Option<u16> {
        None
    }

    fn io_write(&mut self, _addr: u16, _data: u16) -> bool {
        false
    }

    fn io_reset(&mut self) {}

    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        (self.0)()
    }
}

fn resolve_lock<G>(e: std::sync::TryLockResult<G>) -> Option<G> {
    use std::sync::TryLockError;

    match e {
        Ok(guard) => Some(guard),
        Err(TryLockError::WouldBlock) => None,
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner())
    }
}
impl<D: ExternalDevice> ExternalDevice for std::sync::Arc<std::sync::RwLock<D>> {
    fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16> {
        resolve_lock(self.try_write())?
            .io_read(addr, effectful)
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        resolve_lock(self.try_write())
            .map_or(false, |mut g| g.io_write(addr, data))
    }

    fn io_reset(&mut self) {
        if let Some(mut guard) = resolve_lock(self.try_write()) {
            guard.io_reset();
        }
    }

    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        resolve_lock(self.try_write())
            .and_then(|mut g| g.poll_interrupt())
    }
}
impl<D: ExternalDevice> ExternalDevice for std::sync::Arc<std::sync::Mutex<D>> {
    fn io_read(&mut self, addr: u16, effectful: bool) -> Option<u16> {
        resolve_lock(self.try_lock())?
            .io_read(addr, effectful)
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        resolve_lock(self.try_lock())
            .map_or(false, |mut g| g.io_write(addr, data))
    }

    fn io_reset(&mut self) {
        if let Some(mut guard) = resolve_lock(self.try_lock()) {
            guard.io_reset();
        }
    }

    fn poll_interrupt(&mut self) -> Option<Interrupt> {
        resolve_lock(self.try_lock())
            .and_then(|mut g| g.poll_interrupt())
    }
}