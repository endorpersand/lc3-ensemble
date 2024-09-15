//! IO handling for LC-3.
//! 
//! The interface for IO devices is defined with the [`IODevice`] trait.
//! The simulator can be configured to interact with a given IO device with [`Simulator::open_io`].
//! 
//! Besides the trait, this module also includes:
//! - [`EmptyIO`]: An `IODevice` holding the implementation for a lack of IO support.
//! - [`BufferedIO`]: An `IODevice` holding a buffered implementation for IO.
//! - [`BiChannelIO`]: An `IODevice` holding a threaded/channel implementation for IO.
//! 
//! [`Simulator::open_io`]: super::Simulator::open_io

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock, RwLockWriteGuard, TryLockError};

use crossbeam_channel as cbc;

use super::MCR;

const KBSR: u16 = 0xFE00;
const KBDR: u16 = 0xFE02;
const DSR: u16  = 0xFE04;
const DDR: u16  = 0xFE06;
const MCR: u16  = 0xFFFE;

/// An IO device that can be read/written to.
pub trait IODevice: Send + Sync + 'static {
    /// Reads the data at the given memory-mapped address.
    /// 
    /// If successful, this returns the value returned from that address.
    /// If unsuccessful, this returns `None`.
    fn io_read(&mut self, addr: u16) -> Option<u16>;

    /// Writes the data to the given memory-mapped address.
    /// 
    /// This returns whether the write was successful or not.
    fn io_write(&mut self, addr: u16, data: u16) -> bool;

    #[doc(hidden)]
    /// Hacky specialization.
    /// 
    /// This allows [`super::Simulator::open_io`]'s signature to just require an [`IODevice`]
    /// and does not require that we expose IO internals.
    fn _to_sim_io(self, _: internals::ToSimIOToken) -> internals::SimIOKind
        where Self: Sized
    {
        internals::SimIOKind::Custom(Box::new(self))
    }
}
impl dyn IODevice {} // assert IODevice is dyn safe

/// No IO. All reads and writes are unsuccessful.
/// 
/// If IO status registers are accessed while this is the active IO type, 
/// all IO-related traps will hang.
pub struct EmptyIO;
impl IODevice for EmptyIO {
    fn io_read(&mut self, _addr: u16) -> Option<u16> {
        None
    }

    fn io_write(&mut self, _addr: u16, _data: u16) -> bool {
        false
    }
    
    fn _to_sim_io(self, _: internals::ToSimIOToken) -> internals::SimIOKind
        where Self: Sized
    {
        internals::SimIOKind::Empty
    }
}

/// IO that reads from an input buffer and writes to an output buffer.
/// 
/// The input buffer is accessible in the simulator memory through the KBSR and KBDR.
/// The output buffer is accessible in the simulator memory through the DSR and DDR.
/// 
/// The buffers can be accessed in code via [`BufferedIO::get_input`] and [`BufferedIO::get_output`].
/// 
/// Note that if a input/output lock guard is acquired from one of the locks of this IO, 
/// the input/output becomes temporarily inaccessible to the simulator.
/// Thus, a lock guard should never be leaked otherwise the simulator loses access to the input/output.
#[derive(Clone)]
pub struct BufferedIO {
    input: Arc<RwLock<VecDeque<u8>>>,
    output: Arc<RwLock<Vec<u8>>>
}
impl BufferedIO {
    /// Creates a new BufferedIO.
    pub fn new() -> Self {
        Self { input: Default::default(), output: Default::default() }
    }
    /// Creates a new BufferedIO from already defined buffers.
    pub fn with_bufs(input: Arc<RwLock<VecDeque<u8>>>, output: Arc<RwLock<Vec<u8>>>) -> Self {
        Self { input, output }
    }

    fn try_input(&self) -> Option<RwLockWriteGuard<'_, VecDeque<u8>>> {
        match self.input.try_write() {
            Ok(g) => Some(g),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
    fn try_output(&self) -> Option<RwLockWriteGuard<'_, Vec<u8>>> {
        match self.output.try_write() {
            Ok(g) => Some(g),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Gets a reference to the input buffer.
    pub fn get_input(&self) -> &Arc<RwLock<VecDeque<u8>>> {
        &self.input
    }
    /// Gets a reference to the output buffer.
    pub fn get_output(&self) -> &Arc<RwLock<Vec<u8>>> {
        &self.output
    }
}
impl Default for BufferedIO {
    fn default() -> Self {
        Self::new()
    }
}
impl IODevice for BufferedIO {
    fn io_read(&mut self, addr: u16) -> Option<u16> {
        match addr {
            KBSR => {
                // We're ready once we can obtain a write lock to the input
                // AND the input internally is not empty.
                Some(io_bool({
                    self.try_input()
                        .is_some_and(|inp| !inp.is_empty())
                }))
            },
            KBDR => self.try_input()?.pop_front().map(u16::from),
            DSR => {
                // We're ready once we can obtain a lock to the output.
                Some(io_bool(self.try_output().is_some()))
            },
            _ => None
        }
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        match addr {
            DDR => match self.try_output() {
                Some(mut out) => {
                    out.push(data as u8);
                    true
                },
                None => false
            },
            _ => false
        }
    }
    
    fn _to_sim_io(self, _: internals::ToSimIOToken) -> internals::SimIOKind 
        where Self: Sized
    {
        internals::SimIOKind::Buffered(self)
    }
}

/// An IO that reads from one channel and writes to another.
/// 
/// This binds the reader channel to the KBSR and KBDR.
/// When a character is ready from the reader channel,
/// the KBSR status is enabled and the character is accessible from the KBDR.
/// 
/// This binds the writer channel to the DSR and DDR.
/// When a character is ready to be written to the writer channel,
/// the DSR status is enabled and the character can be written to the DDR.
pub struct BiChannelIO {
    read_data: cbc::Receiver<u8>,
    write_data: cbc::Sender<u8>,
}
impl BiChannelIO {
    /// Creates a new bi-channel IO device with the given reader and writer.
    /// 
    /// This invokes the reader's [`Read::read`] method every time the IO input receives a byte.
    /// Note that internally, this uses the [`Read::bytes`] iterator.
    /// Thus, the same cautions of using that iterator apply here.
    /// 
    /// This calls the writer's [`Write::write_all`] method
    /// every time a byte needs to be written to the IO output.
    /// 
    /// This IO calls [`Write::flush`] when the IO is ready to drop.
    /// This function also has a `flush_every_byte` flag, which designates
    /// whether [`Write::flush`] is *also* called for every byte.
    /// This may be useful to enable for displaying real time output.
    /// 
    /// This uses threads to read and write from input and output. As such,
    /// the channels will continue to poll input and output even when the simulator
    /// is not running. As such, care should be taken to not send messages through
    /// the reader thread while the simulator is not running.
    pub fn new(
        reader: impl Read + Send + 'static, 
        mut writer: impl Write + Send + 'static,
        flush_every_byte: bool
    ) -> Self {
        let (read_tx, read_rx) = cbc::bounded(1);
        let (write_tx, write_rx) = cbc::bounded(1);

        // Reader thread.
        // When this channel drops, the thread disconnects once the read_tx.send call passes.
        // If the reader blocks, this thread only disconnects once the reader stops blocking 
        // (which can occur for terminal stdin).
        std::thread::spawn(move || {
            for m_byte in reader.bytes() {
                let Ok(byte) = m_byte else { return };
                let Ok(()) = read_tx.send(byte) else { return };
            }
        });

        // Writer thread.
        // When this channel drops, the thread disconnects when write_rx is queried.
        // If the writer blocks on the write_all or flush calls, 
        // this thread only disconnects once the writer stops blocking.
        std::thread::spawn(move || {
            for byte in write_rx {
                let Ok(()) = writer.write_all(&[byte]) else { return };
                if flush_every_byte {
                    let Ok(()) = writer.flush() else { return };
                }
            }
            // Errors here mean we're just gonna return, soooo...
            let _ = writer.flush();
        });
        
        Self {
            read_data: read_rx,
            write_data: write_tx,
        }
    }

    /// Creates a bi-channel IO device with [`std::io::stdin`] being the reader channel 
    /// and [`std::io::stdout`] being the writer channel.
    /// 
    /// Note that the simulator will only have access to the input data after a new line is typed (in terminal stdin).
    /// Similarly, printed output will only appear once (terminal) stdout is flushed or once a new line is sent.
    pub fn stdio() -> Self {
        Self::new(std::io::stdin(), std::io::stdout(), false)
    }
}

impl IODevice for BiChannelIO {
    fn io_read(&mut self, addr: u16) -> Option<u16> {
        match addr {
            KBSR => Some(io_bool(self.read_data.is_full())),
            KBDR => match self.read_data.try_recv() {
                Ok(b) => Some(u16::from(b)),
                Err(cbc::TryRecvError::Empty) => None,

                // this can occur if the read handler panicked.
                // however, this just means we can't get the data, so just return None
                Err(cbc::TryRecvError::Disconnected) => None,
            },
            DSR => Some(io_bool(self.write_data.is_empty())),
            _ => None
        }
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        match addr {
            DDR => self.write_data.send(data as u8).is_ok(),
            _ => false
        }
    }
    
    fn _to_sim_io(self, _: internals::ToSimIOToken) -> internals::SimIOKind 
        where Self: Sized
    {
        internals::SimIOKind::BiChannel(self)
    }
}
/// Converts boolean data to a register word
fn io_bool(b: bool) -> u16 {
    match b {
        true  => 0x8000,
        false => 0x0000,
    }
}

/// An IO device that handles MCR read/writes 
/// and delegates the rest to the inner SimIOKind.
/// 
/// This isn't exposed publicly because public users 
/// can't really do much with it, since its use
/// is hardcoded into the simulator.
#[derive(Debug, Default)]
pub(super) struct SimIO {
    pub inner: internals::SimIOKind,
    pub mcr: MCR
}
impl IODevice for SimIO {
    fn io_read(&mut self, addr: u16) -> Option<u16> {
        match addr {
            MCR => Some(io_bool(self.mcr.load(Ordering::Relaxed))),
            _ => self.inner.io_read(addr)
        }
    }

    fn io_write(&mut self, addr: u16, data: u16) -> bool {
        match addr {
            MCR => {
                // store whether last bit is 1 (e.g., if data is negative)
                self.mcr.store((data as i16) < 0, Ordering::Relaxed);
                true
            }
            _ => self.inner.io_write(addr, data)
        }
    }
}

pub(super) mod internals {
    use super::{BiChannelIO, BufferedIO, EmptyIO, IODevice};

    /// All the variants of IO accepted by the Simulator.
    #[derive(Default)]
    pub enum SimIOKind {
        /// No IO. This corresponds to the implementation of [`EmptyIO`].
        #[default]
        Empty,
        /// A buffered implementation. See [`BufferedIO`].
        Buffered(BufferedIO),
        /// A bi-channel IO implementation. See [`BiChannelIO`].
        BiChannel(BiChannelIO),
        /// A custom IO implementation.
        Custom(Box<dyn IODevice + Send + Sync + 'static>)
    }
    impl std::fmt::Debug for SimIOKind {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("SimIO")
                .finish_non_exhaustive()
        }
    }
    impl IODevice for SimIOKind {
        fn io_read(&mut self, addr: u16) -> Option<u16> {
            match self {
                SimIOKind::Empty => EmptyIO.io_read(addr),
                SimIOKind::Buffered(io) => io.io_read(addr),
                SimIOKind::BiChannel(io) => io.io_read(addr),
                SimIOKind::Custom(io) => io.io_read(addr),
            }
        }
    
        fn io_write(&mut self, addr: u16, data: u16) -> bool {
            match self {
                SimIOKind::Empty => EmptyIO.io_write(addr, data),
                SimIOKind::Buffered(io) => io.io_write(addr, data),
                SimIOKind::BiChannel(io) => io.io_write(addr, data),
                SimIOKind::Custom(io) => io.io_write(addr, data)
            }
        }
        
        fn _to_sim_io(self, _: ToSimIOToken) -> SimIOKind 
            where Self: Sized
        {
            self
        }
    }

    pub struct ToSimIOToken(pub ());
}