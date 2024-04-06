//! IO handling for LC-3.
//! 
//! The interface for IO devices is defined with the [`IODevice`] trait.
//! This is exposed to the simulator with the [`SimIO`] enum.
//! 
//! Besides those two key items, this module also includes:
//! - [`EmptyIO`]: An `IODevice` holding the implementation for a lack of IO support.
//! - [`BiChannelIO`]: An `IODevice` holding a basic implementation for IO.
//! - [`CustomIO`]: An `IODevice` that can be used to wrap around custom IO implementations.
//! - [`BlockingQueue`]: Not an `IODevice`, but a utility data structure that can be used as an input buffer.

mod queue;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel as cbc;
pub use queue::*;

const KBSR: u16 = 0xFE00;
const KBDR: u16 = 0xFE02;
const DSR: u16  = 0xFE04;
const DDR: u16  = 0xFE06;
const MCR: u16  = 0xFFFE;

/// An IO device that can be read/written to.
pub trait IODevice {
    /// Reads the data at the given memory-mapped address.
    /// 
    /// If successful, this returns the value returned from that address.
    /// If unsuccessful, this returns `None`.
    fn io_read(&self, addr: u16) -> Option<u16>;

    /// Writes the data to the given memory-mapped address.
    /// 
    /// This returns whether the write was successful or not.
    fn io_write(&self, addr: u16, data: u16) -> bool;

    /// Tries to close this IO device.
    fn close(self);
}
impl dyn IODevice {} // assert IODevice is dyn safe

/// No IO. All reads and writes are unsuccessful.
/// 
/// If IO status registers are accessed while this is the active IO type, 
/// all IO-related traps will hang.
pub struct EmptyIO;
impl IODevice for EmptyIO {
    fn io_read(&self, _addr: u16) -> Option<u16> {
        None
    }

    fn io_write(&self, _addr: u16, _data: u16) -> bool {
        false
    }
    
    fn close(self) {}
}

/// A helper struct for [`BiChannelIO::new`], 
/// indicating the channel is closed and no more reads/writes will come from it.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Stop;

/// An IO that reads from one channel and writes to another.
/// 
/// This binds the reader channel to the KBSR and KBDR.
/// When a character is ready from the reader channel,
/// the KBSR status is enabled and the character is accessible from the KBDR.
/// 
/// This binds the writer channel to the DSR and DDR.
/// When a character is ready to be written to the writer channel,
/// the DSR status is enabled and the character can be written to the DDR.
/// 
/// This IO type also exposes the MCR in the MCR MMIO address.
pub struct BiChannelIO {
    read_data:    cbc::Receiver<u8>,
    #[allow(unused)]
    read_handler: JoinHandle<()>,

    write_data:    cbc::Sender<u8>,
    write_handler: JoinHandle<()>,

    mcr: Arc<AtomicBool>
}
impl BiChannelIO {
    /// Creates a new bi-channel IO device with the given reader and writer.
    /// 
    /// This calls the reader function every time the IO input receives a byte.
    /// The reader function should block until a byte is ready, or return Stop
    /// if there are no more bytes to read.
    /// 
    /// This calls the writer function every time a byte needs to be written to the
    /// IO output.
    /// 
    /// This uses threads to read and write from input and output. As such,
    /// the channels will continue to poll input and output even when the simulator
    /// is not running. As such, care should be taken to not send messages through
    /// the reader thread while the simulator is not running.
    pub fn new(
        mut reader: impl FnMut() -> Result<u8, Stop> + Send + 'static, 
        mut writer: impl FnMut(u8) -> Result<(), Stop> + Send + 'static, 
        mcr: Arc<AtomicBool>
    ) -> Self {
        let (read_tx, read_rx) = cbc::bounded(1);
        let (write_tx, write_rx) = cbc::bounded(1);

        // Reader thread:
        let read_handler = std::thread::spawn(move || loop {
            let Ok(byte) = reader() else { return };
            let Ok(()) = read_tx.send(byte) else { return };
        });

        // Writer thread:
        let write_handler = std::thread::spawn(move || {
            for byte in write_rx {
                let Ok(()) = writer(byte) else { return };
            }
        });
        
        Self {
            read_data: read_rx, 
            read_handler, 
            write_data: write_tx, 
            write_handler, 
            mcr
        }
    }

    /// Creates a bi-channel IO device with stdin being the read data and stdout being the write data.
    /// 
    /// Note that due to how stdin works in terminals, data is only sent once a new line is typed.
    /// Additionally, this flushes stdout every time a byte is written.
    pub fn stdio(mcr: Arc<AtomicBool>) -> Self {
        use std::io::{self, BufRead, Write};

        Self::new(
            || {
                let mut stdin = io::stdin().lock();
                let &[byte, ..] = stdin.fill_buf().unwrap() else {
                    // terminal stdin would poll, so this is unreachable with terminal stdin
                    return Err(Stop);
                };

                stdin.consume(1);
                Ok(byte)
            }, 
            |byte| {
                io::stdout().write_all(&[byte]).unwrap();
                io::stdout().flush().unwrap();
                Ok(())
            }, 
            mcr
        )
    }
}

impl IODevice for BiChannelIO {
    fn io_read(&self, addr: u16) -> Option<u16> {
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
            MCR => Some(io_bool(self.mcr.load(Ordering::Relaxed))),
            _ => None
        }
    }

    fn io_write(&self, addr: u16, data: u16) -> bool {
        match addr {
            DDR => self.write_data.send(data as u8).is_ok(),
            MCR => {
                // store whether last bit is 1 (e.g., if data is negative)
                self.mcr.store((data as i16) < 0, Ordering::Relaxed);
                true
            }
            _ => false
        }
    }
    
    fn close(self) {
        let Self {
            read_data,
            read_handler: _,
            write_data,
            write_handler,
            mcr: _
        } = self;

        // Drop the channels.
        std::mem::drop(read_data);
        std::mem::drop(write_data);

        // Wait for the write handler to join.
        // This shouldn't block for long, because we just
        // disconnected the channel.

        // We're not going to wait for the read handler
        // because it can hang on reading, which prevents it from seeing
        // the channel is disconnected.

        // Also, don't error.
        // Skill issue.
        let _ = write_handler.join();
    }
}
/// Converts boolean data to a register word
fn io_bool(b: bool) -> u16 {
    match b {
        true  => 0x8000,
        false => 0x0000,
    }
}

// `Box<dyn IODevice>` does not work.
// It doesn't implement IODevice because it doesn't implement close
// (because you can't close on an unsized dyn IODevice).
// 
// However, changing the signature makes BiChannelIO annoying.
// So, this hack basically puts the device in an Option
// and closes it by taking it out and closing it without consuming the entire object,
// making close only require &mut Self instead of Self.
trait IODeviceMutClosable {
    fn io_read(&self, addr: u16) -> Option<u16>;
    fn io_write(&self, addr: u16, data: u16) -> bool;

    /// Closes but doesn't consume the object.
    /// 
    /// The object should not be used after this point.
    fn take_close(&mut self);
}
impl<D: IODevice> IODeviceMutClosable for Option<D> {
    fn io_read(&self, addr: u16) -> Option<u16> {
        self.as_ref().unwrap().io_read(addr)
    }
    fn io_write(&self, addr: u16, data: u16) -> bool {
        self.as_ref().unwrap().io_write(addr, data)
    }
    fn take_close(&mut self) {
        self.take().unwrap().close()
    }
}

/// An opaque box that holds custom defined IO.
/// 
/// This can be used to use a different implementation of IO
/// than the ones implemented in this module.
pub struct CustomIO(Box<dyn IODeviceMutClosable + Send + Sync>);
impl CustomIO {
    /// Creates a new custom IO.
    pub fn new(device: impl IODevice + Send + Sync + 'static) -> Self {
        CustomIO(Box::new(Some(device)))
    }
}
impl IODevice for CustomIO {
    fn io_read(&self, addr: u16) -> Option<u16> {
        self.0.io_read(addr)
    }

    fn io_write(&self, addr: u16, data: u16) -> bool {
        self.0.io_write(addr, data)
    }

    fn close(mut self) {
        self.0.take_close();
        std::mem::drop(self)
    }
}

/// All the variants of IO accepted by the Simulator.
pub enum SimIO {
    /// No IO. This corresponds to the implementation of [`EmptyIO`].
    Empty,
    /// A bi-channel IO implementation. See [`BiChannelIO`].
    BiChannel(BiChannelIO),
    /// A custom IO implementation. See [`CustomIO`].
    Custom(CustomIO)
}
impl std::fmt::Debug for SimIO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimIO")
            .finish_non_exhaustive()
    }
}
impl From<EmptyIO> for SimIO {
    fn from(_value: EmptyIO) -> Self {
        SimIO::Empty
    }
}
impl From<BiChannelIO> for SimIO {
    fn from(value: BiChannelIO) -> Self {
        SimIO::BiChannel(value)
    }
}
impl From<CustomIO> for SimIO {
    fn from(value: CustomIO) -> Self {
        SimIO::Custom(value)
    }
}
impl IODevice for SimIO {
    fn io_read(&self, addr: u16) -> Option<u16> {
        match self {
            SimIO::Empty => EmptyIO.io_read(addr),
            SimIO::BiChannel(io) => io.io_read(addr),
            SimIO::Custom(io) => io.io_read(addr)
        }
    }

    fn io_write(&self, addr: u16, data: u16) -> bool {
        match self {
            SimIO::Empty => EmptyIO.io_write(addr, data),
            SimIO::BiChannel(io) => io.io_write(addr, data),
            SimIO::Custom(io) => io.io_write(addr, data)
        }
    }

    fn close(self) {
        match self {
            SimIO::Empty => EmptyIO.close(),
            SimIO::BiChannel(io) => io.close(),
            SimIO::Custom(io) => io.close()
        }
    }
}