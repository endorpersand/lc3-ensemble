use std::ops::{Bound, RangeBounds};

use rand::Rng;

use super::ExternalDevice;

#[derive(Clone, Copy)]
struct SampleRange {
    start: u32,
    end: u32,
    end_incl: bool
}
impl SampleRange {
    fn new(r: impl RangeBounds<u32>) -> Self {
        let start = match r.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s.checked_add(1).expect("(Bound::Excluded(u32::MAX), _) is not an acceptable range"),
            Bound::Unbounded    => 0,
        };
        let (end, end_incl) = match r.end_bound() {
            Bound::Included(&s) => (s, true),
            Bound::Excluded(&s) => (s, false),
            Bound::Unbounded => (u32::MAX, true),
        };

        Self { start, end, end_incl }
    }
}
impl std::fmt::Debug for SampleRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleRange { start, end, end_incl: true } => (start..=end).fmt(f),
            SampleRange { start, end, end_incl: false } => (start..end).fmt(f),
        }
    }
}
impl RangeBounds<u32> for SampleRange {
    fn start_bound(&self) -> Bound<&u32> {
        Bound::Included(&self.start)
    }

    fn end_bound(&self) -> Bound<&u32> {
        match self.end_incl {
            true => Bound::Included(&self.end),
            false => Bound::Excluded(&self.end),
        }
    }
}

/// A timer device that triggers an interrupt after a configured number (or range) of instructions.
/// 
/// This is not part of LC-3 specification, so the design decision made 
/// (such as default interrupt vector, priority level) are not reflective of LC-3's choices.
#[derive(Debug)]
pub struct TimerDevice {
    generator: Box<rand::rngs::StdRng>,
    range: SampleRange,
    time: u32,

    /// The interrupt vector.
    pub vect: u8,
    /// The priority. 
    /// 
    /// Note that if this exceeds 7, it will be treated as though it is 7.
    pub priority: u8,

    /// Whether this timer can trigger an interrupt.
    pub enabled: bool,
}
impl TimerDevice {
    /// Creates a new timer device.
    /// - `seed`: Sets the seed for the timer's RNG. This can be `None` 
    ///     if RNG does not need to be deterministic or if range can only be exactly one value.
    /// - `range`: Sets the range of possible number of instructions before interrupt trigger.
    /// - `vect` and `priority`: Initializes the interrupt vector and priority values.
    pub fn new(seed: Option<u64>, range: impl RangeBounds<u32>, vect: u8, priority: u8) -> Self {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let generator = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut timer = Self {
            generator: Box::new(generator),
            range: SampleRange::new(range),
            time: 0,
            vect,
            priority,
            enabled: false
        };
        timer.reset_remaining();

        timer
    }

    /// Gets the range of possible number of instructions before the interrupt is triggered.
    pub fn get_range(&self) -> impl RangeBounds<u32> {
        self.range
    }
    /// Sets the number of instructions before the interrupt is triggered to a range of values.
    pub fn set_range(&mut self, r: impl RangeBounds<u32>) -> &mut Self {
        self.range = SampleRange::new(r);
        self
    }
    /// Sets the number of instructions before the interrupt is triggered to an exact number.
    pub fn set_exact(&mut self, n: u32) -> &mut Self {
        self.set_range(n..=n)
    }

    /// Gets the number of instructions remaining until the interrupt triggers.
    pub fn get_remaining(&self) -> u32 {
        self.time
    }
    /// Resets the remaining time.
    pub fn reset_remaining(&mut self) {
        self.time = self.try_generate_time();
    }
    /// Generates a new random time.
    fn try_generate_time(&mut self) -> u32 {
        match self.range {
            SampleRange { start, end, end_incl: true } => self.generator.gen_range(start..=end),
            SampleRange { start, end, end_incl: false } => self.generator.gen_range(start..end),
        }
    }
}
impl Default for TimerDevice {
    /// Creates a timer with default parameters.
    /// 
    /// The default parameters here are:
    /// - non-deterministic RNG
    /// - Triggers a timer interrupt every 50 instructions
    /// - Binds to interrupt vector `0x81`
    /// - Interrupt priority 4
    /// - Disabled
    /// 
    /// (These are arbitrary.)
    fn default() -> Self {
        Self::new(None, 50..=50, 0x81, 0b100)
    }
}
impl ExternalDevice for TimerDevice {
    fn io_read(&mut self, _addr: u16, _effectful: bool) -> Option<u16> {
        None
    }
    
    fn io_write(&mut self, _addr: u16, _data: u16) -> bool {
        false
    }
    
    fn io_reset(&mut self) {
        self.time = self.try_generate_time();
    }
    fn poll_interrupt(&mut self) -> Option<super::Interrupt> {
        if !self.enabled { return None };
        
        match self.time {
            0 => {
                self.reset_remaining();
                None
            },
            1 => {
                self.time = 0;
                Some(super::Interrupt::vectored(self.vect, self.priority))
            },
            _ => {
                self.time -= 1;
                None
            }
        }
    }
}