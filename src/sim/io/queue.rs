use crossbeam_channel as cbc;

/// A simple multi-producer, multi-consumer blocking queue.
/// 
/// This queue allows producers and consumers to respectively send and receive data
/// from the provided buffer.
/// 
/// A blocking queue can be paired with [`BiChannelIO`] to maintain an input buffer:
/// ```
/// use lc3_ensemble::sim::io::{BiChannelIO, BlockingQueue, Stop};
/// 
/// let queue = BlockingQueue::new(None);
/// 
/// // fn writer(_: u8) -> Result<(), Stop> { ... }
/// // let mcr = ...
/// # fn writer(_: u8) -> Result<(), Stop> { Ok(()) }
/// # let mcr = std::sync::Arc::default();
/// 
/// let io = BiChannelIO::new(queue.reader(), writer, mcr);
/// ```
/// 
/// [`BiChannelIO`]: super::BiChannelIO
pub struct BlockingQueue<T> {
    head: cbc::Sender<T>,
    tail: cbc::Receiver<T>
}
impl<T> BlockingQueue<T> {
    /// Creates a new blocking queue, optionally with a maximum size.
    pub fn new(size: Option<usize>) -> Self {
        let (head, tail) = match size {
            Some(n) => cbc::bounded(n),
            None => cbc::unbounded(),
        };

        Self { head, tail }
    }

    /// Adds an element to the front of the queue, blocking if the queue is full.
    pub fn push(&self, t: T) {
        self.head.send(t).unwrap();
    }
    /// Adds an element to the front of the queue, returning an error if the queue is full.
    /// 
    /// This uses [`crossbeam_channel::TrySendError`] as its error, 
    /// but this function cannot return [`TrySendError::Disconnected`],
    /// since the [`BlockingQueue`] has to exist to call this function.
    /// 
    /// [`TrySendError::Disconnected`]: crossbeam_channel::TrySendError::Disconnected
    pub fn try_push(&self, t: T) -> Result<(), cbc::TrySendError<T>> {
        self.head.try_send(t)
    }

    /// Removes an element from the back of the queue, blocking if the queue is full.
    pub fn pop(&self) -> T {
        self.tail.recv().unwrap()
    }
    /// Removes an element from the back of the queue, returning an error if the queue is full.
    /// 
    /// This uses [`crossbeam_channel::TryRecvError`] as its error, 
    /// but this function cannot return [`TryRecvError::Disconnected`],
    /// since the [`BlockingQueue`] has to exist to call this function.
    /// 
    /// [`TryRecvError::Disconnected`]: crossbeam_channel::TryRecvError::Disconnected
    pub fn try_pop(&self) -> Result<T, cbc::TryRecvError> {
        self.tail.try_recv()
    }

    /// Exposes the sending head of the queue.
    /// 
    /// This enables all methods of [`crossbeam_channel::Sender`] to be used.
    /// However, because the sender's lifetime is not dependent
    /// on the queue's lifetime, users of this function should verify
    /// that the sender is not disconnected when calling [`Sender::send`]
    /// and similar functions.
    /// 
    /// [`Sender::send`]: crossbeam_channel::Sender::send
    pub fn head(&self) -> cbc::Sender<T> {
        self.head.clone()
    }

    /// Exposes the receiving tail of the queue.
    /// 
    /// This enables all methods of [`crossbeam_channel::Receiver`] to be used.
    /// However, because the receiver's lifetime is not dependent
    /// on the queue's lifetime, users of this function should verify
    /// that the sender is not disconnected when calling [`Receiver::recv`]
    /// and similar functions.
    /// 
    /// [`Receiver::recv`]: crossbeam_channel::Receiver::recv
    pub fn tail(&self) -> cbc::Receiver<T> {
        self.tail.clone()
    }

    /// Returns the number of bytes currently in the channel.
    pub fn len(&self) -> usize {
        self.head.len()
    }
    /// Returns whether there are any bytes in the channel (`true` if not).
    pub fn is_empty(&self) -> bool {
        self.head.is_empty()
    }

    /// A utility to allow this queue to interop with [`BiChannelIO`].
    /// 
    /// This can be used as the `reader` parameter of [`BiChannelIO::new`]
    /// to allow the IO device to poll this queue for data.
    /// See the struct-level documentation for an example.
    /// 
    /// [`BiChannelIO`]: super::BiChannelIO
    /// [`BiChannelIO::new`]: super::BiChannelIO::new
    pub fn reader(&self) -> impl Fn() -> Result<T, super::Stop> {
        let tail = self.tail();

        move || tail.recv().map_err(|_| super::Stop)
    }
}
impl<T> Default for BlockingQueue<T> {
    fn default() -> Self {
        Self::new(None)
    }
}
impl<E> Extend<E> for BlockingQueue<E> {
    fn extend<T: IntoIterator<Item = E>>(&mut self, iter: T) {
        for t in iter {
            self.push(t);
        }
    }
}