pub(crate) mod buffer {
    use std::{mem::{self, ManuallyDrop}, ptr::NonNull};
    use std::sync::{atomic::{AtomicUsize, Ordering}, Condvar, Mutex};

    use num_traits::{PrimInt, Zero};

    #[derive(Debug)]
    pub(crate) enum BuffError {
        BufferFull,
        BufferEmpty,
    }

    pub(crate) struct RawData<E> {
        pointer: NonNull<E>,
        capacity: usize,
    }

    impl<E> RawData<E> where E: PrimInt {
        pub(crate) fn new(v: Vec<E>) -> Self {
            let mut v = ManuallyDrop::new(v);
            Self {
                pointer: unsafe { NonNull::new_unchecked(v.as_mut_ptr()) },
                capacity: v.capacity(),
            }
        }

        pub(crate) fn get(&self, idx: usize) -> E {
            unsafe { *self.pointer.as_ptr().offset(idx as isize) }
        }

        pub(crate) fn set(&self, idx: usize, val: E) {
            unsafe { *self.pointer.as_ptr().offset(idx as isize) = val.clone(); } 
        }

        pub(crate) fn iter(&self) -> RawDataIter<E> {
            RawDataIter { data: self, index: 0 }
        }
    }

    impl<E> RawData<E> {
        fn take_as_vec(&mut self) -> Vec<E> {
            let capacity = self.capacity;
            self.capacity = 0;
            unsafe {
                Vec::from_raw_parts(self.pointer.as_ptr(), capacity, capacity)
            }
        }
    }

    impl<E> Drop for RawData<E> {
        fn drop(&mut self) {
            if self.capacity > 0 {
                if mem::needs_drop::<E>() {
                    self.take_as_vec();
                }
            }
        }
    }

    pub struct RawDataIter<'a, E> {
        data: &'a RawData<E>,
        index: usize,
    }

    impl<'a, E> Iterator for RawDataIter<'a, E>
    where E: Copy {
        type Item = E;

        fn next(&mut self) -> Option<Self::Item> {
            if self.index < self.data.capacity {
                let value = unsafe {
                    *self.data.pointer.as_ptr().offset(self.index as isize)
                };
                self.index += 1;
                Some(value)
            } else {
                None
            }
        }  
    }

    pub(crate) struct IOInnerBuffer<E> {
        raw_data: Mutex<RawData<E>>,
        capacity: usize,
        not_full: Condvar,
        not_empty: Condvar,
        rear: AtomicUsize,
        front: AtomicUsize,
    }

    impl<E> IOInnerBuffer<E>
    where E: PrimInt {
        pub(crate) fn new(capacity: usize) -> Self {
            let raw_data = RawData::new(vec![Zero::zero(); capacity]);
            Self {
                raw_data: Mutex::new(raw_data),
                capacity,
                not_full: Condvar::new(),
                not_empty: Condvar::new(),
                rear: AtomicUsize::new(0),
                front: AtomicUsize::new(0),
            }
        }

        pub(crate) fn size(&self) -> usize {
            let rear = self.rear.load(Ordering::SeqCst);
            let front = self.front.load(Ordering::SeqCst);

            let mut size = if rear >= front {
                rear - front
            } else {
                self.capacity - front + rear
            };

            if size == 0 {
                if !self.raw_data.lock()
                    .unwrap()
                    .iter()
                    .all(|item| item == Zero::zero()) {
                    size = self.capacity;
                }
            }
            size
        }

        pub(crate) fn write_in(&self, buff_in: &[E], len: usize) -> Result<usize, BuffError> {
            let current_size = self.size();
            if len > self.capacity {
                return Err(BuffError::BufferFull);
            }
            let mut data = self.raw_data.lock().unwrap();
            let mut written = 0;

            if current_size + len > self.capacity {
                data = self.not_full.wait(data).unwrap();
            }

            let rear = self.rear.load(Ordering::SeqCst);

            for i in 0..len {
                let idx = (rear + i) % self.capacity;
                data.set(idx, buff_in[i]);
                written += 1;
            }
            self.rear.store((rear + written) % self.capacity, Ordering::SeqCst);
            self.not_empty.notify_one();
            Ok(written)
        }

        pub(crate) fn read_out(&self, buff_out: &mut [E], len: usize) -> Result<usize, BuffError> {
            let current_size = self.size();
            if current_size == 0 {
                return Err(BuffError::BufferEmpty);
            }
            let mut data = self.raw_data.lock().unwrap();
            let mut readed = 0;

            if current_size < len {
                data = self.not_empty.wait(data).unwrap();
            }

            let front = self.front.load(Ordering::SeqCst);
            for i in 0..len {
                let idx = (front + i) % self.capacity;
                buff_out[i] = data.get(idx);
                data.set(idx, Zero::zero());
                readed += 1;
            }

            self.front.store((front + readed) % self.capacity, Ordering::SeqCst);
            self.not_full.notify_one();
            Ok(readed)
        }

        pub(crate) fn clear(&self) {
            let capacity = self.capacity;
            let data = self.raw_data.lock().unwrap();

            for idx in 0..capacity {
                data.set(idx, Zero::zero());
            }
            self.front.store(0, Ordering::SeqCst);
            self.rear.store(0, Ordering::SeqCst);
        }
    }

    unsafe impl<E> Send for IOInnerBuffer<E> { }
    unsafe impl<E> Sync for IOInnerBuffer<E> { }
}

pub(crate) mod channel {
    use std::{marker::PhantomData, mem::size_of, sync::Arc};

    use derive_new::new;

    use super::buffer::{BuffError, IOInnerBuffer};

    pub type ChannelElem = u8;
    pub type ChunkBytes<'a> = &'a [ChannelElem];
    
    pub(crate) type InnerBuffer = IOInnerBuffer<ChannelElem>;

    #[derive(Debug)]
    pub(crate) enum ChannelError {
        WriterNotFound,
        WriteChunkFailed,
        ReaderNotFound,
        ReadChunkFailed,
    }

    pub(crate) struct IOChannel<W, R, C> {
        inner_buffer: Arc<InnerBuffer>,
        io_writer: Option<W>,
        io_reader: Option<R>,
        _marker: PhantomData<C>,
    }

    pub(crate) trait IOChunk {
        fn from_chunk_bytes(chunk_bytes: ChunkBytes) -> Self;
        fn to_chunk_bytes(&self) -> &[ChannelElem];
    }

    pub(crate) trait IOWriter {
        fn write(&self, inner_buffer: Arc<InnerBuffer>, buff_in: &[ChannelElem], len: usize) -> Result<usize, BuffError>;
    }

    pub(crate) trait IOReader {
        fn read(&self, inner_buffer: Arc<InnerBuffer>, buff_out: &mut [ChannelElem], len: usize) -> Result<usize, BuffError>;
    }

    impl<'a, W, R, C> IOChannel<W, R, C>
    where
        W: IOWriter + Copy + Send + 'static,
        R: IOReader + Copy + Send + 'static,
        C: IOChunk,
    {
        fn new(inner_buffer: Arc<InnerBuffer>, io_writer: Option<W>, io_reader: Option<R>) -> Self {
            Self {
                inner_buffer,
                io_writer,
                io_reader,
                _marker: PhantomData,
            }
        }

        pub(crate) fn write(&self, chunk: C) -> Result<(), ChannelError> {
            match self.io_writer {
                Some(writer) => {
                    match writer.write(self.inner_buffer.clone(), chunk.to_chunk_bytes(), size_of::<C>()) {
                        Ok(write_len) => if write_len != size_of::<C>() {
                            Err(ChannelError::WriteChunkFailed)
                        } else {
                            Ok(())
                        }
                        Err(_) => Err(ChannelError::WriteChunkFailed),
                    }
                },
                None => Err(ChannelError::WriterNotFound),
            }
        }

        pub(crate) fn read(&self) -> Result<C, ChannelError> {
            match self.io_reader {
                Some(reader) => {
                    let mut buff = vec![0; size_of::<C>()];
                    let len = buff.len();
                    match reader.read(self.inner_buffer.clone(), &mut buff, len) {
                        Ok(read_len) => if read_len == len {
                            Ok(C::from_chunk_bytes(&buff))
                        } else {
                            Err(ChannelError::ReadChunkFailed)
                        },
                        Err(_) => Err(ChannelError::ReadChunkFailed),
                    }
                },
                None => Err(ChannelError::ReaderNotFound),
            }
        }

        pub(crate) fn clear(&self) {
            self.inner_buffer.clear();
        }
    }

    pub(crate) struct IOChannelBuilder<W, R> {
        inner_buffer: Arc<InnerBuffer>,
        io_writer: Option<W>,
        io_reader: Option<R>,
    }

    impl<'a, W, R> IOChannelBuilder<W, R>
    where
        W: IOWriter + Copy + Send + 'static,
        R: IOReader + Copy + Send + 'static,
    {
        pub(crate) fn new(capacity: usize) -> Self {
            Self {
                inner_buffer: Arc::new(IOInnerBuffer::new(capacity)),
                io_reader: None,
                io_writer: None,
            }
        }

        pub(crate) fn with_writer(mut self, io_writer: W) -> Self {
            self.io_writer = Some(io_writer);
            self
        }

        pub(crate) fn with_reader(mut self, io_reader: R) -> Self {
            self.io_reader = Some(io_reader);
            self
        }

        pub(crate) fn build<C: IOChunk>(&self) -> IOChannel<W, R, C> {
            IOChannel::<W, R, C>::new(self.inner_buffer.clone(), self.io_writer, self.io_reader)
        }
    }

    #[derive(new, Clone, Copy)]
    pub(crate) struct DefaultIOWriter;

    #[derive(new, Clone, Copy)]
    pub(crate) struct DefaultIOReader;

    impl IOWriter for DefaultIOWriter {
        fn write(&self, inner_buffer: Arc<InnerBuffer>, buff_in: &[ChannelElem], len: usize) -> Result<usize, BuffError> {
            assert!(buff_in.len() == len);
            inner_buffer.write_in(buff_in, len)
        }
    }

    impl IOReader for DefaultIOReader {
        fn read(&self, inner_buffer: Arc<InnerBuffer>, buff_out: &mut [ChannelElem], len: usize) -> Result<usize, BuffError> {
            assert!(buff_out.len() == len);
            inner_buffer.read_out(buff_out, len)
        }
    }
}