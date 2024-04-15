use core::slice;
use std::{collections::HashMap, io::{self, Read, Write}, mem::size_of, sync::Arc};

use derive_new::new;
use mio::{event::Event, net::{TcpListener, TcpStream}, Events, Interest, Poll, Registry, Token};

use crate::{channel::channel::{ChannelElem, ChannelError, ChunkBytes, DefaultIOReader, DefaultIOWriter, IOChannel, IOChannelBuilder, IOChunk}, sherpa::Sherpa, thread_pool::ThreadPool};

pub(crate) const CHUNK_PAYLOAD_LEN: usize = 6400;

pub(crate) const SHERPA_TOKENS: &str = "../sherpa-onnx/tokens.txt";
pub(crate) const SHERPA_ENCODER: &str = "../sherpa-onnx/encoder-epoch-20-avg-1-chunk-16-left-128.onnx";
pub(crate) const SHERPA_DECODER: &str = "../sherpa-onnx/decoder-epoch-20-avg-1-chunk-16-left-128.onnx";
pub(crate) const SHERPA_JOINER: &str = "../sherpa-onnx/joiner-epoch-20-avg-1-chunk-16-left-128.onnx";

#[derive(Debug, new)]
#[repr(C, packed)]
pub(crate) struct AudioChunk {
    magic_header: u16,
    channel_id: u8,
    cmd_flag: u8,
    chunk_id: u32,
    pay_load: [ChannelElem; CHUNK_PAYLOAD_LEN],
}

impl IOChunk for AudioChunk {
    fn from_chunk_bytes(chunk_bytes: ChunkBytes) -> Self {
        assert!(chunk_bytes.len() == size_of::<AudioChunk>());
        let mut pay_load = [0; CHUNK_PAYLOAD_LEN];
        pay_load.copy_from_slice(&chunk_bytes[8..]);

        Self {
            magic_header: u16::from_le_bytes([chunk_bytes[0], chunk_bytes[1]]),
            channel_id: chunk_bytes[2],
            cmd_flag: chunk_bytes[3],
            chunk_id: u32::from_le_bytes([chunk_bytes[4], chunk_bytes[5], chunk_bytes[6], chunk_bytes[7]]),
            pay_load,
        }
    }
    
    fn to_chunk_bytes(&self) -> &[ChannelElem] {
        unsafe {
            let pointer = self as *const AudioChunk as *const u8;
            slice::from_raw_parts(pointer, size_of::<AudioChunk>())
        }
    }
}

impl AudioChunk {
    pub(crate) fn is_stop(&self) -> bool {
        self.cmd_flag == 1
    }

    pub(crate) fn get_payload(&self) -> &[ChannelElem] {
        &self.pay_load
    }
}

pub(crate) type SherpaChannel = IOChannel<DefaultIOWriter, DefaultIOReader, AudioChunk>;

impl SherpaChannel {
    pub(crate) fn create(capacity: usize) -> Self {
        IOChannelBuilder::new(capacity)
        .with_writer(DefaultIOWriter::new())
        .with_reader(DefaultIOReader::new())
        .build::<AudioChunk>()
    }
}

pub(crate) struct SherpaPipline<'a> {
    token_id: usize,
    chunk_id: u32,
    sherpa: Arc<Sherpa>,
    channel: Arc<SherpaChannel>,
    thread_pool: &'a ThreadPool,
}

impl<'a> SherpaPipline<'a> {
    pub(crate) fn new(token_id: usize, sherpa_channel: SherpaChannel, thread_pool: &'a ThreadPool) -> Self {
        Self {
            token_id,
            chunk_id: 0,
            sherpa: Arc::new(Sherpa::new()),
            channel: Arc::new(sherpa_channel),
            thread_pool,
        }
    }
}

impl<'a> SherpaPipline<'a> {
    pub(crate) fn init(&self, notify: impl FnOnce(usize, usize, String) + Copy + Send + 'static) {
        let input_channel = self.channel.clone();
        let sherpa = self.sherpa.clone();

        self.thread_pool.execute(move || {
            let handler = sherpa.init(SHERPA_TOKENS, SHERPA_ENCODER, SHERPA_DECODER, SHERPA_JOINER);

            loop {
                if let Ok(audio_chunk) = input_channel.read() {
                    if audio_chunk.is_stop() {
                        break;
                    }
                    let sample = audio_chunk.get_payload()
                        .chunks_exact(2)
                        .map(|chunk| {
                            ((chunk[1] as i16) << 8 | chunk[0] as i16 & 0xff) as f32 / 32767f32
                        })
                        .collect::<Vec<f32>>();
                    let ret = sherpa.transcribe(handler, &sample);
                    if !ret.eq("") {
                        notify(audio_chunk.channel_id as usize, audio_chunk.chunk_id as usize, ret);
                    }
                }
            }

            sherpa.close(handler);
        });
    }

    pub(crate) fn write(&mut self, received_data: &[ChannelElem], token_id: usize) -> Result<(), ChannelError> {
        let mut chunk_bytes = [0u8; size_of::<AudioChunk>()];
        chunk_bytes[0] = 0xee;
        chunk_bytes[1] = 0xff;
        chunk_bytes[2] = token_id as u8;
        let chunk_ids = (self.chunk_id as u32).to_le_bytes();
        self.chunk_id += 1;
        chunk_bytes[3] = 0 as u8;
        chunk_bytes[4] = chunk_ids[0];
        chunk_bytes[5] = chunk_ids[1];
        chunk_bytes[6] = chunk_ids[2];
        chunk_bytes[7] = chunk_ids[3];
        chunk_bytes[8..].copy_from_slice(&received_data);
        self.channel.write(AudioChunk::from_chunk_bytes(&chunk_bytes))
    }
}

fn next(current: &mut Token) -> Token {
    let next = current.0;
    current.0 += 1;
    Token(next)
}

fn would_block(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::WouldBlock
}

fn interrupted(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::Interrupted
}

pub fn run(addr: &str, thread_nums: usize, events_num: usize) -> io::Result<()> {
    let thread_pool = ThreadPool::new(thread_nums);
    let mut sherpa_piplines = Vec::<SherpaPipline>::with_capacity(thread_nums);

    for _i in 0..thread_nums {
        let input_channel = SherpaChannel::create(512 * size_of::<AudioChunk>());
        let sherpa_pipline = SherpaPipline::new(0, input_channel, &thread_pool);
        sherpa_pipline.init(|channel_id, chunk_id, result| {
            println!("{}-{}: {}", channel_id, chunk_id, result);
        });
        sherpa_piplines.push(sherpa_pipline);
    }

    const SERVER: Token = Token(0);

    let mut poll = Poll::new()?;
    let mut events = Events::with_capacity(events_num);
    let addr = addr.parse().unwrap();
    let mut server = TcpListener::bind(addr)?;
    poll.registry()
        .register(&mut server, SERVER, Interest::READABLE)?;
    let mut connections = HashMap::new();
    let mut unique_token = Token(SERVER.0 + 1);

    println!("You can connect to the server using `nc`:");
    println!(" $ nc {}", addr);
    println!("You'll see our welcome message and anything you type will be printed here.");

    loop {
        if let Err(err) = poll.poll(&mut events, None) {
            if interrupted(&err) {
                continue;
            }
            return Err(err);
        }

        for event in events.iter() {
            match event.token() {
                SERVER => loop {
                    // Received an event for the TCP server socket, which
                    // indicates we can accept an connection.
                    let (mut connection, address) = match server.accept() {
                        Ok((connection, address)) => (connection, address),
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            // If we get a `WouldBlock` error we know our
                            // listener has no more incoming connections queued,
                            // so we can return to polling and wait for some
                            // more.
                            break;
                        }
                        Err(e) => {
                            // If it was any other kind of error, something went
                            // wrong and we terminate with an error.
                            return Err(e);
                        }
                    };

                    println!("Accepted connection from: {}", address);

                    let token = next(&mut unique_token);
                    poll.registry().register(
                        &mut connection,
                        token,
                        Interest::READABLE.add(Interest::WRITABLE),
                    )?;

                    connections.insert(token, connection);
                },
                token => {
                    // Maybe received an event for a TCP connection.
                    let done = if let Some(connection) = connections.get_mut(&token) {
                        handle_connection_event(poll.registry(), connection, event, &mut sherpa_piplines)?
                    } else {
                        // Sporadic events happen, we can safely ignore them.
                        false
                    };
                    if done {
                        if let Some(mut connection) = connections.remove(&token) {
                            println!("disconnect: {:?}", connection);
                            poll.registry().deregister(&mut connection)?;
                            
                            for sherpa_pipline in sherpa_piplines.iter_mut() {
                                if sherpa_pipline.token_id == token.0 {
                                    sherpa_pipline.token_id = 0;
                                    sherpa_pipline.chunk_id = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn handle_connection_event(
    registry: &Registry,
    connection: &mut TcpStream,
    event: &Event,
    sherpa_piplines: &mut Vec::<SherpaPipline>,
) -> io::Result<bool> {
    if event.is_writable() {
        let ret = "echo";
        match connection.write(ret.as_bytes()) {
            // We want to write the entire `DATA` buffer in a single go. If we
            // write less we'll return a short write error (same as
            // `io::Write::write_all` does).
            Ok(n) if n < ret.len() => return Err(io::ErrorKind::WriteZero.into()),
            Ok(_) => {
                // After we've written something we'll reregister the connection
                // to only respond to readable events.
                registry.reregister(connection, event.token(), Interest::READABLE)?
            }
            // Would block "errors" are the OS's way of saying that the
            // connection is not actually ready to perform this I/O operation.
            Err(ref err) if would_block(err) => {}
            // Got interrupted (how rude!), we'll try again.
            Err(ref err) if interrupted(err) => {
                return handle_connection_event(registry, connection, event, sherpa_piplines)
            }
            // Other errors we'll consider fatal.
            Err(err) => return Err(err),
        }
    }

    if event.is_readable() {
        let mut connection_closed = false;
        let mut received_data = vec![0; CHUNK_PAYLOAD_LEN];
        let mut bytes_read = 0;
        // We can (maybe) read from the connection.
        loop {
            match connection.read(&mut received_data[bytes_read..]) {
                Ok(0) => {
                    // Reading 0 bytes means the other side has closed the
                    // connection or is done writing, then so are we.
                    connection_closed = true;
                    break;
                }
                Ok(n) => {
                    bytes_read += n;
                    if bytes_read == received_data.len() {
                        let select_idx = select_sherpa_pipline(sherpa_piplines, event.token().0);
                        sherpa_piplines[select_idx].write(&received_data, event.token().0).unwrap();
                        // bytes_read = 0;
                        registry.reregister(connection, event.token(), Interest::WRITABLE).unwrap();
                        break;
                    }
                }
                // Would block "errors" are the OS's way of saying that the
                // connection is not actually ready to perform this I/O operation.
                Err(ref err) if would_block(err) => break,
                Err(ref err) if interrupted(err) => continue,
                // Other errors we'll consider fatal.
                Err(err) => return Err(err),
            }
        }

        if connection_closed {
            println!("Connection closed");
            return Ok(true);
        }
    }

    Ok(false)
}

fn select_sherpa_pipline(sherpa_piplines: &mut Vec<SherpaPipline>, token_id: usize) -> usize {
    let mut selected = 0;
    for i in 0..sherpa_piplines.len() {
        if sherpa_piplines[i].token_id == token_id {
            selected = i;
            break;
        }
    }

    if selected == 0 {
        for i in 0..sherpa_piplines.len() {
            if sherpa_piplines[i].token_id == 0 {
                selected = i;
                sherpa_piplines[i].token_id = token_id;
                break;
            }
        }
    }

    selected
}