use core::slice;
use std::{mem::size_of, sync::{Arc, RwLock}, time::Duration};

use derive_new::new;

use signal_hook::{consts::{SIGINT, SIGTERM}, iterator::Signals};
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, net::TcpListener, time::sleep};

use crate::{channel::default::{ChannelElem, ChannelError, ChunkBytes, DefaultIOReader, DefaultIOWriter, IOChannel, IOChannelBuilder, IOChunk}, sherpa::{Sherpa, SherpaHandle}};

pub(crate) const CHUNK_PAYLOAD_LEN: usize = 6400;
pub(crate) const TRANSCRIBE_MAX_LEN: usize = 256;

pub(crate) const SHERPA_TOKENS: &str = "../sherpa-onnx/tokens.txt";
pub(crate) const SHERPA_ENCODER: &str = "../sherpa-onnx/encoder-epoch-20-avg-1-chunk-16-left-128.onnx";
pub(crate) const SHERPA_DECODER: &str = "../sherpa-onnx/decoder-epoch-20-avg-1-chunk-16-left-128.onnx";
pub(crate) const SHERPA_JOINER: &str = "../sherpa-onnx/joiner-epoch-20-avg-1-chunk-16-left-128.onnx";

#[derive(Debug, new)]
#[repr(C, packed)]
pub(crate) struct AudioChunk {
    magic_header: u16,
    channel_id: u8,
    end_flag: u8,
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
            end_flag: chunk_bytes[3],
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
    pub(crate) fn get_payload(&self) -> &[ChannelElem] {
        &self.pay_load
    }
}

pub(crate) type InputChannel = IOChannel<DefaultIOWriter, DefaultIOReader, AudioChunk>;

impl InputChannel {
    pub(crate) fn create(capacity: usize) -> Self {
        IOChannelBuilder::new(capacity)
        .with_writer(DefaultIOWriter::new())
        .with_reader(DefaultIOReader::new())
        .build::<AudioChunk>()
    }
}

#[derive(Debug, new)]
#[repr(C, packed)]
pub(crate) struct TranscribeResult {
    magic_header: u16,
    result_len: u8,
    result_data: [u8; TRANSCRIBE_MAX_LEN],
}

impl IOChunk for TranscribeResult {
    fn from_chunk_bytes(chunk_bytes: ChunkBytes) -> Self {
        assert!(chunk_bytes.len() == size_of::<TranscribeResult>());
        let mut pay_load = [0; TRANSCRIBE_MAX_LEN];
        pay_load.copy_from_slice(&chunk_bytes[3..]);

        Self {
            magic_header: u16::from_le_bytes([chunk_bytes[0], chunk_bytes[1]]),
            result_len: chunk_bytes[2],
            result_data: pay_load,
        }
    }

    fn to_chunk_bytes(&self) -> &[ChannelElem] {
        unsafe {
            let pointer = self as *const TranscribeResult as *const u8;
            slice::from_raw_parts(pointer, size_of::<TranscribeResult>())
        }
    }
}

impl TranscribeResult {
    pub(crate) fn get_result(&self) -> &[ChannelElem] {
        &self.result_data
    }
}

pub(crate) type OutputChannel = IOChannel<DefaultIOWriter, DefaultIOReader, TranscribeResult>;

impl OutputChannel {
    pub(crate) fn create(capacity: usize) -> Self {
        IOChannelBuilder::new(capacity)
        .with_writer(DefaultIOWriter::new())
        .with_reader(DefaultIOReader::new())
        .build::<TranscribeResult>()
    }
}

pub(crate) struct SherpaPipline {
    sherpa_id: usize,
    sherpa: Arc<Sherpa>,
    input_channel: Arc<InputChannel>,
    output_channel: Arc<OutputChannel>,
    handler: Option<Arc<SherpaHandle>>,
}

impl SherpaPipline {
    pub(crate) fn new(channel_id: usize, input_channel: InputChannel, output_channel: OutputChannel) -> Self {
        Self {
            sherpa_id: channel_id,
            sherpa: Arc::new(Sherpa::new()),
            input_channel: Arc::new(input_channel),
            output_channel: Arc::new(output_channel),
            handler: None,
        }
    }
}

impl SherpaPipline {
    pub(crate) fn is_busy(&self) -> bool {
        self.sherpa.is_busy()
    }

    pub(crate) fn set_busy(&self, available: bool) {
        self.sherpa.set_busy(available);
    }

    pub(crate) fn init(&mut self) {
        self.handler = Some(Arc::new(self.sherpa.init(
            SHERPA_TOKENS, SHERPA_ENCODER, SHERPA_DECODER, SHERPA_JOINER)));
    }

    fn create_transcribe(pay_load: &[u8]) -> TranscribeResult {
        let mut chunk_bytes = [0u8; size_of::<TranscribeResult>()];
        chunk_bytes[0] = 0xee;
        chunk_bytes[1] = 0xff;
        chunk_bytes[2] = pay_load.len() as u8;
        
        chunk_bytes[3..(pay_load.len() + 3)].copy_from_slice(pay_load);
        TranscribeResult::from_chunk_bytes(&chunk_bytes)
    }

    pub(crate) fn run(
        &self,
        on_notify: impl Fn(usize, usize, String) + Send + 'static,
        on_end: impl Fn(usize) + Send + 'static) {
        let input_channel = self.input_channel.clone();
        let output_channel = self.output_channel.clone();
        let sherpa = self.sherpa.clone();
        let sherpa_id = self.sherpa_id;
        let handler = self.handler.as_ref().unwrap().clone();

        tokio::spawn(async move {
            loop {
                if !sherpa.is_busy() {
                    on_end(sherpa_id);
                    break;
                }

                if let Ok(audio_chunk) = input_channel.read() {
                    let sample = audio_chunk.get_payload()
                        .chunks_exact(2)
                        .map(|chunk| {
                            ((chunk[1] as i16) << 8 | chunk[0] as i16 & 0xff) as f32 / 32767f32
                        })
                        .collect::<Vec<f32>>();
                    let ret = sherpa.transcribe(*handler, &sample);
                    if !ret.eq("") {
                        let res = {
                            let pay_load = ret.as_bytes();
                            println!("pay_load len: {}", pay_load.len());
                            let transcribe_result = Self::create_transcribe(pay_load);
                            output_channel.write(transcribe_result)
                        };
                        match res {
                            Ok(_) => on_notify(audio_chunk.channel_id as usize, audio_chunk.chunk_id as usize, ret),
                            Err(_) => eprintln!("write transcribe result failed"),
                        }
                    }
                } else {
                    let _ = sleep(Duration::from_millis(10)).await;
                }
            }
        });
    }

    pub(crate) async fn poll_result(&self) -> Result<TranscribeResult, ChannelError> {
        let output_channel = self.output_channel.clone();
        if let Ok(transcribe_result) = output_channel.read() {
            Ok(transcribe_result)
        } else {
            let _ = sleep(Duration::from_millis(10)).await;
            Err(ChannelError::ReadChunkFailed)
        }
    }

    pub(crate) fn write(&self, received_data: &[ChannelElem], chunk_id: u32, is_end: bool) -> Result<(), ChannelError> {
        let mut chunk_bytes = [0u8; size_of::<AudioChunk>()];
        chunk_bytes[0] = 0xee;
        chunk_bytes[1] = 0xff;
        chunk_bytes[2] = self.sherpa_id as u8;
        let chunk_ids = chunk_id.to_le_bytes();
        chunk_bytes[3] = if is_end { 1_u8 } else { 0_u8 };
        chunk_bytes[4] = chunk_ids[0];
        chunk_bytes[5] = chunk_ids[1];
        chunk_bytes[6] = chunk_ids[2];
        chunk_bytes[7] = chunk_ids[3];
        chunk_bytes[8..].copy_from_slice(received_data);
        self.input_channel.write(AudioChunk::from_chunk_bytes(&chunk_bytes))
    }

    pub(crate) fn end(&self) {
        println!("channel-{} end", self.sherpa_id);
        let sherpa = self.sherpa.clone();
        let handler = self.handler.as_ref().unwrap().clone();

        let input_channel = self.input_channel.clone();
        let output_channel = self.output_channel.clone();

        input_channel.clear();
        output_channel.clear();

        sherpa.reset(*handler);
        self.set_busy(false);
        println!("channel-{} end 2", self.sherpa_id);
    }

    pub(crate) fn stop(&self) {
        let handler = self.handler.as_ref().unwrap().clone();
        self.sherpa.close(*handler)
    }
}

#[tokio::main(worker_threads = 40)]
pub async fn run(addr: &'static str, max_clients: usize, debug: bool) -> Result<(), Box<dyn std::error::Error>> {
    let sherpa_piplines = Arc::new(RwLock::new(vec![]));

    for i in 0..max_clients {
        let mut sherpa_pipline = SherpaPipline::new(
            i,
            InputChannel::create(128 * size_of::<AudioChunk>()),
        OutputChannel::create(128 * size_of::<TranscribeResult>()));
        sherpa_pipline.init();
        println!("sherpa pipline-{} init complete", i);
        sherpa_piplines.write().unwrap().push(Arc::new(sherpa_pipline));
    }

    println!("server start complete!");

    let main_loop = {
        let sherpa_piplines = sherpa_piplines.clone();
        tokio::spawn(async move {
            let listener = TcpListener::bind(addr).await.unwrap();

            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                println!("client {:?} accepted", socket);

                let available_idx = {
                    let mut available_idx = -1;
                    let sherpa_piplines = sherpa_piplines.read().unwrap();
                    for sherpa_pipline in sherpa_piplines.iter() {
                        available_idx += 1;
                        if !sherpa_pipline.is_busy() {
                            break;
                        }
                    }
                    available_idx
                };

                if available_idx != -1 {
                    let sherpa_pipline = sherpa_piplines.read().unwrap()[available_idx as usize].clone();
                    sherpa_pipline.set_busy(true);
                    sherpa_pipline.run(
                    move |channel_id, chunk_id, ret| {
                        println!("notify: {}-{} {}", channel_id, chunk_id, ret);
                    },
                    move |sherpa_id| {
                        println!("free: {sherpa_id}");
                    });
                    
                    tokio::spawn(async move {
                        let max_error_count = 100 + if max_clients < 10 {
                            0
                        } else {
                            (((max_clients - 1) as f32 / 10.0f32) * 100.0f32) as usize
                        };

                        if let Err(e) = handle_client(&mut socket, sherpa_pipline, max_error_count, debug).await {

                            eprintln!("error handle {:?}: {:?}", socket, e);
                        };
                    });
                } else {
                    eprintln!("no sherpa_pipline availavle");
                    socket.shutdown().await.unwrap();
                }
            }
        })
    };

    let mut signals = Signals::new([SIGINT, SIGTERM]).unwrap();
    let sherpa_piplines = sherpa_piplines.clone();
    tokio::spawn(async move {
        if let Some(signal) = signals.forever().next() {
            match signal {
                SIGINT | SIGTERM => {
                    println!("Received signal {:?}, exiting gracefully.", signal);
                    for (sherpa_id, sherpa_pipline) in sherpa_piplines.read().unwrap().iter().enumerate() {
                        sherpa_pipline.stop();
                        println!("stop sherpa pipline-{}", sherpa_id);
                    }
                }
                _ => unreachable!(),
            }
        };
        main_loop.abort()
    }).await?;

    Ok(())
}

enum SherpaResult {
    Read(usize),
    Write(Box<TranscribeResult>),
}

#[derive(Debug, PartialEq)]
enum SherpaError {
    SocketReadFailed,
    TranscribeError,
    Timeout,
}

async fn handle_client(
    socket: &mut tokio::net::TcpStream,
    sherpa_pipline: Arc<SherpaPipline>,
    max_error_count: usize,
    debug: bool) -> Result<(), ChannelError> {
    println!("handle {:?}, allowed max_error_count={}", socket, max_error_count);
    let mut buff_in = [0; CHUNK_PAYLOAD_LEN];
    let mut read_len = 0;
    let mut chunk_id = 0;

    let mut error_count = 0;

    loop {
        if error_count > max_error_count {
            eprintln!("transcribe should complete, break");
            if debug {
                socket.write_all("end".as_bytes()).await.unwrap();
            }
            break;
        }
        match tokio::select! {
            read_result = socket.read(&mut buff_in[read_len..]) => {
                match read_result {
                    Ok(read_len) => Ok(SherpaResult::Read(read_len)),
                    Err(_) => Err(SherpaError::SocketReadFailed),
                }
            }
            transcribe_result = sherpa_pipline.poll_result() => {
                match transcribe_result {
                    Ok(transcribe_result) => Ok(SherpaResult::Write(Box::new(transcribe_result))),
                    Err(_) => Err(SherpaError::TranscribeError),
                }
            }
            _ = sleep(Duration::from_secs(15)) => {
                Err(SherpaError::Timeout)
            }
        } {
            Ok(SherpaResult::Read(0)) => break,
            Ok(SherpaResult::Read(n)) => {
                read_len += n;
                if read_len == CHUNK_PAYLOAD_LEN {
                    sherpa_pipline.write(&buff_in, chunk_id, false).unwrap();
                    chunk_id += 1;
                    read_len = 0;
                }
            },
            Ok(SherpaResult::Write(transcribe_result)) => {
                match socket.write_all(transcribe_result.get_result()).await {
                    Ok(_) => error_count = 0,
                    Err(_) => break,
                }
            },
            Err(e) => {
                if !e.eq(&SherpaError::TranscribeError) {
                    break;
                } else {
                    error_count += 1;
                }
            }
        }
    }

    sherpa_pipline.end();
    if debug {
        while (socket.write_all("end".as_bytes()).await).is_err() {
            sleep(Duration::from_millis(10)).await
        }
    }
    if let Err(e)  = socket.shutdown().await {
        eprintln!("error close socket: {:?}", e);
    }
    Ok(())
}