use core::slice;
use std::{mem::size_of, sync::Arc, time::Duration};

use derive_new::new;
use tokio::{io::AsyncReadExt, net::TcpListener, time::sleep};

use crate::{channel::channel::{ChannelElem, ChannelError, ChunkBytes, DefaultIOReader, DefaultIOWriter, IOChannel, IOChannelBuilder, IOChunk}, sherpa::Sherpa};

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

pub(crate) struct SherpaPipline {
    token_id: usize,
    chunk_id: u32,
    sherpa: Arc<Sherpa>,
    channel: Arc<SherpaChannel>,
}

impl SherpaPipline {
    pub(crate) fn new(token_id: usize, sherpa_channel: SherpaChannel) -> Self {
        Self {
            token_id,
            chunk_id: 0,
            sherpa: Arc::new(Sherpa::new()),
            channel: Arc::new(sherpa_channel),
        }
    }
}

impl SherpaPipline {
    pub(crate) fn init(&self, notify: impl Fn(usize, usize, String) + Send + 'static) {
        let input_channel = self.channel.clone();
        let sherpa = self.sherpa.clone();

        tokio::spawn(async move {
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

    pub(crate) fn write(&mut self, received_data: &[ChannelElem]) -> Result<(), ChannelError> {
        let mut chunk_bytes = [0u8; size_of::<AudioChunk>()];
        chunk_bytes[0] = 0xee;
        chunk_bytes[1] = 0xff;
        chunk_bytes[2] = self.token_id as u8;
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

    pub(crate) fn stop(&mut self) -> Result<(), ChannelError> {
        let mut chunk_bytes = [0u8; size_of::<AudioChunk>()];
        chunk_bytes[0] = 0xee;
        chunk_bytes[1] = 0xff;
        chunk_bytes[2] = self.token_id as u8;
        let chunk_ids = (self.chunk_id as u32).to_le_bytes();
        self.chunk_id += 1;
        chunk_bytes[3] = 1 as u8;
        chunk_bytes[4] = chunk_ids[0];
        chunk_bytes[5] = chunk_ids[1];
        chunk_bytes[6] = chunk_ids[2];
        chunk_bytes[7] = chunk_ids[3];
        self.channel.write(AudioChunk::from_chunk_bytes(&chunk_bytes))
    }
}

#[tokio::main(worker_threads = 40)]
pub async fn run(addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(addr).await?;
    let mut token_id = 0;

    loop {
        let (mut socket, _) = listener.accept().await?;
        tokio::spawn(async move {
            println!("handle {:?}", socket);
            if let Err(e) = handle_client(&mut socket, token_id).await {
                println!("error handle {:?}: {:?}", socket, e);
            }
        });
        token_id += 1;
    }
}

async fn handle_client(socket: &mut tokio::net::TcpStream, token_id: usize) -> Result<(), ChannelError> {
    let mut buff_in = [0; CHUNK_PAYLOAD_LEN];
    let mut read_len = 0;

    let mut sherpa_pipline = SherpaPipline::new(token_id, SherpaChannel::create(128 * size_of::<AudioChunk>()));
    sherpa_pipline.init(move |channel_id, chunk_id, ret| {
        println!("receive: {}-{} {}", channel_id, chunk_id, ret);
    });

    loop {
        match tokio::select! {
            read_result = socket.read(&mut buff_in[read_len..]) => {
                match read_result {
                    Ok(read_len) => Ok(read_len),
                    Err(_) => Err(()),
                }
            }
            _ = sleep(Duration::from_millis(30000)) => {
                Err(())
            }
        } {
                Ok(0)  => break,
                Ok(n) => {
                    read_len += n;
                    if read_len == CHUNK_PAYLOAD_LEN {
                        sherpa_pipline.write(&buff_in).unwrap();
                        read_len = 0;
                    }
                },
                Err(_) => {
                    eprintln!("Error reading from socket");
                    break;
                }
        }
    }
    sherpa_pipline.stop()
}