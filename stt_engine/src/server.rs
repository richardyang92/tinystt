use core::slice;
use std::{mem::size_of, sync::{Arc, RwLock}, time::Duration};

use derive_new::new;

use signal_hook::{consts::{SIGINT, SIGTERM}, iterator::Signals};
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, net::TcpListener, time::sleep};

use crate::{channel::channel::{ChannelElem, ChannelError, ChunkBytes, DefaultIOReader, DefaultIOWriter, IOChannel, IOChannelBuilder, IOChunk}, sherpa::{Sherpa, SherpaHandle}};

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
    pub(crate) fn is_end(&self) -> bool {
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
    sherpa_id: usize,
    sherpa: Arc<Sherpa>,
    channel: Arc<SherpaChannel>,
    handler: Option<Arc<SherpaHandle>>,
}

impl SherpaPipline {
    pub(crate) fn new(channel_id: usize, sherpa_channel: SherpaChannel) -> Self {
        Self {
            sherpa_id: channel_id,
            sherpa: Arc::new(Sherpa::new()),
            channel: Arc::new(sherpa_channel),
            handler: None,
        }
    }
}

impl SherpaPipline {
    pub(crate) fn init(&mut self) {
        self.handler = Some(Arc::new(self.sherpa.init(SHERPA_TOKENS, SHERPA_ENCODER, SHERPA_DECODER, SHERPA_JOINER)));
    }

    pub(crate) fn run(
        &self,
        on_notify: impl Fn(usize, usize, String) + Send + 'static,
        on_end: impl Fn(usize) + Send + 'static) {
        let input_channel = self.channel.clone();
        let sherpa = self.sherpa.clone();
        let sherpa_id = self.sherpa_id;
        let handler = self.handler.as_ref().unwrap().clone();

        tokio::spawn(async move {
            loop {
                if let Ok(audio_chunk) = input_channel.read() {
                    if audio_chunk.is_end() {
                        on_end(sherpa_id);
                        break;
                    }
                    let sample = audio_chunk.get_payload()
                        .chunks_exact(2)
                        .map(|chunk| {
                            ((chunk[1] as i16) << 8 | chunk[0] as i16 & 0xff) as f32 / 32767f32
                        })
                        .collect::<Vec<f32>>();
                    let ret = sherpa.transcribe(*handler, &sample);
                    if !ret.eq("") {
                        on_notify(audio_chunk.channel_id as usize, audio_chunk.chunk_id as usize, ret);
                    }
                }
            }
        });
    }

    pub(crate) fn write(&self, received_data: &[ChannelElem], chunk_id: u32) -> Result<(), ChannelError> {
        let mut chunk_bytes = [0u8; size_of::<AudioChunk>()];
        chunk_bytes[0] = 0xee;
        chunk_bytes[1] = 0xff;
        chunk_bytes[2] = self.sherpa_id as u8;
        let chunk_ids = chunk_id.to_le_bytes();
        chunk_bytes[3] = 0 as u8;
        chunk_bytes[4] = chunk_ids[0];
        chunk_bytes[5] = chunk_ids[1];
        chunk_bytes[6] = chunk_ids[2];
        chunk_bytes[7] = chunk_ids[3];
        chunk_bytes[8..].copy_from_slice(&received_data);
        self.channel.write(AudioChunk::from_chunk_bytes(&chunk_bytes))
    }

    pub(crate) fn end(&self) -> Result<(), ChannelError> {
        let mut chunk_bytes = [0u8; size_of::<AudioChunk>()];
        chunk_bytes[0] = 0xee;
        chunk_bytes[1] = 0xff;
        chunk_bytes[2] = self.sherpa_id as u8;
        let chunk_ids = (0 as u32).to_le_bytes();
        chunk_bytes[3] = 1 as u8;
        chunk_bytes[4] = chunk_ids[0];
        chunk_bytes[5] = chunk_ids[1];
        chunk_bytes[6] = chunk_ids[2];
        chunk_bytes[7] = chunk_ids[3];
        self.channel.write(AudioChunk::from_chunk_bytes(&chunk_bytes))
    }

    pub(crate) fn stop(&self) {
        let handler = self.handler.as_ref().unwrap().clone();
        self.sherpa.close(*handler)
    }
}

#[tokio::main(worker_threads = 40)]
pub async fn run(addr: &'static str, max_clients: usize) -> Result<(), Box<dyn std::error::Error>> {
    let sherpa_piplines = Arc::new(RwLock::new(vec![]));
    let sherpa_pipline_states = Arc::new(RwLock::new(vec![0; max_clients]));

    for i in 0..max_clients {
        let mut sherpa_pipline = SherpaPipline::new(i, SherpaChannel::create(128 * size_of::<AudioChunk>()));
        sherpa_pipline.init();
        println!("sherpa pipline-{} init complete", i);
        sherpa_piplines.write().unwrap().push(Arc::new(sherpa_pipline));
    }  

    let main_loop = {
        let sherpa_piplines = sherpa_piplines.clone();
        tokio::spawn(async move {
            let listener = TcpListener::bind(addr).await.unwrap();

            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                println!("client {:?} accepted", socket);
        
                let available_idx = {
                    let mut available_idx = -1;
                    for (i, sherpa_pipline_state) in sherpa_pipline_states.read().unwrap().iter().enumerate() {
                        if sherpa_pipline_state.eq(&0) {
                            available_idx = i as isize;
                            break;
                        }
                    }
                    available_idx
                };
        
                if available_idx == -1 {
                    eprintln!("no sherpa_pipline availavle");
                    socket.shutdown().await.unwrap();
                } else {
                    sherpa_pipline_states.write().unwrap()[available_idx as usize] = 1;
                    if let Some(&ref sherpa_pipline) = sherpa_piplines.read().unwrap().get(available_idx as usize) {
                        let sherpa_pipline = sherpa_pipline.clone();
        
                        sherpa_pipline.run(
                            move |channel_id, chunk_id, ret| {
                                println!("notify: {}-{} {}", channel_id, chunk_id, ret);
                            },
                        {
                            let sherpa_pipline_states = sherpa_pipline_states.clone();
                            move |sherpa_id| {
                                println!("end: {sherpa_id}");
                                sherpa_pipline_states.write().unwrap()[sherpa_id] = 0;
                            }
                        });
                        
                        tokio::spawn(async move {
                            println!("handle {:?}", socket);
                            if let Err(e) = handle_client(&mut socket, sherpa_pipline).await {
                                println!("error handle {:?}: {:?}", socket, e);
                            };
                        });
                    }
                }
            }
        })
    };

    let mut signals = Signals::new(&[SIGINT, SIGTERM]).unwrap();
    let sherpa_piplines = sherpa_piplines.clone();
    tokio::spawn(async move {
        for signal in signals.forever() {
            match signal {
                SIGINT | SIGTERM => {
                    println!("Received signal {:?}, exiting gracefully.", signal);
                    let mut sherpa_id = 0;
                    for sherpa_pipline in sherpa_piplines.read().unwrap().iter() {
                        sherpa_pipline.stop();
                        println!("stop sherpa pipline-{}", sherpa_id);
                        sherpa_id += 1;
                    }
                    break;
                }
                _ => unreachable!(),
            }
        };
        main_loop.abort()
    }).await?;

    Ok(())
}

async fn handle_client(socket: &mut tokio::net::TcpStream, sherpa_pipline: Arc<SherpaPipline>) -> Result<(), ChannelError> {
    let mut buff_in = [0; CHUNK_PAYLOAD_LEN];
    let mut read_len = 0;
    let mut chunk_id = 0;

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
                    sherpa_pipline.write(&buff_in, chunk_id).unwrap();
                    chunk_id += 1;
                    read_len = 0;
                }
            },
            Err(_) => {
                eprintln!("Error reading from socket");
                break;
            }
        }
    }
    sherpa_pipline.end()
}