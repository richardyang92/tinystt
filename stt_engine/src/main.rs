use std::{env, io::{Read, Write}, net::TcpStream, thread};

use server::CHUNK_PAYLOAD_LEN;

mod channel;
mod server;
mod thread_pool;
mod sherpa;

fn main() {
    let args = env::args().collect::<Vec<String>>();
    if args[1].eq("client") {
        let mut futures = vec![];
        for i in 0..30 {
            let future = thread::spawn(move || {
                let mut stream = TcpStream::connect("127.0.0.1:8888").unwrap();
                let mut reader = hound::WavReader::open(format!("./data/segment/split_part_{}.wav", i + 1)).unwrap();
                let samples = reader.samples::<i16>()
                    .flat_map(|sample| match sample {
                        Ok(sample) => sample.to_le_bytes(),
                        Err(_) => [0, 0],
                    })
                    .collect::<Vec<u8>>();
                let n = samples.len() / CHUNK_PAYLOAD_LEN;

                for j in 0..n {
                    let (s, e) = (j * CHUNK_PAYLOAD_LEN, (j + 1) * CHUNK_PAYLOAD_LEN);
                    stream.write(&samples[s..e]).unwrap();
                }

                let mut read_buff = [0; CHUNK_PAYLOAD_LEN];
                loop {
                    match stream.read(&mut read_buff) {
                        Ok(n) => {
                            if n == 0 {
                                break;
                            } else {
                                println!("read: {}", String::from_utf8_lossy(&read_buff[..n]))
                            }
                        },
                        Err(e) => {
                            eprint!("{:?}", e);
                            break;
                        },
                    }
                }

                stream.shutdown(std::net::Shutdown::Both).unwrap();
            });
            futures.push(future);
        }

        for future in futures {
            future.join().unwrap();
        }
    } else if args[1].eq("server") {
        server::run("127.0.0.1:8888", 30, 128).unwrap();
    }
}
