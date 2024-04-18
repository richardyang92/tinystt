use tokio::{io::{AsyncReadExt, AsyncWriteExt}, net::TcpStream};

use crate::server::{CHUNK_PAYLOAD_LEN, TRANSCRIBE_MAX_LEN};

#[tokio::main]
pub async fn run(addr: &'static str, client_nums: usize) {
    let mut futures = vec![];
    for i in 0..client_nums {
        let future = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
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
                println!("write to server [{}, {}]", s, e);
                stream.write_all(&samples[s..e]).await.unwrap();
            }

            let mut buff = [0; TRANSCRIBE_MAX_LEN];
            loop {
                match stream.read(&mut buff).await {
                    Ok(0) => break,
                    Ok(read_len) => {
                        let transcribe_data = String::from_utf8_lossy(&buff[0..read_len]);
                        println!("receive: channel-{} {}", i, transcribe_data);
                        if transcribe_data.eq("end") {
                            break;
                        }
                    },
                    Err(_) => break,
                }
            }

            stream.shutdown().await.unwrap();
        });
        futures.push(future);
    }

    for future in futures {
        if let Err(e) = future.await {
            println!("join error: {:?}", e.to_string());
        }
    }
}