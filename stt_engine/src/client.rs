use tokio::{io::AsyncWriteExt, net::TcpStream};

use crate::server::CHUNK_PAYLOAD_LEN;

#[tokio::main]
pub async fn run(addr: &'static str) {
    let mut futures = vec![];
    for i in 0..30 {
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

            stream.shutdown().await.unwrap();
        });
        futures.push(future);
    }

    for future in futures {
        future.await.unwrap();
    }
}