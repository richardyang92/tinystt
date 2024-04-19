## Tinystt

Tinystt is a new generation of speech recognition software based on Kaldi,which is developed by Rust.

It consists of the following parts: 
  1. sherpa-onnx (https://github.com/k2-fsa/sherpa-onnx.git): speech recognition engine;
  2. sherpa_native: the C interface encapsulation layer of kaldi, which is called by Rust FFI;
  3. stt_engine: a low-latency, concurrent access supported speech recognition server developed based on tokio.

Run commands:
  1. Start the server: `Cargo run server`
  2. Start the client: `Cargo run client`
