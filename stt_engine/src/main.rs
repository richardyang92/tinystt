use std::env;

mod channel;
mod server;
mod client;
mod sherpa;

fn main() {
    let args = env::args().collect::<Vec<String>>();
    if args[1].eq("client") {
        client::run("127.0.0.1:9999");
    } else if args[1].eq("server") {
        server::run("0.0.0.0:9999").unwrap();
    }
}
